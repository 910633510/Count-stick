import XCTest
import UIKit
import ImageIO
import UniformTypeIdentifiers
import CryptoKit
@testable import Count_stick

/// Exports actual App pixels for a separate Torch/CoreML comparison. This helper
/// performs no model inference and never describes synthetic shapes as pipe data.
@MainActor
final class PipePreprocessingExportTests: XCTestCase {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    func testExportSyntheticPreprocessingFixtures() throws {
        let directory = try makeDirectory(prefix: "synthetic")
        var records: [[String: Any]] = []
        for (name, width, height, orientation) in [
            ("landscape", 4097, 1873, 1), ("portrait", 1873, 4097, 1),
            ("odd-size", 1101, 719, 1), ("exif-right", 4097, 1873, 6),
            ("exif-mirrored", 1873, 4097, 2)
        ] {
            let format = UIGraphicsImageRendererFormat()
            format.scale = 1; format.opaque = true; format.preferredRange = .standard
            let synthetic = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format).image { context in
                UIColor(white: 0.8, alpha: 1).setFill()
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
                for row in 0..<7 {
                    for column in 0..<9 {
                        UIColor(hue: CGFloat(row * 9 + column) / 63, saturation: 0.8, brightness: 0.8, alpha: 1).setFill()
                        context.fill(CGRect(x: column * width / 9, y: row * height / 7,
                                            width: max(1, width / 18), height: max(1, height / 14)))
                    }
                }
                UIColor.red.setFill()
                context.fill(CGRect(x: 0, y: 0, width: 33, height: 47))
            }
            let bytes = NSMutableData()
            let destination = try XCTUnwrap(CGImageDestinationCreateWithData(bytes, UTType.jpeg.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(destination, try XCTUnwrap(synthetic.cgImage),
                                      [kCGImagePropertyOrientation: orientation,
                                       kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
            XCTAssertTrue(CGImageDestinationFinalize(destination))
            let data = bytes as Data
            try data.write(to: directory.appendingPathComponent(name + "-source.jpg"), options: .atomic)
            records += try export(data: data, id: name, directory: directory,
                                  provenance: ["kind": "synthetic_geometry_only", "sourceEXIFOrientation": orientation])
        }
        XCTAssertEqual(records.count, 10)
        try receipt(records: records, directory: directory, kind: "synthetic_geometry_only")
    }

    func testExportRequestedTrainValPreprocessingFixtures() throws {
        let requestURL = repository.appendingPathComponent("output/pipes-20261026/preprocessing-request.json")
        guard FileManager.default.fileExists(atPath: requestURL.path) else {
            throw XCTSkip("No explicit train/val preprocessing request is present.")
        }
        let request = try JSONDecoder().decode(ExportRequest.self, from: Data(contentsOf: requestURL))
        XCTAssertEqual(request.status, "FROZEN_TRAIN_VAL_PREPROCESSING_REQUEST")
        XCTAssertFalse(request.testContentIncluded)
        guard request.status == "FROZEN_TRAIN_VAL_PREPROCESSING_REQUEST", !request.testContentIncluded,
              !request.images.isEmpty, request.images.count <= 100,
              request.sourceManifestPath.contains("train-val") else { throw ExportError.invalidRequest }
        let manifestURL = try resolve(request.sourceManifestPath)
        let manifestBytes = try Data(contentsOf: manifestURL)
        guard sha(manifestBytes) == request.sourceManifestSHA256 else { throw ExportError.invalidRequest }
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: manifestBytes) as? [String: Any])
        guard manifest["testContentIncluded"] as? Bool == false,
              let rows = manifest["images"] as? [[String: Any]],
              rows.allSatisfy({ ["train", "val"].contains($0["split"] as? String ?? "") }),
              Set(request.images.map(\.id)).count == request.images.count else { throw ExportError.invalidRequest }
        // Validate every requested identity/hash/split before opening any photo.
        for entry in request.images {
            guard ["train", "val"].contains(entry.split),
                  !entry.id.isEmpty, entry.id.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "_" || $0 == "-") }),
                  rows.filter({ $0["id"] as? String == entry.id && $0["split"] as? String == entry.split &&
                      $0["imagePath"] as? String == entry.imagePath && $0["sha256"] as? String == entry.sha256 }).count == 1 else {
                throw ExportError.invalidRequest
            }
        }
        let directory = try makeDirectory(prefix: "train-val")
        var records: [[String: Any]] = []
        for entry in request.images {
            let bytes = try Data(contentsOf: resolve(entry.imagePath))
            guard sha(bytes) == entry.sha256 else { throw ExportError.invalidRequest }
            records += try autoreleasepool {
                try export(data: bytes, id: entry.id, directory: directory,
                           provenance: ["kind": "frozen_train_val_photo", "split": entry.split,
                                        "sourceImagePath": entry.imagePath, "sourceManifestSHA256": request.sourceManifestSHA256])
            }
        }
        try receipt(records: records, directory: directory, kind: "frozen_train_val_photo")
    }

    private func export(data: Data, id: String, directory: URL, provenance: [String: Any]) throws -> [[String: Any]] {
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] ?? [:]
        let cameraImage = try XCTUnwrap(UIImage(data: data))
        let paths: [(String, UIImage)] = [
            ("imageio-import", try PhotoProcessing.load(data: data)),
            ("camera-normalized", try PhotoProcessing.normalized(cameraImage))
        ]
        var records: [[String: Any]] = []
        for (name, upright) in paths {
            let original = try XCTUnwrap(upright.cgImage)
            XCTAssertEqual(upright.imageOrientation, .up)
            XCTAssertLessThanOrEqual(max(original.width, original.height), 2048)
            let geometry = try PipeLetterbox(width: original.width, height: original.height)
            let network = geometry.prepare(upright)
            let networkImage = try XCTUnwrap(network.cgImage)
            XCTAssertEqual(networkImage.width, 1024); XCTAssertEqual(networkImage.height, 1024)
            let preparedPNG = try XCTUnwrap(upright.pngData()), networkPNG = try XCTUnwrap(network.pngData())
            let preparedURL = directory.appendingPathComponent("\(id)-\(name)-upright.png")
            let networkURL = directory.appendingPathComponent("\(id)-\(name)-network.png")
            try preparedPNG.write(to: preparedURL, options: .atomic)
            try networkPNG.write(to: networkURL, options: .atomic)
            var record = provenance
            record.merge([
                "id": id, "appPath": name, "sourceSHA256": sha(data),
                "sourcePixelWidth": properties[kCGImagePropertyPixelWidth as String] ?? 0,
                "sourcePixelHeight": properties[kCGImagePropertyPixelHeight as String] ?? 0,
                "sourceEXIFOrientation": properties[kCGImagePropertyOrientation as String] ?? 1,
                "preparedWidth": original.width, "preparedHeight": original.height,
                "preparedPNG": relative(preparedURL), "preparedSHA256": sha(preparedPNG),
                "networkPNG": relative(networkURL), "networkSHA256": sha(networkPNG),
                "networkSize": geometry.size, "gain": geometry.gain,
                "resizedWidth": geometry.resizedWidth, "resizedHeight": geometry.resizedHeight,
                "left": geometry.left, "top": geometry.top,
                "right": geometry.size - geometry.resizedWidth - geometry.left,
                "bottom": geometry.size - geometry.resizedHeight - geometry.top,
                "paddingRGB": [114, 114, 114], "outputOrientation": "up"
            ]) { _, new in new }
            records.append(record)
        }
        return records
    }

    private func receipt(records: [[String: Any]], directory: URL, kind: String) throws {
        let record: [String: Any] = [
            "status": "APP_PREPROCESSING_PIXELS_EXPORTED_NO_MODEL_INFERENCE", "kind": kind,
            "createdAt": ISO8601DateFormatter().string(from: Date()), "iOSVersion": UIDevice.current.systemVersion,
            "sourceCodeSHA256": [
                "PhotoProcessing": try sha(Data(contentsOf: repository.appendingPathComponent("Count stick/StickDetector.swift"))),
                "PipeLetterbox": try sha(Data(contentsOf: repository.appendingPathComponent("Count stick/PipeDetector.swift"))),
                "ExportHelper": try sha(Data(contentsOf: URL(fileURLWithPath: #filePath)))
            ], "records": records,
            "limitations": ["No model predictions or accuracy validation performed.",
                            "Use these exact decoded PNG pixels for Torch/CoreML network parity; do not independently resize the original JPEG.",
                            "ImageIO import and camera normalization may produce different resampling pixels/rounded sizes; report each path separately.",
                            "Synthetic color grids are geometry fixtures, not real empty-dish or pipe-negative training images."]
        ]
        let bytes = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
        try bytes.write(to: directory.appendingPathComponent("manifest.json"), options: .atomic)
        let pointer = ["path": relative(directory.appendingPathComponent("manifest.json")), "sha256": sha(bytes)]
        try JSONSerialization.data(withJSONObject: pointer, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.deletingLastPathComponent().appendingPathComponent("latest-\(kind).json"), options: .atomic)
        print("PIPE_PREPROCESSING_EXPORT \(relative(directory.appendingPathComponent("manifest.json")))")
    }

    private func makeDirectory(prefix: String) throws -> URL {
        let output = repository.appendingPathComponent("output/pipes-20261026/preprocessing/\(prefix)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        return output
    }
    private func resolve(_ path: String) throws -> URL {
        let url = repository.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        guard !path.hasPrefix("/"), url.path.hasPrefix(repository.path + "/output/pipes-20261026/") else {
            throw ExportError.invalidRequest
        }
        return url
    }
    private func relative(_ url: URL) -> String { String(url.path.dropFirst(repository.path.count + 1)) }
    private func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private enum ExportError: Error { case invalidRequest }
    private struct ExportRequest: Decodable {
        let status: String
        let testContentIncluded: Bool
        let sourceManifestPath: String
        let sourceManifestSHA256: String
        let images: [Entry]
        struct Entry: Decodable {
            let id: String
            let split: String
            let imagePath: String
            let sha256: String
        }
    }
}
