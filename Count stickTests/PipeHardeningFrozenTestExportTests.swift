import XCTest
import UIKit
import ImageIO
import CryptoKit
@testable import Count_stick

/// Separate, explicitly authorized holdout export. This never loads a model.
/// The original train/val export helper and production preprocessing stay unchanged.
@MainActor
final class PipeHardeningFrozenTestExportTests: XCTestCase {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .standardizedFileURL.resolvingSymlinksInPath()
    }

    func testExportAuthorizedFrozenTestPhotos() throws {
        let requestURL = repository.appendingPathComponent("output/pipes-hardening-20261026/test-preprocessing-request.json")
        guard FileManager.default.fileExists(atPath: requestURL.path) else {
            throw XCTSkip("No authorized frozen-test preprocessing request exists.")
        }
        let pointerURL = repository.appendingPathComponent("output/pipes-hardening-20261026/preprocessing/latest-frozen_test_photo.json")
        guard !FileManager.default.fileExists(atPath: pointerURL.path) else {
            throw XCTSkip("Frozen-test inputs were already exported; preserve the original export.")
        }
        let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: requestURL))
        try require(request.status == "AUTHORIZED_FROZEN_TEST_PREPROCESSING_REQUEST" && request.testContentIncluded)
        let protocolURL = try resolve("output/pipes-hardening-20261026/model/test/pre-test-protocol.json")
        let protocolBytes = try Data(contentsOf: protocolURL)
        try require(sha(protocolBytes) == request.preTestProtocolSHA256)
        let registration = try object(protocolBytes)
        try require(registration["status"] as? String == "FROZEN_SINGLE_HARDENING_TEST_BEFORE_ACCESS")
        let bindings = try XCTUnwrap(registration["bindings"] as? [String: Any])
        let runtimeVersion = try XCTUnwrap(bindings["runtimeVersion"] as? Int)
        try require(runtimeVersion == 3)
        if runtimeVersion == 3 {
            let nms = try XCTUnwrap(bindings["runtimeNMS"] as? [String: String])
            let path = try XCTUnwrap(nms["path"])
            try require(try sha(Data(contentsOf: resolve(path))) == nms["sha256"])
        }
        let expectedCount = try XCTUnwrap(bindings["expectedTestImages"] as? Int)
        try require(expectedCount > 0 && expectedCount <= 100 && request.images.count == expectedCount)
        try require(bindings["inputVariants"] as? [String] == ["gallery", "camera"])
        try require(bindings["sealedManifestPath"] as? String == request.sourceManifestPath)
        try require(bindings["sealedManifestSHA256"] as? String == request.sourceManifestSHA256)
        let helperSHA = try sha(Data(contentsOf: URL(fileURLWithPath: #filePath)))
        try require(bindings["testExportHelperSHA256"] as? String == helperSHA && request.testExportHelperSHA256 == helperSHA)
        try require(bindings["testExportHelperPath"] as? String == "Count stickTests/PipeHardeningFrozenTestExportTests.swift")
        let approved = try XCTUnwrap(bindings["appSourceCodeSHA256"] as? [String: String])
        let sourceFiles = ["PhotoProcessing": "Count stick/StickDetector.swift", "PipeLetterbox": "Count stick/PipeDetector.swift",
                           "ExportHelper": "Count stickTests/PipePreprocessingExportTests.swift", "NMS": "Count stick/PipeRawPostprocessor.swift"]
        try require(approved == request.approvedAppSourceCodeSHA256)
        for (name, path) in sourceFiles {
            try require(try sha(Data(contentsOf: resolve(path))) == approved[name])
        }
        let code = try XCTUnwrap(bindings["codeSHA256"] as? [String: String])
        for (path, hash) in code { try require(try sha(Data(contentsOf: resolve(path))) == hash) }
        let releaseCode = try XCTUnwrap(bindings["releaseCodeSHA256"] as? [String: String])
        for (path, hash) in releaseCode { try require(try sha(Data(contentsOf: resolve(path))) == hash) }
        for (pathKey, hashKey) in [("rootApprovalPath", "rootApprovalSHA256"),
                                   ("parityReportPath", "parityReportSHA256")] {
            let path = try XCTUnwrap(bindings[pathKey] as? String)
            try require(try sha(Data(contentsOf: resolve(path))) == bindings[hashKey] as? String)
        }
        let rootReceipt = try object(Data(contentsOf: resolve(try XCTUnwrap(bindings["rootApprovalPath"] as? String))))
        try require(rootReceipt["status"] as? String == "APPROVED_HARDENING_FRESH_TEST_SINGLE_CANDIDATE")
        try require(try sha(Data(contentsOf: resolve("ml/pipes-hardening-20261026/model/evaluate_fresh_test.py"))) == bindings["evaluatorSHA256"] as? String)
        try require(try sha(Data(contentsOf: resolve("output/pipes-hardening-20261026/model/selection.json"))) == bindings["selectionSHA256"] as? String)
        try require(try sha(Data(contentsOf: resolve("output/pipes-hardening-20261026/model/protocol.json"))) == bindings["protocolSHA256"] as? String)
        try require(!FileManager.default.fileExists(atPath: repository.appendingPathComponent("output/pipes-hardening-20261026/model/test/test-opened.json").path))
        // All approval, code and source hashes above are checked before opening the sealed manifest or photos.
        let manifestBytes = try Data(contentsOf: resolve(request.sourceManifestPath))
        try require(sha(manifestBytes) == request.sourceManifestSHA256)
        let manifest = try object(manifestBytes)
        try require(manifest["status"] as? String == "FROZEN_ANNOTATION_QA_PASSED")
        let rows = try XCTUnwrap(manifest["images"] as? [[String: Any]]).filter { $0["split"] as? String == "test" }
        try require(rows.count == expectedCount && Set(request.images.map(\.id)).count == expectedCount)
        for entry in request.images {
            try require(entry.split == "test" && !entry.id.isEmpty && entry.id.allSatisfy {
                $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "_" || $0 == "-")
            })
            try require(rows.filter { $0["id"] as? String == entry.id && $0["imagePath"] as? String == entry.imagePath &&
                $0["sha256"] as? String == entry.sha256 && $0["width"] as? Int == entry.width && $0["height"] as? Int == entry.height }.count == 1)
        }
        let directory = repository.appendingPathComponent("output/pipes-hardening-20261026/preprocessing/frozen-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var records: [[String: Any]] = []
        for entry in request.images {
            let data = try Data(contentsOf: resolve(entry.imagePath))
            try require(sha(data) == entry.sha256)
            records += try autoreleasepool { try export(data, entry: entry, request: request, directory: directory) }
        }
        try require(records.count == 2 * expectedCount)
        let receipt: [String: Any] = [
            "status": "APP_PREPROCESSING_PIXELS_EXPORTED_NO_MODEL_INFERENCE", "kind": "frozen_test_photo",
            "createdAt": ISO8601DateFormatter().string(from: Date()), "iOSVersion": UIDevice.current.systemVersion,
            "preTestProtocolSHA256": request.preTestProtocolSHA256, "sealedManifestSHA256": request.sourceManifestSHA256,
            "sourceCodeSHA256": approved, "testExportHelperSHA256": helperSHA,
            "sourceCodeHashMeaning": "The production preprocessing, original train/val helper and shared NMS hashes are frozen lineage; this dedicated hardening test helper executed the export.",
            "records": records, "modelInferencePerformed": false,
            "limitations": ["Actual App gallery/camera preprocessing pixels; no predictions or performance claims.",
                            "Use these exact PNGs; do not independently resize source JPEGs."]
        ]
        let bytes = try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
        try bytes.write(to: directory.appendingPathComponent("manifest.json"), options: .withoutOverwriting)
        let pointer = ["path": relative(directory.appendingPathComponent("manifest.json")), "sha256": sha(bytes)]
        try JSONSerialization.data(withJSONObject: pointer, options: [.prettyPrinted, .sortedKeys])
            .write(to: pointerURL, options: .withoutOverwriting)
        print("PIPE_FROZEN_TEST_PREPROCESSING_EXPORT \(relative(directory.appendingPathComponent("manifest.json")))")
    }

    private func export(_ data: Data, entry: Request.Entry, request: Request, directory: URL) throws -> [[String: Any]] {
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] ?? [:]
        try require(properties[kCGImagePropertyPixelWidth as String] as? Int == entry.width)
        try require(properties[kCGImagePropertyPixelHeight as String] as? Int == entry.height)
        let orientation = properties[kCGImagePropertyOrientation as String] as? Int ?? 1
        try require(orientation == 1)
        let cameraImage = try XCTUnwrap(UIImage(data: data))
        let paths: [(String, UIImage)] = [("imageio-import", try PhotoProcessing.load(data: data)),
                                         ("camera-normalized", try PhotoProcessing.normalized(cameraImage))]
        var records: [[String: Any]] = []
        for (name, upright) in paths {
            let photo = try XCTUnwrap(upright.cgImage)
            try require(upright.imageOrientation == .up && max(photo.width, photo.height) <= 2048)
            let geometry = try PipeLetterbox(width: photo.width, height: photo.height)
            let network = geometry.prepare(upright)
            let networkImage = try XCTUnwrap(network.cgImage)
            try require(networkImage.width == 1024 && networkImage.height == 1024)
            let preparedPNG = try XCTUnwrap(upright.pngData()), networkPNG = try XCTUnwrap(network.pngData())
            let photoURL = directory.appendingPathComponent("\(entry.id)-\(name)-upright.png")
            let networkURL = directory.appendingPathComponent("\(entry.id)-\(name)-network.png")
            try preparedPNG.write(to: photoURL, options: .withoutOverwriting)
            try networkPNG.write(to: networkURL, options: .withoutOverwriting)
            records.append([
                "kind": "frozen_test_photo", "split": "test", "id": entry.id, "appPath": name,
                "sourceImagePath": entry.imagePath, "sourceManifestSHA256": request.sourceManifestSHA256,
                "sourceSHA256": sha(data), "sourcePixelWidth": entry.width, "sourcePixelHeight": entry.height,
                "sourceEXIFOrientation": orientation, "preparedWidth": photo.width, "preparedHeight": photo.height,
                "preparedPNG": relative(photoURL), "preparedSHA256": sha(preparedPNG),
                "networkPNG": relative(networkURL), "networkSHA256": sha(networkPNG),
                "networkSize": geometry.size, "gain": geometry.gain, "resizedWidth": geometry.resizedWidth,
                "resizedHeight": geometry.resizedHeight, "left": geometry.left, "top": geometry.top,
                "right": geometry.size - geometry.resizedWidth - geometry.left,
                "bottom": geometry.size - geometry.resizedHeight - geometry.top,
                "paddingRGB": [114, 114, 114], "outputOrientation": "up"
            ])
        }
        return records
    }

    private func resolve(_ path: String) throws -> URL {
        let url = repository.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        try require(!path.hasPrefix("/") && url.path.hasPrefix(repository.path + "/"))
        return url
    }
    private func object(_ data: Data) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any]) }
    private func relative(_ url: URL) -> String { String(url.path.dropFirst(repository.path.count + 1)) }
    private func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private func require(_ condition: Bool) throws { if !condition { throw ExportError.invalidAuthorizationOrInput } }
    private enum ExportError: Error { case invalidAuthorizationOrInput }
    private struct Request: Decodable {
        let status: String
        let testContentIncluded: Bool
        let preTestProtocolSHA256: String
        let sourceManifestPath: String
        let sourceManifestSHA256: String
        let testExportHelperSHA256: String
        let approvedAppSourceCodeSHA256: [String: String]
        let images: [Entry]
        struct Entry: Decodable {
            let id: String
            let split: String
            let imagePath: String
            let sha256: String
            let width: Int
            let height: Int
        }
    }
}
