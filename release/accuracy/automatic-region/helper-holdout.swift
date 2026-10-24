import Foundation
import CoreML
import Vision
import ImageIO
import CryptoKit

// Compile together with the actual production AutoCountRegion.swift.
// This file performs one full-image model inference, then applies the shared selector.
private struct HoldoutInput: Decodable {
    struct Source: Decodable { let path: String; let sha256: String }
    struct Photo: Decodable { let id: String; let path: String; let sha256: String; let width: Int; let height: Int }
    struct RawSettings: Codable, Equatable {
        let confidenceThreshold: Double
        let iouThreshold: Double
        let cropScale: String
        let maximumDimension: Int
        let computeUnits: String
    }
    struct OCRSettings: Codable, Equatable {
        let recognitionLevel: String
        let recognitionLanguages: [String]
        let usesLanguageCorrection: Bool
        let minimumTextHeight: Double
        let minimumCandidateConfidence: Double
        let minimumAlphanumericCharacters: Int
        let candidateRank: Int
        let revisionMode: String
        let failurePolicy: String
    }
    let algorithmFrozen: Bool
    let labelsFrozen: Bool
    let splitLocked: Bool
    let manifestSha256: String
    let checkpointSha256: String
    let compiledHelperSha256: String
    let modelPath: String
    let modelFileHashes: [String: String]
    let sourceFiles: [String: Source]
    let rawSettings: RawSettings
    let ocrSettings: OCRSettings
    let images: [Photo]
}

private final class HoldoutThresholds: MLFeatureProvider {
    var featureNames: Set<String> { ["confidenceThreshold", "iouThreshold"] }
    func featureValue(for name: String) -> MLFeatureValue? {
        switch name {
        case "confidenceThreshold": return MLFeatureValue(double: 0.25)
        case "iouThreshold": return MLFeatureValue(double: 0.45)
        default: return nil
        }
    }
}

private func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
private func fileDigest(_ path: String) throws -> String {
    digest(try Data(contentsOf: URL(fileURLWithPath: path)))
}
private func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}
private func boxObject(_ box: AutoCountBox) -> [String: Double] {
    ["x": box.x, "y": box.y, "width": box.width, "height": box.height]
}
private func pixelBox(_ box: AutoCountBox, width: Int, height: Int) -> [Double] {
    [box.x * Double(width), box.y * Double(height), box.width * Double(width), box.height * Double(height)]
}
private func codableObject<T: Encodable>(_ value: T) throws -> Any {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
}

private func recognizeText(_ image: CGImage, settings: HoldoutInput.OCRSettings) -> ([AutoCountTextArea], [String: Any]) {
    // Kept aligned with StickDetector.recognizedTextAreas; strings are used transiently only.
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = settings.recognitionLanguages
    request.usesLanguageCorrection = false
    request.minimumTextHeight = Float(settings.minimumTextHeight)
    let started = Date()
    do {
        try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    } catch {
        let error = error as NSError
        return ([], ["status": "failed_empty_areas", "effectiveRevision": request.revision,
                     "elapsedSeconds": Date().timeIntervalSince(started),
                     "errorDomain": error.domain, "errorCode": error.code])
    }
    let areas = (request.results ?? []).compactMap { observation -> AutoCountTextArea? in
        guard let candidate = observation.topCandidates(1).first,
              candidate.confidence >= Float(settings.minimumCandidateConfidence) else { return nil }
        let characterCount = candidate.string.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.count
        guard characterCount >= settings.minimumAlphanumericCharacters else { return nil }
        let box = observation.boundingBox.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard !box.isNull, box.width > 0, box.height > 0 else { return nil }
        return AutoCountTextArea(box: AutoCountBox(x: box.minX, y: 1 - box.maxY,
                                                  width: box.width, height: box.height),
                                 confidence: Double(candidate.confidence), characterCount: characterCount)
    }
    return (areas, ["status": "completed", "effectiveRevision": request.revision,
                    "elapsedSeconds": Date().timeIntervalSince(started)])
}

@main
private enum HoldoutEvaluation {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else { fail("Use evaluate-holdout-v3.py with a frozen algorithm checkpoint.") }
        let input = try JSONDecoder().decode(HoldoutInput.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        guard input.algorithmFrozen && input.labelsFrozen && input.splitLocked else { fail("Algorithm, labels, and split must be frozen before inference.") }
        guard input.rawSettings.confidenceThreshold == 0.25, input.rawSettings.iouThreshold == 0.45,
              input.rawSettings.cropScale == "scaleFill", input.rawSettings.maximumDimension == 2048,
              input.rawSettings.computeUnits == "cpuOnly" else { fail("Unexpected raw build2 settings.") }
        guard input.ocrSettings.recognitionLevel == "accurate",
              input.ocrSettings.recognitionLanguages == ["zh-Hans", "zh-Hant", "en-US"],
              !input.ocrSettings.usesLanguageCorrection, input.ocrSettings.minimumTextHeight == 0.015,
              input.ocrSettings.minimumCandidateConfidence == 0.8,
              input.ocrSettings.minimumAlphanumericCharacters == 2, input.ocrSettings.candidateRank == 1,
              input.ocrSettings.revisionMode == "system_default",
              input.ocrSettings.failurePolicy == "empty_text_areas_then_shared_selector" else { fail("Unexpected OCR settings.") }
        guard try fileDigest(CommandLine.arguments[0]) == input.compiledHelperSha256 else { fail("Compiled helper differs from receipt.") }
        for source in input.sourceFiles.values {
            guard try fileDigest(source.path) == source.sha256 else { fail("Bound source changed before inference.") }
        }
        for (relative, expected) in input.modelFileHashes {
            let path = URL(fileURLWithPath: input.modelPath).appendingPathComponent(relative).path
            guard try fileDigest(path) == expected else { fail("Model changed before inference.") }
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let compiledModel = try MLModel.compileModel(at: URL(fileURLWithPath: input.modelPath))
        let model = try MLModel(contentsOf: compiledModel, configuration: configuration)
        let vision = try VNCoreMLModel(for: model)
        let thresholds = HoldoutThresholds()
        vision.featureProvider = thresholds
        var rows: [[String: Any]] = []
        for photo in input.images {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: photo.path))
            guard digest(bytes) == photo.sha256 else { fail("Frozen photo hash changed.") }
            guard let source = CGImageSourceCreateWithData(bytes as CFData, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
                  let rawWidth = properties[kCGImagePropertyPixelWidth as String] as? Int,
                  let rawHeight = properties[kCGImagePropertyPixelHeight as String] as? Int else { fail("Unreadable photo.") }
            let orientation = properties[kCGImagePropertyOrientation as String] as? Int ?? 1
            let rotated = [5, 6, 7, 8].contains(orientation)
            let width = rotated ? rawHeight : rawWidth, height = rotated ? rawWidth : rawHeight
            guard width == photo.width && height == photo.height else { fail("Upright label dimensions differ.") }
            guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 2048,
                kCGImageSourceShouldCacheImmediately: true
            ] as CFDictionary) else { fail("Image normalization failed.") }
            let request = VNCoreMLRequest(model: vision)
            request.imageCropAndScaleOption = .scaleFill
            let modelStarted = Date()
            try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
            let modelElapsed = Date().timeIntervalSince(modelStarted)
            guard let observations = request.results as? [VNRecognizedObjectObservation] else { fail("Unexpected model output.") }
            let detections: [AutoCountDetection] = observations.enumerated().compactMap { index, observation in
                let rect = observation.boundingBox
                guard rect.midX.isFinite, rect.midY.isFinite, rect.width > 0, rect.height > 0 else { return nil }
                // Stable per-image IDs affect neither production selector distances nor thresholds.
                let id = UUID(uuidString: String(format: "00000000-0000-4000-8000-%012llx", UInt64(index + 1)))!
                return AutoCountDetection(id: id, box: AutoCountBox(x: rect.minX, y: 1 - rect.maxY,
                    width: rect.width, height: rect.height), confidence: Double(observation.confidence))
            }
            let ids = Dictionary(uniqueKeysWithValues: detections.enumerated().map { ($0.element.id, "det-\($0.offset)") })
            let raw: [[String: Any]] = detections.map { detection in
                ["id": ids[detection.id]!, "uuid": detection.id.uuidString,
                 "x": min(Double(width), max(0, detection.box.centerX * Double(width))),
                 "y": min(Double(height), max(0, detection.box.centerY * Double(height))),
                 "bbox": pixelBox(detection.box, width: width, height: height),
                 "normalizedBox": boxObject(detection.box), "confidence": detection.confidence]
            }
            let (textAreas, textStatus) = recognizeText(image, settings: input.ocrSettings)
            let regionStarted = Date()
            let selection = AutoCountRegion.select(detections: detections, textAreas: textAreas,
                                                    imageAspectRatio: Double(image.width) / Double(image.height))
            let regionElapsed = Date().timeIntervalSince(regionStarted)
            let kept = Set(selection.keptIDs)
            let filtered = zip(detections, raw).filter { kept.contains($0.0.id) }.map { $0.1 }
            let excluded = selection.excluded.map { ["id": ids[$0.id]!, "reason": $0.reason.rawValue] }
            let selectionData: [String: Any] = ["keptIDs": selection.keptIDs.map { ids[$0]! },
                "excluded": excluded, "regions": selection.regions.map(boxObject),
                "pixelRegions": selection.regions.map { pixelBox($0, width: width, height: height) },
                "fallbackReason": selection.fallbackReason.rawValue, "elapsedSeconds": regionElapsed]
            var ocr = textStatus
            ocr["areas"] = textAreas.map { ["box": boxObject($0.box), "confidence": $0.confidence,
                                            "characterCount": $0.characterCount] as [String: Any] }
            let variants: [String: Any] = [
                "raw_build2": ["predictions": raw, "confidenceThreshold": 0.25, "iouThreshold": 0.45,
                               "cropScale": "scaleFill", "elapsedSeconds": modelElapsed],
                "automatic_region": ["predictions": filtered, "confidenceThreshold": 0.25, "iouThreshold": 0.45,
                                      "cropScale": "scaleFill", "additionalModelInferences": 0]
            ]
            rows.append(["imageId": photo.id, "imageSha256": photo.sha256, "width": width, "height": height,
                         "sourceOrientation": orientation, "processedPixels": [image.width, image.height],
                         "modelInferenceCount": 1, "ocr": ocr, "selection": selectionData, "variants": variants])
            FileHandle.standardError.write(Data(("Evaluated frozen v3 image once: \(photo.id)\n").utf8))
        }
        let output: [String: Any] = ["schemaVersion": 1, "manifestSha256": input.manifestSha256,
            "checkpointSha256": input.checkpointSha256, "compiledHelperSha256": input.compiledHelperSha256,
            "sourceDetectorSha256": input.sourceFiles["detector"]!.sha256,
            "autoCountRegionSha256": input.sourceFiles["region"]!.sha256,
            "modelFileHashes": input.modelFileHashes, "sourceFiles": input.sourceFiles.mapValues { ["path": $0.path, "sha256": $0.sha256] },
            "rawSettings": try codableObject(input.rawSettings), "ocrSettings": try codableObject(input.ocrSettings),
            "coordinateSystem": "upright image pixels, top-left",
            "executionPlatform": "macOS CoreML CPU-only + Vision OCR; not physical iPhone validation",
            "operatingSystem": ProcessInfo.processInfo.operatingSystemVersionString,
            "images": rows]
        let url = URL(fileURLWithPath: CommandLine.arguments[2])
        guard !FileManager.default.fileExists(atPath: url.path) else { fail("Refusing to overwrite existing holdout predictions.") }
        try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
    }
}
