import Foundation
import CoreML
import Vision
import ImageIO
import CryptoKit

/// Offline macOS reproduction of the app's orientation/downsample/scaleFill pipeline.
/// The Python entry point validates frozen labels and source-group splits before invoking this.
struct EvaluationInput: Decodable {
    struct Image: Decodable { let id: String; let path: String; let sha256: String; let width: Int; let height: Int }
    let validationPassed: Bool
    let labelsFrozen: Bool
    let splitLocked: Bool
    let manifestSha256: String
    let modelPath: String
    let modelFileHashes: [String: String]
    let sourceDetectorSha256: String
    let images: [Image]
}
final class ThresholdProvider: MLFeatureProvider {
    let confidence: Double
    let iou: Double
    init(confidence: Double, iou: Double) { self.confidence = confidence; self.iou = iou }
    var featureNames: Set<String> { ["confidenceThreshold", "iouThreshold"] }
    func featureValue(for name: String) -> MLFeatureValue? {
        switch name {
        case "confidenceThreshold": return MLFeatureValue(double: confidence)
        case "iouThreshold": return MLFeatureValue(double: iou)
        default: return nil
        }
    }
}
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8)); exit(2)
}
func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
guard CommandLine.arguments.count == 3 else { fail("Use evaluate.py; this helper needs validated input JSON and an output JSON path.") }
let input = try JSONDecoder().decode(EvaluationInput.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
guard input.validationPassed && input.labelsFrozen && input.splitLocked else { fail("Labels and source-group splits must be frozen before inference.") }
for (relativePath, expected) in input.modelFileHashes {
    let data = try Data(contentsOf: URL(fileURLWithPath: input.modelPath).appendingPathComponent(relativePath))
    guard sha256(data) == expected else { fail("Model file changed before inference: \(relativePath)") }
}
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuOnly
let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: input.modelPath))
let model = try MLModel(contentsOf: compiled, configuration: configuration)
let vision = try VNCoreMLModel(for: model)
let variants: [(String, Double, Double)] = [("baseline", 0.25, 0.70), ("candidate", 0.25, 0.45)]
var results: [[String: Any]] = []
for item in input.images {
    let bytes = try Data(contentsOf: URL(fileURLWithPath: item.path))
    guard sha256(bytes) == item.sha256 else { fail("Image changed after validation: \(item.id)") }
    guard let source = CGImageSourceCreateWithData(bytes as CFData, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
          let rawWidth = properties[kCGImagePropertyPixelWidth as String] as? Int,
          let rawHeight = properties[kCGImagePropertyPixelHeight as String] as? Int else { fail("Unreadable image: \(item.id)") }
    let orientation = properties[kCGImagePropertyOrientation as String] as? Int ?? 1
    let rotated = [5, 6, 7, 8].contains(orientation)
    let width = rotated ? rawHeight : rawWidth
    let height = rotated ? rawWidth : rawHeight
    guard width == item.width && height == item.height else { fail("Annotation upright dimensions do not match image: \(item.id)") }
    guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 2048,
        kCGImageSourceShouldCacheImmediately: true
    ] as CFDictionary) else { fail("Cannot normalize image: \(item.id)") }
    var imageResult: [String: Any] = ["imageId": item.id, "imageSha256": item.sha256, "width": width, "height": height,
                                     "sourceOrientation": orientation, "processedPixels": [image.width, image.height]]
    var variantResults: [String: Any] = [:]
    for (name, confidence, iou) in variants {
        vision.featureProvider = ThresholdProvider(confidence: confidence, iou: iou)
        let request = VNCoreMLRequest(model: vision)
        request.imageCropAndScaleOption = .scaleFill
        let started = Date()
        try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
        guard let observations = request.results as? [VNRecognizedObjectObservation] else { fail("Unsupported model output: \(item.id)") }
        let predictions: [[String: Any]] = observations.enumerated().compactMap { index, observation -> [String: Any]? in
            let rect = observation.boundingBox
            guard rect.midX.isFinite && rect.midY.isFinite && rect.width > 0 && rect.height > 0 else { return nil }
            // Match CountMarker's production clamp after Vision's bottom-left conversion.
            return ["id": "\(name)-\(index)", "x": min(Double(width), max(0, rect.midX * Double(width))), "y": min(Double(height), max(0, (1 - rect.midY) * Double(height))),
                    "bbox": [rect.minX * Double(width), (1 - rect.maxY) * Double(height), rect.width * Double(width), rect.height * Double(height)],
                    "confidence": Double(observation.confidence)]
        }
        variantResults[name] = ["confidenceThreshold": confidence, "iouThreshold": iou, "cropScale": "scaleFill",
                                "elapsedSeconds": Date().timeIntervalSince(started), "predictions": predictions]
    }
    imageResult["variants"] = variantResults
    results.append(imageResult)
    FileHandle.standardError.write(Data(("Evaluated frozen image: \(item.id)\n").utf8))
}
let output: [String: Any] = ["schemaVersion": 1, "manifestSha256": input.manifestSha256,
    "executionPlatform": "macOS CoreML CPU-only offline reproduction; not physical iPhone acceptance",
    "coordinateSystem": "upright image pixels, top-left", "modelFileHashes": input.modelFileHashes,
    "sourceDetectorSha256": input.sourceDetectorSha256, "images": results]
try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
