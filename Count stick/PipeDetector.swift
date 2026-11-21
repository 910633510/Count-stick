import CoreML
import Vision
import UIKit

enum PipeCountError: LocalizedError {
    case unavailableModel, incompatibleModel, unexpectedOutput, detectionFailed
    var errorDescription: String? {
        switch self {
        case .unavailableModel: return L10n.text("pipe.error.model")
        case .incompatibleModel: return L10n.text("pipe.error.configuration")
        case .unexpectedOutput: return L10n.text("pipe.error.output")
        case .detectionFailed: return L10n.text("pipe.error.detect")
        }
    }
}

/// Shipped only with a separately validated pipe detector export. A model alone
/// cannot silently enable guessed operating thresholds.
struct PipeModelConfiguration: Decodable {
    let status: String
    let exportID: UUID
    let checkpointSHA256: String
    let inputSize: Int
    let classLabel: String
    let confidenceThreshold: Double
    let iouThreshold: Double

    func validate() throws {
        guard status == "VALIDATED_COREML_EXPORT", inputSize == 1024, classLabel == "pipe_end",
              checkpointSHA256.count == 64,
              checkpointSHA256.allSatisfy({ "0123456789abcdefABCDEF".contains($0) }),
              confidenceThreshold.isFinite, iouThreshold.isFinite,
              confidenceThreshold > 0, confidenceThreshold < 1,
              iouThreshold > 0, iouThreshold < 1 else { throw PipeCountError.incompatibleModel }
    }

    func validate(metadata: [String: String]) throws {
        try validate()
        guard metadata["countscope.export_id"].flatMap(UUID.init(uuidString:)) == exportID,
              metadata["countscope.checkpoint_sha256"]?.lowercased() == checkpointSHA256.lowercased(),
              metadata["countscope.confidence_threshold"].flatMap(Double.init) == confidenceThreshold,
              metadata["countscope.iou_threshold"].flatMap(Double.init) == iouThreshold,
              metadata["countscope.class_label"] == classLabel,
              metadata["countscope.input_size"].flatMap(Int.init) == inputSize else {
            throw PipeCountError.incompatibleModel
        }
    }
}

/// Ultralytics centered letterbox geometry, auto=False, scale_fill=False.
/// Boxes entering/exiting this helper have a top-left origin.
struct PipeLetterbox {
    let sourceWidth: Double
    let sourceHeight: Double
    let size: Int
    let gain: Double
    let resizedWidth: Int
    let resizedHeight: Int
    let left: Int
    let top: Int

    init(width: Int, height: Int, size: Int = 1024) throws {
        guard width > 0, height > 0, size > 0 else { throw CountError.unreadableImage }
        sourceWidth = Double(width); sourceHeight = Double(height); self.size = size
        gain = min(Double(size) / Double(width), Double(size) / Double(height))
        resizedWidth = Int((Double(width) * gain).rounded(.toNearestOrEven))
        resizedHeight = Int((Double(height) * gain).rounded(.toNearestOrEven))
        guard resizedWidth > 0, resizedHeight > 0 else { throw CountError.unreadableImage }
        left = Int((Double(size - resizedWidth) / 2 - 0.1).rounded(.toNearestOrEven))
        top = Int((Double(size - resizedHeight) / 2 - 0.1).rounded(.toNearestOrEven))
    }

    func originalBox(fromNetwork box: AutoCountBox) -> AutoCountBox? {
        guard box.isValid else { return nil }
        let mapped = AutoCountBox(x: (box.x * Double(size) - Double(left)) / gain / sourceWidth,
                                  y: (box.y * Double(size) - Double(top)) / gain / sourceHeight,
                                  width: box.width * Double(size) / gain / sourceWidth,
                                  height: box.height * Double(size) / gain / sourceHeight).clippedToImage
        return mapped.isValid ? mapped : nil
    }

    func prepare(_ image: UIImage) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1; format.opaque = true; format.preferredRange = .standard
        return UIGraphicsImageRenderer(size: CGSize(width: size, height: size), format: format).image { context in
            UIColor(white: 114.0 / 255.0, alpha: 1).setFill()
            context.fill(CGRect(x: 0, y: 0, width: size, height: size))
            context.cgContext.interpolationQuality = .high
            image.draw(in: CGRect(x: left, y: top, width: resizedWidth, height: resizedHeight))
        }
    }
}

enum PipeCountRegion {
    /// Every valid pipe end is retained, including separate bundles/single pipes.
    /// The outline is a display aid, never an OCR or isolation-based deletion rule.
    static func result(for detections: [AutoCountDetection]) -> CountDetectionResult {
        let valid = detections.filter { $0.box.isValid && $0.box.clippedToImage.isValid && $0.confidence.isFinite }
        let markers = valid.map { CountMarker(id: $0.id, x: $0.box.centerX, y: $0.box.centerY) }
        let bounds = valid.map { $0.box.clippedToImage }.reduce(nil as AutoCountBox?) { $0?.union($1) ?? $1 }
        let region = AutoCountRegionResult(keptIDs: valid.map(\.id), excluded: [], regions: bounds.map { [$0] } ?? [],
                                          fallbackReason: valid.isEmpty ? .empty : .none)
        return CountDetectionResult(markers: markers, rawMarkers: markers, automaticRegion: region)
    }
}

private final class PipeThresholds: MLFeatureProvider {
    let values: [String: Double]
    init(values: [String: Double]) { self.values = values }
    var featureNames: Set<String> { Set(values.keys) }
    func featureValue(for featureName: String) -> MLFeatureValue? { values[featureName].map(MLFeatureValue.init(double:)) }
}

final class PipeDetector: @unchecked Sendable {
    private let bundle: Bundle
    private let queue = DispatchQueue(label: "com.countscope.pipe-detection", qos: .userInitiated)
    private var cached: (VNCoreMLModel, PipeModelConfiguration)?

    init(bundle: Bundle = .main) { self.bundle = bundle }

    private func model() throws -> (VNCoreMLModel, PipeModelConfiguration) {
        if let cached { return cached }
        guard let modelURL = bundle.url(forResource: "PipeEnds", withExtension: "mlmodelc"),
              let configURL = bundle.url(forResource: "PipeEnds", withExtension: "json") else {
            throw PipeCountError.unavailableModel
        }
        do {
            let config = try JSONDecoder().decode(PipeModelConfiguration.self, from: Data(contentsOf: configURL))
            try config.validate()
            let options = MLModelConfiguration()
            options.computeUnits = .cpuAndNeuralEngine
            let model = try MLModel(contentsOf: modelURL, configuration: options)
            try config.validate(metadata: model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:])
            let outputs = model.modelDescription.outputDescriptionsByName
            guard outputs.count == 1, let raw = outputs["raw_detections"], raw.type == .multiArray,
                  raw.multiArrayConstraint?.shape.map(\.intValue) == [1, 5, PipeRawPostprocessor.anchorCount] else {
                throw PipeCountError.incompatibleModel
            }
            let features = model.modelDescription.inputDescriptionsByName
            let imageFeatures = features.values.filter { $0.type == .image }
            guard imageFeatures.count == 1, let input = imageFeatures.first?.imageConstraint,
                  input.pixelsWide == config.inputSize, input.pixelsHigh == config.inputSize else {
                throw PipeCountError.incompatibleModel
            }
            var thresholds: [String: Double] = [:]
            for (name, value) in [("confidenceThreshold", config.confidenceThreshold), ("iouThreshold", config.iouThreshold)] {
                if let feature = features[name] {
                    guard feature.type == .double else { throw PipeCountError.incompatibleModel }
                    thresholds[name] = value
                }
            }
            let vision = try VNCoreMLModel(for: model)
            vision.featureProvider = PipeThresholds(values: thresholds)
            cached = (vision, config)
            return (vision, config)
        } catch let error as PipeCountError { throw error }
        catch { throw PipeCountError.incompatibleModel }
    }

    func detect(from image: UIImage) async throws -> CountDetectionResult {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let (model, config) = try self.model()
                    guard image.imageOrientation == .up, let source = image.cgImage else { throw CountError.unreadableImage }
                    let letterbox = try PipeLetterbox(width: source.width, height: source.height, size: config.inputSize)
                    guard let prepared = letterbox.prepare(image).cgImage else { throw CountError.unreadableImage }
                    let request = VNCoreMLRequest(model: model)
                    request.imageCropAndScaleOption = .scaleFill // Already explicitly letterboxed.
                    try VNImageRequestHandler(cgImage: prepared, orientation: .up).perform([request])
                    guard let features = request.results as? [VNCoreMLFeatureValueObservation],
                          features.count == 1, features[0].featureName == "raw_detections",
                          let raw = features[0].featureValue.multiArrayValue else { throw PipeCountError.unexpectedOutput }
                    let predictions = try PipeRawPostprocessor.predictions(from: raw,
                                                                          confidenceThreshold: Float(config.confidenceThreshold),
                                                                          iouThreshold: config.iouThreshold)
                    var detections: [AutoCountDetection] = []
                    for prediction in predictions {
                        let network = AutoCountBox(x: Double(prediction.x1) / 1024, y: Double(prediction.y1) / 1024,
                                                   width: Double(prediction.x2 - prediction.x1) / 1024,
                                                   height: Double(prediction.y2 - prediction.y1) / 1024)
                        if let original = letterbox.originalBox(fromNetwork: network) {
                            detections.append(AutoCountDetection(id: UUID(), box: original, confidence: Double(prediction.confidence)))
                        }
                    }
                    continuation.resume(returning: PipeCountRegion.result(for: detections))
                } catch let error as PipeCountError { continuation.resume(throwing: error) }
                catch CountError.unreadableImage { continuation.resume(throwing: CountError.unreadableImage) }
                catch { continuation.resume(throwing: PipeCountError.detectionFailed) }
            }
        }
    }
}
