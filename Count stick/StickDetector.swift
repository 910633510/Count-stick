import CoreML
import Vision
import UIKit
import ImageIO

enum CountError: LocalizedError {
    case unreadableImage, unavailableModel, detectionFailed, unexpectedOutput
    var errorDescription: String? {
        switch self {
        case .unreadableImage: return L10n.text("error.image")
        case .unavailableModel: return L10n.text("error.model")
        case .detectionFailed: return L10n.text("error.detect")
        case .unexpectedOutput: return L10n.text("error.output")
        }
    }
}

/// Local deduplication candidate: preserve the model's score threshold while
/// tightening suppression of overlapping boxes. This is not an accuracy guarantee.
private final class StickDetectionThresholds: MLFeatureProvider {
    var featureNames: Set<String> { ["confidenceThreshold", "iouThreshold"] }
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "confidenceThreshold": return MLFeatureValue(double: 0.25)
        case "iouThreshold": return MLFeatureValue(double: 0.45)
        default: return nil
        }
    }
}

struct StickDetectionResult {
    let markers: [CountMarker]
    let rawMarkers: [CountMarker]
    let automaticRegion: AutoCountRegionResult
}

/// Serial work keeps model loading and prediction off the main thread.
final class StickDetector: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.countstick.detection", qos: .userInitiated)
    private var model: VNCoreMLModel?
    private let thresholds = StickDetectionThresholds()

    func detect(from image: UIImage) async throws -> [CountMarker] {
        try await detectWithAutomaticRegion(from: image).markers
    }

    func detectWithAutomaticRegion(from image: UIImage) async throws -> StickDetectionResult {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    guard let cgImage = image.cgImage else { throw CountError.unreadableImage }
                    if self.model == nil {
                        do {
                            let configuration = MLModelConfiguration()
                            let model = try VNCoreMLModel(for: best(configuration: configuration).model)
                            model.featureProvider = self.thresholds
                            self.model = model
                        } catch { throw CountError.unavailableModel }
                    }
                    guard let model = self.model else { throw CountError.unavailableModel }
                    let request = VNCoreMLRequest(model: model)
                    request.imageCropAndScaleOption = .scaleFill
                    do {
                        try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([request])
                    } catch { throw CountError.detectionFailed }
                    guard let observations = request.results as? [VNRecognizedObjectObservation] else {
                        throw CountError.unexpectedOutput
                    }
                    // Vision uses a bottom-left origin; the app uses top-left everywhere.
                    let detections = observations.compactMap { observation -> AutoCountDetection? in
                        let box = observation.boundingBox
                        guard box.midX.isFinite, box.midY.isFinite,
                              box.width > 0, box.height > 0 else { return nil }
                        return AutoCountDetection(id: UUID(), box: AutoCountBox(
                            x: box.minX, y: 1 - box.maxY, width: box.width, height: box.height
                        ), confidence: Double(observation.confidence))
                    }
                    let rawMarkers = detections.map { detection in
                        CountMarker(id: detection.id, x: detection.box.x + detection.box.width / 2,
                                    y: detection.box.y + detection.box.height / 2)
                    }
                    let selection = AutoCountRegion.select(
                        detections: detections, textAreas: self.recognizedTextAreas(in: cgImage),
                        imageAspectRatio: Double(cgImage.width) / Double(cgImage.height)
                    )
                    let kept = Set(selection.keptIDs)
                    continuation.resume(returning: StickDetectionResult(
                        markers: rawMarkers.filter { kept.contains($0.id) }, rawMarkers: rawMarkers,
                        automaticRegion: selection
                    ))
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    /// Text is used only during this local prediction. OCR failures preserve all
    /// detections for the region selector; neither recognized strings nor photos leave the device.
    private func recognizedTextAreas(in image: CGImage) -> [AutoCountTextArea] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
        request.usesLanguageCorrection = false
        request.minimumTextHeight = 0.015
        do {
            try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
        } catch { return [] }
        return (request.results ?? []).compactMap { observation in
            guard let candidate = observation.topCandidates(1).first, candidate.confidence >= 0.8 else { return nil }
            let count = candidate.string.unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.count
            guard count >= 2 else { return nil }
            let box = observation.boundingBox.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
            guard !box.isNull, box.width > 0, box.height > 0 else { return nil }
            return AutoCountTextArea(box: AutoCountBox(x: box.minX, y: 1 - box.maxY,
                                                       width: box.width, height: box.height),
                                     confidence: Double(candidate.confidence), characterCount: count)
        }
    }
}

enum PhotoProcessing {
    static let maximumDimension: CGFloat = 2048

    /// ImageIO applies EXIF rotation and mirroring before Vision and the UI see the image.
    static func load(data: Data) throws -> UIImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maximumDimension,
                kCGImageSourceShouldCacheImmediately: true
              ] as CFDictionary) else { throw CountError.unreadableImage }
        return UIImage(cgImage: cgImage)
    }

    static func normalized(_ image: UIImage, maximumDimension: CGFloat = maximumDimension) throws -> UIImage {
        guard image.size.width > 0, image.size.height > 0 else { throw CountError.unreadableImage }
        let scale = min(1, maximumDimension / max(image.size.width, image.size.height))
        let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            UIColor.white.setFill()
            UIRectFill(CGRect(origin: .zero, size: size))
            image.draw(in: CGRect(origin: .zero, size: size))
        }
    }

    static func shareImage(photo: UIImage, markers: [CountMarker], note: String) -> UIImage {
        let width: CGFloat = min(1600, photo.size.width)
        let scale = width / photo.size.width
        let photoHeight = photo.size.height * scale
        let margin: CGFloat = max(20, width * 0.035)
        let noteFont = UIFont.systemFont(ofSize: min(24, width * 0.032))
        let noteWidth = max(1, width - margin * 2)
        let noteHeight = ceil((note as NSString).boundingRect(
            with: CGSize(width: noteWidth, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: noteFont], context: nil
        ).height)
        let footer: CGFloat = note.isEmpty ? 140 : 125 + noteHeight + 22
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: CGSize(width: width, height: photoHeight + footer), format: format).image { context in
            UIColor.white.setFill()
            context.fill(CGRect(x: 0, y: 0, width: width, height: photoHeight + footer))
            photo.draw(in: CGRect(x: 0, y: 0, width: width, height: photoHeight))
            let radius = max(4, width * 0.006)
            for marker in markers {
                let rect = CGRect(x: marker.x * width - radius, y: marker.y * photoHeight - radius, width: radius * 2, height: radius * 2)
                let path = UIBezierPath(ovalIn: rect)
                AppPalette.coral.setFill()
                path.fill()
                UIColor.white.setStroke()
                path.lineWidth = max(1.5, radius * 0.25)
                path.stroke()
            }
            let title = "Count Stick · \(L10n.count(markers.count))"
            title.draw(in: CGRect(x: margin, y: photoHeight + 22, width: width - margin * 2, height: 48), withAttributes: [
                .font: UIFont.boldSystemFont(ofSize: min(36, width * 0.045)), .foregroundColor: AppPalette.ink
            ])
            L10n.text("share.reviewed").draw(in: CGRect(x: margin, y: photoHeight + 78, width: width - margin * 2, height: 40), withAttributes: [
                .font: UIFont.systemFont(ofSize: min(22, width * 0.03)), .foregroundColor: UIColor.darkGray
            ])
            if !note.isEmpty {
                note.draw(in: CGRect(x: margin, y: photoHeight + 125, width: noteWidth, height: noteHeight), withAttributes: [
                    .font: noteFont, .foregroundColor: AppPalette.ink
                ])
            }
        }
    }
}
