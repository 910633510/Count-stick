import Foundation
import UIKit

enum CountCategory: String, Codable, CaseIterable, Identifiable, Sendable {
    case skewers
    case steelPipes = "steel_pipes"

    var id: String { rawValue }
    var title: String { L10n.text("category.\(rawValue)") }
    func text(_ key: String) -> String {
        L10n.text(self == .skewers ? key : "pipe.\(key)")
    }
    func formattedCount(_ count: Int) -> String {
        String(format: L10n.text(self == .skewers ? "skewer.count.format" : "pipe.count.format"), count)
    }
}

// Preserve the existing skewer result and detector implementation unchanged.
typealias CountDetectionResult = StickDetectionResult

struct CountDetectorRouter {
    typealias Detector = (UIImage) async throws -> CountDetectionResult
    private let skewers: Detector
    private let pipes: Detector

    init(skewers: Detector? = nil, pipes: Detector? = nil) {
        let stickDetector = StickDetector()
        let pipeDetector = PipeDetector()
        self.skewers = skewers ?? { try await stickDetector.detectWithAutomaticRegion(from: $0) }
        self.pipes = pipes ?? { try await pipeDetector.detect(from: $0) }
    }

    func detect(_ image: UIImage, category: CountCategory) async throws -> CountDetectionResult {
        switch category {
        case .skewers: return try await skewers(image)
        case .steelPipes: return try await pipes(image)
        }
    }
}

enum CountExport {
    static func caption(category: CountCategory, count: Int, note: String) -> String {
        let categoryCount = "\(category.title) · \(category.formattedCount(count))"
        return note.isEmpty ? categoryCount : categoryCount + "\n" + note
    }

    static func image(photo: UIImage, markers: [CountMarker], note: String, category: CountCategory) -> UIImage {
        // Use the established renderer, with a category-aware count in its footer.
        PhotoProcessing.shareImage(photo: photo, markers: markers,
                                   note: caption(category: category, count: markers.count, note: note))
    }
}
