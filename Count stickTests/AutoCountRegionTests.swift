import Foundation
import Testing
@testable import Count_stick

struct AutoCountRegionTests {
    @Test func sparseInputFailsOpen() {
        let detections = [point(0.1, 0.1, confidence: 0.2), point(0.8, 0.8, confidence: 0.2)]
        let result = AutoCountRegion.select(detections: detections, textAreas: [], imageAspectRatio: 1)
        #expect(result.keptIDs == detections.map(\.id))
        #expect(result.excluded.isEmpty)
        #expect(result.fallbackReason == .sparse)
    }

    @Test func sparseEndpointsCannotBeErasedByOCR() {
        let detections = [point(0.2, 0.2), point(0.3, 0.2), point(0.4, 0.2)]
        let text = AutoCountTextArea(box: AutoCountBox(x: 0.1, y: 0.15, width: 0.5, height: 0.1), confidence: 0.99, characterCount: 6)
        let result = AutoCountRegion.select(detections: detections, textAreas: [text], imageAspectRatio: 1)
        #expect(result.keptIDs == detections.map(\.id))
        #expect(result.excluded.isEmpty)
        #expect(result.fallbackReason == .sparse)
    }

    @Test func smallSecondaryBundleCannotBeErasedByOCR() {
        let second = [point(0.7, 0.7), point(0.74, 0.7), point(0.72, 0.74)]
        let detections = bundle(x: 0.2, y: 0.2) + second
        let text = AutoCountTextArea(box: AutoCountBox(x: 0.66, y: 0.66, width: 0.18, height: 0.1), confidence: 0.99, characterCount: 4)
        let result = AutoCountRegion.select(detections: detections, textAreas: [text], imageAspectRatio: 1)
        #expect(result.keptIDs == detections.map(\.id))
        #expect(result.excluded.isEmpty)
        #expect(result.regions.count == 2)
    }

    @Test func OCRWithoutNonTextAnchorCannotCreateFalseZero() {
        let detections = bundle(x: 0.2, y: 0.2)
        let text = AutoCountTextArea(box: AutoCountBox(x: 0.1, y: 0.1, width: 0.5, height: 0.25), confidence: 0.99, characterCount: 8)
        let result = AutoCountRegion.select(detections: detections, textAreas: [text], imageAspectRatio: 1)
        #expect(result.keptIDs == detections.map(\.id))
        #expect(result.excluded.isEmpty)
        #expect(result.fallbackReason != .empty)
    }

    @Test func distantLowConfidenceSingletonIsExcludedWithoutRemovingDenseBundle() {
        let cluster = bundle(x: 0.2, y: 0.2)
        let isolated = point(0.93, 0.94, confidence: 0.3)
        let result = AutoCountRegion.select(detections: cluster + [isolated], textAreas: [], imageAspectRatio: 1)
        #expect(Set(result.keptIDs) == Set(cluster.map(\.id)))
        #expect(result.excluded == [.init(id: isolated.id, reason: .isolated)])
        #expect(result.fallbackReason == .none)
    }

    @Test func distantHighConfidenceEndpointAndLowConfidencePairArePreserved() {
        let cluster = bundle(x: 0.2, y: 0.2)
        let high = point(0.91, 0.92, confidence: 0.92)
        let pair = [point(0.8, 0.2, confidence: 0.3), point(0.84, 0.2, confidence: 0.3)]
        let detections = cluster + [high] + pair
        let result = AutoCountRegion.select(detections: detections, textAreas: [], imageAspectRatio: 1)
        #expect(result.keptIDs == detections.map(\.id))
        #expect(result.excluded.isEmpty)
    }

    @Test func smallerSecondBundleAndNearEdgeEndpointArePreserved() {
        let first = bundle(x: 0.15, y: 0.2)
        let second = [point(0.7, 0.7), point(0.74, 0.7), point(0.72, 0.74)]
        let edge = point(0.28, 0.24, confidence: 0.28)
        let detections = first + second + [edge]
        let result = AutoCountRegion.select(detections: detections, textAreas: [], imageAspectRatio: 1)
        #expect(result.keptIDs == detections.map(\.id))
        #expect(result.regions.count == 2)
        #expect(result.excluded.isEmpty)
    }

    @Test func supportedEndpointTextureIsNotDeletedAsText() {
        let cluster = bundle(x: 0.2, y: 0.2)
        let text = AutoCountTextArea(box: AutoCountBox(x: 0.16, y: 0.17, width: 0.25, height: 0.14), confidence: 0.99, characterCount: 5)
        let result = AutoCountRegion.select(detections: cluster, textAreas: [text], imageAspectRatio: 1)
        #expect(result.keptIDs == cluster.map(\.id))
        #expect(result.excluded.isEmpty)
    }

    @Test func confidentSeparatedTextRowIsExcluded() {
        let cluster = bundle(x: 0.2, y: 0.2)
        let letters = (0..<8).map { point(0.2 + Double($0) * 0.04, 0.83) }
        let text = AutoCountTextArea(box: AutoCountBox(x: 0.1, y: 0.79, width: 0.6, height: 0.08), confidence: 0.9, characterCount: 6)
        let result = AutoCountRegion.select(detections: cluster + letters, textAreas: [text], imageAspectRatio: 1)
        #expect(Set(result.keptIDs) == Set(cluster.map(\.id)))
        #expect(Set(result.excluded.map(\.id)) == Set(letters.map(\.id)))
        #expect(result.excluded.allSatisfy { $0.reason == .text })
    }

    @Test func nonfiniteTextConfidenceIsIgnoredAndInvalidDetectionsFailOpen() {
        let single = point(0.3, 0.3)
        let letters = (0..<8).map { point(0.2 + Double($0) * 0.04, 0.83) }
        let complete = bundle(x: 0.2, y: 0.2) + letters
        let text = AutoCountTextArea(box: AutoCountBox(x: 0.1, y: 0.79, width: 0.6, height: 0.08), confidence: .infinity, characterCount: 7)
        let valid = AutoCountRegion.select(detections: complete, textAreas: [text], imageAspectRatio: 1)
        #expect(valid.keptIDs == complete.map(\.id))
        let nan = AutoCountDetection(id: UUID(), box: AutoCountBox(x: .nan, y: 0.3, width: 0.03, height: 0.03), confidence: 0.5)
        let outside = AutoCountDetection(id: UUID(), box: AutoCountBox(x: 100, y: 100, width: 0.1, height: 0.1), confidence: 0.5)
        for bad in [nan, outside] {
            let result = AutoCountRegion.select(detections: [single, bad], textAreas: [], imageAspectRatio: 1)
            #expect(result.keptIDs == [single.id, bad.id])
            #expect(result.excluded.isEmpty)
            #expect(result.regions.isEmpty)
            #expect(result.fallbackReason == .ambiguous)
        }
    }

    @Test func emptyAndRoundTripPreserveFallbackContract() throws {
        let result = AutoCountRegion.select(detections: [], textAreas: [], imageAspectRatio: 1)
        #expect(result.fallbackReason == .empty)
        #expect(result.keptIDs.isEmpty)
        #expect(try JSONDecoder().decode(AutoCountRegionResult.self, from: JSONEncoder().encode(result)) == result)
    }

    private func point(_ x: Double, _ y: Double, confidence: Double = 0.9) -> AutoCountDetection {
        AutoCountDetection(id: UUID(), box: AutoCountBox(x: x - 0.015, y: y - 0.015, width: 0.03, height: 0.03), confidence: confidence)
    }
    private func bundle(x: Double, y: Double) -> [AutoCountDetection] {
        (0..<3).flatMap { row in (0..<3).map { column in point(x + Double(column) * 0.04, y + Double(row) * 0.04) } }
    }
}
