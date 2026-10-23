import Testing
import CoreML
import Vision
import UIKit
@testable import Count_stick

struct StickDetectorDedupRegressionTests {
    @Test func realFixtureMatchesExplicitNMSAndReducesOverlappingDetections() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("release/fixtures/bamboo-skewers-cc0.jpg"))
        let image = try PhotoProcessing.load(data: data)
        let cgImage = try #require(image.cgImage)
        let reference = try VNCoreMLModel(for: best(configuration: MLModelConfiguration()).model)

        let baselineRequest = VNCoreMLRequest(model: reference)
        baselineRequest.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([baselineRequest])
        let baseline = try #require(baselineRequest.results as? [VNRecognizedObjectObservation])

        reference.featureProvider = try MLDictionaryFeatureProvider(dictionary: [
            "confidenceThreshold": 0.25,
            "iouThreshold": 0.45
        ])
        let candidateRequest = VNCoreMLRequest(model: reference)
        candidateRequest.imageCropAndScaleOption = .scaleFill
        try VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([candidateRequest])
        let expected = try #require(candidateRequest.results as? [VNRecognizedObjectObservation])
        let actual = try await StickDetector().detect(from: image)

        #expect(!actual.isEmpty)
        #expect(actual.count < baseline.count)
        #expect(actual.count == expected.count)
        for (marker, observation) in zip(actual, expected) {
            #expect(marker.x.isFinite && marker.y.isFinite)
            #expect(abs(marker.x - observation.boundingBox.midX) < 0.00001)
            #expect(abs(marker.y - (1 - observation.boundingBox.midY)) < 0.00001)
        }
        // This verifies the production feature-provider wiring and real-model
        // deduplication behavior. Neither total is a ground-truth accuracy score.
    }
}
