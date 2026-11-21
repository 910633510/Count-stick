import Foundation
import CoreML

@main
struct VerifyPipeNMSContract {
    static func main() throws {
        let separated = (0..<1205).map { index in
            PipeRawPrediction(anchor: index, x1: Float(index % 40) * 20,
                              y1: Float(index / 40) * 20,
                              x2: Float(index % 40) * 20 + 10,
                              y2: Float(index / 40) * 20 + 10,
                              confidence: 0.8)
        }
        let all = PipeRawPostprocessor.greedyNMS(separated, iouThreshold: 0.7)
        precondition(all.count == 1205, "Must not truncate a dense count at 1000")
        precondition(all.map(\.anchor) == Array(0..<1205), "Equal scores need deterministic ordering")
        let duplicate = PipeRawPrediction(anchor: 5000, x1: 0, y1: 0, x2: 10, y2: 10, confidence: 0.7)
        let cleaned = PipeRawPostprocessor.greedyNMS([duplicate] + separated.reversed(), iouThreshold: 0.7)
        precondition(cleaned.map(\.anchor) == all.map(\.anchor), "Remove duplicates before clipping or counting")
        let wrong = try MLMultiArray(shape: [1, 5, 1000], dataType: .float32)
        do {
            _ = try PipeRawPostprocessor.predictions(from: wrong, confidenceThreshold: 0.3, iouThreshold: 0.7)
            preconditionFailure("Reject truncated raw tensor")
        } catch PipeRawPostprocessError.invalidShape { }
        print("PASS: 1205 retained; deterministic tie order; duplicate suppressed; truncated tensor rejected")
    }
}
