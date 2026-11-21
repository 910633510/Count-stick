import Foundation
import CoreML

struct PipeRawPrediction: Sendable {
    let anchor: Int
    let x1: Float
    let y1: Float
    let x2: Float
    let y2: Float
    let confidence: Float
    var area: Float { (x2 - x1) * (y2 - y1) }
}

enum PipeRawPostprocessError: Error { case invalidShape, invalidValue }

enum PipeRawPostprocessor {
    static let anchorCount = 21_504

    /// Raw YOLO output is [1, 5, 21504]: pixel cx,cy,w,h and class-zero score.
    /// Float32 arithmetic and greedy suppression match the frozen CPU reference.
    static func predictions(from output: MLMultiArray, confidenceThreshold: Float, iouThreshold: Double) throws -> [PipeRawPrediction] {
        guard output.dataType == .float32, output.shape.map(\.intValue) == [1, 5, anchorCount],
              output.strides[1].intValue > 0, output.strides[2].intValue > 0 else { throw PipeRawPostprocessError.invalidShape }
        guard confidenceThreshold.isFinite, confidenceThreshold > 0, confidenceThreshold < 1,
              iouThreshold.isFinite, iouThreshold > 0, iouThreshold < 1 else { throw PipeRawPostprocessError.invalidValue }
        let data = output.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = output.strides[1].intValue, anchorStride = output.strides[2].intValue
        var candidates: [PipeRawPrediction] = []
        for anchor in 0..<anchorCount {
            let offset = anchor * anchorStride
            let x = data[offset], y = data[offset + channelStride]
            let w = data[offset + 2 * channelStride], h = data[offset + 3 * channelStride]
            let score = data[offset + 4 * channelStride]
            guard x.isFinite, y.isFinite, w.isFinite, h.isFinite, score.isFinite,
                  w > 0, h > 0, score >= 0, score <= 1 else { throw PipeRawPostprocessError.invalidValue }
            if score >= confidenceThreshold {
                candidates.append(PipeRawPrediction(anchor: anchor, x1: x - w / 2, y1: y - h / 2,
                                                    x2: x + w / 2, y2: y + h / 2, confidence: score))
            }
        }
        return greedyNMS(candidates, iouThreshold: iouThreshold)
    }

    static func greedyNMS(_ input: [PipeRawPrediction], iouThreshold: Double) -> [PipeRawPrediction] {
        let ordered = input.sorted { $0.confidence == $1.confidence ? $0.anchor < $1.anchor : $0.confidence > $1.confidence }
        var kept: [PipeRawPrediction] = []
        for candidate in ordered {
            var suppressed = false
            for prior in kept {
                let w = max(Float(0), min(candidate.x2, prior.x2) - max(candidate.x1, prior.x1))
                let h = max(Float(0), min(candidate.y2, prior.y2) - max(candidate.y1, prior.y1))
                let intersection = w * h
                let union = candidate.area + prior.area - intersection
                if union > 0 && Double(intersection / union) > iouThreshold { suppressed = true; break }
            }
            if !suppressed { kept.append(candidate) }
        }
        return kept
    }
}
