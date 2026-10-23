import Foundation
import Vision
import ImageIO
import CryptoKit

struct ResearchInput: Codable {
    struct Image: Codable {
        let id: String
        let path: String
        let sha256: String
        let width: Int
        let height: Int
        let detections: [AutoCountDetection]
        let textAreas: [AutoCountTextArea]
    }
    let protocolSha256: String
    let geometrySourceSha256: String
    let images: [Image]
}
@main
struct AutomaticRegionDevelopment {
    static func main() throws {
        let input = try JSONDecoder().decode(ResearchInput.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
        var outputs: [[String: Any]] = []
        for item in input.images {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: item.path))
            let sha = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            guard sha == item.sha256 else { throw NSError(domain: "ImageChanged", code: 1) }
            let textAreas = item.textAreas
            let ocrError: String? = nil
            let settings: [(String, Bool, Bool)] = [("text_only", true, false), ("geometry_only", false, true), ("combined", true, true)]
            var variants: [String: Any] = [:]
            for (name, text, geometry) in settings {
                let result = AutoCountRegion.select(detections: item.detections, textAreas: textAreas,
                    imageAspectRatio: Double(item.width) / Double(item.height), useText: text, useGeometry: geometry)
                variants[name] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(result))
            }
            var row: [String: Any] = ["imageId": item.id, "imageSha256": sha, "variants": variants,
                "textAreas": try JSONSerialization.jsonObject(with: JSONEncoder().encode(textAreas))]
            if let ocrError { row["ocrError"] = ocrError }
            outputs.append(row)
            FileHandle.standardError.write(Data("Processed cached detections and OCR: \(item.id)\n".utf8))
        }
        let output: [String: Any] = ["protocolSha256": input.protocolSha256, "geometrySourceSha256": input.geometrySourceSha256,
            "label": "Already-seen v1/v2 development diagnostics, not blind validation", "images": outputs]
        try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
    }
}
