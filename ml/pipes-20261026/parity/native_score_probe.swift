import Foundation
import CoreML
import Vision
import CoreVideo
import CoreGraphics
import ImageIO
import CryptoKit

struct Failure: Error, CustomStringConvertible { let description: String }
func need(_ value: Bool, _ message: String) throws { if !value { throw Failure(description: message) } }
func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func fileHash(_ path: String) throws -> String { hash(try Data(contentsOf: URL(fileURLWithPath: path))) }
func json(_ path: String) throws -> [String: Any] { try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any] }
func write(_ value: Any, _ path: URL) throws { try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]).write(to: path, options: .atomic) }
func image(_ path: String) throws -> CGImage {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw Failure(description: "Cannot decode PNG") }
    try need(image.width == 1024 && image.height == 1024, "Network image must be1024 square")
    return image
}
func bufferRGBHash(_ value: MLFeatureValue) throws -> String {
    guard let pixel = value.imageBufferValue else { throw Failure(description: "No input pixel buffer") }
    let format = CVPixelBufferGetPixelFormatType(pixel)
    try need(format == kCVPixelFormatType_32BGRA || format == kCVPixelFormatType_32ARGB, "Unexpected input pixel format")
    CVPixelBufferLockBaseAddress(pixel, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixel, .readOnly) }
    let base = CVPixelBufferGetBaseAddress(pixel)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(pixel), width = CVPixelBufferGetWidth(pixel), height = CVPixelBufferGetHeight(pixel)
    var rgb = Data(capacity: width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let p = base + y * stride + x * 4
            if format == kCVPixelFormatType_32BGRA { rgb.append(p[2]); rgb.append(p[1]); rgb.append(p[0]) }
            else { rgb.append(p[1]); rgb.append(p[2]); rgb.append(p[3]) }
        }
    }
    return hash(rgb)
}
final class Thresholds: MLFeatureProvider {
    let confidence: Double, iou: Double
    init(_ confidence: Double, _ iou: Double) { self.confidence = confidence; self.iou = iou }
    var featureNames: Set<String> { ["confidenceThreshold", "iouThreshold"] }
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "confidenceThreshold" { return MLFeatureValue(double: confidence) }
        if featureName == "iouThreshold" { return MLFeatureValue(double: iou) }
        return nil
    }
}
func scalar(_ a: MLMultiArray, _ row: Int, _ column: Int) -> Double { a[[NSNumber(value: row), NSNumber(value: column)]].doubleValue }
func direct(_ model: MLModel, _ input: MLFeatureValue, _ confidence: Double, _ iou: Double) throws -> [String: Any] {
    let provider = try MLDictionaryFeatureProvider(dictionary: ["image": input, "confidenceThreshold": MLFeatureValue(double: confidence), "iouThreshold": MLFeatureValue(double: iou)])
    let start = Date()
    let output = try model.prediction(from: provider)
    guard let scores = output.featureValue(for: "confidence")?.multiArrayValue,
          let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue else { throw Failure(description: "Missing NMS tensors") }
    try need(scores.shape.count == 2 && coordinates.shape.count == 2 && coordinates.shape[1].intValue == 4 && scores.shape[0] == coordinates.shape[0], "NMS shape mismatch")
    var objects: [[String: Any]] = [], paddingMaximum = 0.0
    for r in 0..<scores.shape[0].intValue {
        let score = scalar(scores, r, 0)
        try need(score.isFinite && score >= 0 && score <= 1, "Nonfinite/out-of-range score")
        for c in 1..<scores.shape[1].intValue {
            let value = scalar(scores, r, c)
            try need(value.isFinite, "Nonfinite padding class")
            paddingMaximum = max(paddingMaximum, abs(value))
        }
        let xywh = (0..<4).map { scalar(coordinates, r, $0) }
        try need(xywh.allSatisfy(\.isFinite) && xywh[2] > 0 && xywh[3] > 0, "Invalid NMS box")
        if score >= confidence {
            objects.append(["label": "pipe_end", "confidence": score,
                            "networkXYXY": [(xywh[0]-xywh[2]/2)*1024, (xywh[1]-xywh[3]/2)*1024,
                                            (xywh[0]+xywh[2]/2)*1024, (xywh[1]+xywh[3]/2)*1024]])
        }
    }
    return ["objects": objects, "seconds": Date().timeIntervalSince(start), "scoreShape": scores.shape.map(\.intValue),
            "coordinateShape": coordinates.shape.map(\.intValue), "maximumPaddingClassMagnitude": paddingMaximum]
}
func vision(_ model: VNCoreMLModel, _ image: CGImage, _ confidence: Double) throws -> [String: Any] {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill
    let start = Date()
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    guard let observations = request.results as? [VNRecognizedObjectObservation] else { throw Failure(description: "Vision did not return recognized objects") }
    var objects: [[String: Any]] = []
    for object in observations {
        guard let label = object.labels.first else { throw Failure(description: "Missing Vision label") }
        try need(label.confidence.isFinite, "Invalid Vision confidence")
        let box = object.boundingBox
        try need([box.minX, box.minY, box.width, box.height].allSatisfy(\.isFinite) && box.width > 0 && box.height > 0, "Invalid Vision box")
        if Double(label.confidence) >= confidence {
            objects.append(["label": label.identifier, "confidence": Double(label.confidence), "objectConfidence": Double(object.confidence), "labelConfidence": Double(label.confidence),
                            "networkXYXY": [box.minX*1024, (1-box.maxY)*1024, box.maxX*1024, (1-box.minY)*1024]])
        }
    }
    return ["objects": objects, "seconds": Date().timeIntervalSince(start)]
}
func main() throws {
    try need(CommandLine.arguments.count == 4, "--pixels|--run request.json output-dir")
    let phase = CommandLine.arguments[1], request = try json(CommandLine.arguments[2])
    let output = URL(fileURLWithPath: CommandLine.arguments[3])
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    let rows = request["images"] as! [[String: Any]]
    if phase == "--pixels" {
        var results: [[String: Any]] = []
        for row in rows {
            let im = try image(row["networkImagePath"] as! String)
            let input = try MLFeatureValue(cgImage: im, pixelsWide: 1024, pixelsHigh: 1024, pixelFormatType: kCVPixelFormatType_32BGRA, options: nil)
            results.append(["id": row["id"]!, "variant": row["variant"]!, "actualRGBSHA256": try bufferRGBHash(input),
                            "expectedRGBSHA256": row["decodedRGBSHA256"]!, "sourceColorSpace": im.colorSpace?.name as String? ?? "unknown"])
        }
        try write(["status": "PIXEL_CONVERSION_ONLY_NO_MODEL", "results": results], output.appendingPathComponent("pixel-conversion.json"))
        return
    }
    try need(phase == "--run" && request["status"] as? String == "APPROVED_NATIVE_PIPE_PARITY_RUN", "No approved parity request")
    let confidence = request["confidenceThreshold"] as! Double, iou = request["iouThreshold"] as! Double
    print("STAGE compile start"); fflush(stdout)
    let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: request["packagePath"] as! String))
    print("STAGE compile finished \(compiled.path)"); fflush(stdout)
    let allowed = (ProcessInfo.processInfo.environment["PIPE_DIAG_UNITS"] ?? "cpu,all").split(separator: ",").map(String.init)
    var models: [String: MLModel] = [:], visions: [String: VNCoreMLModel] = [:]
    for (name, unit) in [("cpu", MLComputeUnits.cpuOnly), ("cpu_ne", MLComputeUnits.cpuAndNeuralEngine)] {
        if !allowed.contains(name) { continue }
        print("STAGE model load \(name)"); fflush(stdout)
        let config = MLModelConfiguration(); config.computeUnits = unit
        let model = try MLModel(contentsOf: compiled, configuration: config)
        print("STAGE model loaded \(name)"); fflush(stdout)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        for (key, value) in request["expectedModelMetadata"] as! [String: String] { try need(metadata[key] == value, "Export model metadata mismatch: \(key)") }
        print("STAGE vision load \(name)"); fflush(stdout)
        let visual = try VNCoreMLModel(for: model); visual.featureProvider = Thresholds(confidence, iou)
        models[name] = model; visions[name] = visual
        print("STAGE vision loaded \(name)"); fflush(stdout)
    }
    var results: [[String: Any]] = []
    for row in rows {
        let result: [String: Any] = try autoreleasepool {
            let path = row["networkImagePath"] as! String
            try need(try fileHash(path) == row["networkImageSHA256"] as! String, "Network PNG changed")
            let im = try image(path)
            var modes: [String: Any] = [:], rgbHashes: [String: String] = [:]
            for name in allowed {
                guard let constraint = models[name]!.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else { throw Failure(description: "Missing image constraint") }
                try need(constraint.pixelsWide == 1024 && constraint.pixelsHigh == 1024, "Unexpected image dimensions")
                let input = try MLFeatureValue(cgImage: im, constraint: constraint, options: nil)
                rgbHashes[name] = try bufferRGBHash(input)
                if ProcessInfo.processInfo.environment["PIPE_DIAG_TASK"] != "vision" {
                    print("STAGE direct predict \(name)"); fflush(stdout)
                    modes["direct_\(name)"] = try direct(models[name]!, input, confidence, iou)
                }
                print("STAGE vision predict \(name)"); fflush(stdout)
                modes["vision_\(name)"] = try vision(visions[name]!, im, confidence)
            }
            return ["id": row["id"]!, "variant": row["variant"]!, "networkImageSHA256": row["networkImageSHA256"]!,
                    "decodedRGBSHA256": row["decodedRGBSHA256"]!, "actualInputRGBSHA256": rgbHashes, "modes": modes]
        }
        results.append(result)
        try write(result, output.appendingPathComponent("\(row["variant"]!)__\(row["id"]!).json"))
        print("native parity \(row["variant"]!) \(row["id"]!)")
        fflush(stdout)
    }
    try write(["status": "NATIVE_OUTPUTS_CAPTURED_NOT_YET_QUALIFIED", "macOSVersion": ProcessInfo.processInfo.operatingSystemVersionString,
               "requestSHA256": try fileHash(CommandLine.arguments[2]), "records": results,
               "note": "MacCoreML/Vision outputs, not physicaliPhone performance or automatic parityPASS."], output.appendingPathComponent("native-results.json"))
}
do { try main() }
catch { fputs("Native parity failed: \(error)\n", stderr); exit(1) }
