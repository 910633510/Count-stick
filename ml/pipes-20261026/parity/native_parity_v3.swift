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
func rawResult(_ raw: MLMultiArray, _ confidence: Double, _ iou: Double, start: Date) throws -> [String: Any] {
    let detections = try PipeRawPostprocessor.predictions(from: raw, confidenceThreshold: Float(confidence), iouThreshold: iou)
    let objects: [[String: Any]] = detections.map { ["label": "pipe_end", "anchor": $0.anchor, "confidence": Double($0.confidence),
        "networkXYXY": [Double($0.x1), Double($0.y1), Double($0.x2), Double($0.y2)]] }
    return ["objects": objects, "seconds": Date().timeIntervalSince(start), "rawShape": raw.shape.map(\.intValue),
            "rawDataType": 65568, "maximumPaddingClassMagnitude": 0.0, "paddingClassesPresent": false,
            "postprocessing": "shared Swift Float32 greedy NMS; no AppleNMS"]
}
func direct(_ model: MLModel, _ input: MLFeatureValue, _ confidence: Double, _ iou: Double) throws -> [String: Any] {
    let provider = try MLDictionaryFeatureProvider(dictionary: ["image": input])
    let start = Date()
    let output = try model.prediction(from: provider)
    guard let raw = output.featureValue(for: "raw_detections")?.multiArrayValue else { throw Failure(description: "Missing raw detections") }
    return try rawResult(raw, confidence, iou, start: start)
}
func vision(_ model: VNCoreMLModel, _ image: CGImage, _ confidence: Double, _ iou: Double) throws -> [String: Any] {
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill
    let start = Date()
    try VNImageRequestHandler(cgImage: image, orientation: .up).perform([request])
    guard let observations = request.results as? [VNCoreMLFeatureValueObservation], observations.count == 1,
          observations[0].featureName == "raw_detections", let raw = observations[0].featureValue.multiArrayValue else {
        throw Failure(description: "Vision did not return raw feature tensor")
    }
    return try rawResult(raw, confidence, iou, start: start)
}
func runNative() throws {
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
    let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: request["packagePath"] as! String))
    var models: [String: MLModel] = [:], visions: [String: VNCoreMLModel] = [:]
    for (name, unit) in [("cpu", MLComputeUnits.cpuOnly), ("cpu_ne", MLComputeUnits.cpuAndNeuralEngine)] {
        let config = MLModelConfiguration(); config.computeUnits = unit
        let model = try MLModel(contentsOf: compiled, configuration: config)
        let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        for (key, value) in request["expectedModelMetadata"] as! [String: String] { try need(metadata[key] == value, "Export model metadata mismatch: \(key)") }
        let visual = try VNCoreMLModel(for: model)
        models[name] = model; visions[name] = visual
    }
    var results: [[String: Any]] = []
    for row in rows {
        let result: [String: Any] = try autoreleasepool {
            let path = row["networkImagePath"] as! String
            try need(try fileHash(path) == row["networkImageSHA256"] as! String, "Network PNG changed")
            let im = try image(path)
            var modes: [String: Any] = [:], rgbHashes: [String: String] = [:]
            for name in ["cpu", "cpu_ne"] {
                guard let constraint = models[name]!.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else { throw Failure(description: "Missing image constraint") }
                try need(constraint.pixelsWide == 1024 && constraint.pixelsHigh == 1024, "Unexpected image dimensions")
                let input = try MLFeatureValue(cgImage: im, constraint: constraint, options: nil)
                rgbHashes[name] = try bufferRGBHash(input)
                modes["direct_\(name)"] = try direct(models[name]!, input, confidence, iou)
                modes["vision_\(name)"] = try vision(visions[name]!, im, confidence, iou)
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
@main struct NativeParityMain {
    static func main() {
        do { try runNative() }
        catch { fputs("Native parity failed: \(error)\n", stderr); exit(1) }
    }
}
