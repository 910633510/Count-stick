import Foundation
import CoreML
import Vision
import ImageIO

let args = CommandLine.arguments
let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuOnly
let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: args[1]))
let model = try MLModel(contentsOf: compiled, configuration: configuration)
let vision = try VNCoreMLModel(for: model)
func feature(_ value: MLFeatureDescription) -> [String: Any] {
    var result: [String: Any] = ["type": String(describing: value.type), "optional": value.isOptional]
    if let image = value.imageConstraint {
        result["image"] = ["width":image.pixelsWide,"height":image.pixelsHigh,"pixelFormat":image.pixelFormatType]
    }
    if let array = value.multiArrayConstraint {
        result["array"] = ["shape":array.shape,"dataType":String(describing: array.dataType)]
    }
    return result
}
var result: [String: Any] = [
    "executionPlatform": "macOS CoreML CPU only; not iPhone acceptance",
    "inputs": model.modelDescription.inputDescriptionsByName.mapValues(feature),
    "outputs": model.modelDescription.outputDescriptionsByName.mapValues(feature),
    "visionModelLoaded": true
]
let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String:String] ?? [:]
result["technicalMetadata"] = metadata.filter { ["names","imgsz","task","stride","batch","date"].contains($0.key) }
var samples: [[String:Any]] = []
for path in args.dropFirst(2) {
    let url = URL(fileURLWithPath:path)
    let source = CGImageSourceCreateWithURL(url as CFURL,nil)!
    let properties = CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [String:Any] ?? [:]
    let originalOrientation = (properties[kCGImagePropertyOrientation as String] as? UInt32) ?? 1
    let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 2048,
        kCGImageSourceShouldCacheImmediately: true
    ] as CFDictionary)!
    let request = VNCoreMLRequest(model:vision)
    request.imageCropAndScaleOption = .scaleFill
    let start = Date()
    try VNImageRequestHandler(cgImage:cg,orientation:.up).perform([request])
    let observations = request.results as? [VNRecognizedObjectObservation]
    samples.append([
        "path":path,"processedPixels":[cg.width,cg.height],"sourceOrientation":originalOrientation,"inferenceOrientation":1,
        "imageProcessing":"ImageIO thumbnail max 2048 with EXIF transform, matching PhotoProcessing.load",
        "cropScale":"scaleFill","elapsedSeconds":Date().timeIntervalSince(start),
        "recognizedObjectType":observations != nil,
        "count":observations?.count ?? 0,
        "results": (observations ?? []).map { item -> [String:Any] in
            ["observationConfidence":item.confidence,"labels":item.labels.map { ["identifier":$0.identifier,"confidence":$0.confidence] },
             "box":["x":item.boundingBox.minX,"y":item.boundingBox.minY,"width":item.boundingBox.width,"height":item.boundingBox.height]]
        }
    ])
}
result["samples"] = samples
let colorSpace = CGColorSpaceCreateDeviceRGB()
let blankContext = CGContext(data:nil,width:640,height:640,bitsPerComponent:8,bytesPerRow:640*4,space:colorSpace,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
blankContext.setFillColor(red:1,green:1,blue:1,alpha:1)
blankContext.fill(CGRect(x:0,y:0,width:640,height:640))
let blankRequest = VNCoreMLRequest(model:vision)
blankRequest.imageCropAndScaleOption = .scaleFill
try VNImageRequestHandler(cgImage:blankContext.makeImage()!,orientation:.up).perform([blankRequest])
result["syntheticNegativeSmoke"] = ["description":"Uniform white square; synthetic negative input, not real-world accuracy evidence","count":(blankRequest.results as? [VNRecognizedObjectObservation])?.count ?? -1]
let json = try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
print(String(data:json,encoding:.utf8)!)
