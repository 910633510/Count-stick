import Foundation
import CoreML
import Vision
import ImageIO
import CoreGraphics
import CoreVideo

final class Thresholds: MLFeatureProvider {
    let confidence: Double
    let iou: Double
    init(_ confidence: Double, _ iou: Double) { self.confidence=confidence; self.iou=iou }
    var featureNames: Set<String> { ["confidenceThreshold", "iouThreshold"] }
    func featureValue(for name: String) -> MLFeatureValue? {
        switch name {
        case "confidenceThreshold": return MLFeatureValue(double: confidence)
        case "iouThreshold": return MLFeatureValue(double: iou)
        default: return nil
        }
    }
}

func load(_ path:String) -> CGImage {
    let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:path) as CFURL,nil)!
    return CGImageSourceCreateThumbnailAtIndex(source,0,[kCGImageSourceCreateThumbnailFromImageAlways:true,kCGImageSourceCreateThumbnailWithTransform:true,kCGImageSourceThumbnailMaxPixelSize:2048,kCGImageSourceShouldCacheImmediately:true] as CFDictionary)!
}
func array(_ box:CGRect) -> [Double] {[Double(box.minX),Double(box.minY),Double(box.width),Double(box.height)]}
func iou(_ a:CGRect,_ b:CGRect) -> Double {
    let intersection=a.intersection(b)
    if intersection.isNull {return 0}
    let inter=intersection.width*intersection.height
    return inter/(a.width*a.height+b.width*b.height-inter)
}
func summary(_ observations:[VNRecognizedObjectObservation], includeBoxes:Bool=false) -> [String:Any] {
    let scores=observations.map{Double($0.confidence)}.sorted()
    var pairs=["above03":0,"above05":0,"above07":0]
    var highest:[(Double,Int,Int)] = []
    if observations.count>1 {
        for a in 0..<(observations.count-1) { for b in (a+1)..<observations.count {
            let overlap=iou(observations[a].boundingBox, observations[b].boundingBox)
            if overlap>0.3 { pairs["above03",default:0]+=1;highest.append((overlap,a,b)) }
            if overlap>0.5 { pairs["above05",default:0]+=1 }
            if overlap>0.7 { pairs["above07",default:0]+=1 }
        }}
    }
    var result:[String:Any]=["count":observations.count,"confidenceMin":scores.first ?? 0,"confidenceMax":scores.last ?? 0,"confidenceMedian":scores.isEmpty ? 0 : scores[scores.count/2],"scoreBelow035":scores.filter{$0<0.35}.count,"scoreBelow045":scores.filter{$0<0.45}.count,"scoreBelow055":scores.filter{$0<0.55}.count,"allLabelScoresOne":observations.allSatisfy{$0.labels.allSatisfy{$0.confidence==1}},"overlapPairs":pairs,"highestOverlapPairs":highest.sorted{$0.0>$1.0}.prefix(8).map{["iou":$0.0,"a":$0.1,"b":$0.2]}]
    if includeBoxes { result["detections"]=observations.enumerated().map{index,observation in ["index":index,"confidence":Double(observation.confidence),"labelConfidence":Double(observation.labels.first?.confidence ?? 0),"rectBottomLeft":array(observation.boundingBox)] as [String:Any]} }
    return result
}
func letterbox(_ image:CGImage) -> CGImage {
    let context=CGContext(data:nil,width:640,height:640,bitsPerComponent:8,bytesPerRow:640*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(red:114/255,green:114/255,blue:114/255,alpha:1)
    context.fill(CGRect(x:0,y:0,width:640,height:640))
    let scale=min(640/Double(image.width),640/Double(image.height))
    let width=Double(image.width)*scale,height=Double(image.height)*scale
    context.interpolationQuality = .high
    context.draw(image,in:CGRect(x:(640-width)/2,y:(640-height)/2,width:width,height:height))
    return context.makeImage()!
}
let root=URL(fileURLWithPath:CommandLine.arguments[1])
let outputURL = CommandLine.arguments.count > 2 ? URL(fileURLWithPath:CommandLine.arguments[2]) : root.appendingPathComponent("release/accuracy/results/thresholds.json")
try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
let configuration=MLModelConfiguration();configuration.computeUnits = .cpuOnly
let compiled=try MLModel.compileModel(at:root.appendingPathComponent("Count stick/best.mlpackage"))
let model=try MLModel(contentsOf:compiled,configuration:configuration)
let vision=try VNCoreMLModel(for:model)
let modes:[(String,VNImageCropAndScaleOption)]=[("scaleFill",.scaleFill),("scaleFit",.scaleFit),("centerCrop",.centerCrop)]
var samples:[[String:Any]]=[]
for file in ["bamboo-skewers-cc0.jpg","bamboo-toothpicks-cc0.jpg"] {
    let image=load(root.appendingPathComponent("release/fixtures/"+file).path)
    var baseline:[[String:Any]]=[], sweep:[[String:Any]]=[]
    for (modeName,mode) in modes {
        vision.featureProvider=nil
        let request=VNCoreMLRequest(model:vision);request.imageCropAndScaleOption=mode
        try VNImageRequestHandler(cgImage:image,orientation:.up).perform([request])
        var base=summary(request.results as? [VNRecognizedObjectObservation] ?? [],includeBoxes:true)
        base["mode"]=modeName; baseline.append(base)
        for confidence in [0.10,0.15,0.25,0.35,0.45,0.55,0.65] {
            for overlap in [0.30,0.45,0.50,0.70] {
                vision.featureProvider=Thresholds(confidence,overlap)
                let request=VNCoreMLRequest(model:vision);request.imageCropAndScaleOption=mode
                try VNImageRequestHandler(cgImage:image,orientation:.up).perform([request])
                var row=summary(request.results as? [VNRecognizedObjectObservation] ?? [],includeBoxes:confidence==0.45 && overlap==0.45)
                row["mode"]=modeName;row["confidenceThreshold"]=confidence;row["iouThreshold"]=overlap
                sweep.append(row)
            }
        }
    }
    var direct:[[String:Any]]=[]
    for (name,input) in [("scaleFill",image),("letterboxGray114",letterbox(image))] {
        let imageValue=try MLFeatureValue(cgImage:input,constraint:model.modelDescription.inputDescriptionsByName["image"]!.imageConstraint!,options:[.cropAndScale:VNImageCropAndScaleOption.scaleFill.rawValue])
        for confidence in [0.10,0.25,0.45] {
            let provider=try MLDictionaryFeatureProvider(dictionary:["image":imageValue,"confidenceThreshold":confidence,"iouThreshold":0.45])
            let output=try model.prediction(from:provider)
            let confidences=output.featureValue(for:"confidence")!.multiArrayValue!
            let coordinates=output.featureValue(for:"coordinates")!.multiArrayValue!
            let detections=(0..<confidences.shape[0].intValue).map { i -> [String:Any] in
                ["score":confidences[[NSNumber(value:i),0]].doubleValue,"centerXYWHOriginTopLeft":(0..<4).map{coordinates[[NSNumber(value:i),NSNumber(value:$0)]].doubleValue}]
            }
            direct.append(["mode":name,"confidenceThreshold":confidence,"iouThreshold":0.45,"confidenceShape":confidences.shape,"coordinatesShape":coordinates.shape,"detections":detections])
        }
    }
    samples.append(["fixture":file,"processedPixels":[image.width,image.height],"baseline":baseline,"sweep":sweep,"directPrediction":direct])
    FileHandle.standardError.write(Data("Finished \(file)\n".utf8))
}
let metadata=model.modelDescription.metadata[.creatorDefinedKey] as? [String:String] ?? [:]
let output:[String:Any]=["dateLocal":"2026-10-23","platform":"macOS CoreML CPU-only diagnostic; not physical iPhone accuracy validation","sourceWeightsUnchanged":true,"technicalMetadata":metadata.filter{["names","imgsz","task","stride","batch","date","args","Confidence threshold","IoU threshold"].contains($0.key)},"samples":samples]
let data=try JSONSerialization.data(withJSONObject:output,options:[.prettyPrinted,.sortedKeys])
try data.write(to:outputURL)
