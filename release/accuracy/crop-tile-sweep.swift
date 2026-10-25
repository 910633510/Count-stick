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
let outputURL = CommandLine.arguments.count > 2 ? URL(fileURLWithPath:CommandLine.arguments[2]) : root.appendingPathComponent("release/accuracy/results/crops-and-tiles.json")
try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
let configuration=MLModelConfiguration();configuration.computeUnits = .cpuOnly
let compiled=try MLModel.compileModel(at:root.appendingPathComponent("Count stick/best.mlpackage"))
let model=try MLModel(contentsOf:compiled,configuration:configuration)
let vision=try VNCoreMLModel(for:model)
func predict(_ image:CGImage,_ mode:VNImageCropAndScaleOption,_ confidence:Double,_ threshold:Double) throws -> [VNRecognizedObjectObservation] {
    vision.featureProvider=Thresholds(confidence,threshold)
    let request=VNCoreMLRequest(model:vision);request.imageCropAndScaleOption=mode
    try VNImageRequestHandler(cgImage:image,orientation:.up).perform([request])
    return request.results as? [VNRecognizedObjectObservation] ?? []
}
struct Detection {let rect:CGRect;let score:Double;let source:Int}
func encode(_ detections:[Detection]) -> [[String:Any]] {
    detections.map{["rectTopLeftPixels":array($0.rect),"confidence":$0.score,"tile":$0.source]}
}
func convert(_ observations:[VNRecognizedObjectObservation],_ region:CGRect,_ source:Int) -> [Detection] {
    observations.map{obs in
        let b=obs.boundingBox
        return Detection(rect:CGRect(x:region.minX+b.minX*region.width,y:region.minY+(1-b.maxY)*region.height,width:b.width*region.width,height:b.height*region.height),score:Double(obs.confidence),source:source)
    }
}
func nms(_ list:[Detection],_ threshold:Double) -> [Detection] {
    var sorted=list.sorted{$0.score>$1.score},result:[Detection]=[]
    while let best=sorted.first {result.append(best);sorted.removeFirst();sorted.removeAll{iou($0.rect,best.rect)>threshold}}
    return result
}
let roi=CGRect(x:520,y:700,width:1000,height:800)
let roiImage=load(root.appendingPathComponent("release/accuracy/roi-front-left.png").path)
var cropRows:[[String:Any]]=[]
for (name,mode) in [("scaleFill",VNImageCropAndScaleOption.scaleFill),("scaleFit",.scaleFit)] {
 for confidence in [0.25,0.4,0.5] {for threshold in [0.35,0.45,0.5,0.7] {
    let observations=try predict(roiImage,mode,confidence,threshold)
    cropRows.append(["mode":name,"confidenceThreshold":confidence,"iouThreshold":threshold,"count":observations.count,"detections":encode(convert(observations,roi,-1))])
 }}
}
var tileRows:[[String:Any]]=[],wholeRows:[[String:Any]]=[]
for file in ["bamboo-skewers-cc0.jpg","bamboo-toothpicks-cc0.jpg"] {
 let image=load(root.appendingPathComponent("release/fixtures/"+file).path)
 let source=CGImageSourceCreateWithURL(root.appendingPathComponent("release/fixtures/"+file) as CFURL,nil)!
 let props=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as! [CFString:Any]
 let sourceWidth=Double(props[kCGImagePropertyPixelWidth] as! Int),sourceHeight=Double(props[kCGImagePropertyPixelHeight] as! Int)
 let wholeRegion=CGRect(x:0,y:0,width:sourceWidth,height:sourceHeight)
 for (name,mode) in [("scaleFill",VNImageCropAndScaleOption.scaleFill),("scaleFit",.scaleFit)] {
  for confidence in [0.25,0.4,0.5] { for threshold in [0.35,0.45,0.5,0.7] {
    let list=convert(try predict(image,mode,confidence,threshold),wholeRegion,-1)
    let withinROI=list.filter{roi.contains(CGPoint(x:$0.rect.midX,y:$0.rect.midY))}
    wholeRows.append(["fixture":file,"mode":name,"confidenceThreshold":confidence,"iouThreshold":threshold,"wholeCount":list.count,"roiCount":withinROI.count,"detectionsInROI":encode(withinROI)])
  }}
  for confidence in [0.25,0.4,0.5] {
    for grid in [2,3] {
      let fraction=grid==2 ? 0.6 : 0.4
      let step=(1-fraction)/Double(grid-1)
      var gathered:[Detection]=[]
      let start=Date()
      for y in 0..<grid { for x in 0..<grid {
        let region=CGRect(x:Double(x)*step*Double(image.width),y:Double(y)*step*Double(image.height),width:fraction*Double(image.width),height:fraction*Double(image.height)).integral.intersection(CGRect(x:0,y:0,width:image.width,height:image.height))
        let cropped=image.cropping(to:region)!
        let global=CGRect(x:region.minX/Double(image.width)*sourceWidth,y:region.minY/Double(image.height)*sourceHeight,width:region.width/Double(image.width)*sourceWidth,height:region.height/Double(image.height)*sourceHeight)
        gathered += convert(try predict(cropped,mode,confidence,0.45),global,y*grid+x)
      }}
      for globalThreshold in [0.35,0.45,0.5] {
        let merged=nms(gathered,globalThreshold)
        let withinROI=merged.filter{roi.contains(CGPoint(x:$0.rect.midX,y:$0.rect.midY))}
        tileRows.append(["fixture":file,"mode":name,"grid":grid,"tileFraction":fraction,"confidenceThreshold":confidence,"perTileIoU":0.45,"globalIoU":globalThreshold,"preMergeCount":gathered.count,"count":merged.count,"roiCount":withinROI.count,"seconds":Date().timeIntervalSince(start),"detectionsInROI":encode(withinROI),"detections":encode(merged)])
      }
    }
  }
 }
}
let result:[String:Any]=["coordinateConvention":"All encoded rectangles use original-source pixels, top-left origin, [x,y,width,height]. ROI membership uses center point.","roi":[520,700,1000,800],"roiCropComparisons":cropRows,"wholeImageComparisons":wholeRows,"tileComparisons":tileRows]
try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:outputURL)
