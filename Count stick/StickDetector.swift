import CoreML
import Vision
import UIKit

class StickDetector {
    private let vnModel: VNCoreMLModel

    init?() {
        // 如果你的 CoreML 类名不是 `best`，改这里
        guard let coreml = try? best(configuration: MLModelConfiguration()).model,
              let vnModel = try? VNCoreMLModel(for: coreml) else {
            print("❌ 模型加载失败")
            return nil
        }
        print("✅ 模型加载成功")
        self.vnModel = vnModel
    }

    func detect(from image: UIImage,
                completion: @escaping (_ count: Int, _ boxes: [VNRecognizedObjectObservation]) -> Void) {
        guard let cg = image.cgImage else { return }
        let req = VNCoreMLRequest(model: vnModel) { request, _ in
            let results = (request.results as? [VNRecognizedObjectObservation]) ?? []
            completion(results.count, results)
        }
        req.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cgImage: cg)
        DispatchQueue.global(qos: .userInitiated).async { try? handler.perform([req]) }
    }
}
