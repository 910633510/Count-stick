import SwiftUI
import Vision
import AVFoundation
import UIKit

struct ContentView: View {
    @State private var image: UIImage?
    @State private var detectionResults: [VNRecognizedObjectObservation] = []
    @State private var detectedCount: Int = 0

    // 分离的呈现状态
    @State private var showCamera = false
    @State private var showLibrary = false
    @State private var showSettingsAlert = false

    let detector = StickDetector()

    var body: some View {
        VStack {
            if let img = image {
                GeometryReader { geo in
                    let viewSize = geo.size
                    let imgSize = img.size
                    let scale = min(viewSize.width / imgSize.width,
                                    viewSize.height / imgSize.height)
                    let displaySize = CGSize(width: imgSize.width * scale,
                                             height: imgSize.height * scale)
                    let xOffset = (viewSize.width - displaySize.width) / 2
                    let yOffset = (viewSize.height - displaySize.height) / 2

                    ZStack {
                        Image(uiImage: img)
                            .resizable()
                            .scaledToFit()
                            .frame(width: viewSize.width, height: viewSize.height)

                        ForEach(detectionResults, id: \.uuid) { r in
                            let px = r.boundingBox.midX * imgSize.width
                            let py = (1 - r.boundingBox.midY) * imgSize.height
                            let x = px * scale + xOffset
                            let y = py * scale + yOffset
                            Circle().fill(Color.red).frame(width: 8, height: 8)
                                .position(x: x, y: y)
                        }
                    }
                }
                .frame(height: 400)

                Text("签子数量：\(detectedCount)")
                    .font(.title2)
                    .padding(.top, 8)
            } else {
                Spacer()
                Text("请拍照或选择图片").foregroundColor(.gray)
                Spacer()
            }

            HStack(spacing: 40) {
                Button { openCamera() } label: {
                    Label("拍照", systemImage: "camera")
                }
                Button {
                    showLibrary = true
                } label: {
                    Label("相册", systemImage: "photo.on.rectangle")
                }
            }
            .font(.title2)
            .padding(.vertical, 16)
        }
        // 相册：sheet
        .sheet(isPresented: $showLibrary) {
            ImagePicker(image: $image, sourceType: .photoLibrary) {
                print("📸 Library dismissed")
            }
        }
        // 相机：fullScreenCover（与相册完全分离）
        .fullScreenCover(isPresented: $showCamera) {
            ImagePicker(image: $image, sourceType: .camera) {
                print("📸 Camera dismissed")
            }
        }
        // iOS 17 新签名
        .onChange(of: image) { _, newImage in
            guard let uiImage = newImage else { return }
            detector?.detect(from: uiImage) { count, results in
                DispatchQueue.main.async {
                    print("🎯 检测到 \(count) 个签子")
                    detectedCount = count
                    detectionResults = results
                }
            }
        }
        .onAppear {
            print("NSCameraUsageDescription =",
                  Bundle.main.object(forInfoDictionaryKey: "NSCameraUsageDescription") as Any)
            print("NSPhotoLibraryUsageDescription =",
                  Bundle.main.object(forInfoDictionaryKey: "NSPhotoLibraryUsageDescription") as Any)
        }
        .alert("需要相机权限", isPresented: $showSettingsAlert) {
            Button("去设置") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            Button("取消", role: .cancel) {}
        } message: {
            Text("请在“设置 → 隐私与安全 → 相机”里允许访问。")
        }
    }

    // 只在这里做可用性 + 权限检查；通过才弹相机
    private func openCamera() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            print("⚠️ 设备无相机")
            showSettingsAlert = true
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        self.showCamera = true
                    } else {
                        self.showSettingsAlert = true
                    }
                }
            }
        case .authorized:
            showCamera = true
        case .denied, .restricted:
            showSettingsAlert = true
        @unknown default:
            showSettingsAlert = true
        }

    }
}

#Preview { ContentView() }
