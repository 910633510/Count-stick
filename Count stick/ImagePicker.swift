import SwiftUI
import UIKit

struct ImagePicker: UIViewControllerRepresentable {
    enum SourceType { case camera, photoLibrary }

    @Binding var image: UIImage?
    var sourceType: SourceType
    var onDismiss: (() -> Void)? = nil

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.allowsEditing = false
        picker.modalPresentationStyle = .fullScreen

        switch sourceType {
        case .camera:
            picker.sourceType = .camera                 // ✅ 不做回退
            picker.cameraDevice = .rear
            print("🎥 picker = CAMERA")
        case .photoLibrary:
            picker.sourceType = .photoLibrary
            print("🎞️ picker = PHOTO_LIBRARY")
        }
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        init(_ parent: ImagePicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let uiImage = info[.originalImage] as? UIImage { parent.image = uiImage }
            picker.dismiss(animated: true) { self.parent.onDismiss?() }
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true) { self.parent.onDismiss?() }
        }
    }
}
