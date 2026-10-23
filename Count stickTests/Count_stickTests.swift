import Testing
import UIKit
import ImageIO
import UniformTypeIdentifiers
@testable import Count_stick

struct Count_stickTests {
    @Test func markerEditsMaintainCountAndUndo() {
        let automatic = CountMarker(x: 0.25, y: 0.75)
        var edits = MarkerEdits()
        edits.replace(with: [automatic])
        edits.add(at: CGPoint(x: 0.9, y: 0.1))
        #expect(edits.markers.count == 2)
        #expect(edits.markers.last?.isManual == true)
        edits.remove(id: automatic.id)
        #expect(edits.markers.count == 1)
        edits.undo()
        #expect(edits.markers.count == 2)
        edits.undo()
        #expect(edits.markers == [automatic])
        #expect(!edits.canUndo)
    }

    @Test func invalidTapsAndUnknownRemovalDoNotCreateUndo() {
        var edits = MarkerEdits()
        edits.add(at: CGPoint(x: -0.01, y: 0.5))
        edits.add(at: CGPoint(x: .nan, y: 0.5))
        edits.add(at: CGPoint(x: 0.5, y: .infinity))
        edits.remove(id: UUID())
        #expect(edits.markers.isEmpty)
        #expect(!edits.canUndo)
    }

    @Test func newPhotoResetsUndoAndMarkers() {
        var edits = MarkerEdits()
        edits.add(at: CGPoint(x: 0.5, y: 0.5))
        let detected = [CountMarker(x: 0.2, y: 0.3), CountMarker(x: 0.6, y: 0.8)]
        edits.replace(with: detected)
        edits.undo()
        #expect(edits.markers == detected)
        #expect(!edits.canUndo)
    }

    @Test func fittingRejectsLetterboxAndPreservesCoordinates() throws {
        let rect = ImageGeometry.fittedRect(image: CGSize(width: 400, height: 200), canvas: CGSize(width: 300, height: 400))
        #expect(rect == CGRect(x: 0, y: 125, width: 300, height: 150))
        #expect(ImageGeometry.normalizedPoint(CGPoint(x: 150, y: 50), in: rect) == nil)
        let marker = CountMarker(x: 0.3, y: 0.7)
        let position = ImageGeometry.position(of: marker, in: rect)
        let normalized = try #require(ImageGeometry.normalizedPoint(position, in: rect))
        #expect(abs(normalized.x - marker.x) < 0.00001)
        #expect(abs(normalized.y - marker.y) < 0.00001)
        #expect(ImageGeometry.fittedRect(image: .zero, canvas: CGSize(width: 300, height: 400)) == .zero)
    }

    @Test func normalizingRotationAndMirroringTransformsPixels() throws {
        let source = coloredFixture()
        let cg = try #require(source.cgImage)
        let rotated = try PhotoProcessing.normalized(UIImage(cgImage: cg, scale: 1, orientation: .right))
        #expect(rotated.imageOrientation == .up)
        #expect(rotated.size == CGSize(width: 30, height: 60))
        #expect(try isRed(rotated, at: CGPoint(x: 15, y: 5)))
        #expect(try !isRed(rotated, at: CGPoint(x: 15, y: 55)))
        let mirrored = try PhotoProcessing.normalized(UIImage(cgImage: cg, scale: 1, orientation: .upMirrored))
        #expect(try !isRed(mirrored, at: CGPoint(x: 5, y: 15)))
        #expect(try isRed(mirrored, at: CGPoint(x: 55, y: 15)))
    }

    @Test func decodingAppliesEXIFOrientation() throws {
        let data = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil))
        let cg = try #require(coloredFixture().cgImage)
        CGImageDestinationAddImage(destination, cg, [kCGImagePropertyOrientation: 6] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        let image = try PhotoProcessing.load(data: data as Data)
        #expect(image.size == CGSize(width: 30, height: 60))
        #expect(image.imageOrientation == .up)
        #expect(try isRed(image, at: CGPoint(x: 15, y: 5)))
        #expect(try !isRed(image, at: CGPoint(x: 15, y: 55)))
    }

    @Test func decodingLimitsLargeImagesAndRejectsInvalidData() throws {
        let data = try #require(fixture(size: CGSize(width: 3000, height: 1500)).jpegData(compressionQuality: 0.8))
        let image = try PhotoProcessing.load(data: data)
        #expect(max(image.size.width, image.size.height) <= PhotoProcessing.maximumDimension)
        #expect(image.imageOrientation == .up)
        #expect(throws: CountError.self) { try PhotoProcessing.load(data: Data([0, 1, 2])) }
    }

    @Test func exportedResultContainsPhotoAndFooter() {
        let photo = fixture(size: CGSize(width: 400, height: 200))
        let result = PhotoProcessing.shareImage(photo: photo, markers: [CountMarker(x: 0.5, y: 0.5)], note: "Batch 1")
        #expect(result.size.width == 400)
        #expect(result.size.height > photo.size.height)
        #expect(result.jpegData(compressionQuality: 0.8) != nil)
    }

    @Test func exportGrowsToPreserveLongUnicodeNotes() {
        let photo = fixture(size: CGSize(width: 400, height: 200))
        let short = PhotoProcessing.shareImage(photo: photo, markers: [], note: "Dinner prep")
        let long = PhotoProcessing.shareImage(photo: photo, markers: [], note: String(repeating: "晚餐备料", count: 30))
        #expect(long.size.width == short.size.width)
        #expect(long.size.height > short.size.height)
    }

    @MainActor @Test func historyPersistsReplacesAndDeletesWithoutDuplicates() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let history = CountHistory(directory: directory)
        let photo = fixture(size: CGSize(width: 100, height: 80))
        var record = CountRecord(id: UUID(), date: Date(timeIntervalSince1970: 1000), markers: [CountMarker(x: 0.5, y: 0.2)], automaticCount: 1, note: "Test batch")
        try history.save(record, image: photo)
        record.markers.append(CountMarker(x: 0.2, y: 0.7, isManual: true))
        try history.save(record, image: photo)
        #expect(history.records.count == 1)
        let restored = CountHistory(directory: directory)
        #expect(restored.records == [record])
        #expect(restored.image(for: record) != nil)
        #expect(restored.image(for: record, thumbnail: true) != nil)
        #expect(try directory.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup == true)
        try restored.delete(record)
        #expect(restored.records.isEmpty)
        #expect(restored.image(for: record) == nil)
        #expect(restored.image(for: record, thumbnail: true) == nil)
        #expect(CountHistory(directory: directory).records.isEmpty)
    }

    @MainActor @Test func corruptHistoryShowsErrorWithoutOverwritingFile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("records.json")
        let original = Data("invalid json".utf8)
        try original.write(to: url)
        let history = CountHistory(directory: directory)
        #expect(history.errorMessage != nil)
        let record = CountRecord(id: UUID(), date: Date(), markers: [], automaticCount: 0, note: "Must not overwrite")
        #expect(throws: (any Error).self) { try history.save(record, image: fixture(size: CGSize(width: 100, height: 80))) }
        #expect(try Data(contentsOf: url) == original)
    }

    @Test func bundledDetectorRunsRealModelOnBlankPhoto() async throws {
        let image = fixture(size: CGSize(width: 640, height: 640))
        let markers = try await StickDetector().detect(from: image)
        #expect(markers.allSatisfy { $0.x.isFinite && $0.y.isFinite && (0...1).contains($0.x) && (0...1).contains($0.y) })
        // This is an integration smoke test, not evidence of accuracy on real skewers.
        #expect(markers.isEmpty)
    }

    private func coloredFixture() -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: CGSize(width: 60, height: 30), format: format).image { context in
            UIColor.red.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 30, height: 30))
            UIColor.blue.setFill()
            context.fill(CGRect(x: 30, y: 0, width: 30, height: 30))
        }
    }

    private func isRed(_ image: UIImage, at point: CGPoint) throws -> Bool {
        let source = try #require(image.cgImage)
        let pixel = try #require(source.cropping(to: CGRect(x: point.x, y: point.y, width: 1, height: 1)))
        var bytes = [UInt8](repeating: 0, count: 4)
        try bytes.withUnsafeMutableBytes { buffer in
            let context = try #require(CGContext(data: buffer.baseAddress, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            context.draw(pixel, in: CGRect(x: 0, y: 0, width: 1, height: 1))
        }
        return bytes[0] > bytes[2]
    }

    private func fixture(size: CGSize) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            UIColor.white.setFill()
            context.fill(CGRect(origin: .zero, size: size))
        }
    }
}
