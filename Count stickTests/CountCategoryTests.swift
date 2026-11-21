import XCTest
import UIKit
@testable import Count_stick

@MainActor
final class CountCategoryTests: XCTestCase {
    func testLegacyHistoryDefaultsToSkewersAndBothCategoriesRoundTrip() throws {
        let legacy = """
        {"id":"F94C35C5-C872-4D66-BC38-5C260FB43D60","date":1000,"markers":[],"automaticCount":0,"note":"old"}
        """
        let record = try JSONDecoder().decode(CountRecord.self, from: Data(legacy.utf8))
        XCTAssertEqual(record.category, .skewers)
        XCTAssertNil(record.automaticRegion)
        for category in CountCategory.allCases {
            var current = record
            current.category = category
            XCTAssertEqual(try JSONDecoder().decode(CountRecord.self, from: JSONEncoder().encode(current)), current)
        }
        // Never silently reinterpret an unknown future category as a skewer count.
        let unknown = legacy.replacingOccurrences(of: "\"note\":\"old\"", with: "\"note\":\"old\",\"category\":\"unknown\"")
        XCTAssertThrowsError(try JSONDecoder().decode(CountRecord.self, from: Data(unknown.utf8)))
    }

    func testRouterUsesOnlySelectedDetectorAndDefaultIsSkewers() async throws {
        var stickCalls = 0, pipeCalls = 0
        let router = CountDetectorRouter(skewers: { _ in stickCalls += 1; return Self.result(1) },
                                         pipes: { _ in pipeCalls += 1; return Self.result(2) })
        let storage = TestCountUsage()
        let workspace = CountWorkspace(access: CountingAccess(storage: storage, usesStoreKit: false), router: router)
        XCTAssertEqual(workspace.category, .skewers)
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.automaticCount, 1)
        XCTAssertEqual(stickCalls, 1); XCTAssertEqual(pipeCalls, 0)
        workspace.selectCategory(.steelPipes)
        XCTAssertNil(workspace.image)
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.automaticCount, 2)
        XCTAssertEqual(stickCalls, 1); XCTAssertEqual(pipeCalls, 1)
        XCTAssertEqual(storage.value, 2)
        XCTAssertEqual(workspace.record.category, .steelPipes)
        workspace.reset()
        XCTAssertEqual(workspace.category, .steelPipes)
    }

    func testPipeFailureRetrySuccessAndZeroResultOnlyChargeCompletions() async {
        var shouldFail = true, count = 2
        let storage = TestCountUsage()
        let router = CountDetectorRouter(pipes: { _ in
            if shouldFail { throw PipeCountError.unexpectedOutput }
            return Self.result(count)
        })
        let workspace = CountWorkspace(access: CountingAccess(storage: storage, usesStoreKit: false), router: router)
        workspace.selectCategory(.steelPipes)
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.phase, .failed)
        XCTAssertNil(workspace.automaticCount); XCTAssertEqual(storage.value, 0)
        shouldFail = false
        await workspace.retry()
        XCTAssertEqual(workspace.automaticCount, 2); XCTAssertEqual(storage.value, 1)
        count = 0
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.automaticCount, 0); XCTAssertEqual(storage.value, 2)
        workspace.add(at: CGPoint(x: 0.5, y: 0.5)); workspace.undo(); workspace.markSaved()
        XCTAssertEqual(storage.value, 2)
    }

    func testMissingPipeModelFailsClosedAndManualEditingIsFree() async {
        let storage = TestCountUsage()
        // The test bundle intentionally contains no pipe model or model settings.
        let missing = PipeDetector(bundle: Bundle(for: CountCategoryTests.self))
        let router = CountDetectorRouter(skewers: { _ in XCTFail("Never fall back to the skewer model"); return Self.result(9) },
                                         pipes: { try await missing.detect(from: $0) })
        let workspace = CountWorkspace(access: CountingAccess(storage: storage, usesStoreKit: false), router: router)
        workspace.selectCategory(.steelPipes)
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.phase, .failed)
        XCTAssertEqual(workspace.errorMessage, PipeCountError.unavailableModel.localizedDescription)
        XCTAssertNil(workspace.automaticCount); XCTAssertEqual(storage.value, 0)
        workspace.countManually()
        workspace.add(at: CGPoint(x: 0.2, y: 0.8))
        XCTAssertEqual(workspace.edits.markers.count, 1)
        XCTAssertEqual(workspace.record.category, .steelPipes)
        XCTAssertEqual(storage.value, 0)
    }

    func testSwitchDuringInferenceReleasesReservationAndIgnoresLateResult() async {
        let started = expectation(description: "skewer inference started")
        var completion: CheckedContinuation<CountDetectionResult, Never>?
        let storage = TestCountUsage(); storage.value = 3
        let access = CountingAccess(storage: storage, usesStoreKit: false)
        let router = CountDetectorRouter(skewers: { _ in
            await withCheckedContinuation { continuation in completion = continuation; started.fulfill() }
        }, pipes: { _ in Self.result(2) })
        let workspace = CountWorkspace(access: access, router: router)
        let old = Task { await workspace.load(image: Self.photo()) }
        await fulfillment(of: [started], timeout: 3)
        XCTAssertFalse(access.canStartCount)
        workspace.selectCategory(.steelPipes)
        XCTAssertTrue(access.canStartCount)
        await workspace.load(image: Self.photo())
        let pipeRecord = workspace.record
        XCTAssertEqual(workspace.automaticCount, 2)
        completion?.resume(returning: Self.result(99))
        await old.value
        XCTAssertEqual(workspace.record, pipeRecord)
        XCTAssertEqual(workspace.phase, .ready)
        XCTAssertEqual(workspace.category, .steelPipes)
        XCTAssertEqual(storage.value, 4) // Only the new pipe result consumed the last use.
    }

    func testSwitchWhileAwaitingReservationDoesNotStartOrChargeOldRequest() async {
        let started = expectation(description: "entitlement refresh pending")
        var finishReservation: CheckedContinuation<Void, Never>?
        let storage = TestCountUsage()
        let access = CountingAccess(storage: storage, usesStoreKit: false)
        var calls = 0, first = true
        let workspace = CountWorkspace(access: access,
                                       router: CountDetectorRouter(skewers: { _ in calls += 1; return Self.result(9) },
                                                                   pipes: { _ in calls += 1; return Self.result(1) }),
                                       reservationProvider: {
            if first {
                first = false
                await withCheckedContinuation { continuation in finishReservation = continuation; started.fulfill() }
            }
            return try await access.reserveCount()
        })
        let old = Task { await workspace.load(image: Self.photo()) }
        await fulfillment(of: [started], timeout: 3)
        workspace.selectCategory(.steelPipes)
        await workspace.load(image: Self.photo())
        let current = workspace.record
        finishReservation?.resume()
        await old.value
        XCTAssertEqual(calls, 1)
        XCTAssertEqual(storage.value, 1)
        XCTAssertEqual(workspace.record, current)
        XCTAssertTrue(access.canStartCount)
    }

    func testPipeHistoryRestoresModeAndExportCaptionPreservesCategoryAndNote() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let record = CountRecord(id: UUID(), date: Date(), markers: Self.result(2).markers, automaticCount: 2,
                                 note: "Warehouse 2", category: .steelPipes)
        try CountHistory(directory: directory).save(record, image: Self.photo())
        let history = CountHistory(directory: directory)
        let restored = try XCTUnwrap(history.records.first)
        let storage = TestCountUsage()
        let workspace = CountWorkspace(access: CountingAccess(storage: storage, usesStoreKit: false))
        workspace.restore(restored, image: try XCTUnwrap(history.image(for: restored)))
        XCTAssertEqual(workspace.category, .steelPipes)
        XCTAssertTrue(workspace.isSaved)
        workspace.add(at: CGPoint(x: 0.8, y: 0.2))
        XCTAssertTrue(workspace.hasUnsavedChanges)
        XCTAssertEqual(workspace.record.category, .steelPipes)
        let caption = CountExport.caption(category: restored.category, count: restored.count, note: restored.note)
        XCTAssertTrue(caption.contains(CountCategory.steelPipes.title))
        XCTAssertTrue(caption.contains(CountCategory.steelPipes.formattedCount(2)))
        XCTAssertTrue(caption.contains("Warehouse 2"))
        XCTAssertEqual(storage.value, 0)
    }

    func testPipeRegionKeepsSeparatedAndIsolatedEndsWithoutSkewerFiltering() {
        let detections = [
            AutoCountDetection(id: UUID(), box: AutoCountBox(x: 0.1, y: 0.1, width: 0.1, height: 0.1), confidence: 0.6),
            AutoCountDetection(id: UUID(), box: AutoCountBox(x: 0.85, y: 0.8, width: 0.1, height: 0.1), confidence: 0.3)
        ]
        let result = PipeCountRegion.result(for: detections)
        XCTAssertEqual(result.markers.map(\.id), detections.map(\.id))
        XCTAssertTrue(result.automaticRegion.excluded.isEmpty)
        XCTAssertEqual(result.automaticRegion.regions.count, 1)
        XCTAssertEqual(result.automaticRegion.regions[0].x, 0.1, accuracy: 1e-10)
        XCTAssertEqual(result.automaticRegion.regions[0].width, 0.85, accuracy: 1e-10)
    }

    func testLetterboxMapsLandscapePortraitAndOddPaddingBackToOriginal() throws {
        for (width, height) in [(1200, 600), (600, 1200), (1101, 719)] {
            let geometry = try PipeLetterbox(width: width, height: height)
            let original = AutoCountBox(x: 0.2, y: 0.3, width: 0.25, height: 0.15)
            let network = AutoCountBox(x: (original.x * Double(width) * geometry.gain + Double(geometry.left)) / 1024,
                                       y: (original.y * Double(height) * geometry.gain + Double(geometry.top)) / 1024,
                                       width: original.width * Double(width) * geometry.gain / 1024,
                                       height: original.height * Double(height) * geometry.gain / 1024)
            let restored = try XCTUnwrap(geometry.originalBox(fromNetwork: network))
            XCTAssertEqual(restored.x, original.x, accuracy: 1e-10)
            XCTAssertEqual(restored.y, original.y, accuracy: 1e-10)
            XCTAssertEqual(restored.width, original.width, accuracy: 1e-10)
            XCTAssertEqual(restored.height, original.height, accuracy: 1e-10)
        }
        let landscape = try PipeLetterbox(width: 1200, height: 600)
        XCTAssertEqual(landscape.left, 0); XCTAssertEqual(landscape.top, 256)
        XCTAssertNil(landscape.originalBox(fromNetwork: AutoCountBox(x: 0.2, y: 0.05, width: 0.1, height: 0.1)))
        let full = try XCTUnwrap(landscape.originalBox(fromNetwork: AutoCountBox(x: 0, y: 0, width: 1, height: 1)))
        XCTAssertEqual(full, AutoCountBox(x: 0, y: 0, width: 1, height: 1))
        XCTAssertThrowsError(try PipeLetterbox(width: 0, height: 20))
        XCTAssertThrowsError(try PipeLetterbox(width: 100_000, height: 1))
    }

    func testLetterboxUsesGray114PaddingAndPreservesUprightSource() throws {
        let geometry = try PipeLetterbox(width: 32, height: 24)
        let prepared = geometry.prepare(Self.photo())
        let image = try XCTUnwrap(prepared.cgImage)
        XCTAssertEqual(image.width, 1024); XCTAssertEqual(image.height, 1024)
        XCTAssertEqual(prepared.imageOrientation, .up)
        func pixel(_ x: Int, _ y: Int) throws -> [UInt8] {
            let crop = try XCTUnwrap(image.cropping(to: CGRect(x: x, y: y, width: 1, height: 1)))
            var bytes = [UInt8](repeating: 0, count: 4)
            try bytes.withUnsafeMutableBytes { memory in
                let context = try XCTUnwrap(CGContext(data: memory.baseAddress, width: 1, height: 1, bitsPerComponent: 8,
                                                     bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
                                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
                context.draw(crop, in: CGRect(x: 0, y: 0, width: 1, height: 1))
            }
            return bytes
        }
        XCTAssertEqual(try pixel(10, 10), [114, 114, 114, 255])
        XCTAssertEqual(try pixel(512, 512), [255, 255, 255, 255])
    }

    func testUnvalidatedOrWrongClassPipeConfigurationIsRejected() throws {
        func config(_ status: String, label: String = "pipe_end", confidence: Double = 0.3) -> PipeModelConfiguration {
            PipeModelConfiguration(status: status, exportID: UUID(), checkpointSHA256: String(repeating: "a", count: 64),
                                   inputSize: 1024, classLabel: label,
                                   confidenceThreshold: confidence, iouThreshold: 0.7)
        }
        XCTAssertThrowsError(try config("PENDING").validate())
        XCTAssertThrowsError(try config("VALIDATED_COREML_EXPORT", label: "stick").validate())
        XCTAssertThrowsError(try config("VALIDATED_COREML_EXPORT", confidence: .nan).validate())
        XCTAssertNoThrow(try config("VALIDATED_COREML_EXPORT").validate())
    }

    func testPipeConfigurationMustMatchExportedModelMetadata() throws {
        let config = PipeModelConfiguration(status: "VALIDATED_COREML_EXPORT", exportID: UUID(),
                                            checkpointSHA256: String(repeating: "a", count: 64), inputSize: 1024,
                                            classLabel: "pipe_end", confidenceThreshold: 0.3, iouThreshold: 0.7)
        let metadata = ["countscope.export_id": config.exportID.uuidString,
                        "countscope.checkpoint_sha256": config.checkpointSHA256,
                        "countscope.confidence_threshold": "0.3", "countscope.iou_threshold": "0.7",
                        "countscope.class_label": "pipe_end", "countscope.input_size": "1024"]
        XCTAssertNoThrow(try config.validate(metadata: metadata))
        XCTAssertThrowsError(try config.validate(metadata: [:]))
        for (key, value) in ["countscope.export_id": UUID().uuidString,
                             "countscope.checkpoint_sha256": String(repeating: "b", count: 64),
                             "countscope.confidence_threshold": "0.4", "countscope.iou_threshold": "0.5",
                             "countscope.class_label": "stick", "countscope.input_size": "640"] {
            var mismatch = metadata
            mismatch[key] = value
            XCTAssertThrowsError(try config.validate(metadata: mismatch), "Must reject mismatched \(key)")
        }
    }

    private static func photo() -> UIImage {
        UIGraphicsImageRenderer(size: CGSize(width: 32, height: 24)).image { context in
            UIColor.white.setFill(); context.fill(CGRect(x: 0, y: 0, width: 32, height: 24))
        }
    }
    private static func result(_ count: Int) -> CountDetectionResult {
        let markers = (0..<count).map { _ in CountMarker(x: 0.5, y: 0.5) }
        return CountDetectionResult(markers: markers, rawMarkers: markers,
                                    automaticRegion: AutoCountRegionResult(keptIDs: markers.map(\.id), excluded: [], regions: [], fallbackReason: .empty))
    }
}
