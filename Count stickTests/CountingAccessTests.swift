import XCTest
import UIKit
import Security
@testable import Count_stick

final class TestCountUsage: CountUsageStorage {
    var value = 0
    var readFails = false
    var writeFails = false
    func read() throws -> Int { if readFails { throw CountAccessError.storageUnavailable }; return value }
    func write(_ used: Int) throws { if writeFails { throw CountAccessError.storageUnavailable }; value = used }
}

@MainActor
final class CountingAccessTests: XCTestCase {
    func testFourCompletedCountsPersistAndFifthIsDenied() async throws {
        let storage = TestCountUsage()
        let access = CountingAccess(storage: storage, usesStoreKit: false)
        for remaining in (0...3).reversed() {
            let token = try await access.reserveCount()
            XCTAssertEqual(storage.value, 3 - remaining)
            try access.completeCount(token)
            XCTAssertEqual(access.remainingFreeCounts, remaining)
        }
        let relaunched = CountingAccess(storage: storage, usesStoreKit: false)
        XCTAssertEqual(relaunched.remainingFreeCounts, 0)
        do { _ = try await relaunched.reserveCount(); XCTFail("A fifth count must be denied") }
        catch CountAccessError.subscriptionRequired {} catch { XCTFail("Unexpected \(error)") }
    }

    func testReservationsPreventConcurrentOverspendAndCancelIsFree() async throws {
        let access = CountingAccess(storage: TestCountUsage(), usesStoreKit: false)
        var tokens: [CountingAccess.Reservation] = []
        for _ in 0..<4 { tokens.append(try await access.reserveCount()) }
        XCTAssertEqual(access.remainingFreeCounts, 4)
        do { _ = try await access.reserveCount(); XCTFail("All free uses are reserved") }
        catch CountAccessError.subscriptionRequired {} catch { XCTFail("Unexpected \(error)") }
        access.cancelCount(tokens[0])
        let replacement = try await access.reserveCount()
        try access.completeCount(replacement)
        XCTAssertEqual(access.remainingFreeCounts, 3)
        do { try access.completeCount(replacement); XCTFail("A reservation cannot debit twice") }
        catch CountAccessError.invalidReservation {} catch { XCTFail("Unexpected \(error)") }
        for token in tokens { access.cancelCount(token) }
        XCTAssertEqual(access.remainingFreeCounts, 3)
    }

    func testKeychainPersistenceAcrossNewStoreInstances() async throws {
        let service = "x.Count-stick.unit-tests." + UUID().uuidString
        let storage = KeychainCountUsage(service: service)
        defer { SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service] as CFDictionary) }
        let first = CountingAccess(storage: storage, usesStoreKit: false)
        try first.completeCount(try await first.reserveCount())
        let second = CountingAccess(storage: KeychainCountUsage(service: service), usesStoreKit: false)
        XCTAssertEqual(second.usedFreeCounts, 1)
        XCTAssertEqual(second.remainingFreeCounts, 3)
    }

    func testStorageFailureDoesNotExposeResultOrLoseAllowance() async throws {
        let storage = TestCountUsage()
        storage.writeFails = true
        let access = CountingAccess(storage: storage, usesStoreKit: false)
        let workspace = CountWorkspace(access: access, detector: { _ in Self.result() })
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.phase, .failed)
        XCTAssertNil(workspace.automaticCount)
        XCTAssertEqual(storage.value, 0)
        storage.writeFails = false
        await workspace.retry()
        XCTAssertEqual(workspace.phase, .ready)
        XCTAssertEqual(storage.value, 1)
    }

    func testReadFailureFailsClosedAndRecovers() async throws {
        let storage = TestCountUsage(); storage.readFails = true
        let access = CountingAccess(storage: storage, usesStoreKit: false)
        do { _ = try await access.reserveCount(); XCTFail("Cannot assume an unreadable balance is zero") }
        catch CountAccessError.storageUnavailable {} catch { XCTFail("Unexpected \(error)") }
        storage.readFails = false; storage.value = 4
        do { _ = try await access.reserveCount(); XCTFail("Must reload the actual exhausted balance") }
        catch CountAccessError.subscriptionRequired {} catch { XCTFail("Unexpected \(error)") }
    }

    func testInvalidImageAndDetectorFailureDoNotConsumeUse() async {
        let storage = TestCountUsage()
        let workspace = CountWorkspace(access: CountingAccess(storage: storage, usesStoreKit: false), detector: { _ in throw CountError.unreadableImage })
        await workspace.load(data: Data([0, 1, 2]))
        XCTAssertEqual(storage.value, 0)
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.phase, .failed)
        XCTAssertEqual(storage.value, 0)
        workspace.countManually()
        workspace.add(at: CGPoint(x: 0.5, y: 0.5))
        workspace.undo()
        XCTAssertEqual(storage.value, 0)
    }

    func testZeroResultConsumesOnceHistoryAndEditsAreFreeAndGatePreservesResult() async {
        let storage = TestCountUsage(); storage.value = 3
        let access = CountingAccess(storage: storage, usesStoreKit: false)
        var detectorCalls = 0
        let workspace = CountWorkspace(access: access, detector: { _ in detectorCalls += 1; return Self.result() })
        await workspace.load(image: Self.photo())
        XCTAssertEqual(workspace.automaticCount, 0)
        XCTAssertEqual(storage.value, 4)
        workspace.add(at: CGPoint(x: 0.4, y: 0.6))
        let record = workspace.record
        workspace.restore(record, image: Self.photo())
        workspace.undo(); workspace.markSaved()
        XCTAssertEqual(storage.value, 4)
        await workspace.load(image: Self.photo())
        XCTAssertTrue(workspace.subscriptionRequired)
        XCTAssertEqual(workspace.id, record.id)
        XCTAssertEqual(workspace.edits.markers, record.markers)
        XCTAssertEqual(detectorCalls, 1)
    }

    func testCancelledInferenceAndOverlappingLoadsDoNotDebit() async throws {
        let storage = TestCountUsage()
        var entered = false
        let workspace = CountWorkspace(access: CountingAccess(storage: storage, usesStoreKit: false), detector: { _ in
            entered = true
            try await Task.sleep(for: .milliseconds(200))
            return Self.result()
        })
        let task = Task { await workspace.load(image: Self.photo()) }
        for _ in 0..<50 { if entered { break }; try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(entered)
        await workspace.load(image: Self.photo()) // Already processing: no second inference.
        task.cancel()
        await task.value
        XCTAssertEqual(storage.value, 0)
        XCTAssertNil(workspace.automaticCount)
    }

    private static func photo() -> UIImage {
        UIGraphicsImageRenderer(size: CGSize(width: 32, height: 32)).image { ctx in
            UIColor.white.setFill(); ctx.fill(CGRect(x: 0, y: 0, width: 32, height: 32))
        }
    }
    private static func result() -> StickDetectionResult {
        StickDetectionResult(markers: [], rawMarkers: [], automaticRegion: AutoCountRegionResult(keptIDs: [], excluded: [], regions: [], fallbackReason: .empty))
    }
}
