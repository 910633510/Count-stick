import XCTest
import StoreKit
import StoreKitTest
@testable import Count_stick

/// These are Apple's local StoreKit transactions, never production purchases.
@MainActor
final class SubscriptionStoreKitTests: XCTestCase {
    private var session: SKTestSession!
    override func setUpWithError() throws {
        let file = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "CountScope", withExtension: "storekit"))
        session = try SKTestSession(contentsOf: file)
        session.resetToDefaultState()
        session.clearTransactions()
        session.disableDialogs = true
    }
    override func tearDownWithError() throws {
        session.clearTransactions()
        session.resetToDefaultState()
        session = nil
    }

    func testPurchaseTrialRelaunchRestoreExpireAndRefund() async throws {
        let usage = TestCountUsage(); usage.value = 4
        let access = CountingAccess(storage: usage)
        await access.refresh()
        let product = try XCTUnwrap(access.product)
        XCTAssertEqual(product.id, "com.xmaillc.countscope.pro.monthly")
        XCTAssertEqual(product.price, Decimal(string: "2.99"))
        XCTAssertEqual(product.subscription?.subscriptionPeriod.unit, .month)
        XCTAssertEqual(product.subscription?.subscriptionPeriod.value, 1)
        XCTAssertTrue(access.hasSevenDayTrial)
        XCTAssertFalse(access.isSubscribed)
        await access.purchase()
        try await waitUntil { access.isSubscribed }
        XCTAssertFalse(access.hasSevenDayTrial)
        for _ in 0..<6 { try access.completeCount(try await access.reserveCount()) }
        XCTAssertEqual(usage.value, 4, "Subscriber counts must not change lifetime free allowance")
        let relaunched = CountingAccess(storage: usage)
        await relaunched.refreshEntitlements()
        XCTAssertTrue(relaunched.isSubscribed)
        await relaunched.restore()
        XCTAssertTrue(relaunched.isSubscribed)
        XCTAssertEqual(relaunched.message, L10n.text("subscription.restored"))
        try session.expireSubscription(productIdentifier: CountingAccess.productID)
        try await waitUntil { await relaunched.refreshEntitlements(); return !relaunched.isSubscribed }
        do { _ = try await relaunched.reserveCount(); XCTFail("Expired subscription must enforce exhausted allowance") }
        catch CountAccessError.subscriptionRequired {} catch { XCTFail("Unexpected \(error)") }
        await relaunched.loadProduct()
        await relaunched.purchase()
        try await waitUntil { relaunched.isSubscribed }
        let transaction = try XCTUnwrap(session.allTransactions().last)
        try session.refundTransaction(identifier: transaction.identifier)
        try await waitUntil { await relaunched.refreshEntitlements(); return !relaunched.isSubscribed }
        XCTAssertEqual(usage.value, 4)
    }

    func testPendingPurchaseDoesNotUnlockUntilApprovalAndUpdatesAreObserved() async throws {
        session.askToBuyEnabled = true
        let usage = TestCountUsage(); usage.value = 4
        let access = CountingAccess(storage: usage)
        await access.refresh()
        XCTAssertNotNil(access.product)
        await access.purchase()
        XCTAssertFalse(access.isSubscribed)
        XCTAssertEqual(access.message, L10n.text("subscription.pending"))
        let pending = try XCTUnwrap(session.allTransactions().last)
        try session.approveAskToBuyTransaction(identifier: pending.identifier)
        try await waitUntil { access.isSubscribed }
        XCTAssertEqual(usage.value, 4)
    }

    func testRestoreWithNoPurchaseDoesNotUnlock() async throws {
        let usage = TestCountUsage(); usage.value = 4
        let access = CountingAccess(storage: usage)
        await access.restore()
        XCTAssertFalse(access.isSubscribed)
        XCTAssertEqual(access.message, L10n.text("subscription.nothing.restore"))
        XCTAssertFalse(access.canStartCount)
    }

    func testCancelledAndUnverifiedPurchasesDoNotUnlock() async throws {
        let usage = TestCountUsage(); usage.value = 4
        let access = CountingAccess(storage: usage)
        await access.refresh()
        XCTAssertNotNil(access.product)
        try await session.setSimulatedError(.generic(.userCancelled), forAPI: .purchase)
        await access.purchase()
        XCTAssertFalse(access.isSubscribed)
        XCTAssertNil(access.message)
        try await session.setSimulatedError(nil, forAPI: .purchase)
        try await session.setSimulatedError(.verification(.invalidSignature), forAPI: .verification)
        await access.purchase()
        XCTAssertFalse(access.isSubscribed)
        // iOS 18.5 may reject the purchase before returning an unverified success result.
        // Either path must report the failed verification/purchase and keep access locked.
        XCTAssertNotNil(access.message)
        XCTAssertEqual(usage.value, 4)
    }

    func testUnavailableProductCannotBePurchasedOrAdvertiseTrial() async throws {
        try await session.setSimulatedError(.generic(.notAvailableInStorefront), forAPI: .loadProducts)
        let usage = TestCountUsage(); usage.value = 4
        let access = CountingAccess(storage: usage)
        await access.loadProduct()
        XCTAssertNil(access.product)
        XCTAssertNotNil(access.productError)
        XCTAssertFalse(access.hasSevenDayTrial)
        await access.purchase()
        XCTAssertFalse(access.isSubscribed)
        XCTAssertEqual(session.allTransactions().count, 0)
    }

    private func waitUntil(_ condition: () async -> Bool) async throws {
        for _ in 0..<50 {
            if await condition() { return }
            try await Task.sleep(for: .milliseconds(100))
        }
        XCTFail("StoreKit state did not settle within five seconds")
    }
}
