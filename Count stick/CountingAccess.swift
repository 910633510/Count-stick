import Foundation
import StoreKit
import Security
import SwiftUI

protocol CountUsageStorage {
    func read() throws -> Int
    func write(_ used: Int) throws
}

/// This-device-only Keychain data is not synced or sent to a server.
struct KeychainCountUsage: CountUsageStorage {
    var service = "x.Count-stick.free-counts.v1"
    private var query: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: "successful-counts",
         kSecAttrSynchronizable as String: false]
    }
    func read() throws -> Int {
        var request = query
        request[kSecReturnData as String] = true
        request[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(request as CFDictionary, &result)
        if status == errSecItemNotFound { return 0 }
        guard status == errSecSuccess, let data = result as? Data,
              let value = String(data: data, encoding: .utf8).flatMap(Int.init), (0...4).contains(value) else {
            throw CountAccessError.storageUnavailable
        }
        return value
    }
    func write(_ used: Int) throws {
        let data = Data(String(used).utf8)
        var status = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var item = query
            item[kSecValueData as String] = data
            item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            status = SecItemAdd(item as CFDictionary, nil)
        }
        guard status == errSecSuccess else { throw CountAccessError.storageUnavailable }
    }
}

enum CountAccessError: LocalizedError {
    case subscriptionRequired, storageUnavailable, invalidReservation
    var errorDescription: String? {
        switch self {
        case .subscriptionRequired: return L10n.text("access.limit")
        case .storageUnavailable: return L10n.text("access.storage.error")
        case .invalidReservation: return L10n.text("access.retry.error")
        }
    }
}

@MainActor
final class CountingAccess: ObservableObject {
    static let productID = "com.xmaillc.countscope.pro.monthly"
    static let freeLimit = 4
    static let privacyURL = URL(string: "https://www.xmaillc.com/count-stick-privacy.html")!
    static let termsURL = URL(string: "https://www.apple.com/legal/internet-services/itunes/dev/stdeula/")!
    static let shared: CountingAccess = {
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        if let index = args.firstIndex(of: "--countscope-test-ledger"), args.indices.contains(index + 1),
           UUID(uuidString: args[index + 1]) != nil {
            return CountingAccess(storage: KeychainCountUsage(service: "x.Count-stick.tests." + args[index + 1]))
        }
        #endif
        return CountingAccess()
    }()

    struct Reservation: Hashable { let id: UUID; let usesFreeCount: Bool }
    @Published private(set) var usedFreeCounts: Int = 0
    @Published private(set) var storageAvailable = true
    @Published private(set) var product: Product?
    @Published private(set) var subscriptionValidUntil: Date?
    @Published private(set) var eligibleForTrial = false
    @Published private(set) var loadingProducts = false
    @Published private(set) var purchasing = false
    @Published private(set) var restoring = false
    @Published var message: String?
    @Published private(set) var productError: String?
    private let storage: CountUsageStorage
    private let usesStoreKit: Bool
    private var reservations: Set<Reservation> = []
    private var updatesTask: Task<Void, Never>?
    private var expiryTask: Task<Void, Never>?
    private var entitlementTask: Task<Void, Never>?
    private var entitlementRefreshID = UUID()

    init(storage: CountUsageStorage = KeychainCountUsage(), usesStoreKit: Bool = true) {
        self.storage = storage
        self.usesStoreKit = usesStoreKit
        do { usedFreeCounts = min(Self.freeLimit, max(0, try storage.read())) }
        catch { storageAvailable = false }
        if usesStoreKit {
            updatesTask = Task { [weak self] in
                for await result in Transaction.updates {
                    guard !Task.isCancelled else { return }
                    if case .verified(let transaction) = result, transaction.productID == Self.productID {
                        await self?.refreshEntitlements()
                        await transaction.finish()
                        await self?.refreshTrialEligibility()
                    }
                }
            }
        }
    }
    deinit { updatesTask?.cancel(); expiryTask?.cancel() }

    var isSubscribed: Bool { (subscriptionValidUntil ?? .distantPast) > Date() }
    var remainingFreeCounts: Int { max(0, Self.freeLimit - usedFreeCounts) }
    var canStartCount: Bool { isSubscribed || (storageAvailable && remainingFreeCounts > reservations.filter(\.usesFreeCount).count) }
    var hasSevenDayTrial: Bool {
        guard eligibleForTrial, let offer = product?.subscription?.introductoryOffer, offer.paymentMode == .freeTrial else { return false }
        return (offer.period.unit == .week && offer.period.value * offer.periodCount == 1)
            || (offer.period.unit == .day && offer.period.value * offer.periodCount == 7)
    }

    func refresh() async {
        await refreshEntitlements()
        await loadProduct()
    }

    func loadProduct() async {
        guard usesStoreKit, !loadingProducts else { return }
        loadingProducts = true
        defer { loadingProducts = false }
        do {
            let products = try await Product.products(for: [Self.productID])
            product = products.first { $0.id == Self.productID && $0.type == .autoRenewable }
            productError = product == nil ? L10n.text("subscription.unavailable") : nil
            await refreshTrialEligibility()
            // A verified billing-grace period may extend the entitlement's expiry.
            await refreshEntitlements()
        } catch { productError = L10n.text("subscription.unavailable") }
    }

    func refreshTrialEligibility() async {
        guard let subscription = product?.subscription else { eligibleForTrial = false; return }
        eligibleForTrial = await subscription.isEligibleForIntroOffer
    }

    func refreshEntitlements() async {
        guard usesStoreKit else { return }
        // Queue each requested refresh after the previous scan. In particular, a purchase,
        // refund, or restore event must never join a scan that started before that event.
        let previous = entitlementTask
        let refreshID = UUID()
        entitlementRefreshID = refreshID
        let task = Task { [weak self] in
            if let previous { await previous.value }
            guard let self else { return }
            self.subscriptionValidUntil = await self.readVerifiedEntitlement()
            self.scheduleExpiryRefresh()
        }
        entitlementTask = task
        await task.value
        if entitlementRefreshID == refreshID { entitlementTask = nil }
    }

    private func readVerifiedEntitlement() async -> Date? {
        var expiry: Date?
        for await result in Transaction.currentEntitlements {
            guard case .verified(let transaction) = result,
                  transaction.productID == Self.productID, transaction.revocationDate == nil,
                  !transaction.isUpgraded, let date = transaction.expirationDate, date > Date() else { continue }
            expiry = max(expiry ?? .distantPast, date)
        }
        if let subscription = product?.subscription, let statuses = try? await subscription.status {
            for status in statuses where status.state == .inGracePeriod {
                guard case .verified(let transaction) = status.transaction,
                      case .verified(let renewal) = status.renewalInfo,
                      transaction.productID == Self.productID, transaction.revocationDate == nil,
                      let date = renewal.gracePeriodExpirationDate, date > Date() else { continue }
                expiry = max(expiry ?? .distantPast, date)
            }
        }
        return expiry
    }

    private func scheduleExpiryRefresh() {
        expiryTask?.cancel()
        guard let expiry = subscriptionValidUntil else { return }
        let delay = min(3600, max(1, expiry.timeIntervalSinceNow + 0.5))
        expiryTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            await self?.refreshEntitlements()
        }
    }

    func purchase() async {
        guard !purchasing, !restoring, let product else { return }
        purchasing = true
        message = nil
        defer { purchasing = false }
        do {
            switch try await product.purchase() {
            case .success(let result):
                guard case .verified(let transaction) = result, transaction.productID == Self.productID else {
                    message = L10n.text("subscription.unverified"); return
                }
                await refreshEntitlements()
                await transaction.finish()
                await refreshTrialEligibility()
                if !isSubscribed { message = L10n.text("subscription.not.active") }
            case .pending: message = L10n.text("subscription.pending")
            case .userCancelled: break
            @unknown default: message = L10n.text("subscription.purchase.error")
            }
        } catch StoreKitError.userCancelled {
            // Cancelling Apple’s purchase sheet is not an error and never grants access.
        } catch { message = L10n.text("subscription.purchase.error") }
    }

    func restore() async {
        guard !purchasing, !restoring else { return }
        restoring = true
        message = nil
        defer { restoring = false }
        do {
            try await AppStore.sync()
            await refreshEntitlements()
            await refreshTrialEligibility()
            message = L10n.text(isSubscribed ? "subscription.restored" : "subscription.nothing.restore")
        } catch { message = L10n.text("subscription.restore.error") }
    }

    /// Reserve before starting inference. Main-actor reservations prevent concurrent free-use overspend.
    func reserveCount() async throws -> Reservation {
        await refreshEntitlements()
        if !storageAvailable {
            do { usedFreeCounts = min(Self.freeLimit, max(0, try storage.read())); storageAvailable = true }
            catch { if !isSubscribed { throw CountAccessError.storageUnavailable } }
        }
        let free = !isSubscribed
        if free {
            guard remainingFreeCounts > reservations.filter(\.usesFreeCount).count else { throw CountAccessError.subscriptionRequired }
        }
        let reservation = Reservation(id: UUID(), usesFreeCount: free)
        reservations.insert(reservation)
        return reservation
    }

    /// Persist before exposing the successful result; failed/cancelled inference never consumes a use.
    func completeCount(_ reservation: Reservation) throws {
        guard reservations.contains(reservation) else { throw CountAccessError.invalidReservation }
        if reservation.usesFreeCount {
            do {
                try storage.write(min(Self.freeLimit, usedFreeCounts + 1))
                usedFreeCounts = min(Self.freeLimit, usedFreeCounts + 1)
                storageAvailable = true
            } catch { storageAvailable = false; throw CountAccessError.storageUnavailable }
        }
        reservations.remove(reservation)
    }
    func cancelCount(_ reservation: Reservation) { reservations.remove(reservation) }
}
