import SwiftUI
import StoreKit

struct SubscriptionView: View {
    @ObservedObject var access: CountingAccess
    @Environment(\.dismiss) private var dismiss
    @Environment(\.scenePhase) private var scenePhase
    @State private var showManage = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    Image(systemName: "viewfinder.circle.fill")
                        .font(.system(size: 54)).foregroundStyle(Color(uiColor: AppPalette.ink))
                    Text(L10n.text("subscription.title")).font(.system(.largeTitle, design: .rounded, weight: .bold))
                    Text(L10n.text("subscription.benefit")).font(.title3)
                    Label(L10n.text("subscription.review"), systemImage: "checkmark.shield")
                        .font(.subheadline).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 14) {
                        Text(L10n.text(access.isSubscribed ? "subscription.active" : "subscription.monthly"))
                            .font(.title2.bold())
                        if access.isSubscribed {
                            Text(L10n.text("subscription.active.detail")).font(.subheadline)
                        } else if let product = access.product {
                            Text(String(format: L10n.text("subscription.price"), product.displayPrice))
                                .font(.system(.title, design: .rounded, weight: .bold))
                                .accessibilityIdentifier("subscriptionPrice")
                            if access.hasSevenDayTrial {
                                Text(String(format: L10n.text("subscription.trial.detail"), product.displayPrice))
                                    .font(.subheadline).accessibilityIdentifier("subscriptionTrial")
                            } else {
                                Text(L10n.text("subscription.no.trial")).font(.subheadline)
                            }
                            Button { Task { await access.purchase() } } label: {
                                HStack {
                                    if access.purchasing { ProgressView().tint(.white) }
                                    Text(L10n.text(access.hasSevenDayTrial ? "subscription.start.trial" : "subscription.subscribe"))
                                }.font(.headline).frame(maxWidth: .infinity).padding(.vertical, 16)
                            }
                            .buttonStyle(.borderedProminent).tint(Color(uiColor: AppPalette.ink))
                            .disabled(access.purchasing || access.restoring)
                            .accessibilityIdentifier("subscribeButton")
                        } else {
                            if access.loadingProducts { ProgressView(L10n.text("subscription.loading")) }
                            else {
                                Text(access.productError ?? L10n.text("subscription.unavailable")).font(.subheadline)
                                Button(L10n.text("subscription.reload")) { Task { await access.loadProduct() } }
                            }
                        }
                    }.padding(22).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(red: 0.94, green: 0.69, blue: 0.28).opacity(0.22), in: RoundedRectangle(cornerRadius: 24))
                    if !access.isSubscribed {
                        Text(String(format: L10n.text("access.free.detail"), access.remainingFreeCounts))
                            .font(.subheadline).accessibilityIdentifier("freeCountsDetail")
                    }
                    Text(L10n.text("subscription.renewal"))
                        .font(.footnote).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    if let message = access.message {
                        Text(message).font(.subheadline).accessibilityIdentifier("purchaseMessage")
                    }
                    VStack(alignment: .leading, spacing: 16) {
                        Button { Task { await access.restore() } } label: {
                            if access.restoring { ProgressView() }
                            else { Text(L10n.text("subscription.restore")) }
                        }.disabled(access.purchasing || access.restoring).accessibilityIdentifier("restorePurchases")
                        Button(L10n.text("subscription.manage")) { showManage = true }
                            .accessibilityIdentifier("manageSubscription")
                        Link(L10n.text("guide.policy"), destination: CountingAccess.privacyURL)
                        Link(L10n.text("subscription.terms"), destination: CountingAccess.termsURL)
                    }.font(.subheadline.weight(.semibold))
                }.padding(24).frame(maxWidth: 620).frame(maxWidth: .infinity, alignment: .center)
            }
            .background(Color(red: 0.97, green: 0.96, blue: 0.92))
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button(L10n.text("done")) { dismiss() }.accessibilityIdentifier("closeSubscription") } }
            .manageSubscriptionsSheet(isPresented: $showManage)
        }
        .task { await access.refresh() }
        .onChange(of: scenePhase) { _, phase in if phase == .active { Task { await access.refresh() } } }
        .onChange(of: showManage) { _, shown in if !shown { Task { await access.refreshEntitlements() } } }
    }
}
