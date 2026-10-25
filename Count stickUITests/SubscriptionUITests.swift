import XCTest
import StoreKitTest

@MainActor
final class SubscriptionUITests: XCTestCase {
    private var session: SKTestSession!
    override func setUpWithError() throws {
        continueAfterFailure = false
        let file = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "CountScope", withExtension: "storekit"))
        session = try SKTestSession(contentsOf: file)
        session.resetToDefaultState()
        session.clearTransactions()
        session.disableDialogs = true
    }
    override func tearDownWithError() throws {
        session.clearTransactions()
        session.resetToDefaultState()
    }

    func testLocalizedOfferAndRestoreWithoutPurchase() throws {
        for (language, locale, prefix, emptyRestore) in [
            ("en", "en_US", "en-US", "No active CountScope subscription was found for this Apple Account."),
            ("zh-Hans", "zh_Hans_CN", "zh-Hans", "当前 Apple 账户未找到有效的 CountScope 订阅。")
        ] {
            let app = XCUIApplication()
            app.launchArguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale,
                                   "--countscope-test-ledger", UUID().uuidString]
            app.launch()
            let entry = app.buttons["subscriptionEntry"]
            XCTAssertTrue(entry.waitForExistence(timeout: 15))
            entry.tap()
            XCTAssertTrue(app.staticTexts["subscriptionPrice"].waitForExistence(timeout: 20))
            XCTAssertTrue(app.staticTexts["subscriptionPrice"].label.contains("2.99"))
            XCTAssertTrue(app.staticTexts["subscriptionTrial"].exists)
            XCTAssertTrue(app.buttons["subscribeButton"].isEnabled)
            capture(app, "\(prefix)-subscription-offer")
            let restore = app.buttons["restorePurchases"]
            scrollTo(restore, in: app)
            XCTAssertTrue(app.buttons["manageSubscription"].exists)
            restore.tap()
            XCTAssertTrue(app.staticTexts[emptyRestore].waitForExistence(timeout: 15))
            capture(app, "\(prefix)-subscription-restore-empty")
            app.buttons["closeSubscription"].tap()
            XCTAssertTrue(entry.waitForExistence(timeout: 5))
            app.terminate()
        }
    }

    func testFourFreeCountsThenPaywallWithHistoryStillAvailable() throws {
        #if DEBUG
        let app = XCUIApplication()
        let ledger = UUID().uuidString
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("release/fixtures/bamboo-skewers-cc0.jpg").path
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US",
                               "--countscope-test-ledger", ledger, "--countstick-test-photo", fixture]
        for remaining in (0...3).reversed() {
            app.launch()
            XCTAssertTrue(app.staticTexts["resultCount"].waitForExistence(timeout: 90))
            XCTAssertTrue(app.staticTexts["\(remaining) free counts left · View Pro"].exists)
            if remaining == 3 {
                scrollTo(app.buttons["saveCount"], in: app)
                app.buttons["saveCount"].tap()
                XCTAssertFalse(app.buttons["saveCount"].isEnabled)
            }
            app.terminate()
        }
        app.launch()
        XCTAssertTrue(app.buttons["subscribeButton"].waitForExistence(timeout: 30))
        XCTAssertFalse(app.staticTexts["resultCount"].exists)
        capture(app, "en-US-four-free-counts-exhausted")
        app.buttons["closeSubscription"].tap()
        app.buttons["History"].tap()
        XCTAssertTrue(app.buttons.matching(NSPredicate(format: "label CONTAINS 'sticks'")).firstMatch.waitForExistence(timeout: 5))
        capture(app, "en-US-history-after-free-limit")
        app.terminate()
        #else
        throw XCTSkip("Fixture injection is Debug only; no test import bypass is shipped")
        #endif
    }

    func testPurchaseAndRelaunchSubscriptionUI() throws {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US", "--countscope-test-ledger", UUID().uuidString]
        app.launch()
        XCTAssertTrue(app.buttons["subscriptionEntry"].waitForExistence(timeout: 15))
        app.buttons["subscriptionEntry"].tap()
        XCTAssertTrue(app.buttons["subscribeButton"].waitForExistence(timeout: 20))
        app.buttons["subscribeButton"].tap()
        XCTAssertTrue(app.staticTexts["Pro is active"].waitForExistence(timeout: 20))
        capture(app, "en-US-subscription-active")
        app.buttons["closeSubscription"].tap()
        app.terminate()
        app.launch()
        XCTAssertTrue(app.staticTexts["Pro is active"].waitForExistence(timeout: 15))
        app.terminate()
    }

    private func scrollTo(_ element: XCUIElement, in app: XCUIApplication) {
        for _ in 0..<8 { if element.isHittable { return }; app.swipeUp() }
        XCTAssertTrue(element.isHittable)
    }
    private func capture(_ app: XCUIApplication, _ name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
