import XCTest

@MainActor
final class CountCategoryUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testModeSwitchConfirmsUnsavedResultInBothLanguages() throws {
        #if DEBUG
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let fixture = root.appendingPathComponent("release/fixtures/bamboo-skewers-cc0.jpg")
        for (language, locale, pipes, skewers, cancel, leave, heading) in [
            ("en", "en_US", "Steel pipes", "Skewers", "Cancel", "Continue without saving", "Count the pipe ends."),
            ("zh-Hans", "zh_Hans_CN", "钢管", "签子", "取消", "不保存，继续", "对准端面，核对钢管。")
        ] {
            let app = XCUIApplication()
            app.launchArguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale,
                                   "--countstick-test-photo", fixture.path, "--countscope-test-ledger", UUID().uuidString]
            app.launch()
            let count = app.staticTexts["resultCount"]
            XCTAssertTrue(count.waitForExistence(timeout: 90))
            let originalCount = count.label
            XCTAssertGreaterThan(Int(originalCount.replacingOccurrences(of: ",", with: "")) ?? 0, 0)
            let picker = app.segmentedControls["countCategory"]
            XCTAssertTrue(picker.exists)
            picker.buttons[pipes].tap()
            let confirmation = XCTAttachment(screenshot: app.screenshot())
            confirmation.name = "pipe-mode-confirmation-\(language)"
            confirmation.lifetime = .keepAlways
            add(confirmation)
            XCTAssertTrue(app.buttons[leave].waitForExistence(timeout: 5), app.debugDescription)
            if app.buttons[cancel].exists {
                app.buttons[cancel].tap()
            } else {
                // iOS 26 may present confirmationDialog as a popover, with
                // native outside-tap cancellation instead of a Cancel row.
                let dismiss = app.otherElements["PopoverDismissRegion"]
                XCTAssertTrue(dismiss.exists)
                dismiss.coordinate(withNormalizedOffset: CGVector(dx: 0.05, dy: 0.1)).tap()
            }
            XCTAssertTrue(picker.buttons[skewers].isSelected, app.debugDescription)
            XCTAssertEqual(count.label, originalCount)
            picker.buttons[pipes].tap()
            XCTAssertTrue(app.buttons[leave].waitForExistence(timeout: 3))
            app.buttons[leave].tap()
            XCTAssertTrue(app.staticTexts[heading].waitForExistence(timeout: 3))
            XCTAssertTrue(picker.buttons[pipes].isSelected, app.debugDescription)
            XCTAssertFalse(count.exists)
            let screenshot = XCTAttachment(screenshot: app.screenshot())
            screenshot.name = "pipe-mode-\(language)"
            screenshot.lifetime = .keepAlways
            add(screenshot)
            app.terminate()
        }
        #else
        throw XCTSkip("The actual-photo import hook is available only in Debug.")
        #endif
    }
}
