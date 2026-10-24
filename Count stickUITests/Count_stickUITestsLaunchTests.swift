import XCTest

@MainActor
final class Count_stickUITestsLaunchTests: XCTestCase {
    func testLaunch() {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launch()
        XCTAssertTrue(app.buttons["takePhoto"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.buttons["choosePhoto"].exists)
    }
}
