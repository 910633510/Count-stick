import XCTest

@MainActor
final class Count_stickUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    func testPrepareReleaseScreenshotRecord() throws {
        #if DEBUG
        let app = XCUIApplication()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let fixture = root.appendingPathComponent("release/fixtures/bamboo-skewers-cc0.jpg")
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US", "--countstick-test-photo", fixture.path]
        app.launch()
        let count = app.staticTexts["resultCount"]
        XCTAssertTrue(count.waitForExistence(timeout: 90))
        XCTAssertGreaterThan(Int(count.label) ?? 0, 0)
        capture(app, name: "setup-real-model-inference")
        let note = app.descendants(matching: .any)["resultNote"].firstMatch
        scrollTo(note, app: app)
        note.tap()
        note.typeText("Dinner prep")
        app.buttons["Done"].tap()
        scrollTo(app.buttons["saveCount"], app: app)
        app.buttons["saveCount"].tap()
        XCTAssertFalse(app.buttons["saveCount"].isEnabled)
        app.terminate()
        #else
        throw XCTSkip("Use Debug once to import the real fixture and save it through the UI")
        #endif
    }

    func testCaptureReleaseScreenshots() throws {
        #if DEBUG
        throw XCTSkip("Capture the production UI in Release configuration")
        #else
        let app = XCUIApplication()
        for (language, locale, prefix, noteText) in [("en", "en_US", "en-US", "Dinner prep"), ("zh-Hans", "zh_Hans_CN", "zh-Hans", "晚餐备料")] {
            app.launchArguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale]
            app.launch()
            XCTAssertTrue(app.buttons["takePhoto"].waitForExistence(timeout: 10))
            capture(app, name: "\(prefix)-home")
            tapTab(app, label: language == "en" ? "History" : "记录")
            let savedNote = app.staticTexts["Dinner prep"].firstMatch
            XCTAssertTrue(savedNote.waitForExistence(timeout: 5), "A real record must be saved by the Debug setup test first")
            savedNote.tap()
            let count = app.staticTexts["resultCount"]
            XCTAssertTrue(count.waitForExistence(timeout: 5))
            XCTAssertGreaterThan(Int(count.label) ?? 0, 0)
            let note = app.descendants(matching: .any)["resultNote"].firstMatch
            scrollTo(note, app: app)
            if language != "en" {
                let existingNote = note.value as? String ?? ""
                note.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: 0.5)).tap()
                note.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: existingNote.count))
                let clearedNote = note.value as? String ?? ""
                XCTAssertTrue(clearedNote.isEmpty || clearedNote == note.placeholderValue, "Clear the entire existing note before translating the example")
                note.typeText(noteText)
                XCTAssertEqual(note.value as? String, noteText)
                app.buttons["完成"].tap()
                scrollTo(app.buttons["saveCount"], app: app)
                app.buttons["saveCount"].tap()
            }
            for _ in 0..<3 { app.swipeDown() }
            capture(app, name: "\(prefix)-result")
            scrollTo(app.buttons["enlargePhoto"], app: app)
            app.buttons["enlargePhoto"].tap()
            XCTAssertTrue(app.buttons[language == "en" ? "Done" : "完成"].waitForExistence(timeout: 3))
            capture(app, name: "\(prefix)-correction")
            app.buttons[language == "en" ? "Toggle image zoom" : "切换照片缩放"].tap()
            capture(app, name: "\(prefix)-zoom")
            app.buttons[language == "en" ? "Done" : "完成"].tap()
            tapTab(app, label: language == "en" ? "History" : "记录")
            XCTAssertTrue(app.staticTexts[noteText].waitForExistence(timeout: 5))
            capture(app, name: "\(prefix)-history")
            app.terminate()
        }
        #endif
    }

    func testEnglishPhotoCorrectionSaveReopenDelete() throws {
        try exerciseWorkflow(language: "en", locale: "en_US", prefix: "en-US")
    }

    func testChinesePhotoCorrectionSaveReopenDelete() throws {
        try exerciseWorkflow(language: "zh-Hans", locale: "zh_Hans_CN", prefix: "zh-Hans")
    }

    func testHomeGuideAndCameraUnavailable() {
        let app = XCUIApplication()
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US"]
        app.launch()
        XCTAssertTrue(app.buttons["takePhoto"].waitForExistence(timeout: 10))
        capture(app, name: "en-US-home")
        app.buttons["takePhoto"].tap()
        XCTAssertTrue(app.alerts["Count Stick"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.alerts.staticTexts["A camera is not available on this device. Choose a photo to get started."].exists)
        app.alerts.buttons["OK"].tap()
        tapTab(app, label: "Guide")
        XCTAssertTrue(app.staticTexts["Give every stick some space"].waitForExistence(timeout: 3))
        capture(app, name: "en-US-guide")
        tapTab(app, label: "Count")
        app.buttons["choosePhoto"].tap()
        // PhotosPicker must be the native system picker, without requesting whole-library access.
        XCTAssertTrue(app.buttons["Cancel"].waitForExistence(timeout: 5))
        app.buttons["Cancel"].tap()
        XCTAssertTrue(app.buttons["choosePhoto"].exists)
    }

    private func exerciseWorkflow(language: String, locale: String, prefix: String) throws {
        let app = XCUIApplication()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let fixture = root.appendingPathComponent("release/fixtures/bamboo-skewers-cc0.jpg")
        XCTAssertTrue(FileManager.default.fileExists(atPath: fixture.path))
        app.launchArguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale, "--countstick-test-photo", fixture.path]
        app.launch()
        let count = app.staticTexts["resultCount"]
        XCTAssertTrue(count.waitForExistence(timeout: 90))
        let initial = try XCTUnwrap(Int(count.label.replacingOccurrences(of: ",", with: "")))
        XCTAssertGreaterThan(initial, 0, "The real fixture should run through the model, not a seeded count")
        XCTAssertTrue(app.staticTexts[language == "en" ? "Review your count." : "核对一下，心中有数。"].exists)
        capture(app, name: "\(prefix)-result")

        let photo = app.descendants(matching: .any)["resultPhoto"].firstMatch
        XCTAssertTrue(photo.exists)
        scrollTo(app.segmentedControls["correctionMode"], app: app)
        let picker = app.segmentedControls["correctionMode"]
        picker.buttons[language == "en" ? "Add" : "添加"].tap()
        // The center is within the actual photo, not the aspect-fit margin.
        photo.coordinate(withNormalizedOffset: CGVector(dx: 0.52, dy: 0.5)).tap()
        XCTAssertEqual(Int(count.label.replacingOccurrences(of: ",", with: "")), initial + 1)
        picker.buttons[language == "en" ? "Remove" : "移除"].tap()
        photo.coordinate(withNormalizedOffset: CGVector(dx: 0.52, dy: 0.5)).tap()
        XCTAssertEqual(Int(count.label.replacingOccurrences(of: ",", with: "")), initial)
        app.buttons["undoCorrection"].tap()
        XCTAssertEqual(Int(count.label.replacingOccurrences(of: ",", with: "")), initial + 1)
        app.buttons["undoCorrection"].tap()
        XCTAssertEqual(Int(count.label.replacingOccurrences(of: ",", with: "")), initial)
        picker.buttons[language == "en" ? "View" : "查看"].tap()

        scrollTo(app.buttons["enlargePhoto"], app: app)
        app.buttons["enlargePhoto"].tap()
        XCTAssertTrue(app.navigationBars[language == "en" ? "Fine-tune the count" : "校正计数"].waitForExistence(timeout: 3))
        capture(app, name: "\(prefix)-correction")
        app.buttons[language == "en" ? "Toggle image zoom" : "切换照片缩放"].tap()
        capture(app, name: "\(prefix)-zoom")
        app.buttons[language == "en" ? "Done" : "完成"].tap()

        let note = app.descendants(matching: .any)["resultNote"].firstMatch
        scrollTo(note, app: app)
        note.tap()
        note.typeText(language == "en" ? "Dinner prep" : "晚餐备料")
        app.buttons[language == "en" ? "Done" : "完成"].tap()
        scrollTo(app.buttons["saveCount"], app: app)
        app.buttons["saveCount"].tap()
        XCTAssertFalse(app.buttons["saveCount"].isEnabled)
        tapTab(app, label: language == "en" ? "History" : "记录")
        XCTAssertTrue(app.staticTexts[language == "en" ? "\(initial) sticks" : "\(initial) 根签"].waitForExistence(timeout: 3))
        capture(app, name: "\(prefix)-history")
        app.staticTexts[language == "en" ? "\(initial) sticks" : "\(initial) 根签"].firstMatch.tap()
        XCTAssertEqual(Int(count.label.replacingOccurrences(of: ",", with: "")), initial)
        scrollTo(app.buttons["saveCount"], app: app)
        XCTAssertFalse(app.buttons["saveCount"].isEnabled)
        app.buttons["shareCount"].tap()
        XCTAssertTrue(app.otherElements["ActivityListView"].waitForExistence(timeout: 5) || app.buttons["Close"].exists)
        app.swipeDown()
        // Relaunch retains saved history. The import itself must not create extra records.
        app.terminate()
        app.launchArguments = ["-AppleLanguages", "(\(language))", "-AppleLocale", locale]
        app.launch()
        capture(app, name: "\(prefix)-home")
        tapTab(app, label: language == "en" ? "History" : "记录")
        XCTAssertTrue(app.staticTexts[language == "en" ? "\(initial) sticks" : "\(initial) 根签"].waitForExistence(timeout: 3))
        app.staticTexts[language == "en" ? "\(initial) sticks" : "\(initial) 根签"].firstMatch.tap()
        tapTab(app, label: language == "en" ? "History" : "记录")
        app.buttons[language == "en" ? "Saved count options" : "记录选项"].firstMatch.tap()
        app.buttons[language == "en" ? "Delete" : "删除"].firstMatch.tap()
        app.buttons[language == "en" ? "Delete" : "删除"].firstMatch.tap()
        tapTab(app, label: language == "en" ? "Count" : "数签")
        scrollTo(app.buttons["saveCount"], app: app)
        XCTAssertTrue(app.buttons["saveCount"].isEnabled, "Deleting the current record must allow saving it again")
    }

    private func tapTab(_ app: XCUIApplication, label: String) {
        let phoneTab = app.tabBars.buttons[label]
        if phoneTab.exists { phoneTab.tap() }
        else { app.buttons[label].firstMatch.tap() }
    }

    private func scrollTo(_ element: XCUIElement, app: XCUIApplication) {
        for _ in 0..<5 {
            if element.exists && element.isHittable { return }
            app.swipeUp()
        }
    }

    private func capture(_ app: XCUIApplication, name: String) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
