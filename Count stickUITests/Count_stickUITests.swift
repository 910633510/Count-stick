import XCTest
import UIKit

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
        XCTAssertTrue(app.descendants(matching: .any)["automaticRegionStatus"].firstMatch.waitForExistence(timeout: 5))
        scrollTo(app.buttons["enlargePhoto"], app: app)
        XCTAssertTrue(app.descendants(matching: .any)["automaticRegionOutline"].firstMatch.waitForExistence(timeout: 5),
                      "Build 3 screenshot setup needs a genuine nonempty automatic-region result")
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
            XCTAssertTrue(app.descendants(matching: .any)["automaticRegionStatus"].firstMatch.waitForExistence(timeout: 5))
            XCTAssertTrue(app.descendants(matching: .any)["automaticRegionOutline"].firstMatch.waitForExistence(timeout: 5),
                          "Release capture must show Build 3's authentic automatic reference outline")
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

    func testAutomaticRegionKeepsManualPhotoCorrections() throws {
        #if DEBUG
        let app = XCUIApplication()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let fixture = root.appendingPathComponent("release/fixtures/bamboo-skewers-cc0.jpg")
        let photoSize = try XCTUnwrap(UIImage(contentsOfFile: fixture.path)).size
        app.launchArguments = ["-AppleLanguages", "(en)", "-AppleLocale", "en_US", "--countstick-test-photo", fixture.path]
        app.launch()
        let count = app.staticTexts["resultCount"]
        XCTAssertTrue(count.waitForExistence(timeout: 90))
        let initial = try XCTUnwrap(Int(count.label.replacingOccurrences(of: ",", with: "")))
        XCTAssertGreaterThan(initial, 0)
        let status = app.descendants(matching: .any)["automaticRegionStatus"].firstMatch
        XCTAssertTrue(status.waitForExistence(timeout: 5))
        XCTAssertEqual(status.label, "Estimated counting area. Check for stick ends outside it too.")
        scrollTo(app.buttons["enlargePhoto"], app: app)
        XCTAssertTrue(app.descendants(matching: .any)["automaticRegionOutline"].firstMatch.waitForExistence(timeout: 5),
                      "The visible automatic outline must have its own accessibility element.\n\(app.debugDescription)")
        capture(app, name: "automatic-region-real-photo")

        app.buttons["enlargePhoto"].tap()
        let canvas = app.descendants(matching: .any)["editorPhoto"].firstMatch
        XCTAssertTrue(canvas.waitForExistence(timeout: 5))
        let outlineFrames = app.descendants(matching: .any).matching(identifier: "automaticRegionOutline")
            .allElementsBoundByIndex.map(\.frame)
        XCTAssertFalse(outlineFrames.isEmpty)
        XCTAssertTrue(outlineFrames.allSatisfy { $0.width > 0 && $0.height > 0 })
        let frame = canvas.frame
        let scale = min(frame.width / photoSize.width, frame.height / photoSize.height)
        let imageFrame = CGRect(x: frame.midX - photoSize.width * scale / 2,
                                y: frame.midY - photoSize.height * scale / 2,
                                width: photoSize.width * scale, height: photoSize.height * scale)
        let fractions: [CGPoint] = [CGPoint(x: 0.03, y: 0.03), CGPoint(x: 0.97, y: 0.03),
                                    CGPoint(x: 0.03, y: 0.97), CGPoint(x: 0.97, y: 0.97),
                                    CGPoint(x: 0.03, y: 0.5), CGPoint(x: 0.97, y: 0.5)]
        let targetPoint = try XCTUnwrap(fractions.map {
            CGPoint(x: imageFrame.minX + $0.x * imageFrame.width, y: imageFrame.minY + $0.y * imageFrame.height)
        }.first { point in
            !outlineFrames.contains { $0.insetBy(dx: -2, dy: -2).contains(point) }
        }, "The real fixture has background outside the automatically detected bundle")
        let target = canvas.coordinate(withNormalizedOffset: .zero)
            .withOffset(CGVector(dx: targetPoint.x - frame.minX, dy: targetPoint.y - frame.minY))
        let editorCount = app.staticTexts["editorResultCount"]
        let mode = app.segmentedControls["editorCorrectionMode"]
        mode.buttons["Add"].tap()
        target.tap()
        XCTAssertEqual(editorCount.label, "\(initial + 1) sticks", "Manual correction must work outside the automatic outline")
        mode.buttons["Remove"].tap()
        target.tap()
        XCTAssertEqual(editorCount.label, "\(initial) sticks")
        app.buttons["editorUndoCorrection"].tap()
        XCTAssertEqual(editorCount.label, "\(initial + 1) sticks")
        app.buttons["editorUndoCorrection"].tap()
        XCTAssertEqual(editorCount.label, "\(initial) sticks")
        capture(app, name: "automatic-region-correction-outside-outline")
        app.buttons["Done"].tap()
        XCTAssertEqual(Int(count.label.replacingOccurrences(of: ",", with: "")), initial)
        #else
        throw XCTSkip("The real-fixture import hook is available only in Debug UI tests")
        #endif
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
        XCTAssertTrue(app.staticTexts["Face the ends toward the camera"].waitForExistence(timeout: 3))
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
        let regionStatus = app.descendants(matching: .any)["automaticRegionStatus"].firstMatch
        XCTAssertTrue(regionStatus.waitForExistence(timeout: 5))
        let originalRegionStatus = regionStatus.label
        let hadAutomaticOutline = app.descendants(matching: .any)["automaticRegionOutline"].firstMatch.exists
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
        XCTAssertEqual(regionStatus.label, originalRegionStatus, "Saving and reopening must retain the automatic-region result")
        XCTAssertEqual(app.descendants(matching: .any)["automaticRegionOutline"].firstMatch.exists, hadAutomaticOutline)
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
        // Let transient scroll indicators and sheet animations settle before recording evidence.
        Thread.sleep(forTimeInterval: 1.0)
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
