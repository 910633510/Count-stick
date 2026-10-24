#!/usr/bin/env python3
"""Preserve original XCTest captures and their Release build provenance."""
import hashlib
import json
import plistlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "output/screenshots-build3"
APP = ROOT / "output/screenshots-build3-derived/Build/Products/Release-iphonesimulator/Count stick.app"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((HERE / "source-freeze.json").read_text())
    assert all(sha(ROOT / p) == value for p, value in frozen["sourceFiles"].items())
    info = plistlib.loads((APP / "Info.plist").read_bytes())
    assert (info["CFBundleIdentifier"], info["CFBundleShortVersionString"], info["CFBundleVersion"]) == ("x.Count-stick", "1.0", "3")
    assert b"--countstick-test-photo" not in (APP / "Count stick").read_bytes()
    for family, device in [("iphone", "iPhone 17 Pro"), ("ipad", "iPad Pro 13-inch (M5)")]:
        folder = ARTIFACTS / f"{family}-attachments"
        attachments = json.loads((folder / "manifest.json").read_text())[0]["attachments"]
        for locale, language, apple_locale, effective in [
            ("en-US", "en", "en_US", "English"),
            ("zh-Hans", "zh-Hans", "zh_Hans_CN", "Simplified Chinese"),
        ]:
            destination = HERE / "source" / family / locale
            destination.mkdir(parents=True, exist_ok=True)
            entries = []
            for entry in attachments:
                name = entry["suggestedHumanReadableName"].split("_0_")[0]
                if not name.startswith(locale + "-"):
                    continue
                role = name[len(locale) + 1:]
                original = folder / entry["exportedFileName"]
                saved = destination / (role + ".png")
                shutil.copyfile(original, saved)
                assert sha(original) == sha(saved)
                with Image.open(saved) as opened:
                    dimensions = list(opened.size)
                entries.append({"role": role, "path": saved.name, "sha256": sha(saved),
                                "pixels": dimensions, "originalAttachment": str(original.relative_to(ROOT)),
                                "capturedAtUTC": datetime.fromtimestamp(entry["timestamp"], timezone.utc).isoformat()})
            assert len(entries) == 5
            manifest = {
                "schemaVersion": 1, "bundleID": "x.Count-stick", "appVersion": "1.0", "build": "3",
                "configuration": "Release (ENABLE_TESTABILITY=YES for XCTest; DEBUG import hook absent)",
                "locale": locale, "device": device, "simulatorId": attachments[0]["deviceId"],
                "os": "iOS 26.5 (23F77)", "appleLanguages": [language], "appleLocale": apple_locale,
                "effectiveLanguage": effective, "captureMethod": "XCTest app.screenshot() attachment, copied byte-for-byte",
                "sourceFreeze": str((HERE / "source-freeze.json").relative_to(ROOT)),
                "sourceFreezeSHA256": sha(HERE / "source-freeze.json"),
                "releaseSimulatorExecutableSHA256": sha(APP / "Count stick"),
                "captureTest": "Count_stickUITests/testCaptureReleaseScreenshots",
                "captureTestSourceSHA256": sha(ROOT / "Count stickUITests/Count_stickUITests.swift"),
                "captureResult": str((ARTIFACTS / f"{family}-release-captures.xcresult").relative_to(ROOT)),
                "setupResult": str((ARTIFACTS / f"{family}-setup.xcresult").relative_to(ROOT)),
                "recordOrigin": {
                    "method": "Current Debug 1.0(3) processed the real fixture using production CoreML/OCR/automatic-region code, then saved through app UI. Current Release 1.0(3) reopened that record; no seeded marker coordinates, counts, or regions.",
                    "fixture": "release/fixtures/bamboo-skewers-cc0.jpg",
                    "fixtureSHA256": sha(ROOT / "release/fixtures/bamboo-skewers-cc0.jpg"),
                    "groundTruthCount": None,
                    "noteEnteredThroughUI": "Dinner prep" if locale == "en-US" else "晚餐备料",
                    "automaticRegionAssertions": "Status and nonempty outline exist in both setup and Release capture tests.",
                },
                "visualReview": "PENDING",
                "screenshots": sorted(entries, key=lambda x: ["home", "result", "correction", "zoom", "history"].index(x["role"])),
            }
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            print(destination / "manifest.json")
    shutil.copyfile(ROOT / "release/metadata-draft.json", HERE / "metadata-at-render.json")


if __name__ == "__main__":
    main()
