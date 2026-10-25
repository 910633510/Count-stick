#!/usr/bin/env python3
"""Freeze source and preserve actual Build 4 XCTest captures without inventing evidence.

--freeze-now writes source-freeze.json immediately before building/capturing.
--inputs reads a JSON record containing appPath, setupAppPath, sourceFreeze,
metadata, and families: {iphone|ipad: {xcresult, setupXcresult, attachmentsManifest}}.
Each family may override appPath/setupAppPath and record a distinct setupSourceFreeze.
All input files must already exist. Paths may be absolute or relative to the repo.
Device/OS and timestamps come from the xcresults, never a hard-coded simulator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROLE_ORDER = ["home", "result", "correction", "zoom", "history", "subscription"]
REQUIRED_ROLES = {"result", "correction", "history"}
LOCALES = [("en-US", "en", "en_US", "English"), ("zh-Hans", "zh-Hans", "zh_Hans_CN", "Simplified Chinese")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def recorded_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_fresh(path, content):
    data = (json.dumps(content, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def freeze():
    production = list((ROOT / "Count stick").rglob("*"))
    production += [ROOT / "Count stick.xcodeproj/project.pbxproj"]
    extensions = {".swift", ".strings", ".json", ".plist", ".xcprivacy", ".png", ".mlmodel", ".bin"}
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(production) if p.is_file() and p.suffix in extensions}
    evidence = {"schemaVersion": 1, "frozenAtUTC": datetime.now(timezone.utc).isoformat(),
                "bundleID": "x.Count-stick", "appVersion": "1.0", "build": "4", "sourceFiles": files}
    write_fresh(HERE / "source-freeze.json", evidence)
    print(HERE / "source-freeze.json")


def summary(path):
    value = json.loads(subprocess.check_output(["xcrun", "xcresulttool", "get", "test-results", "summary", "--path", str(path)]))
    if value.get("result") != "Passed" or value.get("failedTests") or value.get("passedTests", 0) < 1:
        raise ValueError(f"Capture/setup tests did not pass: {path}")
    return value


def app_identity(path):
    info = plistlib.loads((path / "Info.plist").read_bytes())
    if (info.get("CFBundleIdentifier"), info.get("CFBundleShortVersionString"), info.get("CFBundleVersion")) != ("x.Count-stick", "1.0", "4"):
        raise ValueError(f"Expected x.Count-stick 1.0(4): {path}")
    return info


def prepare(inputs_path):
    inputs = json.loads(inputs_path.read_text())
    frozen_path = resolve(inputs["sourceFreeze"])
    frozen = json.loads(frozen_path.read_text())
    assert frozen["bundleID"] == "x.Count-stick" and frozen["build"] == "4"
    changed = [name for name, digest in frozen["sourceFiles"].items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    if changed:
        raise ValueError("Source changed after freeze: " + ", ".join(changed))
    metadata_path = resolve(inputs["metadata"])
    metadata = json.loads(metadata_path.read_text())
    assert metadata["targetBuild"] == "4" and metadata["appStoreId"] == "6826601708"
    assert set(inputs["families"]) and set(inputs["families"]).issubset({"iphone", "ipad"})
    for family, evidence in inputs["families"].items():
        app = resolve(evidence.get("appPath", inputs.get("appPath")))
        info = app_identity(app)
        setup_app = resolve(evidence.get("setupAppPath", inputs.get("setupAppPath")))
        setup_info = app_identity(setup_app)
        binary = app / info["CFBundleExecutable"]
        if b"--countstick-test-photo" in binary.read_bytes() or b"--countscope-test-ledger" in binary.read_bytes():
            raise ValueError("Release binary still includes a Debug-only import/ledger hook")
        setup_freeze = resolve(evidence.get("setupSourceFreeze", inputs.get("setupSourceFreeze", inputs["sourceFreeze"])))
        assert setup_freeze.is_file()
        xcresult, setup_result = resolve(evidence["xcresult"]), resolve(evidence["setupXcresult"])
        result, setup = summary(xcresult), summary(setup_result)
        configs = result["devicesAndConfigurations"]
        assert len(configs) == 1
        device = configs[0]["device"]
        assert device["platform"] == "iOS Simulator"
        assert ("iPhone" in device["modelName"]) == (family == "iphone")
        attachments_path = resolve(evidence["attachmentsManifest"])
        all_tests = json.loads(attachments_path.read_text())
        capture_tests = [t for t in all_tests if t.get("testIdentifier") == "Count_stickUITests/testCaptureReleaseScreenshots()"]
        if len(capture_tests) != 1:
            raise ValueError("Expected exactly the genuine Release capture test")
        attachments = capture_tests[0]["attachments"]
        for locale, language, apple_locale, effective in LOCALES:
            destination = HERE / "source" / family / locale
            entries = []
            for attachment in attachments:
                if not attachment["suggestedHumanReadableName"].startswith(locale + "-"):
                    continue
                name = re.sub(r"_\d+_.*$", "", attachment["suggestedHumanReadableName"])
                role = name[len(locale) + 1:]
                if role not in ROLE_ORDER:
                    continue
                assert not attachment.get("isAssociatedWithFailure")
                assert attachment["deviceId"] == device["deviceId"]
                original = attachments_path.parent / attachment["exportedFileName"]
                saved = destination / (role + ".png")
                destination.mkdir(parents=True, exist_ok=True)
                if saved.exists() and sha(saved) != sha(original):
                    raise ValueError(f"Refusing to overwrite a different capture: {saved}")
                if not saved.exists():
                    shutil.copyfile(original, saved)
                assert sha(original) == sha(saved)
                with Image.open(saved) as image:
                    dimensions = list(image.size)
                entries.append({"role": role, "path": saved.name, "sha256": sha(saved), "pixels": dimensions,
                                "originalAttachment": recorded_path(original),
                                "capturedAtUTC": datetime.fromtimestamp(attachment["timestamp"], timezone.utc).isoformat()})
            roles = [e["role"] for e in entries]
            assert len(roles) == len(set(roles)) and REQUIRED_ROLES.issubset(roles)
            manifest = {"schemaVersion": 1, "bundleID": "x.Count-stick", "appVersion": "1.0", "build": "4",
                        "brand": "CountScope", "configuration": "Release (ENABLE_TESTABILITY=YES for XCTest; Debug hooks absent)",
                        "locale": locale, "device": device["modelName"], "simulatorId": device["deviceId"],
                        "os": f"iOS {device['osVersion']} ({device['osBuildNumber']})",
                        "appleLanguages": [language], "appleLocale": apple_locale, "effectiveLanguage": effective,
                        "effectiveLanguageVerification": "PENDING_VISUAL_REVIEW; launch arguments and locale capture labels are not alone proof",
                        "captureMethod": "XCTest app.screenshot() attachments, copied byte-for-byte",
                        "sourceFreeze": recorded_path(frozen_path), "sourceFreezeSHA256": sha(frozen_path),
                        "releaseSimulatorExecutableSHA256": sha(binary),
                        "setupSimulatorExecutableSHA256": sha(setup_app / setup_info["CFBundleExecutable"]),
                        "setupSourceFreeze": recorded_path(setup_freeze), "setupSourceFreezeSHA256": sha(setup_freeze),
                        "captureTest": "Count_stickUITests/testCaptureReleaseScreenshots",
                        "captureTestSourceSHA256": sha(ROOT / "Count stickUITests/Count_stickUITests.swift"),
                        "captureResult": recorded_path(xcresult), "setupResult": recorded_path(setup_result),
                        "captureSummary": result, "setupSummary": setup,
                        "recordOrigin": {"method": "Debug 1.0(4) processed the real fixture through production inference and saved the result through app UI. Release 1.0(4) reopened it. No marker coordinates or count were fabricated.",
                                         "fixture": "release/fixtures/bamboo-skewers-cc0.jpg", "fixtureSHA256": sha(ROOT / "release/fixtures/bamboo-skewers-cc0.jpg"),
                                         "groundTruthCount": None, "noteEnteredThroughUI": "Dinner prep" if locale == "en-US" else "晚餐备料"},
                        "visualReview": "PENDING", "screenshots": sorted(entries, key=lambda x: ROLE_ORDER.index(x["role"]))}
            write_fresh(destination / "manifest.json", manifest)
            print(destination / "manifest.json")
    frozen_copy = HERE / "metadata-at-render.json"
    if frozen_copy.exists() and frozen_copy.read_bytes() != metadata_path.read_bytes():
        raise ValueError("Metadata freeze already exists with different content")
    if not frozen_copy.exists():
        shutil.copyfile(metadata_path, frozen_copy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-now", action="store_true")
    modes.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    freeze() if args.freeze_now else prepare(args.inputs)


if __name__ == "__main__":
    main()
