# Count Stick

A native iPhone and iPad app for counting bamboo skewers from photos, reviewing the detected markers, and saving a local record. The app includes English and Simplified Chinese interfaces.

## Features

- Camera capture and the system photo picker.
- On-device Core ML inference with visible processing, error and empty-result states.
- Automatically estimated bundle regions, with conservative local text/background filtering and visible outlines; no manual region selection.
- Add/remove markers, undo corrections and inspect the photo in a larger editor.
- Manually saved history with notes, reopen/delete, and system sharing of a marked image.
- No account, ads, subscriptions or remote photo-processing API.

Automatic counts and region outlines are estimates; an outline follows detected candidates and may miss undetected ends. Uncertain regions preserve the original detections, and corrections remain possible anywhere in the photo. See [automatic-region evaluation](release/accuracy/automatic-region/README.md) for development results and a separately frozen new-photo test. Review the markers before using a result.

## Build

Open `Count stick.xcodeproj` in Xcode. Select the `Count stick` scheme and an iOS Simulator or a configured iOS device. The minimum deployment target is iOS 17. The app retains its original bundle ID `x.Count-stick`.

```sh
xcodebuild -project 'Count stick.xcodeproj' -scheme 'Count stick' \
  -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath output/DerivedData CODE_SIGNING_ALLOWED=NO test
```

The unit suite exercises coordinate conversion, marker correction, image orientation, local record persistence and real model loading. UI tests exercise the main flows. A DEBUG-only `--countstick-test-photo <absolute path>` argument supports deterministic QA by loading a real photo and running the actual detector; it is excluded from distribution builds.

## Release

See [release/STATUS.md](release/STATUS.md) for the verified package, test evidence and remaining App Store record handoff.

The app is a **paid download, CNY 12 in China**, with no in-app purchases. The new App Store Connect record is **6826601708**. Version **1.0 (3)** was submitted on 2026-10-24 and verified as **Waiting for Review**, with automatic release after approval. The price was applied and independently read back; public availability is not yet confirmed.

- `release/ASO-brief.json`: sourced Chinese/English positioning and screenshot plan.
- `release/metadata-draft.json`: localized product-page text and target pricing.
- `release/scripts/`: scoped signing, metadata and model-smoke helpers.
- `release/ExportOptions.plist`: App Store export configuration.
- `release/fixtures/sources.json`: photo fixture provenance.

The release uses the official API first, with the website for operations the API cannot perform. New-record creation and App Privacy publishing were completed on the website; metadata, pricing, asset placement and submission used the API. The older `com.yang.stickcount` app is unchanged. Credentials and private signing receipts remain outside this repository. Fetch the current Apple schema with `python3 release/scripts/fetch_apple_schema.py` before running release helpers. The metadata helper defaults to a read-only plan; writes require `--apply` and pricing requires `--set-price`.

[Privacy policy](https://www.xmaillc.com/count-stick-privacy.html) · [Support](https://www.xmaillc.com/count-stick-support.html)
