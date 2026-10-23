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

The requested monetization is a **paid download, CNY 12 in China**, with no in-app purchases. App Store prices must be set and verified against the newly created App Store record; a local configuration does not set the live price.

- `release/ASO-brief.json`: sourced Chinese/English positioning and screenshot plan.
- `release/metadata-draft.json`: localized product-page text and target pricing.
- `release/scripts/`: scoped signing, metadata and model-smoke helpers.
- `release/ExportOptions.plist`: App Store export configuration.
- `release/fixtures/sources.json`: photo fixture provenance.

The public App Store Connect API does not create new App Store records. Create the new record for `x.Count-stick` before upload; do not reuse the older `com.yang.stickcount` app. Credentials and private signing receipts remain outside this repository. Fetch the current Apple schema with `python3 release/scripts/fetch_apple_schema.py` before validating or running the metadata helper. Its default is a read-only plan; writes require an explicit `--apply` and pricing requires `--set-price`.

[Privacy policy](https://www.xmaillc.com/count-stick-privacy.html) · [Support](https://www.xmaillc.com/count-stick-support.html)
