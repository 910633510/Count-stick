# CountScope

A native iPhone and iPad photo-counting app. The current automatic mode counts visible bamboo/skewer ends; users review and correct the markers and save local records. Other object-recognition categories are not implemented in this release. The app includes English and Simplified Chinese interfaces.

## Features

- Camera capture and the system photo picker.
- On-device Core ML inference with visible processing, error and empty-result states.
- Automatically estimated bundle regions, with conservative local text/background filtering and visible outlines; no manual region selection.
- Add/remove markers, undo corrections and inspect the photo in a larger editor.
- Manually saved history with notes, reopen/delete, and system sharing of a marked image.
- Four free completed automatic counts per device, persisted in the device-only Keychain. Failed or canceled processing, corrections and reopened history do not consume a count.
- CountScope Pro monthly subscription: an Apple-eligible 7-day free trial, then US$2.99/month in the USA; other storefronts use Apple-localized prices. Active subscribers have unlimited automatic counts.
- StoreKit 2 verified purchases, restoration, expiration/refund handling, and subscription management.
- No app account, ads, analytics or remote photo-processing API. Apple handles purchases and subscription access.

Automatic counts and region outlines are estimates; an outline follows detected candidates and may miss undetected ends. Uncertain regions preserve the original detections, and corrections remain possible anywhere in the photo. See [automatic-region evaluation](release/accuracy/automatic-region/README.md) for development results and a separately frozen new-photo test. Review the markers before using a result.

## Build

Open `Count stick.xcodeproj` in Xcode. Select the `Count stick` scheme and an iOS Simulator or a configured iOS device. The minimum deployment target is iOS 17. The app retains its original bundle ID `x.Count-stick`.

```sh
xcodebuild -project 'Count stick.xcodeproj' -scheme 'Count stick' \
  -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 16 Pro,OS=18.5' \
  -derivedDataPath output/DerivedData CODE_SIGNING_ALLOWED=YES CODE_SIGN_IDENTITY=- test
```

Select an installed simulator matching the destination. StoreKit integration tests were verified on iOS 18.5; the installed iOS 26.5 simulator had a StoreKitTest configuration-service failure, recorded in the release evidence.

The unit suite exercises quota persistence, actual local StoreKit test transactions, coordinate conversion, marker correction, image orientation, local record persistence and real model loading. UI tests exercise the main flows. A DEBUG-only `--countstick-test-photo <absolute path>` argument supports deterministic QA by loading a real photo and running the actual detector; it is excluded from distribution builds.

## Release

See [release/STATUS.md](release/STATUS.md) for the verified package, test evidence and current App Store review state.

The existing new App Store record **6826601708** is named **CountScope: Photo Counter / CountScope 拍照计数**, retaining bundle ID `x.Count-stick`. Build 4 replaces the first paid-download submission with a free app and optional monthly subscription. The original Build 3 submission was withdrawn for this change. App version 1.0 (4), the subscription and subscription group were submitted together on 2026-10-25 and verified as Waiting for Review, with automatic release after approval. See `release/store-status.json`; waiting for review is not approval or public availability.

Current product ID: `com.xmaillc.countscope.pro.monthly`. Download pricing is zero in all 175 existing storefronts; the verified monthly prices are US$2.99 (USA) and CNY22 (China), with Apple equalization elsewhere. All 175 current storefronts offer the eligible 7-day trial. New territories are not enabled automatically.

Current release artifacts live in `release/subscription-build4/` and `release/screenshots/build4/`. Build 3 metadata, packages and reports are preserved as history.

- `release/subscription-build4/ASO-brief.json`: current Chinese/English positioning and screenshot plan.
- `release/subscription-build4/metadata.json`: current subscription product-page copy and pricing.
- `release/scripts/`: scoped signing, metadata and model-smoke helpers.
- `release/ExportOptions.plist`: App Store export configuration.
- `release/fixtures/sources.json`: photo fixture provenance.

The release uses the official API first, with the website for operations the API cannot perform. New-record creation and App Privacy publishing were completed on the website; metadata, pricing, asset placement and submission used the API. The older `com.yang.stickcount` app is unchanged. Credentials and private signing receipts remain outside this repository. Fetch the current Apple schema with `python3 release/scripts/fetch_apple_schema.py` before running release helpers. The original `apply_metadata.py` is a historical paid-download helper and intentionally rejects subscription metadata; do not use it to restore the former CNY12 pricing.

[Privacy policy](https://www.xmaillc.com/count-stick-privacy.html) · [Support](https://www.xmaillc.com/count-stick-support.html)
