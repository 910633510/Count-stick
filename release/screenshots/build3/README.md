# Build 3 screenshots

Current release candidate: **Count Stick 1.0 (3), `x.Count-stick`**. This folder preserves 20 original XCTest PNG attachments and prepares 12 App Store marketing PNGs, without replacing the historical Build 1 assets.

The later live upload pass is recorded separately in `upload-review.json`: all 12 images were uploaded through the official API to new App Store app **6826601708**, processed, placed in the two device groups for both locales, and read back in the intended order. Downloaded full-resolution processed RGB pixels exactly match the approved local marketing images. The capture/composition records below preserve the earlier local preparation phase; they do not claim review submission or public availability.

## Files

- `final/{iphone,ipad}/{en-US,zh-Hans}/01-result.png`, `02-correction.png`, `03-history.png`: upload candidates, in display order.
- `source/{iphone,ipad}/{en-US,zh-Hans}/manifest.json`: genuine UI language, device, OS, Release build, source hash and screenshot origin.
- `final/.../render-manifest.json`: final image hashes and exact screenshot rectangles. Every complete screenshot is proportionally scaled; no UI pixel is translated, retouched, covered, cropped or synthesized.
- `verification.json`: local validation and per-image review.
- `source-freeze.json`: production source hashes verified before and after capture.
- `metadata-at-render.json`: the marketing copy actually rendered. Existing ImageGen backdrop provenance remains in `release/artwork/marketing/generation.json`.

The three screens show an actual automatic reference outline, correction tools, and a saved local record. The displayed **150** is the model's real result on the CC0 fixture; its ground-truth total is unknown. It is not an accuracy claim. Copy and UI explicitly require manual review, including outside the outline.

## Capture provenance

Dedicated simulators on iOS 26.5 (23F77): iPhone 17 Pro `75EDBD64-9DFD-4A86-AA40-D58ABB301A08` and iPad Pro 13-inch (M5) `6C03C109-6C07-4AE8-BCAE-9EBB956CA92D`. Their Count Stick app data was cleared before setup; unrelated simulators were untouched.

1. `testPrepareReleaseScreenshotRecord` ran in current Debug Build 3. It decoded `release/fixtures/bamboo-skewers-cc0.jpg`, ran actual production CoreML/OCR/automatic-region logic, asserted a nonempty automatic outline and saved through the normal app UI. No count, marker coordinates or outline were seeded.
2. The same source was built as Release with `ENABLE_TESTABILITY=YES` for XCTest. The resulting bundle identifies version 1.0/build 3, and its executable contains no `--countstick-test-photo` Debug import hook.
3. `testCaptureReleaseScreenshots` reopened the genuinely saved record in Release, launched with `AppleLanguages=(en)`, `AppleLocale=en_US` and then `(zh-Hans)`, `zh_Hans_CN`. The Chinese note was entered and checked for exact equality through UI. Nonempty automatic-outline assertions passed again.
4. Both devices' setup and Release-capture tests passed. Evidence is in `output/screenshots-build3/{iphone,ipad}-{setup,release-captures}.xcresult` and corresponding logs. The entire PNG attachment bytes are preserved in `source/`.

## Reproduce composition from preserved originals

From the repository root, with Pillow installed:

```sh
python3 release/scripts/render_screenshots.py --manifest release/screenshots/build3/source/iphone/en-US/manifest.json --manifest release/screenshots/build3/source/iphone/zh-Hans/manifest.json --family iphone --output release/screenshots/build3/final
python3 release/scripts/render_screenshots.py --manifest release/screenshots/build3/source/ipad/en-US/manifest.json --manifest release/screenshots/build3/source/ipad/zh-Hans/manifest.json --family ipad --output release/screenshots/build3/final
```

The renderer reads `release/metadata-draft.json`; verify it matches `metadata-at-render.json` before reproducing this revision. Re-rendered receipts start with visual review pending and must be reviewed again. `prepare_manifests.py` is only for re-exporting this capture run's original attachments while the matching build products and xcresults still exist; it also resets visual review to pending.

Final dimensions: iPhone 1206×2622, iPad 2064×2752. Current Apple specification evidence is in `specification-check.json`. All 12 final images were individually inspected and compared to their originals; all embedded screenshot regions matched the proportionally resized originals byte-for-byte. Live App Store placement validation, uploads and review submission are separate parent-owned steps. No upload was performed by this screenshot task.
