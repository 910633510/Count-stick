# Count Stick release status

**Accuracy investigation in progress. The user reported poor recognition; do not submit this candidate until the revised counting behavior is evaluated. Existing build and screenshot evidence below is a checkpoint, not final quality acceptance.**

The local app and signed distribution package are prepared. **The new App Store record does not exist yet; nothing has been uploaded, priced or submitted.** The older StickCount app is not this release target.

| Field | Value |
| --- | --- |
| App | Count Stick: Skewer Counter |
| Bundle ID | `x.Count-stick` |
| Current local candidate | `1.0` / `2` (deduplication experiment; not accepted for release) |
| Publisher team | XM LLC — `5B49H2L7N4` |
| Requested pricing | Paid download, China CNY 12.00; not applied live |
| Languages | English and Simplified Chinese |
| Devices | iPhone and iPad, iOS 17+ |
| App Store app ID | Not created |

## Build 2 local candidate

The candidate is signed at `output/accuracy-candidate/Count stick.ipa`. Its 13 unit tests and 2 focused UI workflows passed; see `candidate-build-review.json`. These tests establish implementation behavior, not restaurant counting accuracy. Build 1 images and package below remain historical preparation artifacts.

## Completed build 1 preparation

- On-device photo counting, native camera/photo selection, manual marker correction, undo, enlarged editing, notes, local history and sharing.
- Bilingual UI and camera explanations, app icon, privacy manifest, ASO research and localized store copy.
- Core unit suite: 12 passed. iPhone workflow suite: 3 passed. iPad English/Chinese workflows: 2 passed. Release screenshot runs passed on both device families.
- Distribution archive and IPA export signed and verified; the Release executable excludes the DEBUG photo-import test hook. The local IPA is `output/app-store-final/Count stick.ipa` (9,222,308 bytes), SHA-256 `b15dbcc046c81270d2e8806a3cd794f2aeb8c9fde654eaeaf311ec6c02f6e8f6`.
- [Privacy policy](https://www.xmaillc.com/count-stick-privacy.html) and [support page](https://www.xmaillc.com/count-stick-support.html) are published and HTTP 200 verified.
- Bundle ID registration and App Store provisioning profile were created through the official API.

## Counting accuracy handoff

The confirmed target is restaurant checkout photos of a bundle of skewer ends. Default inference shows duplicate markers and missed terminal faces in the inspected real-photo region. Raising confidence or adding tiles did not establish a reliable improvement without losing recall, The only adopted local candidate change keeps confidence at 0.25 and tightens overlap suppression from 0.7 to 0.45. Across two small regions of one image it preserves 9 matched endpoints while reducing false marks from 13 to 11; two labeled endpoints remain missed. This does not establish restaurant counting accuracy.

The user delegated online collection and annotation to the assistant. A first frozen set now contains five independent online endpoint photos and499 labeled visible endpoints, with two calibration scenes and three holdout scenes. See `accuracy/web-set/REPORT.md`. Build2 predicts96/8/77 on holdout truths97/28/70 (MAE9.33, exact0/3). One predeclared confidence sweep did not improve holdout counting, so production parameters remain unchanged. More independent target-scene coverage and scale-robust model work are needed before release; no additional user photo upload is required to begin research. Side-view sticks and pointed tips are not substitutes for the confirmed endpoint domain.

A second frozen search-and-test round added four new scenes and 213 certain visible endpoints, all held out from parameter selection. Build 2 predicts 47 and 53 on the two complete visible counts 49 and 52 (MAE 1.5, exact 0/2). Across all four new scenes, point precision is 90.52% and recall is 98.59%; 20 of 22 false positives are advertising text. The two partial-border scenes are excluded from exact-count metrics. See `accuracy/web-set-v2/REPORT.md`. This iteration changed no weights, inference parameters, or App code; release remains blocked.

## Required account handoff

Apple's [Apps API](https://developer.apple.com/documentation/appstoreconnectapi/apps) does not create App Store app records. The current public OpenAPI 4.5.1 exposes only GET on `/v1/apps`. This task is using API-only operations.

Create the new record in App Store Connect with:

- Platform: iOS
- Name: `Count Stick: Skewer Counter`
- Primary language: English (U.S.)
- Bundle ID: `x.Count-stick`
- SKU: `count-stick-20261022`

Provide its numeric app ID or page link. App Privacy also requires the website: the prepared answers are no tracking and no data collected by the developer, based on the app's verified local processing and storage. The public API has no privacy-questionnaire publication endpoint. Other required store declarations must be checked against the new record.

Once the record exists, continue with the official API: verify exact identity, apply the draft metadata and CNY 12 price, upload/select the processed build, place the final screenshots, complete readiness checks, and submit review. Read back every change and the actual submitted state.

## Artifacts and limits

The `screenshots/final/iphone/` and `screenshots/final/ipad/` images are build 1 checkpoints and must be recaptured after the counting-quality work is accepted; their render manifests bind each file to `screenshots/source-release/` captures. The originals preserve actual UI pixels and model-returned counts. Preliminary captures are retained locally and are not the final upload set.

`metadata-draft.json`, `ASO-brief.json`, `build-review.json`, `validation-summary.json`, `store-status.json` and the screenshot manifests contain the detailed evidence. Image-generation prompts and original outputs are in `artwork/`. No credentials or private signing keys are stored in the repository.

All UI validation used simulators. Physical camera behavior and real-world counting accuracy have not been accepted on a physical device. Real-photo smoke checks show inference works, not an accuracy guarantee. A non-failing SwiftUI frame-size diagnostic remains during note-keyboard presentation; no visible malfunction was observed in the exercised flows.
