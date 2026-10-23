# Count Stick release status

The local app and signed distribution package are prepared. **The new App Store record does not exist yet; nothing has been uploaded, priced or submitted.** The older StickCount app is not this release target.

| Field | Value |
| --- | --- |
| App | Count Stick: Skewer Counter |
| Bundle ID | `x.Count-stick` |
| Version / build | `1.0` / `1` |
| Publisher team | XM LLC — `5B49H2L7N4` |
| Requested pricing | Paid download, China CNY 12.00; not applied live |
| Languages | English and Simplified Chinese |
| Devices | iPhone and iPad, iOS 17+ |
| App Store app ID | Not created |

## Completed

- On-device photo counting, native camera/photo selection, manual marker correction, undo, enlarged editing, notes, local history and sharing.
- Bilingual UI and camera explanations, app icon, privacy manifest, ASO research and localized store copy.
- Core unit suite: 12 passed. iPhone workflow suite: 3 passed. iPad English/Chinese workflows: 2 passed. Release screenshot runs passed on both device families.
- Distribution archive and IPA export signed and verified; the Release executable excludes the DEBUG photo-import test hook. The local IPA is `output/app-store-final/Count stick.ipa` (9,222,308 bytes), SHA-256 `b15dbcc046c81270d2e8806a3cd794f2aeb8c9fde654eaeaf311ec6c02f6e8f6`.
- [Privacy policy](https://www.xmaillc.com/count-stick-privacy.html) and [support page](https://www.xmaillc.com/count-stick-support.html) are published and HTTP 200 verified.
- Bundle ID registration and App Store provisioning profile were created through the official API.

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

Use `screenshots/final/iphone/` and `screenshots/final/ipad/` for final store images; their render manifests bind each file to `screenshots/source-release/` captures. The originals preserve actual UI pixels and model-returned counts. Preliminary captures are retained locally and are not the final upload set.

`metadata-draft.json`, `ASO-brief.json`, `build-review.json`, `validation-summary.json`, `store-status.json` and the screenshot manifests contain the detailed evidence. Image-generation prompts and original outputs are in `artwork/`. No credentials or private signing keys are stored in the repository.

All UI validation used simulators. Physical camera behavior and real-world counting accuracy have not been accepted on a physical device. Real-photo smoke checks show inference works, not an accuracy guarantee. A non-failing SwiftUI frame-size diagnostic remains during note-keyboard presentation; no visible malfunction was observed in the exercised flows.
