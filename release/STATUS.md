# CountScope release status

**Version 1.0 (4), its monthly subscription version and subscription-group version are all `WAITING_FOR_REVIEW`. This confirms submission, not approval or public availability.**

Submitted **2026-10-25 00:27:52 EDT**, using the official App Store Connect API. Release is `AFTER_APPROVAL`.

| Field | Verified value |
| --- | --- |
| App | CountScope: Photo Counter / CountScope 拍照计数 |
| App Store ID / bundle | `6826601708` / `x.Count-stick` |
| Version / build | `1.0` / `4` |
| Publisher team | XM LLC / `5B49H2L7N4` |
| Download | Free in all 175 existing storefronts |
| Free use | 4 completed automatic counts per device; device-only Keychain; no daily reset |
| Monthly subscription | `com.xmaillc.countscope.pro.monthly` |
| Price | USA US$2.99/month; China CNY22/month; other Apple-equalized local prices |
| Trial | 7 days for Apple-eligible first-time subscribers; then monthly auto-renewal |
| Subscriber benefit | Unlimited automatic counts while the subscription/trial is active |
| Availability | Same 175 territories; future territories not automatically enabled |
| Privacy | Data Not Collected retained after source/data-flow review |
| Screenshots | 12 current Build 4 images; iPhone/iPad × English/Simplified Chinese × 3 |
| Submission | `1e69912e-a6a8-44f6-a202-3c67aa632d4e` |
| App version | `d0c1b249-a423-475d-892f-f53130b2eb69` |
| Subscription / group | `6826741970` / `22511744` |
| Selected build | `e240daf9-6d06-42df-9eb7-ec1687022404`, `VALID` |

[App Store Connect](https://appstoreconnect.apple.com/apps/6826601708/distribution)

## Changes and verification

- Rebranded the app, both localized desktop names, exported-image title, store listing and website to CountScope. The selected full-bleed camera-and-number icon replaces the rejected abstract draft. Bundle and app identity remain unchanged.
- Replaced the pending paid-download release with free download and StoreKit 2 monthly access. Quota is charged only for completed automatic results (including zero detections); failed/canceled processing, manual correction and reopening saved history do not consume another count. Entitlement verification, purchase, pending/cancel, restore, expiry and refunds are implemented. Saved history stays accessible after access ends.
- The old Build 3 submission was withdrawn. Its original state/artifacts are preserved in `STATUS-build3-submitted-20261024.md`, `store-status-build3-submitted-20261024.json`, and the Build 3 asset/review files. The older separate `com.yang.stickcount` App Store record was not modified.
- **46 functional tests passed** (8 quota/Keychain, 5 local StoreKit transactions, 26 existing tests, 7 UI flows). **4 further Release capture tests passed** on iPhone/iPad in English/Chinese. iOS 18.5 was used after a StoreKitTest configuration-service failure in the installed iOS 26.5 simulator; failed environmental runs remain recorded. These are local tests, not production billing or physical-camera acceptance.
- Final source, IPA identity/signing, both localized names, selected camera icon, absence of Debug hooks and `.storekit` test resources were independently verified. Final IPA: `10,405,279` bytes, SHA-256 `681e04826e105732701c833dd50ce1bcbf209a4617f069e6a971586e7906a920`.
- Captured authentic Release UI, rendered 12 promotional images using the established ImageGen background, and independently compared embedded UI pixels. Uploaded assets were downloaded from Apple and all 12 matched the approved RGB pixels. The exact old 12 placements were replaced; original library images remain preserved.
- Support/privacy pages were deployed from website commit `5d7d5e8e6fb78ccc2ee7da3a9f51f9d8f1112405` and verified HTTP 200 with identical content. Apple handles purchases/restoration; no photos, OCR strings or counts are uploaded to a developer service. No privacy-label change was required.
- App, subscription version and subscription-group version were added to the same review draft and submitted together. Independent API readback confirmed all three version states and the overall submission as `WAITING_FOR_REVIEW`. Review item resources use a separate state machine and still report `READY_FOR_REVIEW`; this is recorded without relabeling those items.

## Evidence

- `store-status.json`: current sanitized release state and identifiers.
- `subscription-build4/`: current metadata, ASO brief, API readbacks, independent implementation/monetization/build/privacy/submission passes and captain gate.
- `screenshots/build4/`: authentic source manifests, final assets, visual/pixel checks and Apple placement/readback evidence.
- `artwork/countscope/`: built-in ImageGen source images, final camera-icon prompt and verification. Abstract draft is historical only.
- Final archive/IPA: `output/subscription-build4/CountScope-1.0-4-final.xcarchive` and `output/subscription-build4/export-final/Count stick.ipa`.
- Private API receipts: `~/.codex/app-store-connect-release/runs/count-stick-subscription-20261025/`. Credentials, review-contact details and signed upload URLs remain private.

## Accuracy and scope

This release changes branding and monetization, not the recognition model. Automatic recognition currently targets visible skewer ends; steel pipes, colonies and other future categories are not implemented or advertised as available. Counts and reference outlines remain estimates requiring whole-photo manual verification.

All frozen prior experiments remain unchanged. The model's complete automatic-checkout accuracy and physical iPhone acceptance have not been established. See `accuracy/automatic-region/README.md`, `accuracy/web-set-v3/REPORT.md` and `subscription-build4/quality-review.json`.

During final capture, disk pressure was resolved by deleting only three completed-build `ModuleCache.noindex` folders created in this turn. Source, IPA/archive and frozen accuracy evidence hashes were retained. The private/local audit is `output/subscription-build4/owned-cache-cleanup.json`; optional simulator diagnostics were interrupted after tests passed so XCTest could finalize its result bundle.
