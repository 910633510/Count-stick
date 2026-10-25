# Count Stick release status

**Submitted to Apple App Review. Version 1.0 (3) and submission both read back as `WAITING_FOR_REVIEW`. This is not approval or a public-release confirmation.**

Submitted 2026-10-24 11:44:44 EDT through the official App Store Connect API. The release option is `AFTER_APPROVAL`, so Apple can publish this version automatically after approval.

| Field | Verified value |
| --- | --- |
| New app | Count Stick: Skewer Counter |
| App Store app ID | `6826601708` |
| Bundle ID | `x.Count-stick` |
| Version / build | `1.0` / `3` |
| Publisher team | XM LLC — `5B49H2L7N4` |
| China storefront price | Paid download, **CNY 12.00**, applied and independently read back |
| Other storefront prices | Apple equalization from the China base price |
| Configured storefronts | 175 current storefronts; future territories are not automatically added |
| Languages | English (U.S.) and Simplified Chinese |
| Devices | iPhone and iPad, iOS 17+ |
| App Privacy | Data Not Collected, published and verified on the website |
| Age rating | 4+ (`FOUR_PLUS`) |
| Screenshots | 12 current Build 3 images; 4 locale/device groups of 3, verified placements/order |
| Review submission ID | `f9e44425-25f0-40ee-8d5e-ed77913a0f1a` |
| State | `WAITING_FOR_REVIEW` |

[App Store Connect](https://appstoreconnect.apple.com/apps/6826601708/distribution)

## Completed release work

- Created a new app record for the original bundle ID; the older `6752362515` / `com.yang.stickcount` record was not modified.
- Used the website for new-app creation and App Privacy publishing because those operations have no public API endpoint in the fetched Apple OpenAPI 4.5.1. The user clarified that the preferred workflow is API-first with website fallback when needed, not API-only.
- Applied and re-read English/Chinese names, subtitles, keywords, descriptions, promotional text, support/privacy URLs, review notes/contact, category, age declarations, content declaration and price through the API.
- Uploaded the signed Build 3 using Apple's upload tool with API-key authentication, verified `VALID`, and selected the exact processed build.
- Captured current Release 1.0(3) UI on dedicated iPhone/iPad simulators in both languages. Rendered 12 marketing images using the existing ImageGen-derived backgrounds and unmodified real UI pixels. Uploaded through Asset Library, verified processing, downloaded all processed images for pixel comparison, and read back every placement/order. Historical Build 1 images were not used for this release.
- Published the verified no-collection privacy answers. Reused the single review draft, added the intended version, and submitted it. Apple automatically includes its assigned assets with the first app version.
- Read back both the submission and version as waiting for review, with the selected Build 3 unchanged.

## Verification evidence

- `store-status.json`: sanitized live submission receipt and identifiers.
- `pre-submit-build3-review.json`: independent identity, signing, implementation and privacy review.
- `aso-build3-review.json` and `aso-build3-copy-independent-review.json`: live copy/price and independent copy/asset review.
- `screenshots/build3/verification.json`, `independent-visual-review.json`, `upload-review.json`: authentic capture, pixel fidelity, processing and placement evidence.
- `automatic-region-build-review.json`: prior Build 3 package/test evidence. Its pre-submission status fields are historical; use `store-status.json` for current store state.
- Private API receipts: `~/.codex/app-store-connect-release/runs/count-stick-20261024/`. Credentials, contacts and signed upload URLs are not committed.

The submitted IPA remains `output/automatic-region-candidate/Count stick.ipa`, 9,313,754 bytes, SHA-256 `4218999b24aca4f62a29e564aa09be71c441de33fa5876821c7069732fdf417b`. Bundle, version, team entitlements, strict code signing and absence of the Debug photo hook were verified. No production source/model change was required in this release-preparation turn.

## Accuracy and product scope

The submitted product is assisted counting with explicit manual review, editable markers, local history and sharing. The automatic outline is a reference derived from detected candidates, not a guaranteed complete bundle boundary. Store copy and current UI tell users to check all visible ends inside and outside the outline. The release request does not turn earlier recognition failures into a pass.

The frozen experiments remain unchanged: on 9 previously seen development scenes, filtering reduced FP83 to34 with TP674/FN38 unchanged; in 2 new scenes, 31 of140 confirmed endpoints remained missed. The single nominal new-set FP improvement is sensitive to an uncertainty boundary and is not treated as reliable background-removal evidence. Complete automatic checkout accuracy and physical iPhone acceptance have not been established. See `accuracy/automatic-region/README.md` and `accuracy/web-set-v3/REPORT.md`.

The user explicitly requested creation and submission on 2026-10-24 after the prior limitation disclosure. Historical preparation and investigation status is preserved in `STATUS-before-submission-20261023.md`; it is not the current release state.
