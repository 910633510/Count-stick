# CountScope ASO naming review

Researched: 2026-10-27. Target: app 6826601708 / `x.Count-stick`. This is a read-only naming recommendation, not a live metadata readback or an Apple Ads performance report.

## Recommendation

Retain **CountScope** as the brand. English title: **CountScope: Photo Counter**. Simplified Chinese title: **拍照计数 CountScope**. The Chinese change puts the recognizable task before the less familiar English brand; this is an editorial hypothesis, not a measured ranking improvement. The English title already combines an extensible brand with the actual photo-counting task. Rebranding again has no demonstrated search-demand advantage.

Current automatic counting remains limited to visible skewer ends. The subtitle and opening description should make that scope visible. Do not add steel pipe, colony, pills, general-purpose object recognition, guaranteed accuracy, or one-second results to current metadata. Pipe mode in the local checkout has no qualified model, and is not part of submitted Build 4.

| Option | en-US title | Characters | zh-Hans title | Characters | Judgment |
| --- | --- | ---: | --- | ---: | --- |
| 1 | CountScope: Photo Counter | 25 | 拍照计数 CountScope | 15 | Recommended; retains existing brand and clear task |
| 2 | CountScope: Count from Photos | 29 | 数照 CountScope - 拍照计数 | 20 | More conversational English; introduces an extra Chinese brand without evidence of benefit |
| 3 | CountScope: Photo Tally | 23 | CountScope 点数相机 | 15 | Short; `tally` can imply manual click counters, and 点数相机 is a crowded phrase |
| 4 | CountFrame: Photo Counter | 25 | 点数镜 - 拍照计数 | 10 | Reserve only; no same-name App Store result found in bounded query, but CountFrame is also used by a counting toy, and it abandons existing brand continuity |

Name checks do not establish trademark clearance or App Store name availability. An exact ASC mutation response, when the app status permits editing, is needed to establish whether Apple accepts the chosen localized name. Do not rename subscription product IDs or bundle IDs just for ASO.

## Recommended companion fields

| Locale | Subtitle | Characters | Keywords | UTF-8 bytes |
| --- | --- | ---: | --- | ---: |
| en-US | Skewers, markers & history | 26 | `bamboo,wooden,stick,barbecue,hotpot,restaurant,offline,correction,notes,share` | 77 |
| zh-Hans | 竹签识别、手动校对与本地记录 | 14 | `串串,烧烤,火锅,餐饮,木签,点数,相册,标记,分享,备注,离线` | 76 |

These are relevance-based candidates awaiting any available Apple Ads evidence. Extra bytes are deliberately unfilled rather than used for unsupported objects or filler. The keyword fields omit title/subtitle words, competitor brands, and unsupported capabilities. Description and promotional text can continue to explain four free completed automatic counts, Apple-eligible seven-day trial, current monthly subscription, and on-device processing after the parent verifies current live facts.

Suggested description openings:

- en-US: “Count skewer ends from a photo, check each marker, and save the result. CountScope combines automatic suggestions with manual correction and local history.”
- zh-Hans: “拍下签束端头，查看自动标记，逐点校对后保存结果。CountScope 将照片识别、手动修正和本地记录放在一起，方便回看和分享。”

The first three screenshots can retain the authentic Build 4 narrative: automatic reference outline with review / add-remove-undo markers / saved result and sharing. Do not imply outline accuracy or fabricate new pipe screens.

## Keyword intent seeds for Apple Ads research

- Current direct intent (EN): photo counter, skewer counter, count skewers, stick counter, count from photos.
- Current direct intent (ZH): 拍照计数, 数签子, 竹签计数, 串串计数, 木签计数, 点数相机.
- Secondary current workflow: offline counter, tally photo, count history, 照片标记, 离线计数, 计数记录.
- Future-only excluded from current metadata/ads targeting: pipe counter, colony counter, 数钢管, 菌落计数.
- Competitor brands: research comparison only, never organic metadata keywords.

No popularity, volume, competition index, conversion, CPI or revenue values are asserted here. Public iTunes search results are metadata/collision evidence, not Apple Ads search-popularity values and not personalized App Store ranking observations.

## Evidence and collision review

1. Existing local `release/subscription-build4/metadata.json` names CountScope, en-US and zh-Hans, and explicitly excludes pipe/colony/general recognition. Current `README.md` confirms no qualified pipe model is bundled. These files establish product scope, not current store approval status.
2. [CountThings from Photos](https://apps.apple.com/us/app/countthings-from-photos/id1196810823) uses a task-based name and an automatic-counting subtitle. Its own description emphasizes specialized templates, offline processing, and organizational workflows.
3. [Count This - Counting App](https://apps.apple.com/us/app/count-this-counting-app/id1553863458) already occupies the Count This naming space and describes photo marking/manual correction. Several unrelated App Store products use very similar names, making a new Count This variation difficult to distinguish.
4. [精点相机](https://apps.apple.com/cn/app/id1627219849) and [计数相机-拍照计数](https://apps.apple.com/cn/app/id6747957693) show established Chinese task language around 拍照计数/相机, while making their own broader feature claims. Those claims are not evidence that CountScope supports the same objects.
5. CountLens is already used by multiple counting apps: [CountLens — AI Object Counter](https://apps.apple.com/ca/app/id6809061893), [CountLens](https://apps.apple.com/se/app/id6793931747), and [CountLens: Count Pipes & More](https://apps.apple.com/mo/app/id6775705623). Reject as a new brand.
6. [TallyLens official site](https://tallylens.pro/) already presents a photo object-counting app. Reject as a new brand.
7. [CountFrame counting toy](https://morgans-minis.com/products/countframe) is an existing adjacent use. This is a naming observation, not a legal conclusion.
8. Bounded Apple public iTunes Search API queries for CountScope, CountFrame, 数照 and 点数镜 in US and CN found no literal-substring title matches. Full returned title/ID lists, query URL and UTC retrieval time are saved in `itunes-name-query.json`. The API returns loosely related names for some queries; total results are not exact-match counts or search demand. The absence of an indexed result also does not establish the target app's current review/availability state.

## Official rules used

- [Apple product page guidance](https://developer.apple.com/app-store/product-page/): distinctive memorable name, up to 30 characters; subtitle up to 30; relevant keywords; no competing names; promotional text does not affect search ranking. The page's general guidance says 100 characters for keywords, while the specific field reference below says 100 bytes; use the stricter byte preflight.
- [App information reference](https://developer.apple.com/help/app-store-connect/reference/app-information/app-information/): name/subtitle lengths and name editability depend on review/version state.
- [Platform version information](https://developer.apple.com/help/app-store-connect/reference/app-information/platform-version-information/): keyword field maximum 100 bytes. Count commas and UTF-8 bytes in exact outgoing text.
- [View and edit app information](https://developer.apple.com/help/app-store-connect/create-an-app-record/view-and-edit-app-information/): app name, subtitle, keywords and company name contribute to searchability; only editable-state properties can be changed.

This review used primary Apple pages and source-owned competitor product pages. No live App Store Connect writes, advertising mutations, campaign creation, or paid research occurred.
