# CountScope ASO 与命名建议

查询日期：2026-10-27。目标：App 6826601708，Bundle ID `x.Count-stick`，1.0 / Build 4。

本次已完成 Apple Ads API 查询、竞品重名检查、中英文修改稿和技能更新。当前 App Store Connect 返回 `WAITING_FOR_REVIEW`；以下修改尚未写入商店，也未撤回现有审核。

## 推荐名称

| 字段 | en-US | zh-Hans |
| --- | --- | --- |
| 名称 | **CountScope: Photo Counter** | **拍照计数 CountScope** |
| 副标题 | Skewers, markers & history | 竹签识别、手动校对与本地记录 |
| 关键词长度 | 77 UTF-8 字节 | 86 UTF-8 字节 |

保留 CountScope 作为品牌。它可以容纳后续计数类别；目前的标题描述照片计数，副标题和介绍明确当前自动识别面向签头。中文将“拍照计数”前置，便于读懂用途。这是编辑判断，不是已测得的排名或转化提升。

CountLens、TallyLens 已有同类产品，排除。另比较了 CountScope: Count from Photos、CountScope: Photo Tally、CountFrame: Photo Counter，均没有足够依据优于现名。未检出 CountScope 的字面同名标题，不等于商标审查或 Apple 名称可用保证。完整对比和查询记录见 [名称研究](research/names.md)。

## Apple Ads 实测

通过现有本地客户端验证了 XM LLC (US) 账户身份和访问权限，调用的是 Apple Ads Platform API v1。没有创建广告、投放或改变预算。

周窗口：2026-10-18 至 2026-10-24；下面返回行均为美国、`PRODUCTIVITY_UTILITIES` 类别。

| 返回搜索词 | searchPopularity1to100 | searchPopularityInGenre | 处理 |
| --- | ---: | ---: | --- |
| counter | 58 | 64 | 与计数相关，支持保留 Counter 功能描述；不能推断 Photo Counter 短语有相同热度 |
| countdown | 66 | 73 | 与当前功能无关，排除 |
| countdown widget | 57 | 64 | 与当前功能无关，排除 |

英文细分词、中文种子词没有返回记录，含“计数”的扩展查询也没有返回行。其含义是本次查询未提供数据，不能写成零搜索量。短语查询成功但结果为空，且该接口没有已文档化的国家过滤。

App 专属关键词建议失败，结构化错误为 `VALIDATION_ERROR / INVALID_INPUT adamId App not found or access denied`。这说明当前账户下该 App 查询不可用；尚不能证明仅由待审核状态导致。未用其他 App 的建议冒充本 App 的数据。

这些分数是相对热度，不是月搜索次数、自然排名或竞争难度。[原始请求与脱敏证据索引](apple-ads-evidence.json) 保留了请求范围、日期、结果和失败记录。

## 本次具体优化

- 中文名称前置用途；英文保留现有品牌与 Photo Counter。
- 副标题明确现有签头识别、校对和历史记录能力，避免通用物体识别暗示。
- 关键词清理标题/副标题重复项，补充与当前用途相关的词；不加入钢管、菌落、竞品名或倒计时词。
- 描述首段直接说明“拍照—核对—保存”的用途，其余功能、订阅、隐私和支持内容保持现有版本原文。
- 保留 4 次免费识别、符合资格的 7 天试用及 US$2.99/月规则；未修改任何价格或订阅产品。
- 保留现有中英文 iPhone/iPad 各 3 张真实 Build 4 截图。12 个线上素材均已重新下载，与本地批准图片的 RGB 像素一致；12 个 placement 均 ACTIVE。图片叙事仍是自动参考范围、手动校对、记录与分享。

完整可提交字段在 [metadata-candidate.json](metadata-candidate.json)，逐字段前后差异在 [metadata-diff.json](metadata-diff.json)。所有名称、副标题、关键词、宣传文本与描述均已完成长度检查。

## 应用到商店

Apple 对名称等字段的可编辑状态有限制。当前版本正在等待审核；如要将这份标题、关键词等改动用于本次发布，需要授权撤回现有提交、修改后重新提交。Apple 说明重新提交会重新开始审核流程。当前未执行该动作。

如果保留本轮审核，则在下一个可编辑版本应用此稿，并重新核对届时实际功能。不能因钢管/菌落有实验代码而提前宣传这些能力。

## 可复用流程

已更新 `/Users/xue/.codex/skills/app-store-connect-release/SKILL.md`，新增 `references/apple-ads-aso.md`，包含：已有客户端发现、账户核实、按国家与周期查询、空数据与错误解释、候选名称比较、重名核查、字段长度验证、真实截图核对、ASC 写入后回读，以及待审核时保留提交的处理方式。技能结构校验通过。

主要官方依据：[关键词建议](https://developer.apple.com/documentation/apple-ads-platform-api/query-keyword-suggestions)、[搜索热度](https://developer.apple.com/documentation/apple-ads-platform-api/query-app-search-term-popularity-data)、[产品页](https://developer.apple.com/app-store/product-page/)、[名称字段](https://developer.apple.com/help/app-store-connect/reference/app-information/app-information/)、[撤回审核的影响](https://developer.apple.com/help/app-store-connect/manage-submissions-to-app-review/remove-a-submission-from-review/)。
