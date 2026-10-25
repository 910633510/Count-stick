# 自动候选区域：实现与证据

自动区域 v2 已实现，用户无需手动框选。它能过滤部分文字与离群误报，但**不能恢复底层模型漏掉的端头，也不能证明轮廓覆盖整束签子**。新 v3 两场景仍漏掉31个已确认端面，不足以支持无需人工核对直接结账。

本文记录2026-10-23的固定候选与冻结证据，没有触发新推理或修改算法、标签。自动区域算法的“开发v1/v2”与照片批次“web-set v1/v2/v3”是不同版本序列。

**实际管线。** [StickDetector.swift](../../../Count%20stick/StickDetector.swift) 先将照片按EXIF转正并限制最长边为2048，再全图运行一次原有640×640 Core ML模型：`scaleFill`、confidence `0.25`、NMS IoU `0.45`。每个观察获得本次结果内稳定的UUID，框转换为左上原点的归一化坐标；使用真实`observation.confidence`，不使用单类别归一化后的label分数。

随后进行本机Vision OCR：`.accurate`，语言`zh-Hans/zh-Hant/en-US`，关闭语言纠错，最小文字高度`0.015`；第一候选confidence≥0.8且至少两个字母/数字/CJK字符才进入区域选择器。只传框、置信度及字符数，不持久化OCR字符串。OCR失败时文字区域为空，几何筛选仍可执行。

[AutoCountRegion.swift](../../../Count%20stick/AutoCountRegion.swift) 只筛选现有全图检测ID：没有手动ROI、裁图重推理或新检测点。结果包含`keptIDs`、带`text/isolated`原因的`excluded`、`regions`和`none/sparse/ambiguous/empty`状态，均可Codable/Sendable。`StickDetectionResult`同时保留原始与筛选后标记；`CountWorkspace`防止旧请求覆盖新结果，历史记录中的自动区域字段可选。旧记录兼容，手动标记校正不被自动框限制。

**固定规则和回退。** 完整常量见[protocol-v2.json](protocol-v2.json)。尺度`s`为考虑图像宽高比后的检测框面积平方根的中位数。

- 分组使用中心距离≤3s。OCR删除需要可靠的非文字锚点组：至少6点，横纵跨度均≥2s；候选近锚点≤2.5s或属于原始少于6点的小组件时保留。
- 文字框长轴须≥2s、长短轴比≥1.4；检测中心在文字框内且至少70%检测面积在框内，才可删除。OCR文字附近的点不能充当自己的非文字锚点。
- 几何删除仅限单点组件：confidence<0.7、距支持组>6s、且在支持框外扩3s之外。成对/较大组件、高置信孤点、附近伸出端保留；支持组件至少3点，可靠组件至少6点，支持率至少65%。
- 原输入为空才返回`empty`；原输入<6在OCR前保留全部原检测。无效输入、过滤后稀疏/空或支持不足时回滚原检测，清空排除与轮廓，返回`sparse/ambiguous`。无可靠非文字锚点时OCR不删点。回退避免误删，不代表原计数准确。

轮廓由**保留的检测框**合并并加边距，解释候选分组；它不是独立的整束像素分割。`none`仅表示分组规则成立，不表示数量可靠。OCR误认大组端头纹理时仍可能出错，不能保证所有多束布局下绝不误删。

**保留的v1缺陷与v2修复。** [v1源码](candidates/AutoCountRegion-v1.swift)、[协议](protocol-v1.json)、[开发结果](development-v1-report.json)未被覆盖。

| 历史反例 | v1失败 | v2修复与独立复核 |
|---|---|---|
| 3个高置信端头被OCR框覆盖 | 先被全部删掉，再返回`empty`，形成假零 | 输入<6先fail-open；raw3/kept3/excluded0。 |
| 主束9点，OCR覆盖三角副束3点 | OCR邻域保护与分组门槛不一致，12→9 | 原始小组件保护；12点全部保留，副束3点未删。 |
| HKTV两行真实文字 | 文字误报互相提供二维邻域支持，OCR虽识别却0删除 | 使用独立非文字锚点；开发集删除17个文字误报，TP44不变。 |

反例输入和回归测试见[AutoCountRegionTests.swift](../../../Count%20stickTests/AutoCountRegionTests.swift)，独立复核见[review-v2.json](review-v2.json)。26项集成单测通过（26 passed/0 failed/0 skipped），证据为[automatic-region-unit-v2.xcresult](../../../output/automatic-region-unit-v2.xcresult)；其中11个纯几何测试体也独立执行通过。本文不认定最终UI、签名、提交或上架状态。

**九张已见场景：开发诊断。** 使用冻结web-set v1/v2缓存检测，原checkpoint在报告`provenance`中。区域v1运行OCR；区域v2复用同一OCR缓存。数据已被观察和用于修订，不能称盲测。

| 处理 | TP | FP | FN | 完整计数MAE（6图） | 完全正确 |
|---|---:|---:|---:|---:|---:|
| 原始build2 | 674 | 83 | 38 | 11.667 | 0/6 |
| 历史区域v1组合 | 674 | 58 | 38 | 7.500 | 0/6 |
| 区域v2仅文字 | 674 | 39 | 38 | 7.167 | 0/6 |
| 区域v2仅几何 | 674 | 78 | 38 | 10.833 | 0/6 |
| 固定区域v2组合 | 674 | 34 | 38 | 6.333 | 0/6 |

组合规则在开发集未增加标注端头损失：flatblack文字图FP44→17，HKTV图FP20→3，餐饮图77→72；Pospal两根高出的真实端头均保留，其近邻手部误报仍保留。28根近拍图依旧只有8个输出，返回`ambiguous`，不能据此声称计数改善已足够。逐图TP/FP/FN、忽略区和完整计数资格见[development-v2-report.json](development-v2-report.json)。

**冻结后一次运行的新v3两场景。** [checkpoint](holdout-v3-checkpoint.json)冻结于`2026-10-23T16:29:15.197289+00:00`，报告完成于`16:29:41.256695+00:00`。照片与人工标签先冻结，再比较同一次全图检测的原始输出与共享选择器输出。环境是macOS26.4.1、Core ML CPU-only与Vision OCR，不是真机验收。

| 图片 | 人工参考 | 原始→自动输出 | 原始TP/FP/FN | 自动TP/FP/FN |
|---|---|---:|---:|---:|
| Vietnam错落圆端面 | 37个确认端面，部分真值/忽略区 | 19→18 | 18/1/19 | 18/0/19 |
| Allegro方端面 | 103个完整可见端面 | 92→92 | 91/1/12 | 91/1/12 |
| 合计 | 140个确认端面 | 111→110 | 109/2/31 | 109/1/31 |

Vietnam不参与整图MAE：37只是明确标注端面数，不是物理整束总数；其中仍漏掉19个，且有10个确认端面落在自动轮廓外。Allegro净少11，源于12次漏检和1次重复误报；唯一完整计数样本仍不正确，0/1、MAE11。

**唯一表面上的FP收益有边界敏感性。** Vietnam被删除的`det-17`在`(5.452,591.875)`，置信度约0.302；冻结忽略区上边界为`y=592`。它仅在边界外**0.125像素**，因此按既定协议被计为FP。这不能证明可靠去除了普通背景物体；标签边界没有在看结果后调整。两张新图的OCR合格区域均为0，本轮也没有独立验证文字过滤收益。

召回仍为109/140（77.86%），自动输出精确率109/110（99.09%）**不是计数准确率**。完整解释、独立匹配复算和审计见[v3报告](../web-set-v3/REPORT.md)、[运行回执](../../../output/web-annotation-v3/evaluation-auto-region/report.json)与[审计回执](holdout-v3-review.json)。评估完整性通过不等于自动结账质量通过。

**冻结指纹与运行命令。** 以下SHA256对应这次运行；完整源码、模型、照片和标签绑定在checkpoint中。后续修改必须另建实验，不能回写这些证据。

```text
e8e17cf3fbb2e3f399ba0452ca72ec022faf685b0b9a1bee3ed98bbbce141112  Count stick/AutoCountRegion.swift
50462cf49c71e06c947bca0e870ff4cca36e3ef63414d0cdf23bdd12db8f10fa  Count stick/StickDetector.swift
21717eef227d96f45419d0b5730fa4114e6877f4a6548a71efe641fc23240090  Count stick/best.mlpackage/Data/com.apple.CoreML/weights/weight.bin
0d1274cd5feac851a9382e2b91e6c54421fd5b5ff3354b675843ad8caaba4dc8  release/accuracy/automatic-region/protocol-v2.json
6d2238f44c2b71a36ee347c43f887a5a61cbaddba649bca05693a37029955ff3  release/accuracy/automatic-region/development-v2-report.json
bfd3d87885e8add72be93919688e6103c3379dc3c019ec02296b40b6a24f2af9  release/accuracy/web-set-v3/manifest.json
067adaa5091813d2f856a127de73b11c37e6cb2bcc5dd13f6cc617fcf4c4cf79  release/accuracy/automatic-region/holdout-v3-checkpoint.json
165d056e14d5fdbace78da918c59cfbd8b418b66a8ad3f593404efab8ce82080  output/web-annotation-v3/evaluation-auto-region/report.json
```

从仓库根只读验证，不触发推理：

```sh
python3 release/accuracy/automatic-region/evaluate-holdout-v3.py --validate-only
```

仅重评分已有绑定预测时，使用新的空输出目录；不要再运行v3推理：

```sh
python3 release/accuracy/automatic-region/evaluate-holdout-v3.py \
  --checkpoint release/accuracy/automatic-region/holdout-v3-checkpoint.json \
  --predictions output/web-annotation-v3/evaluation-auto-region/predictions.json \
  --output-dir output/web-annotation-v3/recheck-from-cached-01
```

已有首次运行使用同一checkpoint与`--run-inference`，前置输入、预测和报告都保留。源码/模型与checkpoint不一致时验证器拒绝执行。历史开发驱动`evaluate-development.py`和`evaluate-development-v2.py`使用当前工作区源码及固定输出路径，不应直接重跑覆盖结果；重现历史版本须在隔离副本绑定对应`candidates/`快照并改用新输出目录。本次文档工作没有执行上述命令。

**关键限制与下一步。** 后处理无法恢复从未进入候选集的端头，自动框也不能用来证明没有漏检。现有v3已暴露结果，只能作为后续回归案例；新的模型选型、尺度处理或训练必须用不同来源/拍摄批次的全新测试集验收。

下一阶段应补充错落长短端头、方端头、大近拍、小密集束，以及肤色/木纹/文字背景的训练样本，标每个可见端面的中心、真实框或必要的面掩膜，并保留不确定区。按来源和拍摄批次隔离训练/校准/测试，增强图与裁片不得跨组。找回可训练检查点/训练源码，或建立轻量端头专用模型；本阶段没有训练或换权重。若需覆盖整束的自动边界，应另外验证基于原图像素的束区域模型，而不是把候选框并集当作完整边界。

验收以整图完全正确率、MAE、严重少数/多数和人工校正量为主，TP/FP/FN用于诊断；不能让平均指标掩盖37个确认端面只找到18个的失败。最终还需验证Core ML导出后的真实iPhone计数一致性、延迟、内存与连续使用表现。
