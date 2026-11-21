# 圆钢管端面数据

**后续状态：** 本文记录数据收集和候选划分阶段。候选现已完成独立复核并冻结为 `pipe-ends-maco-v1`，模型训练和验证进度见[本轮总报告](../README.md)。下文的 proposal、未训练等用语保留为当时的阶段记录，不代表最新状态。

冻结文件：[全量清单](../../../output/datasets/pipe-ends-maco-v1/manifest.json)（SHA256 `960005aa6c7956db8ebf5962693071e0ea01888f0d0380431437894b4fddec41`）、[不含测试记录的 TrainVal 清单](../../../output/datasets/pipe-ends-maco-v1/train-val-manifest.json)（SHA256 `505d0a3de3424b75d53632e5349453d9e9f82faf2d831778f07f6be5a4072476`）、[独立冻结复核](freeze-independent-review.json)。

本轮主来源是作者公开的 [MACO / Zenodo 17165968](https://zenodo.org/records/17165968)。圆管包与方管包合计 **1,786,361,902 字节**，均已核对完整 MD5、SHA256、成员 CRC 和解包路径。圆管包实际含 **1,025 张图及 HBB XML**；方管 HBB/Mask/OBB 中重复保存照片，不能把不同标注格式当成独立数据。方管本轮只存档，不训练。

三方逐页场景与目标筛选后，再查看所选图片的原图/标注对照并排除明显非钢材、手圈、数字计数笔迹、遮盖目标、来源编辑不明或范围不清的图片。最终 proposal 含：

- **223 个圆管正例、35,218 个框**。复用作者 HBB，完成全量几何检查和场景/粗覆盖视觉审核；没有声称重新独立数过每一个原框。
- **11 张真实 non-pipe 整图**：实心钢筋与开放 I 型材；原材质标注仅作佐证，零管口标签由本轮视觉审查确认。带橙色笔迹的 Rebar1 已隔离。
- **7 个真实背景裁剪**：来自已批准正例的无 GT 交集地面/土/屋檐区域，经视觉确认没有可辨管口；像素不改，与父照片及其别名严格同组。它们不是 7 个额外独立拍摄场景。

目标是一根可辨圆管的一个前端，框住可见外缘及内孔；不把内孔和外环标两次。管壁/管身可辨时，灰土堵住孔口仍可视为管端；不承诺识别仅有封盖、材质不明、复杂弯头或多开口配件。矩形管、实心钢筋、钢坯与焊缝缺陷没有混成圆管正例。

共检查到 7 个原 XML 中存在重复坐标框，逐个放大查看后确认重复罩住同一前端，另存修正版并记录移除的 21 条索引/坐标；**原 XML 未改动**。这些修正不自动使相应照片合格；例如同源编辑方向不明的图片仍隔离。最终子集实际采用的修正数量见 [QA](proposal-qa.json)。

## 分组与当前 proposal

无作者场景 ID 时，先用精确字节/RGB 哈希及非学习型 SIFT/RANSAC 查找关联，再合并人工确认和保守疑似家族。算法未匹配不等于已证明独立；场景组是保守防泄漏单元，不是可认证的物理堆料身份。

| proposal | 正例图 | 零管口图 | 正例组 | 全部组 |
| --- | ---: | ---: | ---: | ---: |
| train | 156 | 8 | 148 | 150 |
| val | 33 | 7 | 33 | 35 |
| test | 34 | 3 | 33 | 34 |

种子为 `CountScope-pipe-ends-real-20261026-v1`。正例按组的目标数量中位数分三档后哈希划分；负例按保守场景/材质族固定分配，背景裁剪继承父组。验证和测试都至少有 10 个正例组，并有 non-pipe 组。组、字节 SHA 和 RGB 像素 SHA 跨 split 交集为零。最大单图 **1,153 个目标**，模型/后处理容量不得默默限制为 1,000 后再声称无漏检。

- [全量 proposal](../../../output/pipes-20261026/data/proposal/all-splits-manifest.json)：含测试记录，仅供主代理冻结/评估。
- [TrainVal proposal](../../../output/pipes-20261026/data/proposal/train-val-manifest.json)：不含测试图片或逐图 GT；当前状态仍为 `PROPOSED_QA_PASSED_PENDING_ROOT_FREEZE`。
- [统计与哈希](split-proposal-summary.json) · [241 个视图的文件/像素/标签 QA](proposal-qa.json) · [重复框修正](annotation-corrections.json)。

“真实”是出处与视觉一致性筛选，不是司法图像真实性或金属成分鉴定。17/716 的对照可证明同一拍摄家族存在背景编辑差异，但不能仅凭两图确定哪个是原片、是添加还是删除目标；首次较强的合成猜测与后续更正均保留，最终按 `unknown_edited_variant` 隔离，不指称已证明作者生成了虚假物体。

已核实的补充来源：[ASCE 2022 作者仓库](https://github.com/lyang95/PipeCounting)、[IvanFenster/PipesCounting](https://github.com/IvanFenster/PipesCounting)、[Kawai-Senpai/PipeCounter](https://github.com/Kawai-Senpai/PipeCounter)。前两者有外部数据链接但未作为独立新测试源加入；Kawai 的示例 YAML 将 train/val 指向同一目录，不能引用其自报 mAP 作为本项目成绩。本轮没有注册、付款、发邮件、下载第三方 `.pt` 或运行计数模型。所有来源元数据、许可字段、原档和审查证据保留。

主代理冻结与授权后训练器才可开始。当前没有新模型准确率、CoreML/iPhone 通过结果或上架结果。
