# 大菌落与真实空皿补充数据来源

本轮冻结 **51 个训练/验证图像文件：11 个正例、59 个可见实例框，以及 40 个真实空皿负例**。正例是 **9 个显式可数裁剪区域和 2 张整图**，来自高光谱图像转换的 RGB，**不是 11 张手机照片**。本文件说明数据与标注，不报告尚未完成的训练效果。

| 一手来源 | 本轮取得的照片 | 针对本轮逐图复核 | 最终纳入 |
| --- | ---: | ---: | ---: |
| [Zenodo 20178469：Semantic Segmentation for Automated Identification of Bacterial Colonies](https://zenodo.org/records/20178469) | 111 | 15 个正例候选原图 | 11 个样本编号，导出 9 crop + 2 整图 |
| [Mendeley xrbtd74pfj v3](https://data.mendeley.com/datasets/xrbtd74pfj/3)，[作者研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC10024077/) | 48 个预先按哈希选定的 Negative 样本 | 48 | 28 个零计数负例 |
| [BRUKERAGAR 作者仓库](https://github.com/michalcicatka/BRUKERAGAR-DATASET)，[作者论文](https://doi.org/10.1016/j.heliyon.2024.e25714) | 12 个公开样本，来自两皿的不同光照/视角 | 12 | 9 个零计数负例 |
| [Zenodo 18703636：curated micro-colonies](https://zenodo.org/records/18703636)，原入口 [18505210](https://zenodo.org/records/18505210) | COCO 原档中的 165 张导出照片 | 其中 4 张作者零框照片 | 3 个零计数负例 |

共取得 336 张照片，但并未把整包照片自动当成新训练标签；其中 79 个候选做了针对本轮的逐图复核。正例候选中 4 个没有获得可接受的计数区域；64 个负例候选中 24 个隔离。下载与复制均无需登录、付款或联系作者；原档、元数据、来源掩膜和校验记录保留在 `output/research-20261025-hardcases/`。

## 标注做法与限制

正例使用作者的**语义掩膜**辅助提出区域，再由 Codex 逐图判断可见实例、拆分有清晰主体/边界的相接目标、修正框或限定可数 ROI，并由另一代理复核。它们不是直接照搬的作者实例真值，也不是从零绘制的精确实例掩膜。部分经复核保留的框与作者掩膜外接框相同，记录中已明确区分。

例如，Sepi_2 的两个连通区域各被拆为 3 个可辨实例；Kpne_4 的大圆与其较小相接邻体分别标框。相反，Ecoli_5 的大波状区域无法排除多菌落融合，独立复核拒绝了初稿中的“3 个实例”，该草稿保留审计但不进入训练。这是不确定的实例解释，不是证明作者语义标签错误。其他无法完整计数的原图也不作为零样本或未标注背景；只有明确记录边界、逐个标注的 ROI 可以使用。

正例裁剪逐像素等于原始 RGB 对应区域，没有把复核时的显示增亮写入训练图片。少量边缘截断保留可辨主体，坐标按真实 ROI 交集裁切，独立复核已重算。Mendeley 原片 EXIF 为方向 6，另存了全分辨率转正副本；JPEG quality 98/subsampling 0 的编码过程也已逐张复现核对，原始字节保留。

负例的 0 表示**复核照片中没有可辨菌落**，不证明生物学无菌。强过曝、可疑斑点、无法区别沉淀/反光与生长的图片均隔离，不能只凭作者的 Negative 或零框类别赋值。纯白、灰、黑控制图不计入这 40 个真实样本。这些标注没有微生物专家共识认证。

## 冻结与防止重复泄漏

[冻结清单](../../output/datasets/colony-hardcases-retention-v1/manifest.json)使用种子 `CountScope-hardcases-retention-20261025-v1`。正例按 `SHA256(seed|species|groupId)` 排序：有多个样本组的物种取首组进入验证，单组物种留在训练。同一样本编号的所有年龄、日期、原图、裁剪与其他视图必须绑定。

| 划分 | 正例 | 真实空皿 | 保守分组数 |
| --- | ---: | ---: | ---: |
| train | 7 | 34 | 9 |
| val | 4 | 6 | 6 |

这 51 个文件仅构成 **15 个保守防泄漏组**，不能声称 51 个独立培养皿。11 个正例按源 CSV 样本编号分组；Mendeley 缺少公开皿 ID，整个 Negative 来源合并为一个训练组；BRUKER 的 AEBM/BESBL 各皿所有光照视图分别绑定；三张 Zenodo 空皿属于同一短时间采集会话。负例固定为 Mendeley+AEBM 训练、BESBL+Zenodo 会话验证。

[独立完整性审查](split-independent-review.json)确认：候选内容除 split 外未变化；输入候选与策略哈希吻合；全局 train/val 组及文件 SHA 交集为 0；51 个来源与派生文件的 SHA 与旧 369 图、临床 train/val 149 图无重叠；裁剪像素、坐标、合法边界截断与空标签均正确。主代理另以封存数据的指纹元数据排除重叠，本复核没有打开封存 36 图的像素、框或清单。精确 SHA 检查不等于排除了所有重编码近重复，受控手机/机器人/高光谱采集也不能替代自由手持手机验收。

标注证据：[正例 QA](positives/qa-receipt.json) · [正例二审与拒绝记录](positives/second-review-decisions.json) · [负例来源与逐图记录](negatives/README.md)。训练和旧场景保持结果由主代理另行追加，不从数据准备本身推断模型已经改善。
