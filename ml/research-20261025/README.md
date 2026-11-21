# CountScope 计数研究与实验记录 · 2026-10-25

**当前结论：新增临床数据上的计数改善明显，但旧数据退步，尚无可替换 App 模型的新方案。** 本轮已完成论文核实、尺度融合筛选、76 张跨研究数据评估、LOCA 无示例框筛选，以及菌株隔离的混合训练。8 轮训练后的 18 个配置全部未通过保护条件；新 36 张测试图保留，未用于模型评估。这里的 MAE 是每张图片平均数错多少个，不是“准确率”；检测召回率、计数误差、真实手机表现需要分别报告。

## 论文支持什么，哪些可以落地

以下链接来自出版商、会议论文集或作者官方代码。期刊、会议和预印本分开列出，论文原始成绩不作为 CountScope 的实测成绩。

### 高水平期刊

| 论文与正式发表信息 | 对当前问题的启发与落地边界 |
| --- | --- |
| [Cellpose 2.0](https://www.nature.com/articles/s41592-022-01663-4)，**Nature Methods，2022**；[官方代码](https://github.com/MouseLand/cellpose/tree/main/paper/2.0) | 形态、采集域和标注风格的覆盖很重要。当前先补充独立菌株与来源；若试分割，需要人工修正的像素掩膜，不能把现有检测框直接填成真值掩膜。本文的少量标注结果不保证适用于菌落。 |
| [Cellpose3](https://www.nature.com/articles/s41592-025-02595-5)，**Nature Methods，2025**；[官方代码](https://github.com/MouseLand/cellpose/tree/main/paper/3.0) | 面向分割优化的去噪、去模糊、欠采样恢复，可作为低对比度图像的小规模验证实验。论文主要是显微细胞；图像变清楚不等于菌落数正确，必须检查新增假目标和错误拆分。 |
| [Omnipose](https://www.nature.com/articles/s41592-022-01639-4)，**Nature Methods，2022**；[官方代码](https://github.com/kevinjohncutler/omnipose) | 距离场方法处理复杂、细长或分叉的微观细菌形态。它识别的是单个细胞，不能直接当作培养皿菌落计数器；需要新的代表性掩膜监督。 |
| [CellSAM](https://www.nature.com/articles/s41592-025-02879-w)，**Nature Methods，2025**；[官方代码](https://github.com/vanvalenlab/cellSAM) | 自动检测框再提示 SAM，适合研究自动定位与训练标注辅助。不能把测试集真值框输入后声称“自动识别”。本机完整复现训练成本不合适，MPS/CoreML 适配未验证。 |
| [SQLNet](https://doi.org/10.1109/TIP.2025.3588255)，**IEEE Transactions on Image Processing，2025**；[作者论文](https://arxiv.org/html/2311.10011v2)、[官方代码](https://github.com/HCPLab-SYSU/SQLNet) | 尺度调制、位置与尺寸监督值得借鉴，但发布模型仍依赖示例框；不是无需手动选择目标的直接替代品。 |
| [Focus for Free in Density-Based Counting](https://link.springer.com/article/10.1007/s11263-024-01990-3)，**IJCV，2024**；[官方代码链接](https://github.com/shizenglin/Counting-with-Focus-for-Free) | 从点标注生成保持计数一致的遮挡增强和前景监督，可研究背景误数。主要证据来自人群计数；出版商链接的仓库仍是旧 ICCV 2019 / TensorFlow 1 实现，不能声称已经复现 2024 论文。 |

Cellpose3 官方文档提供 MPS 路径，但 Mac 上部分掩膜后处理走 CPU；这不是 iPhone 可部署证明。若尝试，应使用隔离环境和小样本验证，不改动现有训练环境。

### 会议与补充证据

| 论文 | 可直接借鉴的部分与限制 |
| --- | --- |
| [SAHI，ICIP 2022](https://doi.org/10.1109/ICIP46576.2022.9897990)；[官方实现](https://github.com/obss/sahi/blob/main/sahi/predict.py) | **会议论文。** 官方切片推理保留整图预测以帮助大目标。支持比较整图、切片和融合；本项目的硬尺寸阈值融合是工程变体，不是原论文完整复现。 |
| [LOCA，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Dukic_A_Low-Shot_Object_Counting_Network_With_Iterative_Prototype_Adaptation_ICCV_2023_paper.html)；[官方代码](https://github.com/djukicn/loca) | **会议论文。** 有独立的零样本权重和无示例框路径，已完成本机筛选。无需框不代表能理解混合场景中用户想数哪一类。 |
| [DAVE，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Pelhan_DAVE_-_A_Detect-and-Verify_Paradigm_for_Low-Shot_Counting_CVPR_2024_paper.html)；[官方代码](https://github.com/jerpelhan/DAVE) | **会议论文。** 先产生高召回候选再验证外观，可借鉴用于抑制误检。零样本多数类选择可能选错目标；聚类、ROIAlign 等增加手机部署工作。 |
| [CountGD，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/57c56985d9afe89bf78a8264c91071aa-Paper-Conference.pdf)；[官方代码](https://github.com/niki-amini-naieni/CountGD) | **会议论文，非 ECCV 2024。** 文本提示可免画框，并明确要数的类别。约 1.2 GB 权重和自定义 CUDA 算子需要先移植；尚未验证 Mac/iPhone。其输出主要监督中心位置，不能假定可靠框边界。 |
| [GeCo2，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37780)；[官方代码](https://github.com/jerpelhan/GECO2) | **会议论文，非 ICCV 2025。** 多尺度查询聚合适合研究大小目标兼顾，但仍依赖示例框；SAM2 和可变形注意力不是当前端侧模型的直接替换。 |

[菌落 SAHI 应用论文](https://www.nature.com/articles/s41598-026-55724-1)发表于 **Scientific Reports（2026）**；[新临床数据论文](https://www.nature.com/articles/s41597-026-07095-5)发表于 **Scientific Data（2026）**。两者是补充的应用/数据证据，不能因为属于 Nature 品牌就称作 Nature 或 Nature Methods 顶刊。前者的 24 类数据划分与本项目不同，其 mAP 不能套用成本项目正确率。

另有 [Cellpose-SAM](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1.full)：本次核实到的官方引用仍为 **bioRxiv 预印本**，不要与上述已发表于 Nature Methods 的 **CellSAM** 混淆。详细出处与运行限制见 [菌落文献记录](colony-papers.json)、[通用计数文献记录](general-papers.json)。

## 已完成实验：保留失败证据

### 1. 旧菌落数据暴露出形态不均衡

旧 63 图中，sp21 占全部标注目标的 **94.96%**。总体召回率约 **91.87%**，但物种平均召回率仅 **50.59%**；sp08 / sp17 / sp21 分别为 **8.97% / 47.32% / 95.49%**。因此高总体指标掩盖了大而模糊等形态的漏检。固定四切片改善小目标，但旧 63 图计数 MAE 从整图 **11.63** 变为 **13.03**；FP 与 FN 的抵消关系使“少一些检测错误”并不必然等于“数得更准”。这些旧测试结果已知，只能继续作为回归证据。[审计记录](experiment-audit.json)

### 2. 融合在旧 56 图验证集上仍未通过保护条件

| 方法 | 计数 MAE ↓ | 召回率 ↑ | 物种平均 MAE ↓ |
| --- | ---: | ---: | ---: |
| 整图 | 25.68 | 76.08% | 24.52 |
| 自动四切片 | 15.27 | 84.99% | 18.06 |
| 平均 MAE 最低的融合候选 `side16-nms0.5` | 11.54 | 87.64% | 12.82 |

平均值改善仍不合格：例如 sp04 的 MAE 相对整图从 **5.67 → 8.83**，sp13 从 **16.83 → 25.67**。硬尺寸分流还会在两个分支的尺寸估计不一致时漏掉原本能检测到的目标。6 个尺度融合配置与追加预冻结的 1 个来源感知去重配置均未通过；没有放宽门槛，也没有选出可推广配置。[融合结果](fusion-results.json) · [边界审计](fusion-boundary-audit.json) · [最终判定](fusion-final-selection.json)

对三个按固定规则选出的旧验证失败样本进行了缓存叠图和目视检查：`sp04_img04` 的大而贴连目标，整图 TP29/FN21，切片变成 TP10/FN40；`sp09_img12` 的 446 个微小目标，切片 TP312 高于整图 26，但仍漏掉 134 个；`sp13_img06` 的切片 FP166 中，有 110 个未匹配预测的中心位于已有真值框内部，图中也可见重复/碎片候选与皿缘高光候选。这里的 FP/FN 是相对冻结框的匹配结果，不能据此断言原标注错误或证明切片边界是唯一原因。[定性诊断](qualitative/report.json) · [大目标对照图](../../output/research-20261025/qualitative/sp04_img04-overview.png) · [密集目标局部图](../../output/research-20261025/qualitative/sp13_img06-detail.png)

### 3. 76 图跨研究外部评估失败

从独立研究的 950 张原图中，在推理前按固定哈希选取 19 个物种各 4 张。完整性与标注对齐检查通过后，对预冻结的整图、四切片和上述内部已失败的探索融合做同图对照：

| 方法 | 计数 MAE ↓ | 精确率 ↑ | 召回率 ↑ | AP50 ↑ |
| --- | ---: | ---: | ---: | ---: |
| 整图 | 24.71 | 47.40% | 27.66% | 20.56% |
| 自动四切片 | 22.55 | 55.76% | 36.89% | 32.77% |
| 探索融合 | 25.29 | 52.06% | 31.59% | 24.14% |

融合的外部相对保护条件也失败。四切片虽在这组三者中较好，仍有明显漏检，不能称为可靠泛化。这 76 图现在已暴露，后续不再算新盲测；它们来自专业受控拍摄，也不能代表手机使用验收。[外部结果](external-comparison.json) · [独立数据 QA](external-data-review.json)

### 4. LOCA 无示例框筛选：有进展，但空白控制失败

使用作者零样本权重，严格加载完整参数、保持原模型结构；实际输入仅图片，没有示例框或文本。预冻结的 29 类各 2 张旧 clean-val 子集，共 58 图：LOCA 的 MAE / RMSE 为 **22.50 / 68.62**，同图缓存 FamNet 参照为 **35.04 / 137.17**。FamNet 获得示例框，二者输入条件不同；这不是新盲测或完整基准优胜结论。

此前 FamNet 短程微调在完整 clean-val 上的 MAE 从 **24.27 → 25.58**，低数量图片改善，但中高数量图片和低估偏差恶化；这个完整集合的成绩不能与上述 58 图子集直接横比。继续训练也不会自动消除 FamNet 对示例框的输入依赖。[历史实验审计](experiment-audit.json)

关键失败保留：纯白图误数 **90.41**、纯灰图误数 **70.44**；101–500 数量桶只有 2 图，LOCA MAE **56.07**，劣于 FamNet **24.92**。未添加空白图特判。MPS 单图计数调用平均约 **87.5 ms**，包含前向、张量传输及计数求和，不含文件读取、预处理和加载，也不是 iPhone/CoreML 延迟。当前只保留研究候选身份。[完整筛选结果](../zero_shot_loca/reports/20261025/screen-result.json) · [复现入口](../zero_shot_loca/README.md)

## 已准备的新数据：按菌株隔离，而不是只增加切片

[Zenodo 原始数据](https://zenodo.org/records/18256864)包含 19 个物种、151 个临床菌株、950 张原图。新划分仅使用匿名图像/物种/菌株代码，不导入患者字段，也不下载 augmentation 档案。先冻结哈希选择，再下载；不根据目标数量、模型分数或图像难度筛选、替换样本。

| 新临床子集 | 菌株数 | 原图数 | 角色 |
| --- | ---: | ---: | --- |
| train | 113 | 113 | 每菌株取 1 图，补充来源与形态覆盖 |
| val | 19 | 36 | 每物种一个未暴露菌株，最多取 2 图 |
| test | 19 | 36 | 另一个未暴露菌株，留待冻结方案后的独立评估 |

三个集合的菌株和原图 SHA 交集均为零；新 val/test 也排除了旧 76 图暴露的全部 57 个菌株，并通过原图 SHA 去重。185 图全部通过大小、CRC、SHA、解码、EXIF、坐标与 YOLO 标签回转检查；每物种抽查一份 XML，共 19 份，与 COCO 坐标完美匹配，允许源格式最多 1 px 的取整差。保留原 COCO 数值精度，只转成浮点 xyxy，不制造更细的标注。

旧 76 图中有 16 张原图被复用于新 train；因此混合训练后更不能把旧 76 图当独立测试。旧 63 图也保持为已知回归集。新 36 图尚不参与训练、选阈值或选模型；训练代理只收到不含 test 内容的 train/val 清单。菌株隔离仍不等同于拍摄 session 或培养皿完全独立，SHA 检查也不能排除重编码的近重复。

[数据准备报告](../../output/datasets/colony-mixed-isolate-v1/dataset-prep-report.json) · [只含 train/val 的清单](../../output/datasets/colony-mixed-isolate-v1/train-val-manifest.json) · [独立复核](../../output/datasets/colony-mixed-isolate-v1/independent-prep-check.json) · [准备脚本](prepare_isolate_mixture.py)

## 混合训练完成：临床来源改善，但旧来源退步

旧 247 train + 新 113 train，共 **360 train**；旧 56 val + 新 36 val，共 **92 val**。从现有 YOLO11n 继续训练 **8 epoch**，学习率 0.0001、输入 1024、batch 4，不使用旋转或 mosaic。实际 DataLoader 按来源、来源内物种、组内图片均衡抽样；8 轮实际消费 2,880 张次，旧来源 1,455、新来源 1,425。均衡图片曝光不等于均衡目标级损失；框曝光记录是抽中原图的 GT 框数累计，并非增强后进入损失的框数。

只比较预注册的第 3/5/8 轮 × 整图/自动四切片 × 置信度 0.2/0.3/0.4，共 18 配置。主指标先在每个来源内对物种平均，再对来源等权，避免新来源的 19 个物种压过旧来源的 6 个验证物种。基线也在相同验证集、相同有限参数中冻结为旧权重自动四切片/0.3。

下表是**开发集宏 MAE 最低但未通过验收**的第 8 轮、自动四切片/0.3；不是选出的发布模型，也不是独立测试成绩。

| 同图验证指标 | 冻结基线 | 最低宏 MAE 的研究候选 |
| --- | ---: | ---: |
| 来源等权的物种平均 MAE ↓ | 15.306 | 13.306（下降 13.1%） |
| 全部 92 图的平均计数 MAE ↓ | 13.989 | 13.511 |
| 旧来源 56 图 MAE ↓ | 15.268 | **21.000（退步）** |
| 新临床来源 36 图 MAE ↓ | 12.000 | **1.861** |
| 旧来源精确率 / 召回率 ↑ | 82.38% / 84.99% | **79.12% / 84.10%** |
| 新临床来源精确率 / 召回率 ↑ | 54.14% / 43.73% | **92.06% / 91.65%** |
| 全部 TP / FP / FN | 6,675 / 1,978 / 2,143 | 7,566 / 1,673 / 1,252 |
| 全部 AP50 / AP50:95 ↑ | 73.36% / 33.21% | 84.82% / 43.37% |

新增来源改善是真实的开发集结果；代价也很明确：旧来源误检从 **1,242 增至 1,516**，旧来源物种平均 MAE 从 **18.060 增至 24.716**。最佳平均值候选仍有 **19 项**保护条件失败，涉及旧来源误差、精确率、多个菌种与尺度召回，以及新来源的少数菌种。所有 18 配置的白/灰/黑正常推理计数均为 0，但全部未通过其他条件，最终 `selectedConfig=null`。没有调整保护门槛，也没有用新测试集挑选候选。

训练结束时框架额外触发一次回调，报告代码误按第 9 轮检查，进程因此退出 1。8 轮参数更新、8 份真实消费日志和 3/5/8 检查点已完整保存；保留原失败日志、训练启动时源代码和修复后的代码，重放 2,880 个采样 ID 并核对 checkpoint 内轮数与哈希后恢复完成报告，**没有重训**。MPS 训练诊断验证还出现 NMS 限时警告，因此这些日志指标不用于选择；最终 CPU 预测缓存无此警告。最终结果审查与训练恢复证据在独立复核报告中单独列明。

[冻结协议](../../output/research-20261025/mixed-training/protocol.json) · [完整实验结果](../../output/research-20261025/mixed-training/experiment-result.json) · [18 配置与全部保护条件](../../output/research-20261025/mixed-training/selection.json) · [独立复核](mixed-independent-review.json) · [测试集保留记录](holdout-disposition.json)

**新 36 图未用于模型评估；CoreML/iPhone 验收未完成；App/ASC 本轮未修改。** [发布源码核验](production-preservation.json)确认 17 个签名发布绑定文件与原哈希一致。

下一轮应针对已经观察到的错误：补充大而模糊、相互接触的菌落与真实空皿，建立经人工修正的小型实例掩膜集，比较自动定位加分割和现有检测器；同时把旧来源退步作为硬性回归条件。Cellpose/CellSAM 适合作为标注和结构研究起点，仍需验证培养皿尺度与手机拍摄适用性。自动区域定位继续作为约束，不能把测试真值框或用户手动画框作为输入后声称全自动。通用计数方向先解决 LOCA 的空白误数与目标类别选择，再考虑端侧蒸馏。
