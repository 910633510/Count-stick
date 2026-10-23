# 网络实拍端头标注与评测 v1

已完成第一批 **5张独立来源照片、499个明确可见端头的逐点标注**。4张产品实拍、1张真实餐后手持签束；另有5处阴影/遮挡区域明确忽略。数值指图中可见端头，不推断被完全遮挡的签子，也不引用商品标称包装数量。

标签由助手视觉逐点标注并交叉复核；密集黑底图先用非学习型像素峰值作为未审核点击候选，再逐一纠错、补点，独立复核补入8个漏标。所有标签、匹配容差、原图哈希、来源组和2张校准/3张留出划分均在首次模型运行前锁定。完整清单是`manifest.json`，权威标签在`annotations/`。原图与标注预览只保存在本地`output/web-annotation/`。

## 当前结果

| 场景 | 分组 | 标注可见端头 | 原版预测 | build 2预测 |
| --- | --- | ---: | ---: | ---: |
| 密集平切端头 | calibration | 212 | 297 | 251 |
| 白底签束（部分不确定） | calibration | 92（另有5处不确定区域） | 96 | 95 |
| 粗签端头 | holdout | 97 | 96 | 96 |
| 大端头近景 | holdout | 28 | 9 | 8 |
| 真实餐后手持签束 | holdout | 70 | 80 | 77 |

原版为confidence0.25/IoU0.7；build2为confidence0.25/IoU0.45，权重和scaleFill不变。04有不确定区域，不能用92直接计算整图误差。

3张留出图的build2结果为 **96/8/77**，标注为 **97/28/70**。平均绝对计数误差9.33，完整数量吻合0/3；原版平均误差10。这个小样本是诊断集，不代表所有餐厅或手机上的总体准确率。模型原训练集未知，无法排除与网上图片的既有训练重叠。

独立匹配复算通过。距离容差在0.5–1.5倍变化会影响端点定位指标，但不会改变上述整图计数偏差。原始预测、逐点一对一匹配、重复误报、漏检和哈希绑定在`predictions-v1.json`、`benchmark-v1.json`和`evaluation-review.json`。

## 预声明的单次阈值试验

在两张校准图上一次比较confidence0.25/0.35/0.45/0.55，IoU固定0.45；按逐图F1宏平均选择0.35，先保存选择记录，再对3张留出图仅测试一次。其预测为96/4/73，平均计数误差仍9.33，完整数量吻合仍0/3，留出集宏F1还下降。因此没有修改生产阈值，也未继续根据留出结果调参。详见`confidence-experiment-v2.json`。

## 复现

需要macOS、Xcode命令行工具和原项目Core ML模型；网络原图当前保存在本机。命令从项目根执行：

```sh
python3 release/accuracy/web-set/evaluate-tests.py
python3 release/accuracy/web-set/evaluate.py release/accuracy/web-set/manifest.json --validate-only
python3 release/accuracy/web-set/evaluate.py release/accuracy/web-set/manifest.json --output-dir output/web-annotation/evaluation-reproduced
```

模型只在冻结标签和哈希校验通过后运行。图片/标注改变须建立新数据版本，不可覆盖v1后重写成绩。来源地址逐张列在清单中；重复图片已排除为独立样本。

## 发布状态

这批标注已完成，但模型仍未通过结账场景的质量验收。本轮未更换模型、未更改生产代码、未提交App Store。后续模型改进须保留独立的新场景验证，不能只优化已看过的这5张图。历史build1商店截图仍不可直接上传当前候选版。
