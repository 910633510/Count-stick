# CountScope public-data training — 2026-10-25

The [2026-10-26 GitHub source review](github-research-20261026/README.md) records current stars/commits, concrete training defaults, bounded CPU/MPS checks, and prioritized next experiments. It does not introduce new trained-model accuracy claims.

The latest [large/touching-colony and real-empty-plate retention round](research-20261025-hardcases/README.md) adds visually reviewed instance labels, physical-group separation, and strict regression gates against the frozen model.

The subsequent [paper review and optimization experiments](research-20261025/README.md) cover Nature Methods, IEEE TIP and IJCV methods, automatic counting, cross-study evaluation, and the isolate-separated mixed-data experiment. That report keeps failures and deployment limits separate from aggregate improvements.

Both no-login datasets were downloaded and checked. Two separate experiments were completed; neither changes the submitted app or its skewer model.

| Experiment | Data / training | Verified result |
| --- | --- | --- |
| Colony detector | Figshare369 images/56,865 source boxes;247train/56validation/63test after audit;20epochs of YOLO11n fine-tuning | Selected automatic4tile pipeline: precision92.8%, recall91.9%, AP50 94.3%, AP50–95 60.3%; mean absolute count error13.03; exactly correct5/63images |
| General few-shot counter | Complete FSC147 archive;3,658training queries;one full FamNet fine-tuning epoch | Fine-tuned validation MAE25.58 is worse than original baseline24.27, so it was not promoted. Selected original baseline test MAE22.54 over1,189clean test queries |

## Important limits

The colony detector's aggregate scores are dominated by one test species, accounting for about95% of objects. Species-macro recall is only50.6%; large diffuse colonies are often missed. Tiling improved validation counting and test detection metrics, but test count MAE13.03 is worse than the preserved whole-image reference11.63. The primary pipeline was not switched after observing test results.

The general counter needs dataset-provided or user-provided example rectangles. It does not automatically determine what object category to count. Neither experiment is ready to be treated as a reliable general counting feature or physical-device acceptance.

## Artifacts

- [Colony methods and frozen evidence](colony/README.md); [colony result JSON](colony/reports/20261025/summary.json).
- [FSC147 methods and evidence](fsc147/README.md); [FSC147 result JSON](fsc147/reports/20261025/summary.json).
- [Colony PyTorch checkpoint](../output/training-20261025/runs/colony-yolo11n-v2-cpuassign/weights/best.pt).
- [Colony FP32 CoreML experimental package](../output/models/CountScope-Colony-v1-fp32/CountScopeColony1024.mlpackage).
- [FSC147 offline model-bundle recipe](../output/models/CountScope-FSC147-FamNet-v1/recipe.json), including the original baseline and the rejected fine-tuned candidate.
- Original datasets: `output/datasets/colony-figshare-22022540-v3/` and `output/datasets/fsc147/`.

Colony FP32 CoreML matched the selected tiled pipeline's counts on three validation photos. Whole-image counts still differed by up to6; this is a Mac smoke test, not complete numerical or iPhone parity. FP16 and all failed/interrupted attempts are preserved.

The app's25-file Build4 production-source fingerprint was checked unchanged. No app model replacement, store metadata change, upload or review resubmission occurred during these training experiments.

Original colony data and pictured evaluation examples are from Norbert Solymosi and Sára Nagy, [Figshare22022540v3](https://doi.org/10.6084/m9.figshare.22022540.v3), CC BY4.0. Diagnostic images overlay model predictions and resize the original photos; original downloads remain unchanged. FSC147 and FamNet are from [the authors' CVPR2021 release](https://github.com/cvlab-stonybrook/LearningToCountEverything).
