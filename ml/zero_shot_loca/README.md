# LOCA zero-shot feasibility screen

The official [LOCA (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Dukic_A_Low-Shot_Object_Counting_Network_With_Iterative_Prototype_Adaptation_ICCV_2023_paper.html) zero-shot model runs on this Mac with CPU and MPS. This is a research candidate, not an accepted App model. Solid white and gray controls incorrectly produced approximately 90 and 70 objects. No exception or threshold was added to conceal these failures.

## Run image-only inference

From the repository root, using the existing environment and downloaded artifacts:

```sh
output/training-venv/bin/python ml/zero_shot_loca/predict.py \
  --image /absolute/path/to/image.jpg \
  --device cpu \
  --output output/loca-20261025/my-image
```

`--device mps` is also supported on this Mac. The command writes `prediction.json` and `density.npy`. It accepts neither exemplar boxes nor category text. It does not promise to select a particular object class in a mixed scene.

The CLI smoke test reused the first frozen pilot image, `7649.jpg`, and reproduced the CPU pilot count of `22.214698791503906`. Evidence is in `output/loca-20261025/cli-smoke/`. No additional evaluation images were selected for the smoke test.

Artifacts required by the CLI:

- Official source commit `45433064e36e48e491b430f6126f439d77a9aa3c` under `output/loca-20261025/source-archive/`, located through `source-receipt.json`.
- Official zero-shot checkpoint `output/loca-20261025/checkpoints/loca_zero_shot.pt`, 142,474,397 bytes, SHA256 `fe373e981328ce7057b9638802ded24633f8191a7b905e458f041d1ec6e0b87f`.
- Existing `output/training-venv` with PyTorch, torchvision, NumPy and Pillow. No installed packages were upgraded for this experiment.

Source and checkpoint came from the [author repository](https://github.com/djukicn/loca) and its linked public [zero-shot checkpoint](https://drive.google.com/file/d/11-gkybBmBhQF2KZyo-c2-4IGUmor_JMu/view). Both were downloaded without login or payment. Receipts and exact revisions are preserved in `reports/20261025/`.

The source architecture is unchanged. The adapter bypasses the author's CUDA/DDP runner and skips redundant initialization downloads before loading every official checkpoint tensor strictly. It verifies checkpoint hash and equality of all loaded state values. Inference calls `model(x, None)`; a guard throws if exemplar ROIAlign is reached. Three learned prototype slots are model parameters, not three supplied examples. RGB preprocessing uses the author validation resize and ImageNet normalization, with explicit `antialias=False` to reproduce the older torchvision tensor-resize default.

## Frozen screen and results

`screen.py` created a 58-image manifest before inference: two images from each of 29 categories, chosen solely by the SHA256 ordering of a fixed seed and filename. The source was the project's existing clean validation split. This is an already-used validation screen, not a new blind test, full benchmark, or production accuracy estimate. FamNet predictions were read from its existing clean-validation report; FamNet received visual exemplars, whereas LOCA received only the image. The comparison therefore has different input conditions.

| Metric on the same 58 images | LOCA, zero examples | FamNet, supplied examples |
|---|---:|---:|
| Mean absolute count error | 22.50 | 35.04 |
| Root mean square count error | 68.62 | 137.17 |
| Mean signed error | -18.63 | -20.09 |
| Rounded count exactly correct | 1/58 | 1/58 |

LOCA's MAE was 35.8% lower on this screen and its absolute error was lower on 35/58 images. This is not an accuracy percentage. Dense images remain problematic:

| True count | Images | LOCA MAE | FamNet MAE |
|---|---:|---:|---:|
| 1–20 | 30 | 5.57 | 7.21 |
| 21–100 | 24 | 12.93 | 17.05 |
| 101–500 | 2 | 56.07 | 24.92 |
| More than 500 | 2 | 357.83 | 678.40 |

Only two samples occupy each dense bucket, so these are warning signals, not stable estimates. Synthetic blank white and blank gray controls had true count zero and predictions 90.41 and 70.44. LOCA is unsuitable for direct App replacement based on this evidence.

CPU/MPS predictions on the two pilot images differed by less than 0.00008 objects. On MPS, mean forward/transfer/count-sum time was 0.0875 seconds per image; image decoding, resizing and model initialization are excluded. The 58-image and two-negative-control loop took 5.36 seconds. Process peak RSS was 2.15 GB, including loading and CPU pilot; observed MPS driver allocation reached 1.81 GB. These desktop measurements do not establish iPhone latency, peak memory or CoreML compatibility.

## Exclusions and integrity

The original official validation set contains 1,286 images; the existing clean validation set contains 1,281. Its excluded members are `3477.jpg` (density-label mismatch) and `1896.jpg`, `4549.jpg`, `4707.jpg`, `4719.jpg` (exact cross-split image duplicates). None appears in the 58-image selection.

The [CountGD NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/57c56985d9afe89bf78a8264c91071aa-Paper-Conference.pdf) notes bad visual exemplars for `7171.jpg` and a target description correction from “lego” to “yellow lego stud” for `7611.jpg`. Both belong to the official test split and were absent from both validation and this screen. No original annotations or old reports were rewritten, and no old test images were inferred.

`reports/20261025/screen-result.json` contains per-image predictions, density buckets, runtime, memory, paired comparison and the explicit exclusion audit. `protocol.json` preserves the frozen selection and input conditions. The original output evidence remains in `output/loca-20261025/`; the tracked-size report copies make the conclusions reviewable without committing weights or images. `screen.py` refuses to overwrite a completed result.

No training, parameter tuning, App integration, CoreML export or iPhone testing was done. Future work should address actual background/empty-scene rejection and target selection with independently collected data, and evaluate dense objects separately before any model promotion. The broader source-verified literature assessment is in `ml/research-20261025/general-papers.json`.
