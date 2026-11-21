# FSC147 reference-conditioned counting experiment

This is a separate research pipeline, not a shipped CountScope feature. It needs 3–6 example rectangles identifying what to count. Automatic object/example selection is not implemented.

## Acquisition and provenance

The complete author-linked FSC147 archive was downloaded without login or browser cookies. ZIP CRC and safe paths passed; 6,146 image/density pairs are retained, of which 6,135 belong to official train/validation/test splits (89/29/29 disjoint categories). Annotation files match their Git blob hashes. Data: [Learning To Count Everything, CVPR 2021](https://github.com/cvlab-stonybrook/LearningToCountEverything), Viresh Ranjan, Udbhav Sharma, Thu Nguyen and Minh Hoai. The authors' original code and MIT notice are preserved in the model bundle.

The original density maps for `3471.jpg` (training) and `3477.jpg` (validation) disagree with dot counts. They are excluded without overwriting the source. An exact-photo audit found five groups crossing official splits. Four validation images (`1896.jpg`, `4707.jpg`, `4549.jpg`, `4719.jpg`) and one test image (`3494.jpg`) are excluded before the test. Within-split repeated pixels can have different exemplar/target queries and remain present. This is a class-disjoint benchmark with exact cross-split duplicates removed; it is not an independently acquired phone-photo test.

Final cohorts: **3,658 training / 1,281 validation / 1,189 test queries**.

## Training

The published FamNet CountRegressor is initialized with the author's public checkpoint, and the original ImageNet ResNet50 weights from torchvision 0.5 are frozen as the extractor. The extractor/regressor passed a CPU-versus-MPS pilot. Frozen float32 features are cached locally. A complete one-epoch fine-tune ran 3,658 Adam updates at learning rate 1e-6 using mean-squared density loss; interpolation preserves density mass. Validation selects between the original baseline and the trained candidate before test. No test-time adaptation or test-driven retraining is performed.

A supervising process allowed the full epoch to finish and save its weights, then interrupted only the original evaluation phase. This applied the exact-photo exclusions before any test inference. The clean evaluation and original training logs are retained.

## Results

| Model | Clean validation MAE | Decision |
| --- | ---: | --- |
| Original public FamNet baseline | 24.27 | Selected before test |
| One-epoch fine-tuned candidate | 25.58 | Retained, not promoted |

The selected **original baseline**, not the new fine-tuned candidate, obtained **MAE 22.54 / RMSE 101.32** over 1,189 clean test queries, with 35 rounded exact counts. The fine-tune therefore did not establish an improvement. Do not present the baseline's test score as a result achieved by the fine-tuned weights.

`reports/20261025/` contains the frozen split, training-completion record, numeric parameter-change check, selection receipt and full test results. Full checkpoints, feature caches and logs remain under `output/training-fsc147-20261025/`.

## Offline use

Bundle: `output/models/CountScope-FSC147-FamNet-v1/`, containing both regressors, the frozen backbone, inspected original model utilities and a recipe with hashes. No network download is performed by the inference command.

```sh
output/training-venv/bin/python ml/fsc147/predict.py \
  --image path/to/photo.jpg --boxes path/to/examples.json \
  --weights baseline --device cpu --output output/fsc-prediction
```

`examples.json` is a list of 3–6 `[x1, y1, x2, y2]` rectangles in original image pixels. The output includes a continuous count, rounded count and density map. Use `--weights finetuned` only to inspect the rejected research candidate. No iPhone runtime, automatic reference selection or App Store integration was performed.
