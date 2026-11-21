# CountScope colony training experiment

This is a separate research candidate. It does not change the app binary, its submitted model, subscription configuration or store review.

## Data acquired without sign-in

- [Figshare bacterial colonies, version 3](https://figshare.com/articles/dataset/Annotated_dataset_for_deep-learning-based_bacterial_colony_detection/22022540/3): 369 images and 56,865 original bounding boxes across 24 species. Original files are under `output/datasets/colony-figshare-22022540-v3/raw`; every published file is checked against the source's byte count and MD5. Downloaded Python attachments are retained as source material, not executed.
- [Author's FSC147 release](https://github.com/cvlab-stonybrook/LearningToCountEverything): the public Google Drive archive, annotations and official splits are under `output/datasets/fsc147`. No logged-in cookies were used. The archive contains 6,146 image/density-map pairs; 6,135 belong to the official train/validation/test splits, with 89/29/29 disjoint categories. It is a separate point/density-counting task and is not mixed into the colony bounding-box training labels.

Full data, environments, checkpoints and intermediate diagnostics remain in gitignored `output/`.

## Colony audit and split

The original files remain unchanged. EXIF orientation is normalized in prepared images so annotations align with the displayed photo. All 24 species become one detection class, `colony`; this model does not identify species.

Three entire images with zero-area annotations are quarantined. Valid boxes extending beyond an image are clipped to the image boundary, with their original and corrected coordinates recorded. Exact source hashes and potential pHash neighbors are audited before fitting. Entire species stay together, and species connected by a near-similarity candidate also stay in one split conservatively. Similar circular dishes can trigger pHash even when the colonies differ; these are not claimed to be proven duplicate images.

The fixed seed is 20261025. Oversized similarity components go into training; the remaining seeded components fill validation and test to roughly 15% each with at least three components. The frozen result is:

| Split | Images | Valid object boxes |
| --- | ---: | ---: |
| Train | 247 | 37,611 |
| Validation | 56 | 6,831 |
| Test | 63 | 11,752 |

`output/datasets/colony-figshare-22022540-v3/yolo-species-v1/manifest.json` is immutable once created. The scripts refuse to overwrite it. No test inference is run before checkpoint and confidence selection have been frozen using validation data.

## Training and evaluation

The original attempt used a 30-epoch maximum and was interrupted at epoch11 by an MPS indexing failure. The completed successor fine-tuned YOLO11n for20epochs at1024pixels, with CPU target assignment, lower learning rate and no arbitrary-angle rotation. Dense-plate mosaic augmentation is disabled. Rotation, flips and modest image transformations apply only to the training split. Model choice uses validation mAP50-95; the counting confidence threshold uses only validation count MAE on a fixed threshold grid.

```sh
python3 ml/colony/download_figshare.py
output/training-venv/bin/python ml/colony/prepare_dataset.py
output/training-venv/bin/python ml/colony/train.py --epochs 20 --batch 4 --imgsz 1024 \
  --name colony-yolo11n-v2-cpuassign --lr 0.0005 --degrees 0 --patience 8 --warmup-bias-lr 0.0005
output/training-venv/bin/python ml/colony/evaluate_pipeline.py \
  --run output/training-20261025/runs/colony-yolo11n-v2-cpuassign
```

The evaluation reports standard AP50/AP50-95, fixed-threshold IoU-0.50 TP/FP/FN, precision/recall, small-object recall, count MAE/RMSE/bias and exact-count rate. It also reports a training-median-count baseline. An annotated error gallery includes the largest count errors as well as the smallest. These dataset results are not physical-device, phone-photo or production acceptance.

The first process completed and saved epoch 1 but its progress logger expected a tensor where the installed Ultralytics returned a loss dictionary. The logger was corrected and training resumed from the saved checkpoint with the same frozen data and optimizer protocol. The failed process log and original logger source remain in `output/training-20261025/`.

## Known source-data issues

FSC147's official density maps for `3471.jpg` and `3477.jpg` differ from the dot counts: approximately 77 vs 78 and 13 vs 344. Original files and discrepancy evidence remain intact. Future density training must explicitly resolve or quarantine these cases, and must not infer accurate labels from a readable archive alone.

## MPS recovery experiment

At epoch 11, the first run failed in MPS advanced boolean indexing inside the no-gradient target assigner (`tal.py`). The first ten complete epochs and all checkpoints/logs remain preserved; no heldout test was run on that attempt. The second candidate starts again from the original pretrained weights with the same frozen dataset, lower learning rate, no arbitrary-angle box rotation, and a 20-epoch maximum. `mps_compat.py` routes the unchanged target-assignment algorithm through CPU and transfers targets back to MPS; network forward/backward stays on the GPU. An exact CPU/MPS-transfer equivalence check is included. Final predictions and metrics use CPU inference as a separate backend check.

## Completed result

Automatic four-tile inference was selected using only validation count error plus precision/recall/F1 gates, then frozen before test. Test:63images,11,752objects, precision92.8%, recall91.9%, AP50 94.3%, AP50–95 60.3%, countMAE13.03, exact count5/63. The frozen whole-image reference had countMAE11.63, so tiling did not improve heldout count error despite better aggregate detection. Large/diffuse colonies remain poorly detected; one species dominates the object-weighted score. See the full per-species results and failure gallery; this is not general deployment acceptance.

CoreML FP16 and FP32 experimental exports are retained. FP32 matched counts on three validation photos using the selected four-tile pipeline, but whole-image counts still differed by up to6. No app or physical-iPhone integration was performed.
