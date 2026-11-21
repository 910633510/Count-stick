# Real empty-plate annotation candidates

64 real photographs were downloaded or copied from verified public source data and visually reviewed. **40 images receive zero visible-colony annotations; 24 remain quarantined with no count assignment.** No model was run and no train/validation split was assigned here. Root must freeze the global capture/plate groups before training.

| Source | Reviewed | Accepted zero | Quarantined | Leakage-control grouping |
|---|---:|---:|---:|---|
| Mendeley xrbtd74pfj v3 | 48 | 28 | 20 | Entire Negative source is one conservative group because physical-plate IDs are unavailable |
| BRUKERAGAR public sample | 12 | 9 | 3 | Two identified physical plates, six lighting/views each; all variants stay together |
| Zenodo 18703636 | 4 | 3 | 1 | One acquisition-date/session group; root must union same-session positive images |

These are **40 files in four conservative leakage-control groups**, not 40 independent plates. Two groups are physical plates, one is a source-level uncertainty group, and one is a capture-session group. Pure white/gray/black synthetic controls are not included in these counts.

- [Accepted candidate manifest](negative-candidates.json): full-resolution image path/hash, original source/hash, zero count, empty xyxy, empty YOLO label/hash, source/group, visual notes, provenance. Status remains `ANNOTATION_QA_PASSED_PENDING_ROOT_GLOBAL_SPLIT`.
- [All 64 visual decisions](visual-review.json): per-image reasons and review resolution.
- [Quarantine](quarantine.json): excluded records have `count=null`, `xyxy=null`, and no generated training label. Quarantine is uncertainty, not a diagnosis or a confirmed source-label error.
- [Review gallery](../../../output/research-20261025-hardcases/negatives/review-gallery.html): accepted and excluded full frames; click to open the corresponding full-resolution image.

The public [Mendeley dataset](https://data.mendeley.com/datasets/xrbtd74pfj/3) was photographed with a Samsung Galaxy Note9 inside a controlled light chamber. Its [author paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10024077/) defines Negative as no growth and places 1 CFU cases in Uncertain. Nevertheless, the category was not treated as sufficient evidence for zero: for example, N406 has distinct cyan/green points and N151 has pale round rim marks, so both were quarantined. Native-resolution detail crops are preserved under `output/research-20261025-hardcases/negatives/review/`; they are display evidence, not additional training samples.

48 Mendeley images were selected before download by lowest SHA256 of a fixed seed and filename among the 500 Negative originals. None was selected by model score, and quarantined records were not replaced with easier examples. All original byte counts and published SHA256 hashes were verified. The originals have EXIF orientation6; full-resolution copies were rotated with `ImageOps.exif_transpose`, converted to RGB and saved as JPEG quality98/subsampling0 with EXIF removed. Original bytes remain intact. Visual review used complete1536×1536 previews, with native-resolution details for specified ambiguous cases.

The [BRUKERAGAR author repository](https://github.com/michalcicatka/BRUKERAGAR-DATASET), fixed at `5cbee6da8785b9bd6abf1b23a0a86e60afd7076d`, publicly exposes12 sample files covering only AEBM/BESBL plate0000001. The paper describes a larger dataset, but the current README requires contacting the author for full access; no contact, login or paid download was performed. LFS object size and SHA256 were checked for every public sample. Every image was inspected as a full frame. Three saturated BESBL views were quarantined because their agar could not be independently inspected.

The four [Zenodo original-source photographs](https://zenodo.org/records/18703636) were copied byte-for-byte from the positive-data agent's verified archive. Its SHA256 is `2c9762e8d89234b4dcd9c38a37b2d6478e4356ec4317c1f8d48b1ba7391bcef5`. All four have zero COCO boxes, but image `20230710_141728831...` contains many fine pale points that cannot be confidently separated from backing/condensation texture by this review; it remains quarantined. The other three show smooth red agar and real camera/rim reflections. All four were captured within46seconds and share `zenodo18703636_session_20230710`. The author's original train/valid/test folder names are provenance, not our new split assignment.

A zero label means **no visible colony in the reviewed photograph**; it does not certify microbiological sterility. These Codex annotations are not certified microbiologist ground truth. Controlled phone/robot imaging also does not establish freehand phone performance. No previous project data, App file or App Store state was changed, and the previously reserved36 clinical test images were not opened.

Acquisition receipts, frozen selection, source catalog and originals are under `output/research-20261025-hardcases/negatives/`. `acquire_mendeley_negatives.py` performs bounded public acquisition. `build_manifest.py` binds the reviewed decisions; do not regenerate a frozen manifest during training. Root owns the final global split and dataset receipt.
