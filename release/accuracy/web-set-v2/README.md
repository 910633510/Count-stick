# New endpoint scenes, evaluation v2

This set extends the diagnostic coverage of `../web-set/`; it does not replace or modify the frozen v1 labels. The user requested another online data search and trial.

## Protocol declared before inference

- Select four visually distinct new scenes: cotton-candy skewer ends, Relaxdays large ends, HKTVmall large ends with advertising text, and the clean photographic panel of a Pospal handheld-skewer demonstration.
- All four source groups are **holdout** for this run. No threshold, input transform, matching radius, or model selection is tuned using their results.
- Use the unchanged original weights with the fixed baseline (`confidence=0.25`, `IoU=0.70`) and existing build 2 candidate (`confidence=0.25`, `IoU=0.45`), both `scaleFill`.
- Manually mark visible blunt terminal faces. A second reviewer checks labels before freezing hashes and running inference. Partial/uncertain border faces are explicitly ignored; those photos do not contribute to exact whole-image count or MAE.
- The Pospal source collage includes another system's displayed result. The first annotator receives only the unmarked photo crop and does not see the claim. Acquisition/root reviewers have seen the collage; any review must check each visible face rather than accept its count. This is disclosed rather than described as a completely blinded acquisition process.
- Keep repeated vendor copies, resized versions, and all panels of one collage in the same source group. Use SHA checks, perceptual hashes, and visual scene comparison; perceptual hashes alone do not establish independence.

## Local source images

Source URLs, original and normalized image hashes, and crop derivations are in `root-acquisition.json`, `reviewer-acquisition.json`, and `acquisition.json`. Downloaded originals and annotation/result previews remain in gitignored `output/web-annotation-v2/`; they are not redistributed in the repository. No images are synthesized or repaired for evaluation.

The Pospal image is a published POS demonstration, not verified post-meal restaurant usage. Three other selected scenes are product photographs. Unknown overlap with the original model's unavailable training data cannot be ruled out.

## Reproduce after labels are frozen

From the repository root on a Mac with Swift/CoreML:

```sh
python3 release/accuracy/web-set/evaluate.py release/accuracy/web-set-v2/manifest.json --validate-only
python3 release/accuracy/web-set/evaluate.py release/accuracy/web-set-v2/manifest.json --output-dir output/web-annotation-v2/evaluation
```

This reuses the already-tested v1 evaluator. It rejects changed labels, photos, manifests, model files, or detector source. Scores use one-to-one point matching and are not bounding-box AP or field accuracy. macOS CPU inference is not physical iPhone acceptance.

See `REPORT.md` for measured results after the run and `dataset-prospects.json` plus `acquisition.json` for larger dataset leads and their current access limitations.
