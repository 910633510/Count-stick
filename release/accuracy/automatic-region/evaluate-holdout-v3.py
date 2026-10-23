#!/usr/bin/env python3
"""Frozen v3 holdout: raw build2 detections versus the actual production region selector.

Default / --validate-only: no inference, no output files.
--compile-only: compile the shared Swift selector + helper, never execute it.
--freeze-checkpoint PATH: write immutable code/model/OCR bindings after candidate approval.
--checkpoint PATH --run-inference: one full-image CoreML pass per photo, then shared selection.
--checkpoint PATH --predictions PATH: score bound existing outputs into a fresh output directory.

Do not freeze or run until the release captain confirms that the candidate is fixed.
This script never edits v1/v2 data, labels, production code, or previous results.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
EXPECTED_MANIFEST_SHA = "bfd3d87885e8add72be93919688e6103c3379dc3c019ec02296b40b6a24f2af9"
VARIANTS = ("raw_build2", "automatic_region")
RAW_SETTINGS = {"confidenceThreshold": 0.25, "iouThreshold": 0.45, "cropScale": "scaleFill",
                "maximumDimension": 2048, "computeUnits": "cpuOnly"}
OCR_SETTINGS = {"recognitionLevel": "accurate", "recognitionLanguages": ["zh-Hans", "zh-Hant", "en-US"],
                "usesLanguageCorrection": False, "minimumTextHeight": 0.015,
                "minimumCandidateConfidence": 0.8, "minimumAlphanumericCharacters": 2,
                "candidateRank": 1, "revisionMode": "system_default",
                "failurePolicy": "empty_text_areas_then_shared_selector"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_scorer():
    path = ROOT / "release/accuracy/web-set/evaluate.py"
    spec = importlib.util.spec_from_file_location("frozen_v1_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bindings(manifest_path):
    paths = {"detector": ROOT / "Count stick/StickDetector.swift",
             "region": ROOT / "Count stick/AutoCountRegion.swift",
             "helper": Path(__file__).with_name("helper-holdout.swift"),
             "harness": Path(__file__).resolve(),
             "scorer": ROOT / "release/accuracy/web-set/evaluate.py"}
    source = paths["detector"].read_text()
    # Fail closed on production OCR changes instead of silently using stale settings.
    for pattern in [r"recognitionLevel\s*=\s*\.accurate", r'recognitionLanguages\s*=\s*\["zh-Hans",\s*"zh-Hant",\s*"en-US"\]',
                    r"usesLanguageCorrection\s*=\s*false", r"minimumTextHeight\s*=\s*0\.015",
                    r"candidate\.confidence\s*>=\s*0\.8", r"guard count\s*>=\s*2", r"topCandidates\(1\)",
                    r"candidate\.string\.unicodeScalars", r"CharacterSet\.alphanumerics"]:
        require(re.search(pattern, source), "Production OCR changed; align and review helper before freezing")
    model = ROOT / "Count stick/best.mlpackage"
    model_hashes = {str(path.relative_to(model)): digest(path) for path in sorted(model.rglob("*")) if path.is_file()}
    v1 = read(ROOT / "release/accuracy/web-set/benchmark-v1.json")
    require(model_hashes == v1["modelFileHashes"], "Raw build2 comparison requires unchanged v1 model files")
    return {"manifestSha256": digest(manifest_path), "modelPath": str(model), "modelFileHashes": model_hashes,
            "sourceFiles": {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in paths.items()},
            "rawSettings": RAW_SETTINGS, "ocrSettings": OCR_SETTINGS}


def compile_helper(current, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = output_dir / "helper-holdout-v3"
    require(not binary.exists(), "Use a fresh output directory; compiled helper already exists")
    subprocess.run(["xcrun", "swiftc", "-O", "-parse-as-library",
                    current["sourceFiles"]["region"]["path"], current["sourceFiles"]["helper"]["path"],
                    "-o", str(binary)], check=True)
    for source in current["sourceFiles"].values():
        require(digest(source["path"]) == source["sha256"], "Source changed during compilation")
    return binary


def aggregate(rows):
    result = {}
    for variant in VARIANTS:
        metrics = [row["variants"][variant] for row in rows]
        totals = {key: sum(metric[key] for metric in metrics)
                  for key in ("TP", "FP", "FN", "ignoredPredictions", "duplicateFalsePositives")}
        tp, fp, fn = (totals[key] for key in ("TP", "FP", "FN"))
        counts = [metric for metric in metrics if metric["exactCountEligible"]]
        result[variant] = {**totals, "images": len(rows), "sourceGroups": len({row["sourceGroup"] for row in rows}),
                           "precision": tp / (tp + fp) if tp + fp else None,
                           "recall": tp / (tp + fn) if tp + fn else None,
                           "countEligibleImages": len(counts),
                           "countMAE": sum(metric["absoluteCountError"] for metric in counts) / len(counts) if counts else None,
                           "exactCountRate": sum(metric["exactCount"] for metric in counts) / len(counts) if counts else None}
    return result


def verify_predictions(predictions, checkpoint, checkpoint_path, active):
    require(predictions["checkpointSha256"] == digest(checkpoint_path), "Prediction checkpoint differs")
    for key in ("manifestSha256", "modelFileHashes", "sourceFiles", "rawSettings", "ocrSettings"):
        require(predictions[key] == checkpoint["bindings"][key], "Prediction binding differs: " + key)
    require(predictions["sourceDetectorSha256"] == checkpoint["bindings"]["sourceFiles"]["detector"]["sha256"]
            and predictions["autoCountRegionSha256"] == checkpoint["bindings"]["sourceFiles"]["region"]["sha256"],
            "Explicit production source hashes differ")
    require(predictions["coordinateSystem"] == "upright image pixels, top-left", "Unexpected prediction coordinates")
    rows = {row["imageId"]: row for row in predictions["images"]}
    require(len(rows) == len(predictions["images"]) and set(rows) == {item["id"] for item in active}, "Prediction image identities differ")
    for item in active:
        row = rows[item["id"]]
        require(row["imageSha256"] == item["sha256"] and (row["width"], row["height"]) == (item["width"], item["height"]), "Photo identity/dimensions differ")
        require(row["modelInferenceCount"] == 1, "Each image must have exactly one full-image model pass")
        require(set(row["variants"]) == set(VARIANTS), "Unexpected comparison variants")
        raw = row["variants"]["raw_build2"]["predictions"]
        raw_by_id = {point["id"]: point for point in raw}
        require(len(raw_by_id) == len(raw), "Duplicate raw detection ID")
        for variant in VARIANTS:
            data = row["variants"][variant]
            for key in ("confidenceThreshold", "iouThreshold", "cropScale"):
                require(data[key] == RAW_SETTINGS[key], "Fixed raw settings changed")
            points = data["predictions"]
            require(len({point["id"] for point in points}) == len(points), "Duplicate filtered detection ID")
            for point in points:
                require(point["id"] in raw_by_id and point == raw_by_id[point["id"]], "Selector changed or created a detection")
                require(all(isinstance(point[key], (float, int)) and math.isfinite(point[key]) for key in ("x", "y", "confidence")), "Nonfinite prediction")
        filtered = row["variants"]["automatic_region"]["predictions"]
        kept = row["selection"]["keptIDs"]
        excluded = row["selection"]["excluded"]
        require(set(kept) == {point["id"] for point in filtered} and len(kept) == len(filtered), "Selected and retained IDs differ")
        require(len({entry["id"] for entry in excluded}) == len(excluded), "Duplicate exclusion ID")
        require(set(kept).isdisjoint({entry["id"] for entry in excluded})
                and set(kept) | {entry["id"] for entry in excluded} == set(raw_by_id), "Selection is not a partition of raw IDs")
        require(all(entry["reason"] in ("text", "isolated") for entry in excluded), "Unexpected exclusion reason")
        require(row["variants"]["automatic_region"]["additionalModelInferences"] == 0, "Automatic region unexpectedly reran model")
        for area in row["ocr"]["areas"]:
            require(set(area) == {"box", "confidence", "characterCount"}, "Unexpected OCR content; strings must not be persisted")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "release/accuracy/web-set-v3/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/web-annotation-v3/evaluation-auto-region")
    parser.add_argument("--checkpoint", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--compile-only", action="store_true")
    modes.add_argument("--freeze-checkpoint", type=Path, metavar="PATH")
    modes.add_argument("--run-inference", action="store_true")
    modes.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    require(digest(manifest_path) == EXPECTED_MANIFEST_SHA, "The approved v3 frozen manifest changed")
    scorer = load_scorer()
    manifest, active, excluded = scorer.validate_manifest(manifest_path, ROOT)
    require(len(active) == 2 and all(item["split"] == "holdout" for item in active), "Expected the two unseen v3 holdout groups")
    current = bindings(manifest_path)
    validation = {"validationPassed": True, "datasetId": manifest["datasetId"], "manifestSha256": EXPECTED_MANIFEST_SHA,
                  "eligibleImages": len(active), "manualEndpointCount": sum(len(item["annotation"]["points"]) for item in active),
                  "sourceGroups": len({item["sourceGroup"] for item in active}), "inferencePerformed": False}
    if args.freeze_checkpoint:
        write_new(args.freeze_checkpoint.resolve(), {"schemaVersion": 1, "algorithmFrozen": True,
                  "frozenAtUTC": now(), "datasetId": manifest["datasetId"], "bindings": current,
                  "protocol": "One full-image raw build2 pass per new scene; aligned Vision OCR plus shared production AutoCountRegion.select; no model/threshold/label tuning on v3 outputs."})
        print(json.dumps({**validation, "checkpoint": str(args.freeze_checkpoint.resolve()), "checkpointSha256": digest(args.freeze_checkpoint)}, indent=2))
        return
    if args.compile_only:
        binary = compile_helper(current, args.output_dir.resolve())
        print(json.dumps({**validation, "compiledHelper": str(binary), "binarySha256": digest(binary), "executed": False}, indent=2))
        return
    if args.validate_only or not (args.run_inference or args.predictions):
        print(json.dumps(validation, indent=2))
        return
    require(args.checkpoint is not None, "Inference/scoring requires an explicit frozen --checkpoint")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = read(checkpoint_path)
    require(checkpoint.get("algorithmFrozen") is True and checkpoint.get("frozenAtUTC"), "Algorithm checkpoint is not frozen")
    require(checkpoint["bindings"] == current, "Source/model/OCR changed after checkpoint; do not rerun holdout after tuning")
    output_dir = args.output_dir.resolve()
    require(not (output_dir / "report.json").exists(), "Refusing to overwrite a previous holdout report")
    prediction_path = args.predictions.resolve() if args.predictions else output_dir / "predictions.json"
    if args.run_inference:
        require(not prediction_path.exists(), "Refusing to overwrite or silently rerun holdout predictions")
        binary = compile_helper(current, output_dir)
        receipt = {"schemaVersion": 1, "algorithmFrozen": True, "labelsFrozen": True, "splitLocked": True,
                   "preparedAtUTC": now(), "checkpointSha256": digest(checkpoint_path),
                   "compiledHelperSha256": digest(binary), **current,
                   "images": [{key: item[key] for key in ("id", "path", "sha256", "width", "height")} for item in active]}
        receipt_path = output_dir / "validated-input.json"
        write_new(receipt_path, receipt)
        subprocess.run([str(binary), str(receipt_path), str(prediction_path)], check=True)
    predictions = read(prediction_path)
    require(bindings(manifest_path) == current, "Bound source/model/manifest changed during evaluation")
    require(digest(checkpoint_path) == predictions["checkpointSha256"], "Checkpoint changed during evaluation")
    for item in active:
        require(digest(item["path"]) == item["sha256"] and digest(item["annotationPath"]) == item["annotationSha256"], "Frozen photo/labels changed during evaluation")
    by_id = verify_predictions(predictions, checkpoint, checkpoint_path, active)
    rows = []
    for item in active:
        predicted = by_id[item["id"]]
        scores = {variant: scorer.score_image(item, predicted["variants"][variant]["predictions"]) for variant in VARIANTS}
        rows.append({"imageId": item["id"], "sourceGroup": item["sourceGroup"], "split": item["split"],
                     "imageSha256": item["sha256"], "annotationSha256": item["annotationSha256"],
                     "variants": scores, "selection": predicted["selection"],
                     "ocrSummary": {key: value for key, value in predicted["ocr"].items() if key != "areas"},
                     "ocrAreaCount": len(predicted["ocr"]["areas"]),
                     "change": {"truePositiveChange": scores["automatic_region"]["TP"] - scores["raw_build2"]["TP"],
                                "falsePositiveChange": scores["automatic_region"]["FP"] - scores["raw_build2"]["FP"],
                                "removedDetectionCount": len(predicted["selection"]["excluded"])}})
    report = {"schemaVersion": 1, "datasetId": manifest["datasetId"], "evaluatedAtUTC": now(),
              "validation": {key: value for key, value in validation.items() if key != "inferencePerformed"},
              "checkpointSha256": digest(checkpoint_path),
              "checkpointFrozenAtUTC": checkpoint["frozenAtUTC"], "modelFileHashes": current["modelFileHashes"],
              "sourceDetectorSha256": current["sourceFiles"]["detector"]["sha256"],
              "autoCountRegionSha256": current["sourceFiles"]["region"]["sha256"],
              "sourceFiles": current["sourceFiles"], "rawSettings": RAW_SETTINGS, "ocrSettings": OCR_SETTINGS,
              "predictionFileSha256": digest(prediction_path), "executionPlatform": predictions["executionPlatform"],
              "operatingSystem": predictions["operatingSystem"],
              "inferencePerformedThisInvocation": args.run_inference, "excludedImages": excluded,
              "matchingProtocol": "Unmodified v1 score_image: maximum-cardinality, minimum-distance one-to-one matches within frozen radii; unmatched predictions inside frozen ignore polygons are excluded.",
              "countProtocol": "Exact count and MAE only for completeCountVerified + full_image + no ignored uncertainty. Vietnam is partial; Allegro square ends are the sole complete-count image.",
              "selectionProtocol": "Full-image CoreML inference once. Shared production AutoCountRegion.select filters original detection IDs using aligned OCR areas and geometry; no manual ROI, input crop, extra prediction, or new detection.",
              "overall": aggregate(rows), "bySplit": {split: aggregate([row for row in rows if row["split"] == split]) for split in ("calibration", "holdout")},
              "bySourceGroup": {group: aggregate([row for row in rows if row["sourceGroup"] == group]) for group in sorted({row["sourceGroup"] for row in rows})},
              "images": rows, "limitations": ["Two new online product scenes are a diagnostic holdout, not restaurant field accuracy.",
                  "Square-ended Allegro image is a shape stress case separate from the usual round-skewer population.",
                  "Unknown original-model training-image overlap cannot be ruled out.",
                  "Vision OCR revisions/platform behavior may differ on physical iPhone; this run uses recorded macOS default revision and CPU-only CoreML.",
                  "Only OCR rectangles/confidence/character counts are stored; recognized strings are never emitted.",
                  "Independent score audit and physical-device validation remain separate from this evaluator's completion."]}
    write_new(output_dir / "report.json", report)
    print(json.dumps({"report": str(output_dir / "report.json"), "overall": report["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print("Holdout evaluation blocked:", str(error), file=sys.stderr)
        raise SystemExit(2)
