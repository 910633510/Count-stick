"""One fixed pipe candidate on the sealed scene holdout; never select on test.

authorize: stdlib-only gate, then exclusively register the pre-test protocol.
evaluate: recheck the same gate, exclusively mark test opened, then load test.
Neither phase accesses test, imports ML libraries, or loads weights if selected=null.
Do not erase the exclusive protocol/opened markers to retry a different candidate.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/pipes-20261026/model"
SELF = Path(__file__).resolve()
SOURCE = SELF.parent
FROZEN_CODE = [
    "pipe_ml.py", "train_pipe_ends.py", "evaluate_pipe_ends.py",
    "export_pipe_coreml.py", "prepare_photos.swift", "audit_official_init.py",
    "bind_app_inputs.py",
]
APP_FILES = {
    "PhotoProcessing": "Count stick/StickDetector.swift",
    "PipeLetterbox": "Count stick/PipeDetector.swift",
    "ExportHelper": "Count stickTests/PipePreprocessingExportTests.swift",
}
SEALED_REL = "output/datasets/pipe-ends-maco-v1/manifest.json"
SEALED_SHA = "960005aa6c7956db8ebf5962693071e0ea01888f0d0380431437894b4fddec41"
VARIANTS = ["gallery", "camera"]
RUNTIME_V2_CHANGES = [
    "computeUnits_all_to_cpuAndNeuralEngine",
    "vision_score_label_to_object_confidence",
]
RUNTIME_V3_CHANGES = [
    "computeUnits_all_to_cpuAndNeuralEngine",
    "coreml_raw_nms_false",
    "shared_swift_greedy_nms",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(directory):
    directory = resolved(directory)
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        require(not path.is_symlink(), "Package tree contains symlink")
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def resolved(path):
    path = Path(path)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    require(path.is_relative_to(ROOT), "Path escapes repository")
    return path


def exclusive(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def code_hashes():
    paths = [SOURCE / name for name in FROZEN_CODE] + [ROOT / "ml/colony/mps_compat.py"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def letterbox_source(source):
    start, end = "struct PipeLetterbox {", "\nenum PipeCountRegion {"
    require(source.count(start) == 1 and source.count(end) == 1, "Ambiguous PipeLetterbox source block")
    require(source.index(start) < source.index(end), "Invalid PipeLetterbox source order")
    return source[source.index(start):source.index(end)]


def runtime_source_binding(protocol, receipt, expected):
    """Versioned authorization, not a bypass of the frozen training source hash.

    v1 requires original source. v2 has an exact two-change allowlist. v3 binds
    independently reviewed raw output/shared-NMS source and export identities.
    All versions preserve the training protocol/eight ML files and image geometry.
    """
    old = protocol["appSourceCodeSHA256"]
    current = {name: sha(ROOT / path) for name, path in APP_FILES.items()}
    names = ["runtimeAmendmentPath", "runtimeAmendmentSHA256",
             "runtimeAmendmentReviewPath", "runtimeAmendmentReviewSHA256"]
    if not any(name in receipt for name in names):
        require(old == current, "App preprocessing source changed without runtime amendment")
        return {"trainingAppSourceCodeSHA256": old, "appSourceCodeSHA256": current, "runtimeVersion": 1}
    require(all(name in receipt for name in names), "Incomplete runtime amendment approval binding")
    amendment_path = resolved(receipt["runtimeAmendmentPath"])
    review_path = resolved(receipt["runtimeAmendmentReviewPath"])
    require(sha(amendment_path) == receipt["runtimeAmendmentSHA256"], "Runtime amendment changed")
    require(sha(review_path) == receipt["runtimeAmendmentReviewSHA256"], "Runtime amendment review changed")
    amendment, review = read(amendment_path), read(review_path)
    version = amendment["version"]
    require(version in [2, 3] and amendment["status"] == f"FROZEN_PIPE_RUNTIME_V{version}_AMENDMENT", "Unsupported runtime amendment")
    require(review["status"] == f"PASS_PIPE_RUNTIME_V{version}_SOURCE_AMENDMENT", "Runtime amendment independent review not PASS")
    allowed_changes = RUNTIME_V2_CHANGES if version == 2 else RUNTIME_V3_CHANGES
    require(review["amendmentSHA256"] == receipt["runtimeAmendmentSHA256"], "Review binds another amendment")
    for document in [amendment, review]:
        for key in ["protocolSHA256", "selectionSHA256", "modelSHA256", "codeSHA256", "operatingPoint"]:
            require(document[key] == expected[key], "Runtime changes frozen model/protocol/threshold: " + key)
        require(document["oldAppSourceCodeSHA256"] == old, "Runtime old hash does not match frozen training protocol")
        require(document["newAppSourceCodeSHA256"] == current, "Runtime new hash does not match current source")
        require(document["allowedChanges"] == allowed_changes and document["pixelGeometryUnchanged"] is True, "Unapproved runtime change scope")
    training_authorization = read(OUT / "training-authorization.json")
    training_receipt_sha = sha(resolved(training_authorization["rootReceiptPath"]))
    require(training_authorization["rootReceiptSHA256"] == training_receipt_sha, "Original training root receipt changed")
    prior_export_path = OUT / "export/export-status.json"
    export_path = prior_export_path if version == 2 else resolved(amendment["exportStatusPath"])
    export = read(export_path)
    require(export["modelSHA256"] == expected["modelSHA256"] and export["selectionSHA256"] == expected["selectionSHA256"], "Runtime export binds different weights/selection")
    require({"confidence": export["confidenceThreshold"], "nmsIoU": export["iouThreshold"]} == expected["operatingPoint"], "Runtime export thresholds changed")
    require(tree_sha(export["packagePath"]) == export["packageTreeSHA256"], "CoreML package changed during runtime repair")
    runtime_artifacts = {"trainingRootReceiptSHA256": training_receipt_sha, "exportStatusSHA256": sha(export_path),
                         "packageTreeSHA256": export["packageTreeSHA256"], "exportID": export["exportID"]}
    nms_binding = {}
    if version == 3:
        require(export_path != prior_export_path, "Preserve the original embedded-NMS export")
        require(export["nms"] is False and export["quantize"] == 32 and export["inputSize"] == 1024 and export["rawLocationCapacity"] == 21504, "v3 must use unchanged1024 FP32 raw outputs without embedded NMS")
        prior_export = read(prior_export_path)
        require(prior_export["modelSHA256"] == expected["modelSHA256"] and prior_export["selectionSHA256"] == expected["selectionSHA256"], "Preserved export model binding mismatch")
        require(tree_sha(prior_export["packagePath"]) == prior_export["packageTreeSHA256"], "Original embedded-NMS package was changed")
        runtime_artifacts.update({"exportStatusPath": str(export_path.relative_to(ROOT)),
                                  "priorExportStatusSHA256": sha(prior_export_path),
                                  "priorPackageTreeSHA256": prior_export["packageTreeSHA256"]})
        nms_path = resolved(amendment["nmsSourcePath"])
        require(nms_path.suffix == ".swift" and nms_path.is_relative_to(ROOT / "Count stick"), "Shared NMS must be production Swift source")
        nms_binding = {"path": str(nms_path.relative_to(ROOT)), "sha256": sha(nms_path)}
        for document in [amendment, review]:
            require(document["nmsSourcePath"] == nms_binding["path"] and document["nmsSourceSHA256"] == nms_binding["sha256"], "Reviewed shared NMS source changed")
    for document in [amendment, review]:
        for key, value in runtime_artifacts.items():
            require(document[key] == value, "Runtime artifact binding mismatch: " + key)
    parity_path = resolved(review["parityReportPath"])
    require(sha(parity_path) == review["parityReportSHA256"], "Reviewed runtime parity report changed")
    parity = read(parity_path)
    require(parity["status"] == "PASS_PIPE_TORCH_COREML_VISION_PARITY", "Runtime parity is not PASS")
    for key in ["modelSHA256", "selectionSHA256", "protocolSHA256", "operatingPoint"]:
        require(parity[key] == expected[key], "Runtime parity binding mismatch: " + key)
    require(parity["exportID"] == export["exportID"] and parity["packageTreeSHA256"] == export["packageTreeSHA256"], "Runtime parity package mismatch")
    require(parity["inputVariants"] == VARIANTS, "Runtime parity missing an App path")
    if version == 3:
        require(parity["runtimeVersion"] == 3 and parity["nmsSourceSHA256"] == nms_binding["sha256"], "Parity does not bind the v3 shared NMS")
    require(old["PhotoProcessing"] == current["PhotoProcessing"] and old["ExportHelper"] == current["ExportHelper"], "PhotoProcessing or original export helper changed")
    snapshot_path = resolved(amendment["oldPipeDetectorSnapshotPath"])
    require(sha(snapshot_path) == old["PipeLetterbox"] == amendment["oldPipeDetectorSHA256"], "Old PipeDetector snapshot mismatch")
    require(current["PipeLetterbox"] == amendment["newPipeDetectorSHA256"], "New PipeDetector snapshot mismatch")
    old_source = snapshot_path.read_bytes().decode("utf-8")
    new_source = (ROOT / APP_FILES["PipeLetterbox"]).read_bytes().decode("utf-8")
    if version == 2:
        require(old_source.count("options.computeUnits = .all") == 1, "Unexpected old compute-device expression")
        require(old_source.count("label.confidence") == 3, "Unexpected old confidence expressions")
        allowed_source = old_source.replace("options.computeUnits = .all", "options.computeUnits = .cpuAndNeuralEngine")
        allowed_source = allowed_source.replace("label.confidence", "object.confidence")
        require(new_source == allowed_source, "Runtime source exceeds exact two-change allowlist")
    require(letterbox_source(old_source) == letterbox_source(new_source), "PipeLetterbox geometry/pixels changed")
    return {"trainingAppSourceCodeSHA256": old, "appSourceCodeSHA256": current, "runtimeVersion": version,
            "runtimeAmendmentPath": str(amendment_path.relative_to(ROOT)),
            "runtimeAmendmentSHA256": sha(amendment_path),
            "runtimeAmendmentReviewPath": str(review_path.relative_to(ROOT)),
            "runtimeAmendmentReviewSHA256": sha(review_path),
            "runtimeParityReportPath": str(parity_path.relative_to(ROOT)), "runtimeParityReportSHA256": sha(parity_path),
            "runtimeArtifacts": runtime_artifacts, "runtimeNMS": nms_binding,
            "PipeLetterboxBlockSHA256": hashlib.sha256(letterbox_source(old_source).encode()).hexdigest(),
            "runtimeAllowedChanges": allowed_changes}


def gate(receipt_path):
    # The only file read before this early return is the validation selection.
    selection_path = OUT / "selection.json"
    selection = read(selection_path)
    require("selected" in selection, "Selection has no selected field")
    if selection["selected"] is None:
        return None
    require(selection["status"] == "VAL_QUALIFIED_PENDING_FROZEN_SCENE_TEST", "Unqualified selection")
    chosen = selection["selected"]
    protocol_path = OUT / "protocol.json"
    protocol = read(protocol_path)
    code = code_hashes()
    require(selection["codeSHA256"] == code, "Frozen model source changed")
    require(selection["protocolSHA256"] == sha(protocol_path), "Protocol changed")
    require(protocol["evaluationVariants"] == VARIANTS, "Unexpected input variants")
    require(protocol["candidateCount"] == 18 and len(selection["candidates"]) == 18, "Incomplete candidate grid")
    require(chosen in selection["candidates"] and chosen["status"] == "PASS", "Selected candidate not qualified")
    require(set(chosen["variants"]) == set(VARIANTS), "Selected candidate missing an App path")
    require(all(chosen["variants"][v]["gates"]["status"] == "PASS" for v in VARIANTS), "Selected variant failed")
    point = {"confidence": chosen["confidence"], "nmsIoU": chosen["nmsIoU"]}
    require(chosen["epoch"] in protocol["checkpointEpochs"], "Unknown selected epoch")
    require(point["confidence"] in protocol["confidenceGrid"] and point["nmsIoU"] in protocol["nmsIoUGrid"], "Unknown operating point")
    require(all(math.isfinite(x) and 0 < x < 1 for x in point.values()), "Invalid thresholds")
    receipt_path = resolved(receipt_path)
    receipt = read(receipt_path)
    require(receipt["status"] == "APPROVED_PIPE_FROZEN_TEST_SINGLE_CANDIDATE", "No dedicated root test approval")
    expected = {
        "selectionSHA256": sha(selection_path), "modelSHA256": chosen["modelSHA256"],
        "protocolSHA256": sha(protocol_path), "codeSHA256": code,
        "evaluatorSHA256": sha(SELF), "epoch": chosen["epoch"],
        "operatingPoint": point, "inputVariants": VARIANTS,
        "sealedManifestPath": SEALED_REL, "sealedManifestSHA256": SEALED_SHA,
        "expectedTestImages": 37,
    }
    for name, value in expected.items():
        require(receipt[name] == value, "Root approval binding mismatch: " + name)
    app_binding = runtime_source_binding(protocol, receipt, expected)
    helper_path = resolved(receipt["testExportHelperPath"])
    require(sha(helper_path) == receipt["testExportHelperSHA256"], "Dedicated test export helper changed")
    audit_path = resolved(receipt["validationAuditPath"])
    require(sha(audit_path) == receipt["validationAuditSHA256"], "Independent audit changed")
    audit = read(audit_path)["validationAudit"]
    require(str(audit["status"]).startswith("PASS"), "Independent validation audit not PASS")
    for name in ["selectionSHA256", "modelSHA256", "protocolSHA256", "codeSHA256"]:
        require(audit[name] == expected[name], "Independent audit binding mismatch: " + name)
    require(audit["selected"] == chosen, "Independent audit selected candidate mismatch")
    weights = OUT / "checkpoints" / ("epoch%03d.pt" % chosen["epoch"])
    require(sha(weights) == chosen["modelSHA256"], "Checkpoint bytes changed")
    metadata = read(weights.with_suffix(".json"))
    require(metadata["sha256"] == chosen["modelSHA256"], "Checkpoint registration mismatch")
    completion = read(OUT / "training-complete.json")
    require(completion["status"] == "COMPLETE_20_EPOCHS_WITH_ACTUAL_REGISTERED_CHECKPOINTS", "Training incomplete")
    require(set(completion["checkpoints"]) == {"8", "14", "20"}, "Registered checkpoints incomplete")
    require(completion["checkpoints"][str(chosen["epoch"])] == metadata, "Chosen registration not in completion")
    return {**expected, "rootApprovalPath": str(receipt_path.relative_to(ROOT)),
            "rootApprovalSHA256": sha(receipt_path), "validationAuditPath": str(audit_path.relative_to(ROOT)),
            "validationAuditSHA256": sha(audit_path), **app_binding,
            "testExportHelperPath": str(helper_path.relative_to(ROOT)),
            "testExportHelperSHA256": sha(helper_path),
            "qualification": protocol["qualification"], "weightsPath": str(weights.relative_to(ROOT))}


def authorize(receipt_path):
    bindings = gate(receipt_path)
    if bindings is None:
        return {"status": "SKIP_NO_VALIDATION_QUALIFIED_CANDIDATE", "testAccessed": False}
    registration = {
        "status": "FROZEN_SINGLE_PIPE_TEST_PROTOCOL_BEFORE_TEST_ACCESS", "createdUTC": now(),
        "bindings": bindings, "testAccessedByRegistration": False,
        "policy": "One fixed candidate and two App paths. No selection, threshold/epoch switch, retraining, or repeat holdout run after opening.",
        "scope": "37 sealed real scene views, true split=test; unchanged author/corrected outer-rim boxes.",
        "rawCapture": {"dtype": "float32", "confidenceFloor": 0.001, "rawLocations": 21504,
                       "NMSApplied": False, "outputCapApplied": False},
        "metrics": "Reuse frozen evaluator count_metrics/AP/postprocess; each gallery/camera path must independently pass original qualification.",
    }
    exclusive(OUT / "test/pre-test-protocol.json", registration)
    return {"status": registration["status"], "path": str((OUT / "test/pre-test-protocol.json").relative_to(ROOT)),
            "sha256": sha(OUT / "test/pre-test-protocol.json"), "testAccessed": False}


def load_test_inputs(bindings, inputs_path, input_audit_path, registration_sha):
    # Caller has exclusively recorded test-opened.json before entering here.
    import numpy as np
    from PIL import Image
    import pipe_ml as p
    inputs_path, input_audit_path = resolved(inputs_path), resolved(input_audit_path)
    audit = read(input_audit_path)
    require(str(audit["status"]).startswith("PASS"), "Test App input audit not PASS")
    require(audit["inputsManifestSHA256"] == sha(inputs_path), "Input audit manifest mismatch")
    require(audit["preTestProtocolSHA256"] == registration_sha, "Input audit protocol mismatch")
    require(audit["sealedManifestSHA256"] == bindings["sealedManifestSHA256"], "Input audit dataset mismatch")
    require(audit["testExportHelperSHA256"] == bindings["testExportHelperSHA256"], "Input audit export helper mismatch")
    inputs = read(inputs_path)
    require(inputs["status"] == "VALIDATED_APP_FROZEN_TEST_INPUTS", "Unvalidated App test inputs")
    require(inputs["testContentIncluded"] is True and inputs["variants"] == VARIANTS, "Wrong split/variant scope")
    require(inputs["preTestProtocolSHA256"] == registration_sha, "Inputs were not bound to exclusive pre-test protocol")
    require(inputs["sealedManifestSHA256"] == bindings["sealedManifestSHA256"], "Inputs dataset mismatch")
    require(inputs["approvedAppSourceCodeSHA256"] == bindings["appSourceCodeSHA256"], "Production App source mismatch")
    require(inputs["testExportHelperSHA256"] == bindings["testExportHelperSHA256"], "Inputs dedicated export helper mismatch")
    require(inputs["appExportManifests"], "Missing actual App export receipts")
    exported = {}
    path_to_variant = {"imageio-import": "gallery", "camera-normalized": "camera"}
    for binding in inputs["appExportManifests"]:
        path = resolved(binding["path"])
        require(sha(path) == binding["sha256"], "App export receipt changed")
        manifest = read(path)
        require(manifest["status"] == "APP_PREPROCESSING_PIXELS_EXPORTED_NO_MODEL_INFERENCE", "Invalid App export status")
        require(manifest["kind"] == "frozen_test_photo", "App receipt is not real held-out test photos")
        require(manifest["preTestProtocolSHA256"] == registration_sha, "App export protocol mismatch")
        # The approved three hashes are lineage/unchanged-source evidence. The
        # distinct test helper below is the code that actually exports test PNGs.
        require(manifest["sourceCodeSHA256"] == bindings["appSourceCodeSHA256"], "App receipt source lineage mismatch")
        require(manifest["testExportHelperSHA256"] == bindings["testExportHelperSHA256"], "Actual test export helper mismatch")
        for row in manifest["records"]:
            key = (path_to_variant[row["appPath"]], row["id"])
            require(key not in exported and row["split"] == "test", "Duplicate/non-test export record")
            require(row["sourceManifestSHA256"] == bindings["sealedManifestSHA256"], "Export source manifest mismatch")
            require(row["sourceEXIFOrientation"] in [1, None] and row["outputOrientation"] == "up", "Unexpected image orientation")
            require(row["networkSize"] == 1024 and row["paddingRGB"] == [114, 114, 114], "Incorrect network preprocessing")
            exported[key] = row
    sealed_path = resolved(bindings["sealedManifestPath"])
    require(sha(sealed_path) == bindings["sealedManifestSHA256"], "Sealed manifest changed")
    manifest = read(sealed_path)
    require(manifest["status"] == "FROZEN_ANNOTATION_QA_PASSED", "Dataset not frozen")
    test = [r for r in manifest["images"] if r["split"] == "test"]
    require(len(test) == bindings["expectedTestImages"] and len({r["id"] for r in test}) == len(test), "Wrong test identities")
    other = [r for r in manifest["images"] if r["split"] != "test"]
    for key in ["groupId", "sha256", "RGBPixelSHA256"]:
        require(not {r[key] for r in test} & {r[key] for r in other}, "Cross-split identity overlap")
    expected_keys = {(v, r["id"]) for v in VARIANTS for r in test}
    require(set(exported) == expected_keys, "Actual App receipt coverage mismatch")
    require(len(inputs["images"]) == len(expected_keys) and {(r["variant"], r["id"]) for r in inputs["images"]} == expected_keys, "Input rows incomplete, duplicate or extra")
    records = {}
    for row in test:
        require(row["annotationQA"] == "PASS" and row["dataNature"] == "real" and row["isSynthetic"] is False and row["hasAnnotationOverlay"] is False, "Invalid real-data QA")
        require(row["count"] == len(row["xyxy"]) and 0 <= row["count"] <= 21504, "Invalid count/capacity")
        require(sha(resolved(row["imagePath"])) == row["sha256"], "Original test image changed")
        for a, b, c, d in row["xyxy"]:
            require(all(math.isfinite(x) for x in [a, b, c, d]) and 0 <= a < c <= row["width"] and 0 <= b < d <= row["height"], "Invalid test GT geometry")
        if row["count"]:
            require(row["subtype"] == "round" and row["annotationBoundary"] == "outer_rim", "Incorrect target scope")
        else:
            require(row["negativeKind"] in ["empty_scene", "non_pipe", "side_only"], "Invalid negative scope")
        records[row["id"]] = {**row, "originalWidth": row["width"], "originalHeight": row["height"], "originalXYXY": row["xyxy"]}
    for row in inputs["images"]:
        require(row["split"] == "test", "Never relabel test as validation")
        record = records[row["id"]]
        origin = exported[(row["variant"], row["id"])]
        require(row["sourceImageSHA256"] == record["sha256"] == origin["sourceSHA256"], "Source image binding mismatch")
        require(row["sourceManifestSHA256"] == bindings["sealedManifestSHA256"], "Input manifest provenance mismatch")
        for key, original_key in [("networkImagePath", "networkPNG"), ("photoImagePath", "preparedPNG")]:
            require(resolved(row[key]) == resolved(origin[original_key]), "Input path differs from actual App export")
        for key, original_key in [("networkImageSHA256", "networkSHA256"), ("photoImageSHA256", "preparedSHA256"), ("photoWidth", "preparedWidth"), ("photoHeight", "preparedHeight")]:
            require(row[key] == origin[original_key], "Input identity/size differs from App export")
        geometry = p.pipeline_geometry(row["photoWidth"], row["photoHeight"])
        require(set(row["geometry"]) == set(geometry) and all(abs(row["geometry"][k] - value) < 1e-8 for k, value in geometry.items()), "Invalid letterbox geometry")
        for key in ["gain", "left", "top", "resizedWidth", "resizedHeight"]:
            require(abs(origin[key] - geometry[key]) < 1e-8, "App recorded geometry mismatch")
        require(origin["right"] == 1024 - geometry["left"] - geometry["resizedWidth"] and origin["bottom"] == 1024 - geometry["top"] - geometry["resizedHeight"], "App odd padding mismatch")
        require(0 < row["photoWidth"] <= 2048 and 0 < row["photoHeight"] <= 2048, "Unexpected prepared image size")
        for prefix, size in [("network", (1024, 1024)), ("photo", (row["photoWidth"], row["photoHeight"]))]:
            path = resolved(row[prefix + "ImagePath"])
            require(sha(path) == row[prefix + "ImageSHA256"], "Prepared pixels changed")
            with Image.open(path) as image:
                require(image.format == "PNG" and image.size == size and image.getexif().get(274, 1) in [1, None], "Invalid PNG geometry/orientation")
                if prefix == "network":
                    pixels = np.asarray(image.convert("RGB"))
                    padding = np.ones((1024, 1024), dtype=bool)
                    top, left = geometry["top"], geometry["left"]
                    padding[top:top+geometry["resizedHeight"], left:left+geometry["resizedWidth"]] = False
                    require(bool(np.all(pixels[padding] == 114)), "Padding pixels are not gray114")
    return inputs, records


def evaluate(receipt_path, inputs_path, input_audit_path):
    bindings = gate(receipt_path)
    if bindings is None:
        return {"status": "SKIP_NO_VALIDATION_QUALIFIED_CANDIDATE", "testAccessed": False}
    registration_path = OUT / "test/pre-test-protocol.json"
    registration = read(registration_path)
    require(registration["status"] == "FROZEN_SINGLE_PIPE_TEST_PROTOCOL_BEFORE_TEST_ACCESS" and registration["bindings"] == bindings, "Pre-test registration differs from current fixed approval")
    registration_sha = sha(registration_path)
    require(inputs_path is not None and input_audit_path is not None, "Actual App test inputs/audit required")
    exclusive(OUT / "test/test-opened.json", {"status": "OPENED_ONCE_FIXED_CANDIDATE", "createdUTC": now(),
                                             "preTestProtocolSHA256": registration_sha, "bindings": bindings})
    try:
        # No heavyweight imports, sealed label parsing or pixel reads occur before the marker.
        import numpy as np
        import torch
        from PIL import Image
        from ultralytics import YOLO
        from evaluate_pipe_ends import postprocess, count_metrics, ap_metrics, eligible
        initial_input_sha = sha(resolved(inputs_path))
        initial_audit_sha = sha(resolved(input_audit_path))
        inputs, records = load_test_inputs(bindings, inputs_path, input_audit_path, registration_sha)
        model = YOLO(str(resolved(bindings["weightsPath"]))).model.float().cpu().eval()
        require(model.names == {0: "pipe_end"} and model.model[-1].nc == 1 and not model.model[-1].end2end, "Wrong model architecture/classes")
        model.fuse()
        torch.set_num_threads(8)
        operating, ap_predictions = {}, {}
        for row in inputs["images"]:
            require(sha(resolved(row["networkImagePath"])) == row["networkImageSHA256"], "Network PNG changed after input audit")
            with Image.open(resolved(row["networkImagePath"])) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).float() / 255
            with torch.inference_mode():
                output = model(tensor)
                output = output[0] if isinstance(output, tuple) else output
            require(tuple(output.shape) == (1, 5, 21504) and bool(torch.isfinite(output).all()), "Invalid raw model output")
            raw = output[0].T
            scores = raw[:, 4]
            require(bool(((scores >= 0) & (scores <= 1)).all()), "Invalid raw confidence range")
            xywh = raw[:, :4]
            boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            keep = scores >= 0.001
            arrays = {"xyxy": boxes[keep].numpy(), "confidence": scores[keep].numpy()}
            rawpath = OUT / "test/raw" / row["variant"] / (row["id"] + ".npz")
            rawpath.parent.mkdir(parents=True, exist_ok=True)
            with rawpath.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            exclusive(rawpath.with_suffix(".json"), {"id": row["id"], "split": "test", "variant": row["variant"],
                "modelSHA256": bindings["modelSHA256"], "networkImageSHA256": row["networkImageSHA256"],
                "preTestProtocolSHA256": registration_sha, "npzSHA256": sha(rawpath), "rawLocations": 21504,
                "retainedAfterFloor": int(keep.sum()), "NMSApplied": False, "outputCapApplied": False})
            key = (row["variant"], row["id"])
            point = bindings["operatingPoint"]
            operating[key] = postprocess(arrays, row["geometry"], point["confidence"], point["nmsIoU"])
            ap_predictions[key] = postprocess(arrays, row["geometry"], 0.001, point["nmsIoU"])
            print("test raw", row["variant"], row["id"], int(keep.sum()), flush=True)
        results = {}
        for variant in VARIANTS:
            rows = [r for r in inputs["images"] if r["variant"] == variant]
            result = count_metrics({r["id"]: operating[(variant, r["id"])] for r in rows}, records, rows, bindings["operatingPoint"]["confidence"])
            result["summary"].update(ap_metrics({r["id"]: ap_predictions[(variant, r["id"])] for r in rows}, records, rows))
            result["gates"] = eligible(result)
            result["predictions"] = {r["id"]: operating[(variant, r["id"])] for r in rows}
            results[variant] = result
        # Reject concurrent source/selection/approval changes before publishing a result.
        require(gate(receipt_path) == bindings, "Bindings changed during test")
        require(sha(resolved(inputs_path)) == initial_input_sha and sha(resolved(input_audit_path)) == initial_audit_sha, "Input metadata changed during test")
        require(sha(resolved(bindings["sealedManifestPath"])) == bindings["sealedManifestSHA256"], "Sealed labels changed during test")
        input_binding = {"path": str(resolved(inputs_path)), "sha256": initial_input_sha}
        report = {"status": "PASS_PIPE_FROZEN_GROUP_HOLDOUT" if all(r["gates"]["status"] == "PASS" for r in results.values()) else "FAIL_PIPE_FROZEN_GROUP_HOLDOUT",
            "createdUTC": now(), "modelSHA256": bindings["modelSHA256"], "selectionSHA256": bindings["selectionSHA256"],
            "protocolSHA256": bindings["protocolSHA256"], "operatingPoint": bindings["operatingPoint"],
            "inputVariants": VARIANTS, "inputManifestBindings": {v: input_binding for v in VARIANTS},
            "inputAuditBinding": {"path": str(resolved(input_audit_path)), "sha256": initial_audit_sha},
            "preTestProtocolSHA256": registration_sha, "sealedManifestSHA256": bindings["sealedManifestSHA256"],
            "evaluatorSHA256": bindings["evaluatorSHA256"], "codeSHA256": bindings["codeSHA256"],
            "runtimeSourceBinding": {key: value for key, value in bindings.items() if key.startswith("runtime") or key in ["trainingAppSourceCodeSHA256", "appSourceCodeSHA256", "PipeLetterboxBlockSHA256"]},
            "epoch": bindings["epoch"], "split": "test", "testImages": len(records), "variants": results,
            "selectionBasedOnTest": False, "newTrainingPerformed": False, "arbitraryMarkerCapApplied": False,
            "limitations": ["MACO source imagery/author boxes and conservative scene groups; not a prospective phone field validation.",
                            "PyTorch fixed-model qualification only. This report does not establish CoreML/Vision parity or iPhone performance.",
                            "Any failed gate retains this result; do not choose another model/threshold on this holdout."]}
        exclusive(OUT / "test/test-report.json", report)
        return {"status": report["status"], "path": str((OUT / "test/test-report.json").relative_to(ROOT))}
    except BaseException as error:
        exclusive(OUT / "test/test-failure.json", {"status": "ERROR_TEST_OPENED_NO_QUALIFICATION_REPORT", "createdUTC": now(),
            "preTestProtocolSHA256": registration_sha, "errorType": type(error).__name__, "error": str(error),
            "retryPolicy": "Keep protocol/opened/cache evidence. A technical resume needs explicit independent audit; never erase markers or change candidate."})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["authorize", "evaluate"])
    parser.add_argument("--receipt", type=Path, default=OUT / "root-frozen-test-approval.json")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--input-audit", type=Path)
    args = parser.parse_args()
    result = authorize(args.receipt) if args.phase == "authorize" else evaluate(args.receipt, args.inputs, args.input_audit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
