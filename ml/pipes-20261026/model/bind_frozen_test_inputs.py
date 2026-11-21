"""Authorized test identity request and actual App PNG audit; no model inference.

Do not run either phase before evaluate_frozen_test.py authorize succeeds.
The module itself imports only stdlib and does not read any dataset on import.
"""
import argparse
import json
from pathlib import Path
import evaluate_frozen_test as h

REQUEST = h.ROOT / "output/pipes-20261026/test-preprocessing-request.json"
INPUTS = h.OUT / "test-app-inputs.json"
AUDIT = h.OUT / "test-app-input-audit.json"
VARIANTS = {"imageio-import": "gallery", "camera-normalized": "camera"}


def authorized(receipt_path):
    bindings = h.gate(receipt_path)
    h.require(bindings is not None, "No qualified validation candidate: do not open test")
    h.require(bindings["runtimeVersion"] in [2, 3], "Dedicated helper requires a reviewed runtime amendment")
    path = h.OUT / "test/pre-test-protocol.json"
    registration = h.read(path)
    h.require(registration["status"] == "FROZEN_SINGLE_PIPE_TEST_PROTOCOL_BEFORE_TEST_ACCESS" and registration["bindings"] == bindings, "Exclusive pre-test registration mismatch")
    h.require(not (h.OUT / "test/test-opened.json").exists(), "Do not create/change inputs after model test begins")
    return bindings, h.sha(path)


def identities(bindings):
    path = h.resolved(bindings["sealedManifestPath"])
    h.require(h.sha(path) == bindings["sealedManifestSHA256"], "Frozen manifest changed")
    manifest = h.read(path)
    h.require(manifest["status"] == "FROZEN_ANNOTATION_QA_PASSED", "Dataset not frozen")
    # Extract identities only; never use annotations/counts to select export views.
    rows = [{key: r[key] for key in ["id", "split", "imagePath", "sha256", "width", "height"]}
            for r in manifest["images"] if r["split"] == "test"]
    h.require(len(rows) == 37 and len({r["id"] for r in rows}) == 37, "Test identities incomplete")
    return rows


def request(receipt_path):
    bindings, protocol_sha = authorized(receipt_path)
    h.require(not REQUEST.exists(), "Preserve existing test preprocessing request")
    rows = identities(bindings)
    value = {"status": "AUTHORIZED_FROZEN_TEST_PREPROCESSING_REQUEST", "testContentIncluded": True,
             "preTestProtocolSHA256": protocol_sha, "sourceManifestPath": bindings["sealedManifestPath"],
             "sourceManifestSHA256": bindings["sealedManifestSHA256"],
             "testExportHelperSHA256": bindings["testExportHelperSHA256"],
             "approvedAppSourceCodeSHA256": bindings["appSourceCodeSHA256"], "images": rows,
             "createdUTC": h.now(), "requestScriptSHA256": h.sha(__file__)}
    h.exclusive(REQUEST, value)
    return {"status": value["status"], "path": str(REQUEST.relative_to(h.ROOT)), "sha256": h.sha(REQUEST),
            "testPhotosOpened": False, "modelInferencePerformed": False}


def geometry(width, height):
    gain = min(1024 / width, 1024 / height)
    rw, rh = round(width * gain), round(height * gain)
    return {"gain": gain, "resizedWidth": rw, "resizedHeight": rh,
            "left": round((1024 - rw) / 2 - .1), "top": round((1024 - rh) / 2 - .1),
            "sourceWidth": width, "sourceHeight": height, "size": 1024}


def bind(receipt_path, export_manifest_path):
    bindings, protocol_sha = authorized(receipt_path)
    h.require(not INPUTS.exists() and not AUDIT.exists(), "Preserve existing test input binding/audit")
    request_data = h.read(REQUEST)
    h.require(request_data["preTestProtocolSHA256"] == protocol_sha and request_data["testExportHelperSHA256"] == bindings["testExportHelperSHA256"], "Export request changed")
    rows = identities(bindings)
    h.require(request_data["images"] == rows, "Request differs from all sealed test identities")
    by_id = {r["id"]: r for r in rows}
    export_path = h.resolved(export_manifest_path)
    export = h.read(export_path)
    h.require(export["status"] == "APP_PREPROCESSING_PIXELS_EXPORTED_NO_MODEL_INFERENCE" and export["kind"] == "frozen_test_photo", "Not actual App holdout export")
    h.require(export["preTestProtocolSHA256"] == protocol_sha and export["sealedManifestSHA256"] == bindings["sealedManifestSHA256"], "App export protocol/data mismatch")
    h.require(export["sourceCodeSHA256"] == bindings["appSourceCodeSHA256"] and export["testExportHelperSHA256"] == bindings["testExportHelperSHA256"], "App source identity mismatch")
    expected = {(variant, identity) for variant in h.VARIANTS for identity in by_id}
    h.require(len(export["records"]) == 74 and {(VARIANTS[r["appPath"]], r["id"]) for r in export["records"]} == expected, "Missing/extra/duplicate App paths")
    # The authorization/protocol/identity checks above precede all image imports/reads.
    import numpy as np
    from PIL import Image
    images = []
    checks = []
    for row in export["records"]:
        source = by_id[row["id"]]
        h.require(row["split"] == "test" and row["kind"] == "frozen_test_photo", "Incorrect export split")
        h.require(row["sourceImagePath"] == source["imagePath"] and row["sourceSHA256"] == source["sha256"], "Source identity changed")
        h.require(row["sourceManifestSHA256"] == bindings["sealedManifestSHA256"], "Source manifest changed")
        h.require((row["sourcePixelWidth"], row["sourcePixelHeight"]) == (source["width"], source["height"]), "Source dimensions changed")
        h.require(h.sha(h.resolved(source["imagePath"])) == source["sha256"], "Source image bytes changed")
        h.require(row["sourceEXIFOrientation"] == 1 and row["outputOrientation"] == "up", "Unexpected EXIF orientation")
        h.require(row["networkSize"] == 1024 and row["paddingRGB"] == [114, 114, 114], "Wrong preprocessing dimensions/padding")
        w, height = row["preparedWidth"], row["preparedHeight"]
        h.require(0 < w <= 2048 and 0 < height <= 2048, "Invalid prepared dimensions")
        g = geometry(w, height)
        for key in ["gain", "resizedWidth", "resizedHeight", "left", "top"]:
            h.require(abs(row[key] - g[key]) < 1e-8, "Letterbox metadata changed")
        h.require(row["right"] == 1024 - g["left"] - g["resizedWidth"] and row["bottom"] == 1024 - g["top"] - g["resizedHeight"], "Odd padding changed")
        photo_path, network_path = h.resolved(row["preparedPNG"]), h.resolved(row["networkPNG"])
        h.require(h.sha(photo_path) == row["preparedSHA256"] and h.sha(network_path) == row["networkSHA256"], "PNG byte hashes changed")
        with Image.open(photo_path) as im:
            h.require(im.format == "PNG" and im.size == (w, height) and im.getexif().get(274, 1) in [None, 1], "Invalid upright PNG")
            im.load()
        with Image.open(network_path) as im:
            h.require(im.format == "PNG" and im.size == (1024, 1024) and im.getexif().get(274, 1) in [None, 1], "Invalid network PNG")
            pixels = np.asarray(im.convert("RGB"))
        pad = np.ones((1024, 1024), dtype=bool)
        pad[g["top"]:g["top"] + g["resizedHeight"], g["left"]:g["left"] + g["resizedWidth"]] = False
        h.require(bool(np.all(pixels[pad] == 114)), "Actual padding is not gray114")
        images.append({"id": row["id"], "split": "test", "variant": VARIANTS[row["appPath"]], "kind": "frozen_test_photo",
                       "sourceImageSHA256": source["sha256"], "sourceManifestSHA256": bindings["sealedManifestSHA256"],
                       "sourceEXIFOrientation": 1, "networkImagePath": str(network_path), "networkImageSHA256": row["networkSHA256"],
                       "photoImagePath": str(photo_path), "photoImageSHA256": row["preparedSHA256"], "photoWidth": w, "photoHeight": height,
                       "geometry": g})
        checks.append({"id": row["id"], "variant": VARIANTS[row["appPath"]], "hashesDimensionsGeometryPadding": "PASS"})
    h.require(authorized(receipt_path) == (bindings, protocol_sha), "Approval source changed during input audit")
    value = {"status": "VALIDATED_APP_FROZEN_TEST_INPUTS", "testContentIncluded": True, "preTestProtocolSHA256": protocol_sha,
             "sealedManifestSHA256": bindings["sealedManifestSHA256"], "approvedAppSourceCodeSHA256": bindings["appSourceCodeSHA256"],
             "testExportHelperSHA256": bindings["testExportHelperSHA256"], "variants": h.VARIANTS, "images": images,
             "appExportManifests": [{"path": str(export_path), "sha256": h.sha(export_path)}],
             "modelInferencePerformedByBinding": False, "bindingScriptSHA256": h.sha(__file__)}
    h.exclusive(INPUTS, value)
    audit = {"status": "PASS_FROZEN_TEST_APP_PIXELS_IDENTITY_HASH_GEOMETRY_GRAY114", "createdUTC": h.now(),
             "inputsManifestSHA256": h.sha(INPUTS), "preTestProtocolSHA256": protocol_sha,
             "sealedManifestSHA256": bindings["sealedManifestSHA256"], "testExportHelperSHA256": bindings["testExportHelperSHA256"],
             "bindingScriptSHA256": h.sha(__file__), "images": 37, "inputVariants": h.VARIANTS, "networkPNGs": 74,
             "checks": checks, "modelInferencePerformed": False, "annotationsUsedForSelection": False,
             "limitations": ["Byte/identity/geometry audit of actual App export; no model predictions or accuracy acceptance.",
                             "No claim that UIKit pixels equal independently resized OpenCV pixels."]}
    h.exclusive(AUDIT, audit)
    return {"status": audit["status"], "inputsPath": str(INPUTS.relative_to(h.ROOT)), "inputsSHA256": h.sha(INPUTS),
            "auditPath": str(AUDIT.relative_to(h.ROOT)), "auditSHA256": h.sha(AUDIT)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["request", "bind"])
    parser.add_argument("--receipt", type=Path, default=h.OUT / "root-frozen-test-approval.json")
    parser.add_argument("--export-manifest", type=Path)
    args = parser.parse_args()
    if args.phase == "bind":
        h.require(args.export_manifest is not None, "Explicit export manifest required")
    result = request(args.receipt) if args.phase == "request" else bind(args.receipt, args.export_manifest)
    print(json.dumps(result, indent=2))
