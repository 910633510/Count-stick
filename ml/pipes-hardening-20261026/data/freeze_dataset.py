"""Root-reviewed fresh cohort; immutable manifest, no model inference."""
from pathlib import Path
import collections
import copy
import datetime
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "ml/pipes-hardening-20261026/data"
DEST = ROOT / "output/datasets/pipe-ends-hardening-v1"
BRIDGE = ROOT / "output/pipes-20261026/hardening-fresh-val-source"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")

def main():
    positive_path = ART / "positives/positive-proposal.json"
    negative_path = ROOT / "output/pipes-hardening-20261026/data/negatives/negative-candidates-v2.json"
    assert sha(positive_path) == "029cd02e27858b661c53cb218b7ccb83b1765dc777286ac75c75605bce79b6ba"
    assert sha(negative_path) == "4ba65082e727bc86c0c6e22937b9d69d62cbc153e2157a728bd904331e1dba8c"
    review_paths = [ART / "positives/positive-qa.json",
                    ART / "positives/root-annotations-independent-review.json",
                    ART / "positives/general-annotations-independent-review.json",
                    ART / "negatives/negative-data-qa-v2.json"]
    for path in review_paths:
        assert read(path)["status"].startswith("PASS"), path
    review_bindings = {str(p.relative_to(ROOT)): sha(p) for p in review_paths + [positive_path, negative_path]}
    root_manual = read(ART / "positives/root-kaggle-manual-annotations.json")["images"]
    manual = {"kaggle_%03d" % r["number"]: r for r in root_manual}
    manual.update({r["id"]: r for r in read(ART / "positives/root-commons-manual-annotations.json")["images"]})
    manual.update({r["id"]: r for r in read(ART / "positives/general-manual-annotations.json")["images"]})
    old_path = ROOT / "output/datasets/pipe-ends-maco-v1/manifest.json"
    assert sha(old_path) == "960005aa6c7956db8ebf5962693071e0ea01888f0d0380431437894b4fddec41"
    old = read(old_path)["images"]
    old_sha = {r["sha256"] for r in old}
    old_rgb = {r["RGBPixelSHA256"] for r in old}
    old_groups = {r["groupId"] for r in old}
    positives = read(positive_path)["images"]
    negatives = read(negative_path)["images"]
    assert read(positive_path)["modelInferencePerformed"] is False
    assert read(negative_path)["modelInferencePerformed"] is False
    assert len(positives) == 21 and sum(r["count"] for r in positives) == 100
    assert len(negatives) == 48 and all(r["count"] == 0 for r in negatives)
    rows = copy.deepcopy(positives + negatives)
    ids = set()
    identity_splits = {key: {} for key in ["groupId", "sha256", "RGBPixelSHA256", "parent"]}
    for row in rows:
        if "modelInferencePerformed" not in row:
            row["modelInferencePerformed"] = False
            row["modelInferenceDeclarationInheritedFromProposal"] = True
        assert re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]) and row["id"] not in ids
        ids.add(row["id"])
        assert row["split"] in {"train", "val", "test"}
        assert row["count"] == len(row["xyxy"]) and 0 <= row["count"] <= 20
        assert row["isSynthetic"] is False and row["hasAnnotationOverlay"] is False
        assert row["modelInferencePerformed"] is False
        image_path = (ROOT / row["imagePath"]).resolve()
        assert image_path.is_relative_to(ROOT) and sha(image_path) == row["sha256"]
        with Image.open(image_path) as im:
            assert im.size == (row["width"], row["height"])
            assert im.getexif().get(274, 1) in (None, 1)
            rgb_sha = hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
        if "RGBPixelSHA256" in row:
            assert row["RGBPixelSHA256"] == rgb_sha
        row["RGBPixelSHA256"] = rgb_sha
        assert row["sha256"] not in old_sha and rgb_sha not in old_rgb and row["groupId"] not in old_groups
        assert all(0 <= a < c <= row["width"] and 0 <= b < d <= row["height"] for a,b,c,d in row["xyxy"])
        assert len(row["xyxy"]) == len({tuple(box) for box in row["xyxy"]})
        if row["id"] in manual:
            assert row["xyxy"] == manual[row["id"]]["xyxy"]
            assert row["count"] == manual[row["id"]]["count"]
            assert row["sha256"] == manual[row["id"]]["sha256"]
        elif row["count"]:
            assert row["id"] == "new_maco_0021" and row["count"] == 17
            xml = image_path.with_suffix(".xml")
            boxes = [[float(obj.find("bndbox").find(key).text) for key in ["xmin","ymin","xmax","ymax"]]
                     for obj in ET.parse(xml).getroot().findall("object")]
            assert row["xyxy"] == boxes
        if row["dataNature"] == "real_crop":
            assert row["split"] == "train" and row["count"] == 0 and row["cropReview"] == "PASS"
            assert row["sourceHadBlackMask"] is True and row["parentGroupId"] == row["groupId"]
            parent = ROOT / row["sourceImagePath"]
            assert sha(parent) == row["sourceImageSHA256"] and sha(parent) not in old_sha
            with Image.open(parent) as im:
                assert im.getexif().get(274, 1) in (None, 1)
                parent_rgb = im.convert("RGB")
                assert hashlib.sha256(parent_rgb.tobytes()).hexdigest() not in old_rgb
                expected = parent_rgb.crop(tuple(row["cropRectPixels"]))
            with Image.open(image_path) as im:
                assert im.format == "PNG" and im.mode == "RGB" and im.size == expected.size
                assert im.tobytes() == expected.tobytes()
        else:
            assert row["dataNature"] == "real" and not row.get("sourceHadBlackMask", False)
        for key in ["groupId", "sha256", "RGBPixelSHA256"]:
            prior = identity_splits[key].setdefault(row[key], row["split"])
            assert prior == row["split"], (key, row["id"])
        parent_id = row.get("sourceImageSHA256", row["sha256"])
        assert identity_splits["parent"].setdefault(parent_id, row["split"]) == row["split"]
        row["proposalAnnotationQA"] = row.get("annotationQA")
        row["annotationQA"] = "PASS"
        row["label"] = "pipe_end"
        row["bucket"] = "positive" if row["count"] else "negative"
        row["finalTrainingAdmission"] = row["split"] == "train"
        if row["count"]:
            assert row["subtype"] == "round" and row["annotationBoundary"] == "outer_rim"
    assert not DEST.exists(), "Never overwrite a frozen cohort"
    BRIDGE.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["split"] == "val":
            original = ROOT / row["imagePath"]
            target = BRIDGE / (row["id"] + original.suffix)
            if not target.exists():
                shutil.copy2(original, target)
            assert sha(target) == row["sha256"]
            row["originalImagePath"] = row["imagePath"]
            row["imagePath"] = str(target.relative_to(ROOT))
            row["pathBridge"] = "Byte-identical copy for unchanged original App test helper whitelist. Not a new view."
    DEST.mkdir()
    labels = DEST / "labels"
    labels.mkdir()
    for row in rows:
        path = labels / (row["id"] + ".txt")
        w,h = row["width"], row["height"]
        text = "\n".join(f"0 {(a+c)/2/w:.10f} {(b+d)/2/h:.10f} {(c-a)/w:.10f} {(d-b)/h:.10f}" for a,b,c,d in row["xyxy"])
        path.write_text(text + ("\n" if text else ""))
        row["labelPath"] = str(path.relative_to(ROOT))
        row["labelSHA256"] = sha(path)
    summary = {s: {"images": sum(r["split"] == s for r in rows),
                   "positiveImages": sum(r["split"] == s and r["count"] > 0 for r in rows),
                   "negativeImages": sum(r["split"] == s and r["count"] == 0 for r in rows),
                   "groups": len({r["groupId"] for r in rows if r["split"] == s}),
                   "objects": sum(r["count"] for r in rows if r["split"] == s)} for s in ["train","val","test"]}
    assert summary["train"]["images"] == 42 and summary["val"]["images"] == 7 and summary["test"]["images"] == 20
    manifest = {"status": "FROZEN_ANNOTATION_QA_PASSED", "createdUTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "testContentIncluded": True, "images": rows, "summary": summary, "reviewBindings": review_bindings,
                "freezerSHA256": sha(__file__), "previousFullManifestSHA256": sha(old_path),
                "original37TestReusedInTrainingOrSelection": False, "newModelInferencePerformed": False,
                "splitPolicy": "Approved coverage allocation before optimization; all known capture/parent groups stay within one split.",
                "limits": "Train crop groups are conservative source/diameter pools, not certified distinct piles. Natural public-photo holdout is small and not prospective iPhone validation."}
    write_new(DEST / "manifest.json", manifest)
    tv = {**manifest, "testContentIncluded": False, "images": [r for r in rows if r["split"] != "test"],
          "summary": {s: summary[s] for s in ["train","val"]}, "sealedFullManifestSHA256": sha(DEST / "manifest.json")}
    write_new(DEST / "train-val-manifest.json", tv)
    identity = {"status": "FROZEN_IDENTITY_ONLY_NO_GT_OR_PIXELS", "fullManifestSHA256": sha(DEST / "manifest.json"),
                "images": [{k:r[k] for k in ["id","split","groupId","sha256","RGBPixelSHA256"]} for r in rows]}
    write_new(DEST / "identity-only.json", identity)
    bridge_manifest = BRIDGE / "hardening-train-val-manifest.json"
    assert not bridge_manifest.exists()
    shutil.copy2(DEST / "train-val-manifest.json", bridge_manifest)
    assert sha(bridge_manifest) == sha(DEST / "train-val-manifest.json")
    assert all(sha(ROOT / p) == digest for p,digest in review_bindings.items())
    qa = {"status": "PASS_ROOT_FRESH_DATA_FREEZE_QA", "createdUTC": manifest["createdUTC"], "summary": summary,
          "fullManifestPath": str((DEST / "manifest.json").relative_to(ROOT)), "fullManifestSHA256": sha(DEST / "manifest.json"),
          "trainValManifestPath": str((DEST / "train-val-manifest.json").relative_to(ROOT)), "trainValManifestSHA256": sha(DEST / "train-val-manifest.json"),
          "identityOnlyPath": str((DEST / "identity-only.json").relative_to(ROOT)), "identityOnlySHA256": sha(DEST / "identity-only.json"),
          "helperManifestCopyPath": str(bridge_manifest.relative_to(ROOT)), "reviewBindings": review_bindings,
          "checks": ["69 identities unique; coordinates/counts/hash/RGB/EXIF valid", "all20 human new positive annotations independently reviewed; MACO21 count17 separately reviewed",
                     "32 crop pixels exactly equal unmasked parent rectangles, train only", "no old241 byte/RGB/group or crop-parent overlap", "cross-split group/file/RGB/parent identities exclusive",
                     "fresh-val bridge copies byte-identical", "no recognition model run on fresh data before freeze"],
          "old37Policy": "Preserved historical failure; no images or GT in new training/selection.", "newTrainingPerformed": False}
    write_new(ART / "root-freeze-qa.json", qa)
    print(json.dumps({"status": qa["status"], "summary": summary, "fullSHA256": qa["fullManifestSHA256"], "tvSHA256": qa["trainValManifestSHA256"]}, indent=2))

if __name__ == "__main__":
    main()
