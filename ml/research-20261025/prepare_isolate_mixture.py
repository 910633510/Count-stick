#!/usr/bin/env python3
"""Freeze an isolate-disjoint clinical cohort, then acquire original images.

Selection uses image/species/isolate identifiers only. Model outputs and target
counts are never selection inputs. Existing external datasets are read-only.
"""
from pathlib import Path, PurePosixPath
import argparse
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import struct
import time
import xml.etree.ElementTree as ET
import zlib
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "output/research-20261025"
OUT = ROOT / "output/datasets/colony-mixed-isolate-v1"
OLD = ROOT / "output/datasets/colony-external-zenodo18256864-v1"
SEED = "CountScope-isolate-mixture-20261025-v1"
PROTOCOL = OUT / "selection-before-acquisition.json"
SOURCE_COCO = OLD / "source-coco.json"
SPLITS = ("train", "val", "test")
SPEC = importlib.util.spec_from_file_location("range_acquisition", Path(__file__).with_name("acquire_external_colonies.py"))
RANGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RANGE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(path.read_bytes())


def rank(value):
    return sha((SEED + "|" + value).encode())


def read(path):
    return json.loads(path.read_text())


def write_new_or_same(path, obj):
    """Keep pre-existing artifacts immutable unless content is identical."""
    if path.exists():
        assert read(path) == obj, f"Existing artifact changed: {path}"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def valid_payload(path, entry):
    if not path.exists():
        return False
    assert not path.is_symlink(), str(path)
    data = path.read_bytes()
    assert len(data) == entry["size"], f"Size mismatch: {path}"
    assert zlib.crc32(data) & 0xffffffff == entry["CRC"], f"CRC mismatch: {path}"
    return True


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    identity_path = WORK / "isolate-records.json"
    identity = read(identity_path)
    # Explicit allowlist: discard all count/clinical fields before sampling.
    rows = [{k: r[k] for k in ("image", "species", "isolate")} for r in identity["records"]]
    assert len(rows) == 950 and len({r["image"] for r in rows}) == 950
    image_entries = {PurePosixPath(e["filename"]).name: e for e in read(WORK / "images.zip-directory.json") if e["filename"].endswith(".jpg")}
    xml_entries = {PurePosixPath(e["filename"]).stem: e for e in read(WORK / "label.zip-directory.json") if e["filename"].endswith(".xml")}
    assert len(image_entries) == len(xml_entries) == 950
    assert set(image_entries) == {r["image"] for r in rows}
    old_protocol = read(OLD / "selection-before-inference.json")
    old_names = {r["filename"] for r in old_protocol["selected"]}
    exposed = {r["isolate"] for r in rows if r["image"] in old_names}
    assert exposed == set(identity["exposedIsolatesFrom76"]) and len(exposed) == 57
    by_species = collections.defaultdict(set)
    by_isolate = collections.defaultdict(list)
    isolate_species = {}
    for r in rows:
        assert r["image"] == PurePosixPath(r["image"]).name
        assert PurePosixPath(image_entries[r["image"]]["filename"]).parent.name == r["species"]
        assert isolate_species.setdefault(r["isolate"], r["species"]) == r["species"]
        by_species[r["species"]].add(r["isolate"])
        by_isolate[r["isolate"]].append(r)
    assert len(by_species) == 19 and len(by_isolate) == 151
    split_isolates = {s: [] for s in SPLITS}
    for species, isolates in sorted(by_species.items()):
        unexposed = sorted(isolates - exposed, key=rank)
        assert len(unexposed) >= 2, f"Insufficient unexposed isolates: {species}"
        split_isolates["test"].append(unexposed[0])
        split_isolates["val"].append(unexposed[1])
        split_isolates["train"].extend(sorted(isolates - set(unexposed[:2])))
    assert {s: len(v) for s, v in split_isolates.items()} == {"train": 113, "val": 19, "test": 19}
    selected = []
    for split in SPLITS:
        for isolate in sorted(split_isolates[split]):
            chosen = sorted(by_isolate[isolate], key=lambda r: rank(r["image"]))[:1 if split == "train" else 2]
            for r in chosen:
                selected.append({**r, "split": split, "imageArchiveEntry": image_entries[r["image"]]})
    assert len({r["image"] for r in selected}) == len(selected)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(split_isolates[a]) & set(split_isolates[b])
    assert not (set(split_isolates["val"]) | set(split_isolates["test"])) & exposed
    # One deterministic original XML crosscheck per species, no annotation filtering.
    xml_check = [min((r for r in selected if r["species"] == species), key=lambda r: rank("xml-check|" + r["image"]))["image"] for species in sorted(by_species)]
    reused_names = [r["image"] for r in selected if (OLD / "images" / r["image"]).exists()]
    new_rows = [r for r in selected if r["image"] not in reused_names]
    reused_xml = [n for n in xml_check if (OLD / "labels" / (Path(n).stem + ".xml")).exists()]
    estimate = {
        "imageCount": len(selected),
        "imagesPerSplit": dict(collections.Counter(r["split"] for r in selected)),
        "reusedOldOriginalImages": len(reused_names),
        "newOriginalImages": len(new_rows),
        "localUncompressedImageBytes": sum(r["imageArchiveEntry"]["size"] for r in selected),
        "newCompressedImagePayloadBytes": sum(r["imageArchiveEntry"]["compressedSize"] for r in new_rows),
        "xmlCrosschecks": len(xml_check),
        "newCompressedXMLPayloadBytes": sum(xml_entries[Path(n).stem]["compressedSize"] for n in xml_check if n not in reused_xml),
        "note": "Estimate excludes small ZIP header/HTTP overhead. Local copies preserve old images without shared write ownership."
    }
    protocol = {
        "schema": "CountScope.isolate-cohort.v1", "seed": SEED,
        "sourceRecord": "https://zenodo.org/records/18256864",
        "sourcePaper": "https://www.nature.com/articles/s41597-026-07095-5",
        "selectionRule": "Within each species, rank previously unexposed isolate codes by SHA256(seed|isolate): first=newTest, second=newVal; all other isolates=newTrain. Rank each isolate's original filenames by SHA256(seed|filename), take one train or up to two val/test. No target-count, morphology or model-output selection.",
        "augmentationArchivesExcluded": ["images_agu.zip", "label_agu.zip"],
        "inputSHA256": {"isolateRecords": digest(identity_path), "imageDirectory": digest(WORK / "images.zip-directory.json"), "xmlDirectory": digest(WORK / "label.zip-directory.json"), "old76SelectionProtocol": digest(OLD / "selection-before-inference.json"), "old76Manifest": digest(OLD / "manifest.json"), "sourceCOCO": digest(SOURCE_COCO)},
        "exposedOld76Isolates": sorted(exposed), "isolates": split_isolates,
        "selected": selected, "xmlCrosscheckImages": xml_check, "downloadEstimate": estimate,
        "testPolicy": "New test isolate-disjoint from new train/val and all old76 exposed isolates. Old63 and old76 remain regression/exposed sets, not fresh blind evaluation. No inference in this preparation.",
        "oldData": "Old247train/56val may be mixed later with newTrain/newVal; not copied, reassigned or modified here.",
        "coordinatePolicy": "Preserve source COCO xywh numerics and convert to floating-point xyxy without fabricated precision. XML used only for independent <=1px rounding-tolerance crosscheck.",
    }
    if PROTOCOL.exists():
        frozen = read(PROTOCOL)
        assert {k: v for k, v in frozen.items() if k != "frozenUTC"} == protocol
        return frozen
    protocol["frozenUTC"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_new_or_same(PROTOCOL, protocol)
    return protocol


def extract_checked(meta, entry):
    # Add filename verification while reusing the reviewed bounded range extractor.
    head = RANGE.get_range(meta["links"]["self"], meta["size"], entry["headerOffset"], 30)
    fields = struct.unpack("<4s5H3I2H", head)
    assert fields[0] == b"PK\x03\x04"
    name_bytes = RANGE.get_range(meta["links"]["self"], meta["size"], entry["headerOffset"] + 30, fields[-2])
    name = name_bytes.decode("utf-8" if fields[2] & 0x800 else "cp437")
    assert name == entry["filename"], (name, entry["filename"])
    return RANGE.extract_one(meta, entry)


def obtain(path, oldpath, meta, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid_payload(path, entry):
        return "already_complete"
    if valid_payload(oldpath, entry):
        data = oldpath.read_bytes()
        origin = "reused_old_original"
    else:
        data = extract_checked(meta, entry)
        origin = "anonymous_bounded_range"
    assert len(data) == entry["size"] and zlib.crc32(data) & 0xffffffff == entry["CRC"]
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    return origin


def perfect_coordinate_match(xml_boxes, coco_boxes):
    assert len(xml_boxes) == len(coco_boxes)
    adjacent = [[j for j, b in enumerate(coco_boxes) if max(abs(x-y) for x,y in zip(a,b)) <= 1.0000001] for a in xml_boxes]
    matched = {}
    def add(i, visited):
        for j in adjacent[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in matched or add(matched[j], visited):
                matched[j] = i
                return True
        return False
    assert all(add(i, set()) for i in sorted(range(len(adjacent)), key=lambda j: len(adjacent[j])))
    return max((max(abs(x-y) for x,y in zip(xml_boxes[i],coco_boxes[j])) for j,i in matched.items()), default=0)


def acquire(protocol):
    # Verify frozen inputs before acquisition or reading annotation content.
    assert digest(SOURCE_COCO) == protocol["inputSHA256"]["sourceCOCO"]
    source = read(WORK / "zenodo-18256864-metadata.json")
    image_meta = next(f for f in source["files"] if f["key"] == "images.zip")
    xml_meta = next(f for f in source["files"] if f["key"] == "label.zip")
    xml_entries = {PurePosixPath(e["filename"]).stem: e for e in read(WORK / "label.zip-directory.json") if e["filename"].endswith(".xml")}
    def download(row):
        path = OUT / "images" / row["split"] / row["image"]
        result = obtain(path, OLD / "images" / row["image"], image_meta, row["imageArchiveEntry"])
        return row["image"], result
    origins = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, r) for r in protocol["selected"]]
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            name, origin = f.result()
            origins[name] = origin
            print(f"Acquired {i+1}/{len(futures)} {name} {origin}", flush=True)
    for name in protocol["xmlCrosscheckImages"]:
        stem = Path(name).stem
        obtain(OUT / "qa-source-xml" / (stem + ".xml"), OLD / "labels" / (stem + ".xml"), xml_meta, xml_entries[stem])
    coco = read(SOURCE_COCO)
    ci = {r["file_name"]: r for r in coco["images"]}
    categories = {r["id"]: r["name"] for r in coco["categories"]}
    ca = collections.defaultdict(list)
    assert len(ci) == len(coco["images"]) == 950
    assert len({r["id"] for r in coco["annotations"]}) == len(coco["annotations"])
    for a in coco["annotations"]:
        ca[a["image_id"]].append(a)
    old_hashes = {r["sha256"] for r in read(ROOT / "output/datasets/colony-figshare-22022540-v3/yolo-species-v1/manifest.json")["images"]}
    records = []
    xml_results = []
    for row in protocol["selected"]:
        filename, split = row["image"], row["split"]
        path = OUT / "images" / split / filename
        valid_payload(path, row["imageArchiveEntry"])
        source_image = ci[filename]
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            assert image.size == (source_image["width"], source_image["height"]), filename
            assert image.getexif().get(274, 1) == 1, filename
            assert image.mode == "RGB", (filename, image.mode)
        boxes = []
        for a in ca[source_image["id"]]:
            assert categories[a["category_id"]] == row["species"], filename
            x, y, w, h = map(float, a["bbox"])
            box = [x, y, x+w, y+h]
            assert all(math.isfinite(v) for v in box), filename
            assert 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height, (filename, box)
            assert a.get("iscrowd", 0) == 0, filename
            boxes.append(box)
        assert len({tuple(b) for b in boxes}) == len(boxes), filename
        image_hash = digest(path)
        assert image_hash not in old_hashes, filename
        label = OUT / "labels" / split / (Path(filename).stem + ".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        yolo = "".join("0 " + " ".join(f"{v:.12f}" for v in ((b[0]+b[2])/(2*width), (b[1]+b[3])/(2*height), (b[2]-b[0])/width, (b[3]-b[1])/height)) + "\n" for b in boxes)
        if label.exists():
            assert label.read_text() == yolo
        else:
            label.write_text(yolo)
        # Verify the exported YOLO coordinates round-trip to original pixel boxes.
        for line, b in zip(yolo.splitlines(), boxes):
            _, cx, cy, bw, bh = map(float, line.split())
            back = [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]
            assert max(abs(a-bb) for a,bb in zip(back,b)) < 1e-7
        record = {"filename": filename, "species": row["species"], "isolate": row["isolate"], "split": split, "imagePath": str(path.relative_to(ROOT)), "labelPath": str(label.relative_to(ROOT)), "width": width, "height": height, "sha256": image_hash, "labelSHA256": digest(label), "count": len(boxes), "xyxy": boxes, "coordinateSource": "source COCO bbox xywh converted to float xyxy"}
        records.append(record)
        if filename in protocol["xmlCrosscheckImages"]:
            xml_path = OUT / "qa-source-xml" / (Path(filename).stem + ".xml")
            xml = ET.fromstring(xml_path.read_bytes())
            assert xml.findtext("filename") == filename
            assert int(xml.findtext("size/width")) == width and int(xml.findtext("size/height")) == height
            assert all(o.findtext("name") == row["species"] for o in xml.findall("object"))
            xb = [[float(o.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")] for o in xml.findall("object")]
            delta = perfect_coordinate_match(xb, boxes)
            xml_results.append({"filename": filename, "species": row["species"], "isolate": row["isolate"], "matched": True, "maxCoordinateDifferencePx": delta, "xmlSHA256": digest(xml_path)})
    hashes = {s: {r["sha256"] for r in records if r["split"] == s} for s in SPLITS}
    assert len({r["sha256"] for r in records}) == len(records), "Duplicate original bytes"
    for a,b in (("train","val"),("train","test"),("val","test")):
        assert not hashes[a] & hashes[b]
    # Protect old datasets from any accidental mutation in this preparation.
    assert digest(OLD / "manifest.json") == protocol["inputSHA256"]["old76Manifest"]
    assert digest(OLD / "selection-before-inference.json") == protocol["inputSHA256"]["old76SelectionProtocol"]
    manifest = {"status": "READY_ISOLATE_DISJOINT_CLINICAL_COHORT_NO_INFERENCE", "sourceRecord": protocol["sourceRecord"], "sourcePaper": protocol["sourcePaper"], "selectionProtocolSHA256": digest(PROTOCOL), "sourceCOCOSHA256": digest(SOURCE_COCO), "images": records, "imageCounts": dict(collections.Counter(r["split"] for r in records)), "isolateCounts": {s: len(v) for s,v in protocol["isolates"].items()}, "speciesPerSplit": {s: len({r["species"] for r in records if r["split"] == s}) for s in SPLITS}, "allImageBytesAndCRCsChecked": True, "augmentationFilesIncluded": False, "dataOnlyNoInference": True}
    write_new_or_same(OUT / "manifest.json", manifest)
    write_new_or_same(OUT / "train-val-manifest.json", {
        "status": "READY_TRAIN_VAL_ONLY_NO_TEST_CONTENT",
        "sealedFullManifestSHA256": digest(OUT / "manifest.json"),
        "source": "zenodo18256864-clinical-originals",
        "sourceRecord": protocol["sourceRecord"],
        "imageCounts": {s: manifest["imageCounts"][s] for s in ("train", "val")},
        "isolateCounts": {s: manifest["isolateCounts"][s] for s in ("train", "val")},
        "images": [{**r, "source": "zenodo18256864-clinical-originals"} for r in records if r["split"] in ("train", "val")],
        "testContentIncluded": False,
        "coordinateSource": "Source COCO numeric xywh converted to float xyxy; YOLO labels use class0=colony."
    })
    write_new_or_same(OUT / "xml-crosscheck.json", {"status": "PASS", "method": "One SHA-ranked image per species; order-independent perfect matching with <=1px source-format rounding tolerance", "imagesChecked": len(xml_results), "results": xml_results})
    report = {"status": manifest["status"], "selectionProtocol": str(PROTOCOL.relative_to(ROOT)), "selectionProtocolSHA256": digest(PROTOCOL), "manifest": str((OUT / "manifest.json").relative_to(ROOT)), "manifestSHA256": digest(OUT / "manifest.json"), "scriptSHA256": digest(Path(__file__)), "cohortImages": manifest["imageCounts"], "cohortIsolates": manifest["isolateCounts"], "speciesPerSplit": manifest["speciesPerSplit"], "downloadEstimate": protocol["downloadEstimate"], "checks": {"selectionUsesOnlyIdentityHashes": True, "trainValTestIsolateIntersectionsZero": True, "testAndValVsOldExposedIsolatesIntersectionZero": True, "sameRawSHAAcrossSplitsZero": True, "sameRawSHAVsOldFigshareZero": True, "originalArchiveOnly": True, "allImageSizeCRCDecodeEXIFDimensionsAndCoordinatesValid": True, "allYOLOLabelsPixelRoundTripVerified": True, "xmlCOCOPerfectMatchingPassed": True, "old76ManifestAndProtocolUnchanged": True}, "coordinatePrecision": "COCO source values are represented as float; no precision beyond the author-provided values is claimed.", "patientData": "Only anonymized image/species/isolate codes are included; no patient fields are imported.", "testPolicy": protocol["testPolicy"], "mixingPolicy": protocol["oldData"], "limitations": ["Controlled professional imaging, not phone field acceptance.", "Isolate separation does not prove separation of every acquisition session or plate.", "Full remote ZIP archive MD5 unverified; each selected member size and CRC plus local SHA256 verified.", "Raw SHA comparison does not exclude resized/re-encoded or unknown pretraining overlap."], "modelInferencePerformed": False, "imagesOrSourceAnnotationsModified": False, "issues": []}
    write_new_or_same(OUT / "dataset-prep-report.json", report)
    print(json.dumps({"status": report["status"], "images": report["cohortImages"], "isolates": report["cohortIsolates"], "manifestSHA256": report["manifestSHA256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquire", action="store_true")
    args = parser.parse_args()
    protocol = prepare()
    print(json.dumps({"status": "SELECTION_FROZEN", "protocolSHA256": digest(PROTOCOL), "downloadEstimate": protocol["downloadEstimate"]}), flush=True)
    if args.acquire:
        try:
            acquire(protocol)
        except Exception as exc:
            failure = {"status": "STOPPED_ON_ANOMALY_NO_REPLACEMENT", "type": type(exc).__name__, "error": str(exc), "protocolSHA256": digest(PROTOCOL), "modelInferencePerformed": False}
            p = OUT / ("failure-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".json")
            p.write_text(json.dumps(failure, indent=2) + "\n")
            raise


if __name__ == "__main__":
    main()
