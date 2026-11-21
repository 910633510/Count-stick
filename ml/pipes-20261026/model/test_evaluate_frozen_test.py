"""Synthetic/stdlib-only contract checks. No real dataset or model is opened."""
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("evaluate_frozen_test.py")
spec = importlib.util.spec_from_file_location("frozen_pipe_test", PATH)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TestGuardContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.out = self.root / "output/pipes-20261026/model"
        self.source = self.root / "ml/pipes-20261026/model"
        self.patches = [patch.object(h, "ROOT", self.root), patch.object(h, "OUT", self.out), patch.object(h, "SOURCE", self.source)]
        for p in self.patches:
            p.start()
        self.addCleanup(self.tmp.cleanup)
        for p in self.patches:
            self.addCleanup(p.stop)

    def fixture(self):
        for name in h.FROZEN_CODE:
            p = self.source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# synthetic source binding only\n")
        p = self.root / "ml/colony/mps_compat.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic source binding only\n")
        for rel in h.APP_FILES.values():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("// synthetic source binding only\n")
        (self.root / h.APP_FILES["PipeLetterbox"]).write_text(
            "struct PipeLetterbox {\n let padding = 114\n}\n"
            "\nenum PipeCountRegion { }\n"
            "options.computeUnits = .all\n"
            "guard label.confidence.isFinite else { fail() }\n"
            "guard Double(label.confidence) >= threshold else { continue }\n"
            "let score = Double(label.confidence)\n")
        code = h.code_hashes()
        protocol = {"evaluationVariants": h.VARIANTS, "candidateCount": 18,
                    "checkpointEpochs": [8, 14, 20], "confidenceGrid": [.2, .3, .4], "nmsIoUGrid": [.5, .7],
                    "appSourceCodeSHA256": {k: h.sha(self.root / v) for k, v in h.APP_FILES.items()},
                    "qualification": {"precisionMin": .95}}
        write(self.out / "protocol.json", protocol)
        models = {}
        for epoch in protocol["checkpointEpochs"]:
            path = self.out / "checkpoints" / ("epoch%03d.pt" % epoch)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic not loadable weights " + str(epoch)).encode())
            meta = {"epoch": epoch, "sha256": h.sha(path), "protocolSHA256": h.sha(self.out / "protocol.json")}
            write(path.with_suffix(".json"), meta)
            models[str(epoch)] = meta
        candidates = [{"epoch": e, "confidence": c, "nmsIoU": i, "modelSHA256": models[str(e)]["sha256"],
                       "status": "PASS", "variants": {v: {"gates": {"status": "PASS"}} for v in h.VARIANTS}}
                      for e in [8, 14, 20] for c in [.2, .3, .4] for i in [.5, .7]]
        chosen = candidates[0]
        selection = {"status": "VAL_QUALIFIED_PENDING_FROZEN_SCENE_TEST", "selected": chosen,
                     "candidates": candidates, "codeSHA256": code, "protocolSHA256": h.sha(self.out / "protocol.json")}
        write(self.out / "selection.json", selection)
        audit = {"status": "PASS_SYNTHETIC_CONTRACT_ONLY", "selectionSHA256": h.sha(self.out / "selection.json"),
                 "modelSHA256": chosen["modelSHA256"], "protocolSHA256": selection["protocolSHA256"],
                 "codeSHA256": code, "selected": chosen}
        audit_path = self.root / "synthetic-validation-audit.json"
        write(audit_path, {"validationAudit": audit})
        receipt = {k: audit[k] for k in ["selectionSHA256", "modelSHA256", "protocolSHA256", "codeSHA256"]}
        helper_path = self.root / "SyntheticDedicatedTestHelper.swift"
        helper_path.write_text("// synthetic dedicated helper binding only\n")
        receipt.update({"status": "APPROVED_PIPE_FROZEN_TEST_SINGLE_CANDIDATE", "evaluatorSHA256": h.sha(h.SELF),
                        "epoch": chosen["epoch"], "operatingPoint": {"confidence": .2, "nmsIoU": .5},
                        "inputVariants": h.VARIANTS, "sealedManifestPath": h.SEALED_REL,
                        "sealedManifestSHA256": h.SEALED_SHA, "expectedTestImages": 37,
                        "validationAuditPath": str(audit_path), "validationAuditSHA256": h.sha(audit_path),
                        "testExportHelperPath": str(helper_path), "testExportHelperSHA256": h.sha(helper_path)})
        receipt_path = self.root / "synthetic-root-approval.json"
        write(receipt_path, receipt)
        write(self.out / "training-complete.json", {"status": "COMPLETE_20_EPOCHS_WITH_ACTUAL_REGISTERED_CHECKPOINTS", "checkpoints": models})
        return receipt_path, receipt, audit_path, audit

    def runtime_fixture(self):
        path, receipt, *_ = self.fixture()
        app_path = self.root / h.APP_FILES["PipeLetterbox"]
        old_source = app_path.read_bytes()
        snapshot = self.root / "preserved-runtime-v1/PipeDetector.swift"
        snapshot.parent.mkdir()
        snapshot.write_bytes(old_source)
        app_path.write_bytes(old_source.replace(b"options.computeUnits = .all", b"options.computeUnits = .cpuAndNeuralEngine")
                             .replace(b"label.confidence", b"object.confidence"))
        old = h.read(self.out / "protocol.json")["appSourceCodeSHA256"]
        current = {k: h.sha(self.root / v) for k, v in h.APP_FILES.items()}
        common = {key: receipt[key] for key in ["protocolSHA256", "selectionSHA256", "modelSHA256", "codeSHA256", "operatingPoint"]}
        common.update({"oldAppSourceCodeSHA256": old, "newAppSourceCodeSHA256": current,
                       "allowedChanges": h.RUNTIME_V2_CHANGES, "pixelGeometryUnchanged": True})
        training_receipt_path = self.root / "synthetic-training-receipt.json"
        write(training_receipt_path, {"status": "SYNTHETIC_ONLY"})
        write(self.out / "training-authorization.json", {"rootReceiptPath": str(training_receipt_path), "rootReceiptSHA256": h.sha(training_receipt_path)})
        package_path = self.out / "export/Synthetic.mlpackage"
        package_path.mkdir(parents=True)
        (package_path / "synthetic.txt").write_text("not a model")
        export = {"modelSHA256": receipt["modelSHA256"], "selectionSHA256": receipt["selectionSHA256"],
                  "confidenceThreshold": .2, "iouThreshold": .5, "packagePath": str(package_path),
                  "packageTreeSHA256": h.tree_sha(package_path), "exportID": "synthetic-export-id"}
        write(self.out / "export/export-status.json", export)
        common.update({"trainingRootReceiptSHA256": h.sha(training_receipt_path),
                       "exportStatusSHA256": h.sha(self.out / "export/export-status.json"),
                       "packageTreeSHA256": export["packageTreeSHA256"], "exportID": export["exportID"]})
        parity_path = self.root / "synthetic-runtime-parity.json"
        write(parity_path, {**{key: receipt[key] for key in ["protocolSHA256", "selectionSHA256", "modelSHA256", "operatingPoint"]},
                           "status": "PASS_PIPE_TORCH_COREML_VISION_PARITY", "inputVariants": h.VARIANTS,
                           "packageTreeSHA256": export["packageTreeSHA256"], "exportID": export["exportID"]})
        amendment = {**common, "status": "FROZEN_PIPE_RUNTIME_V2_AMENDMENT", "version": 2,
                     "oldPipeDetectorSnapshotPath": str(snapshot), "oldPipeDetectorSHA256": h.sha(snapshot),
                     "newPipeDetectorSHA256": h.sha(app_path)}
        amendment_path = self.root / "runtime-v2-amendment.json"
        review_path = self.root / "runtime-v2-review.json"
        write(amendment_path, amendment)
        review = {**common, "status": "PASS_PIPE_RUNTIME_V2_SOURCE_AMENDMENT", "amendmentSHA256": h.sha(amendment_path),
                  "parityReportPath": str(parity_path), "parityReportSHA256": h.sha(parity_path)}
        write(review_path, review)
        receipt.update({"runtimeAmendmentPath": str(amendment_path), "runtimeAmendmentSHA256": h.sha(amendment_path),
                        "runtimeAmendmentReviewPath": str(review_path), "runtimeAmendmentReviewSHA256": h.sha(review_path)})
        write(path, receipt)
        return path, receipt, amendment_path, amendment, review_path, review

    def rewrite_runtime_bindings(self, fixture):
        path, receipt, amendment_path, amendment, review_path, review = fixture
        write(amendment_path, amendment)
        review["amendmentSHA256"] = h.sha(amendment_path)
        write(review_path, review)
        receipt.update({"runtimeAmendmentSHA256": h.sha(amendment_path), "runtimeAmendmentReviewSHA256": h.sha(review_path)})
        write(path, receipt)

    def runtime_v3_fixture(self):
        fixture = self.runtime_fixture()
        path, receipt, amendment_path, amendment, review_path, review = fixture
        app = self.root / h.APP_FILES["PipeLetterbox"]
        app.write_text(app.read_text() + "// synthetic reviewed raw decoder, no executable model\n")
        nms = self.root / "Count stick/SyntheticPipeNMS.swift"
        nms.write_text("enum SyntheticPipeNMS {}\n")
        original_export_path = self.out / "export/export-status.json"
        prior = h.read(original_export_path)
        package = self.out / "export-raw-v3/SyntheticRaw.mlpackage"
        package.mkdir(parents=True)
        (package / "synthetic.txt").write_text("raw synthetic fixture, not a model")
        raw_export = {**prior, "packagePath": str(package), "packageTreeSHA256": h.tree_sha(package),
                      "exportID": "synthetic-raw-v3", "nms": False, "quantize": 32,
                      "inputSize": 1024, "rawLocationCapacity": 21504}
        raw_export_path = self.out / "export-raw-v3/export-status.json"
        write(raw_export_path, raw_export)
        for document in [amendment, review]:
            document.update({"allowedChanges": h.RUNTIME_V3_CHANGES,
                             "exportStatusPath": str(raw_export_path.relative_to(self.root)),
                             "exportStatusSHA256": h.sha(raw_export_path), "exportID": raw_export["exportID"],
                             "packageTreeSHA256": raw_export["packageTreeSHA256"],
                             "priorExportStatusSHA256": h.sha(original_export_path),
                             "priorPackageTreeSHA256": prior["packageTreeSHA256"],
                             "nmsSourcePath": str(nms.relative_to(self.root)), "nmsSourceSHA256": h.sha(nms),
                             "newAppSourceCodeSHA256": {**document["newAppSourceCodeSHA256"], "PipeLetterbox": h.sha(app)}})
        amendment.update({"status": "FROZEN_PIPE_RUNTIME_V3_AMENDMENT", "version": 3, "newPipeDetectorSHA256": h.sha(app)})
        review["status"] = "PASS_PIPE_RUNTIME_V3_SOURCE_AMENDMENT"
        parity_path = Path(review["parityReportPath"])
        parity = h.read(parity_path)
        parity.update({"runtimeVersion": 3, "nmsSourceSHA256": h.sha(nms),
                       "exportID": raw_export["exportID"], "packageTreeSHA256": raw_export["packageTreeSHA256"]})
        write(parity_path, parity)
        review["parityReportSHA256"] = h.sha(parity_path)
        self.rewrite_runtime_bindings(fixture)
        return fixture

    def test_null_returns_before_receipt_protocol_test_or_ml_imports(self):
        selection_path = self.out / "selection.json"
        write(selection_path, {"selected": None, "status": "FAIL_VALIDATION_QUALIFICATION"})
        real_read = h.read
        reads = []
        def restricted(path):
            reads.append(Path(path))
            self.assertEqual(Path(path), selection_path)
            return real_read(path)
        modules = set(sys.modules)
        with patch.object(h, "read", restricted):
            self.assertEqual(h.authorize(self.root / "absent-approval")["status"], "SKIP_NO_VALIDATION_QUALIFIED_CANDIDATE")
            self.assertEqual(h.evaluate(self.root / "absent-approval", None, None)["status"], "SKIP_NO_VALIDATION_QUALIFIED_CANDIDATE")
        self.assertEqual(reads, [selection_path, selection_path])
        self.assertFalse((self.out / "test").exists())
        self.assertFalse({"torch", "numpy", "pipe_ml", "ultralytics"} & (set(sys.modules) - modules))

    def test_authorize_never_reads_missing_sealed_data_and_is_exclusive(self):
        receipt, *_ = self.fixture()
        self.assertFalse((self.root / h.SEALED_REL).exists())
        self.assertEqual(h.authorize(receipt)["status"], "FROZEN_SINGLE_PIPE_TEST_PROTOCOL_BEFORE_TEST_ACCESS")
        self.assertFalse((self.out / "test/test-opened.json").exists())
        with self.assertRaises(FileExistsError):
            h.authorize(receipt)

    def test_gate_rejects_each_root_binding_change(self):
        path, receipt, *_ = self.fixture()
        for key, value in {"status": "PENDING", "modelSHA256": "0" * 64,
                           "protocolSHA256": "0" * 64, "evaluatorSHA256": "0" * 64,
                           "epoch": 20, "expectedTestImages": 36,
                           "inputVariants": ["gallery"], "operatingPoint": {"confidence": .4, "nmsIoU": .5}}.items():
            with self.subTest(field=key):
                changed = {**receipt, key: value}
                write(path, changed)
                with self.assertRaises(ValueError):
                    h.gate(path)
        self.assertFalse((self.out / "test").exists())

    def test_failed_independent_audit_is_rejected_even_if_receipt_hash_matches(self):
        path, receipt, audit_path, audit = self.fixture()
        audit["status"] = "PENDING"
        write(audit_path, {"validationAudit": audit})
        receipt["validationAuditSHA256"] = h.sha(audit_path)
        write(path, receipt)
        with self.assertRaisesRegex(ValueError, "not PASS"):
            h.gate(path)

    def test_independent_audit_cannot_bind_other_selected_candidate(self):
        path, receipt, audit_path, audit = self.fixture()
        audit["selected"] = copy.deepcopy(audit["selected"])
        audit["selected"]["confidence"] = .4
        write(audit_path, {"validationAudit": audit})
        receipt["validationAuditSHA256"] = h.sha(audit_path)
        write(path, receipt)
        with self.assertRaisesRegex(ValueError, "selected candidate mismatch"):
            h.gate(path)

    def test_source_and_weight_mutation_rejected_before_test(self):
        path, *_ = self.fixture()
        source = self.source / "evaluate_pipe_ends.py"
        original = source.read_bytes()
        source.write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            h.gate(path)
        source.write_bytes(original)
        (self.out / "checkpoints/epoch008.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checkpoint bytes changed"):
            h.gate(path)

    def test_incomplete_training_is_rejected(self):
        path, *_ = self.fixture()
        completion_path = self.out / "training-complete.json"
        completion = h.read(completion_path)
        completion["status"] = "PARTIAL"
        write(completion_path, completion)
        with self.assertRaisesRegex(ValueError, "Training incomplete"):
            h.gate(path)

    def test_input_arguments_required_before_opening(self):
        path, *_ = self.fixture()
        h.authorize(path)
        with self.assertRaisesRegex(ValueError, "inputs/audit required"):
            h.evaluate(path, None, None)
        self.assertFalse((self.out / "test/test-opened.json").exists())

    def test_test_opened_marker_precedes_heavy_import_and_blocks_second_run(self):
        path, *_ = self.fixture()
        h.authorize(path)
        importer = builtins.__import__
        def refuse_ml(name, *args, **kwargs):
            if name in {"numpy", "torch", "PIL", "ultralytics", "evaluate_pipe_ends"}:
                self.assertTrue((self.out / "test/test-opened.json").is_file())
                raise ImportError("synthetic stop before any ML/data read")
            return importer(name, *args, **kwargs)
        with patch("builtins.__import__", refuse_ml):
            with self.assertRaisesRegex(ImportError, "synthetic stop"):
                h.evaluate(path, "absent-inputs", "absent-audit")
        self.assertEqual(h.read(self.out / "test/test-failure.json")["status"], "ERROR_TEST_OPENED_NO_QUALIFICATION_REPORT")
        with self.assertRaises(FileExistsError):
            h.evaluate(path, "absent-inputs", "absent-audit")
        self.assertFalse((self.root / h.SEALED_REL).exists())

    def test_runtime_v2_exact_changes_pass_and_keep_training_hash_lineage(self):
        path, receipt, *_ = self.runtime_fixture()
        binding = h.gate(path)
        self.assertEqual(binding["runtimeVersion"], 2)
        self.assertNotEqual(binding["trainingAppSourceCodeSHA256"]["PipeLetterbox"], binding["appSourceCodeSHA256"]["PipeLetterbox"])
        self.assertEqual(binding["trainingAppSourceCodeSHA256"]["PhotoProcessing"], binding["appSourceCodeSHA256"]["PhotoProcessing"])
        h.authorize(path)
        registered = h.read(self.out / "test/pre-test-protocol.json")
        self.assertEqual(registered["bindings"]["runtimeAmendmentSHA256"], receipt["runtimeAmendmentSHA256"])
        self.assertFalse((self.root / h.SEALED_REL).exists())

    def test_runtime_change_without_explicit_amendment_rejected(self):
        path, receipt, *_ = self.runtime_fixture()
        receipt = {key: value for key, value in receipt.items() if not key.startswith("runtime")}
        write(path, receipt)
        with self.assertRaisesRegex(ValueError, "without runtime amendment"):
            h.gate(path)

    def test_runtime_amendment_and_review_bytes_bound(self):
        fixture = self.runtime_fixture()
        path, receipt, amendment_path, amendment, review_path, review = fixture
        original = amendment_path.read_bytes()
        amendment_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "Runtime amendment changed"):
            h.gate(path)
        amendment_path.write_bytes(original)
        review_path.write_bytes(review_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Runtime amendment review changed"):
            h.gate(path)

    def test_runtime_independent_review_must_pass(self):
        fixture = self.runtime_fixture()
        fixture[5]["status"] = "PENDING"
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "independent review not PASS"):
            h.gate(fixture[0])

    def test_runtime_cannot_change_operating_threshold_even_with_bound_review(self):
        fixture = self.runtime_fixture()
        for document in [fixture[3], fixture[5]]:
            document["operatingPoint"] = {"confidence": .4, "nmsIoU": .5}
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "frozen model/protocol/threshold"):
            h.gate(fixture[0])

    def test_runtime_old_snapshot_must_match_frozen_training_hash(self):
        fixture = self.runtime_fixture()
        snapshot = Path(fixture[3]["oldPipeDetectorSnapshotPath"])
        snapshot.write_text(snapshot.read_text() + "// hidden old edit\n")
        fixture[3]["oldPipeDetectorSHA256"] = h.sha(snapshot)
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "Old PipeDetector snapshot mismatch"):
            h.gate(fixture[0])

    def test_runtime_extra_geometry_change_rejected_even_if_metadata_is_consistent(self):
        fixture = self.runtime_fixture()
        app = self.root / h.APP_FILES["PipeLetterbox"]
        app.write_text(app.read_text().replace("padding = 114", "padding = 0"))
        for document in [fixture[3], fixture[5]]:
            document["newAppSourceCodeSHA256"] = {**document["newAppSourceCodeSHA256"], "PipeLetterbox": h.sha(app)}
        fixture[3]["newPipeDetectorSHA256"] = h.sha(app)
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "exact two-change allowlist"):
            h.gate(fixture[0])

    def test_runtime_photo_processing_change_rejected_even_if_metadata_is_consistent(self):
        fixture = self.runtime_fixture()
        app = self.root / h.APP_FILES["PhotoProcessing"]
        app.write_text(app.read_text() + "// changed resizing\n")
        for document in [fixture[3], fixture[5]]:
            document["newAppSourceCodeSHA256"] = {**document["newAppSourceCodeSHA256"], "PhotoProcessing": h.sha(app)}
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "PhotoProcessing or original export helper changed"):
            h.gate(fixture[0])

    def test_runtime_failed_parity_rejected_with_matching_receipt_hashes(self):
        fixture = self.runtime_fixture()
        parity_path = Path(fixture[5]["parityReportPath"])
        parity = h.read(parity_path)
        parity["status"] = "FAIL_RUNTIME_ALL"
        write(parity_path, parity)
        fixture[5]["parityReportSHA256"] = h.sha(parity_path)
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "Runtime parity is not PASS"):
            h.gate(fixture[0])

    def test_runtime_package_change_rejected(self):
        fixture = self.runtime_fixture()
        (self.out / "export/Synthetic.mlpackage/synthetic.txt").write_text("changed model package")
        with self.assertRaisesRegex(ValueError, "CoreML package changed"):
            h.gate(fixture[0])

    def test_runtime_v3_raw_export_and_shared_nms_hash_bound(self):
        fixture = self.runtime_v3_fixture()
        binding = h.gate(fixture[0])
        self.assertEqual(binding["runtimeVersion"], 3)
        self.assertEqual(binding["runtimeNMS"]["sha256"], fixture[3]["nmsSourceSHA256"])
        nms = self.root / fixture[3]["nmsSourcePath"]
        nms.write_text(nms.read_text() + "// changed threshold\n")
        with self.assertRaisesRegex(ValueError, "shared NMS source changed"):
            h.gate(fixture[0])

    def test_runtime_v3_rejects_embedded_nms_even_with_consistent_metadata(self):
        fixture = self.runtime_v3_fixture()
        path = self.root / fixture[3]["exportStatusPath"]
        export = h.read(path)
        export["nms"] = True
        write(path, export)
        for document in [fixture[3], fixture[5]]:
            document["exportStatusSHA256"] = h.sha(path)
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "without embedded NMS"):
            h.gate(fixture[0])

    def test_runtime_v3_rejects_geometry_change_despite_approval_metadata(self):
        fixture = self.runtime_v3_fixture()
        app = self.root / h.APP_FILES["PipeLetterbox"]
        app.write_text(app.read_text().replace("padding = 114", "padding = 0"))
        for document in [fixture[3], fixture[5]]:
            document["newAppSourceCodeSHA256"] = {**document["newAppSourceCodeSHA256"], "PipeLetterbox": h.sha(app)}
        fixture[3]["newPipeDetectorSHA256"] = h.sha(app)
        self.rewrite_runtime_bindings(fixture)
        with self.assertRaisesRegex(ValueError, "PipeLetterbox geometry/pixels changed"):
            h.gate(fixture[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
