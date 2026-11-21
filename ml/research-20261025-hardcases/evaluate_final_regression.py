#!/usr/bin/env python3
"""One fixed-candidate OLD63 regression, after a hash-bound final audit.

No test data or ML framework is read/imported by the no-selection branch.
The data fingerprint binding contains identities/hashes only, never GT boxes.
Once protocol-before-old63.json exists this entry point cannot be run again.
Use --self-test for synthetic-only verification and --snapshot-code to freeze
local source dependencies. Neither command opens real test data.
"""
from pathlib import Path
import argparse
import ast
import collections
import copy
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUN_REL = Path('output/research-20261025-hardcases/retention-savefix-v2')
OUT_REL = RUN_REL / 'final-old63-regression'
OLD_REL = Path('output/datasets/colony-figshare-22022540-v3/yolo-species-v1')
ANCHOR_REL = Path('output/training-20261025/runs/colony-yolo11n-v2-cpuassign/weights/best.pt')
HISTORICAL_REL = ANCHOR_REL.parent.parent / 'evaluation'
REVIEW_REL = Path('ml/research-20261025-hardcases/independent-retention-review.json')
SNAPSHOT_REL = Path('ml/research-20261025-hardcases/final-regression-dependencies.json')
# Root's immutable fingerprints-only binding remains at its original path.
BINDING_REL = Path('output/research-20261025-hardcases/retention/final-regression-input-binding.json')
ANCHOR_SHA = '91d70c3067f2fdb77d2ba2541e6105bf3caf730da7bbc32e5f4d2c9310b0b337'
OLD_MANIFEST_SHA = 'f28b0409993893f4874845c77a06207a5f98f5c620b1a5b478b733de64e1d8fa'
# Supplied by root from a fingerprints-only operation; this implementation did
# not open the fingerprint file or any OLD63 image/label/prediction values.
INPUT_BINDING_SHA = 'f139fde6c8b0fdeef9782b1226f74f8c079530aca4964f630d5174d77ba1eac4'
FIXED = {'pipeline': 'automatic4tile', 'confidence': .3, 'nmsIoU': .7,
         'imgsz': 1024, 'maxDetections': 1000}
EPS = 1e-9
LEGACY_BOUNDARY_EPS_PX = 1e-4  # Validate serialization rounding; NEVER clip GT.
DEPENDENCIES = (
    'ml/research-20261025-hardcases/evaluate_final_regression.py',
    'ml/research-20261025-hardcases/train_retention_savefix.py',
    'ml/research-20261025-hardcases/evaluate_retention_savefix.py',
    'ml/colony/train_isolate_mixture.py',
    'ml/colony/evaluate.py',
    'ml/colony/mps_compat.py',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def save_progress(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(temporary, path)


def checked_bytes(path, expected):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected, 'Fingerprint changed: ' + str(path))
    return payload


def local_hashes(root):
    return {name: sha(root / name) for name in DEPENDENCIES}


def snapshot_code(root):
    result = {'status': 'FROZEN_LOCAL_CODE_NO_TEST_ACCESS', 'createdUTC': now(),
              'filesSHA256': local_hashes(root),
              'allowedRunDirectory': str(RUN_REL),
              'expectedInputBindingSHA256FromRoot': INPUT_BINDING_SHA,
              'testDataRead': False, 'modelInferencePerformed': False}
    write_exclusive(root / SNAPSHOT_REL, result)
    return result


def verify_snapshot(root, snapshot):
    require(snapshot.get('status') == 'FROZEN_LOCAL_CODE_NO_TEST_ACCESS', 'Invalid code snapshot status')
    require(snapshot.get('allowedRunDirectory') == str(RUN_REL), 'Code snapshot belongs to a different attempt')
    require(snapshot.get('expectedInputBindingSHA256FromRoot') == INPUT_BINDING_SHA, 'Fingerprint binding expectation changed')
    require(snapshot.get('filesSHA256') == local_hashes(root), 'Local code dependency drift; independent re-review required')


def resolve_repo_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    require(resolved.is_relative_to(root.resolve()), 'Path outside repository')
    return resolved


def validate_run_directory(root, value):
    """Only the root-authorized technical retry is eligible, never attempt1."""
    if value is None or str(value) == RUN_REL.name:
        return root / RUN_REL
    path = resolve_repo_path(root, value)
    require(path == (root / RUN_REL).resolve(), 'Only retention-savefix-v2 is an authorized final-regression run')
    return path


def check_hash_string(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'Expected a lowercase SHA256 fingerprint')


def validate_binding(root, binding):
    require(binding.get('status') == 'FROZEN_OLD63_FINGERPRINTS_NO_INFERENCE', 'Fingerprint binding not frozen')
    paths = {
        'oldManifest': OLD_REL / 'manifest.json',
        'baselineCache': HISTORICAL_REL / 'test-predictions.json',
        'baselineSelection': HISTORICAL_REL / 'selection-before-test.json',
        'baselineInferenceProtocol': HISTORICAL_REL / 'inference-protocol-before-test.json',
        'anchor': ANCHOR_REL,
    }
    for name, relative in paths.items():
        require(binding[name + 'Path'] == str(relative), 'Unexpected fixed input path: ' + name)
        check_hash_string(binding[name + 'SHA256'])
    require(binding['oldManifestSHA256'] == OLD_MANIFEST_SHA, 'Historical dataset identity changed')
    require(binding['anchorSHA256'] == ANCHOR_SHA, 'Historical model identity changed')
    rows = binding.get('records', [])
    require(len(rows) == 63, 'Fingerprint cohort must contain exactly OLD63')
    required = {'filename', 'imagePath', 'labelPath', 'width', 'height', 'species',
                'imageSHA256', 'labelSHA256'}
    require(len({r['filename'] for r in rows}) == 63, 'Duplicate OLD63 identity')
    for r in rows:
        require(set(r) == required, 'Fingerprint rows must contain metadata only, no GT/count/predictions')
        name = r['filename']
        require(Path(name).name == name and name.endswith('.jpg'), 'Unsafe image identity')
        require(r['imagePath'] == str(OLD_REL / 'images/test' / name), 'Image must retain split=test')
        require(r['labelPath'] == str(OLD_REL / 'labels/test' / Path(name).with_suffix('.txt')), 'Label must retain split=test')
        require(r['species'] == name[:4], 'Species identity mismatch')
        require(isinstance(r['width'], int) and isinstance(r['height'], int)
                and r['width'] > 0 and r['height'] > 0, 'Invalid dimensions')
        check_hash_string(r['imageSHA256']); check_hash_string(r['labelSHA256'])
    # Deliberately no is_file/hash/read on any bound test path here.
    return binding


def verify_eligibility(root, selection, selection_sha):
    require(selection.get('status') == 'PASS_VALIDATION_PENDING_FIXED_OLD63_REGRESSION', 'Selection did not PASS')
    chosen = selection['selectedConfig']
    require(chosen is not None, 'No selected candidate')
    require(selection.get('candidateCount') == 12 and len(selection.get('candidates', [])) == 12,
            'All twelve validation candidates must be completed')
    require(selection.get('old63ReadOrInferred') is False and selection.get('new36ReadOrInferred') is False,
            'Selection used protected test data')
    require(selection.get('appModelChanged') is False, 'Unexpected app mutation')
    require(selection['selectedResult']['config'] == chosen and selection['selectedResult']['gates']['status'] == 'PASS',
            'Selected config is not the validated PASS config')
    expected_grid = {(e, a) for e in (3, 5, 8) for a in (.05, .15, .35, 1.)}
    require({(x['config']['epoch'], x['config']['alpha']) for x in selection['candidates']} == expected_grid,
            'Candidate grid differs from frozen twelve')
    require((chosen['epoch'], chosen['alpha']) in expected_grid and chosen['alpha'] > 0,
            'Not an eligible trained candidate')
    require(chosen['pipeline'] == FIXED['pipeline'] and chosen['confidence'] == FIXED['confidence'],
            'Pipeline/threshold switching prohibited')
    expected_model = root / RUN_REL / 'candidates' / ('epoch%03d-alpha%g.pt' % (chosen['epoch'], chosen['alpha']))
    require(resolve_repo_path(root, chosen['path']) == expected_model.resolve(), 'Unexpected model path')
    check_hash_string(chosen['sha256'])
    require(chosen['anchorSHA256'] == ANCHOR_SHA, 'Selected blend has wrong anchor')

    review_path = root / REVIEW_REL
    review = read_json(review_path)
    require(str(review.get('status', '')).startswith('PASS'), 'Independent review has not passed')
    audit = review.get('finalCandidateAudit', {})
    require(str(audit.get('status', '')).startswith('PASS'), 'Final candidate audit has not passed')
    require(audit.get('selectionSHA256') == selection_sha, 'Final audit is not bound to this selection')
    require(audit.get('candidateCount') == 12 and audit.get('selectedConfig') == chosen,
            'Final audit must cover twelve candidates and this exact selected config')
    run = root / RUN_REL
    bindings = {'protocolSHA256': run / 'protocol.json', 'recordsSHA256': run / 'records.json',
                'operatorSHA256': root / DEPENDENCIES[1], 'evaluatorSHA256': root / DEPENDENCIES[2]}
    for key, path in bindings.items():
        require(audit.get(key) == selection.get(key) == sha(path), 'Final audit/code/data binding drift: ' + key)
    protocol = read_json(run / 'protocol.json')
    require(protocol['frozenOperationalPipeline'] == FIXED, 'Operational configuration changed')
    require(protocol['new36TestAccessAllowed'] is False and protocol['appChangesAllowed'] is False,
            'Unexpected scope change')
    snapshot = read_json(root / SNAPSHOT_REL)
    verify_snapshot(root, snapshot)
    authorization = read_json(run / 'training-authorization.json')
    for key in ('operatorSHA256', 'evaluatorSHA256', 'protocolSHA256'):
        require(authorization[key] == selection[key], 'Training authorization mismatch: ' + key)
    # Hashes come from this run's selection and training authorization. Optional
    # explicit code paths must also agree with the allowed retry's code pair.
    for document in (selection, authorization):
        for key, relative in (('operatorPath', DEPENDENCIES[1]), ('evaluatorPath', DEPENDENCIES[2])):
            if key in document:
                require(resolve_repo_path(root, document[key]) == (root / relative).resolve(), 'Wrong code pair for selected run')
    require(authorization['dataBindingSHA256'] == sha(run / 'data-binding.json'), 'Training data binding changed')
    require(authorization['localDependencySHA256'] == {p: sha(root / p) for p in DEPENDENCIES[3:]},
            'Inference/scoring helper changed after training authorization')
    require(sha(resolve_repo_path(root, authorization['rootReceiptPath'])) == authorization['rootReceiptSHA256'],
            'Root training receipt changed')
    data_binding = read_json(run / 'data-binding.json')
    require(data_binding['recordsSHA256'] == selection['recordsSHA256'], 'Records changed after training')
    require(sha(resolve_repo_path(root, data_binding['manifestPath'])) == data_binding['manifestSHA256'],
            'Hardcase annotation manifest changed')
    completion = read_json(run / 'training-complete.json')
    require(completion['status'] == 'COMPLETE_8_EPOCHS_CLEAN_EXIT', 'Training not complete')
    require(completion['protocolSHA256'] == selection['protocolSHA256'], 'Training protocol changed')
    require(sha(expected_model) == chosen['sha256'] and sha(root / ANCHOR_REL) == ANCHOR_SHA,
            'Model bytes changed')
    require(sha(root / BINDING_REL) == INPUT_BINDING_SHA, 'Root OLD63 fingerprint binding changed')
    fingerprint = validate_binding(root, read_json(root / BINDING_REL))
    versions = runtime_versions()
    require(all(versions.get(p) for p in ('numpy', 'scipy', 'Pillow', 'torch', 'torchvision', 'ultralytics')),
            'Inference runtime packages unavailable; use the verified training environment before opening OLD63')
    return {'chosen': chosen, 'reviewSHA256': sha(review_path), 'snapshot': snapshot,
            'authorizationSHA256': sha(run / 'training-authorization.json'),
            'trainingCompletionSHA256': sha(run / 'training-complete.json'), 'binding': fingerprint,
            'runtimeVersions': versions}


def parse_yolo(text, width, height):
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = list(map(float, line.split()))
        require(len(parts) == 5 and parts[0] == 0 and all(math.isfinite(v) for v in parts), 'Invalid frozen YOLO label')
        _, x, y, w, h = parts
        box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
        require(w > 0 and h > 0 and box[0] >= -LEGACY_BOUNDARY_EPS_PX and box[1] >= -LEGACY_BOUNDARY_EPS_PX
                and box[2] <= width+LEGACY_BOUNDARY_EPS_PX and box[3] <= height+LEGACY_BOUNDARY_EPS_PX,
                'GT outside documented serialization tolerance')
        boxes.append(box)  # No clipping, relabeling or split reassignment.
    return boxes


def load_old63_after_protocol(root, binding):
    from PIL import Image
    manifest = json.loads(checked_bytes(root / binding['oldManifestPath'], binding['oldManifestSHA256']))
    require(set(manifest['splits']['test']) == {r['filename'] for r in binding['records']}, 'OLD63 membership changed')
    metadata = {r['file']: r for r in manifest['images']}
    provenance = json.loads(checked_bytes(root / binding['baselineSelectionPath'], binding['baselineSelectionSHA256']))
    require(provenance['modelSHA256'] == ANCHOR_SHA and provenance['manifestSHA256'] == OLD_MANIFEST_SHA,
            'Historical reference binding invalid')
    require(provenance['selectedPipeline'] == 'automatic4tile' and provenance['selectedConfidence'] == .3
            and provenance['nmsIoU'] == .7, 'Cached reference is not oldv2 automatic4tile/conf0.3')
    prior_protocol = json.loads(checked_bytes(root / binding['baselineInferenceProtocolPath'], binding['baselineInferenceProtocolSHA256']))
    require(prior_protocol['modelSHA256'] == ANCHOR_SHA and prior_protocol['manifestSHA256'] == OLD_MANIFEST_SHA,
            'Historical inference protocol mismatch')
    reference = json.loads(checked_bytes(root / binding['baselineCachePath'], binding['baselineCacheSHA256']))
    require(set(reference) == set(manifest['splits']['test']), 'Baseline cache is not exactly OLD63')
    records = []
    for r in binding['records']:
        m = metadata[r['filename']]
        require((m['width'], m['height'], m['sha256']) == (r['width'], r['height'], r['imageSHA256']),
                'Fingerprint/manifest image identity mismatch')
        require(r['species'] == 'sp%02d' % m['species'], 'Historical species mismatch')
        label = checked_bytes(root / r['labelPath'], r['labelSHA256']).decode('utf-8')
        boxes = parse_yolo(label, r['width'], r['height'])
        require(len(boxes) == m['validBoxCount'], 'Frozen GT count changed')
        require(sha(root / r['imagePath']) == r['imageSHA256'], 'Frozen OLD63 image bytes changed')
        with Image.open(root / r['imagePath']) as image:
            require(image.size == (r['width'], r['height']) and image.getexif().get(274, 1) in (None, 1),
                    'Image dimensions/orientation changed')
        records.append({**r, 'id': r['filename'], 'split': 'test', 'source': 'figshare22022540v3', 'xyxy': boxes})
    return records, reference


def iou_matrix(a, b):
    import numpy as np
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(high-low, 0).prod(2)
    return intersection / (np.maximum(a[:, 2:]-a[:, :2], 0).prod(1)[:, None]
                           + np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)[None, :] - intersection + 1e-9)


def matched_gt(overlaps):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not overlaps.size:
        return set()
    ii, jj = linear_sum_assignment(-((overlaps >= .5).astype(float)*1001 + overlaps))
    return {int(i) for i, j in zip(ii, jj) if overlaps[i, j] >= .5}


def summary_rows(rows):
    require(bool(rows), 'Cannot summarize an empty cohort')
    tp, fp, fn = (sum(r[k] for r in rows) for k in ('TP', 'FP', 'FN'))
    errors = [r['signedError'] for r in rows]
    summary = {'images': len(rows), 'groundTruthObjects': tp+fn, 'TP': tp, 'FP': fp, 'FN': fn,
               'precision': tp/max(1, tp+fp), 'recall': tp/max(1, tp+fn), 'F1': 2*tp/max(1, 2*tp+fp+fn),
               'countMAE': sum(abs(e) for e in errors)/len(errors),
               'countRMSE': math.sqrt(sum(e*e for e in errors)/len(errors)),
               'countBias': sum(errors)/len(errors), 'sizeRecall': {}}
    for key in ('under16px', '16to32px', 'over32px'):
        gt = sum(r['sizeRecall'][key]['groundTruth'] for r in rows)
        stp = sum(r['sizeRecall'][key]['TP'] for r in rows)
        summary['sizeRecall'][key] = {'groundTruth': gt, 'TP': stp, 'FN': gt-stp, 'recall': stp/gt if gt else None}
    return summary


def score_test(predictions, records, confidence=.3):
    """General OLD63 scorer: retains split=test, never calls val-only scorer."""
    import numpy as np
    require(set(predictions) == {r['id'] for r in records}, 'Prediction membership differs from test records')
    rows = []
    for r in records:
        require(r['split'] == 'test', 'This final-regression scorer requires split=test')
        truth = np.asarray(r['xyxy'], dtype=float).reshape(-1, 4)
        p = predictions[r['id']]
        boxes = np.asarray(p['xyxy'], dtype=float).reshape(-1, 4)
        scores = np.asarray(p['confidence'], dtype=float)
        require(scores.shape == (len(boxes),) and len(boxes) <= 1000, 'Invalid prediction shape/count')
        require(np.isfinite(boxes).all() and np.isfinite(scores).all()
                and ((scores >= 0) & (scores <= 1)).all(), 'Nonfinite/out-of-range model output')
        require(((boxes[:, 2:] - boxes[:, :2]) >= 0).all(), 'Negative prediction extent')
        boxes = boxes[scores >= confidence]
        matched = matched_gt(iou_matrix(truth, boxes))
        sizes = {k: {'groundTruth': 0, 'TP': 0} for k in ('under16px', '16to32px', 'over32px')}
        for i, box in enumerate(truth):
            side = min(box[2]-box[0], box[3]-box[1])*1024/max(r['width'], r['height'])
            key = 'under16px' if side < 16 else '16to32px' if side < 32 else 'over32px'
            sizes[key]['groundTruth'] += 1
            sizes[key]['TP'] += int(i in matched)
        rows.append({'id': r['id'], 'source': r['source'], 'species': r['species'], 'split': 'test',
                     'truth': len(truth), 'predicted': len(boxes), 'signedError': len(boxes)-len(truth),
                     'TP': len(matched), 'FP': len(boxes)-len(matched), 'FN': len(truth)-len(matched), 'sizeRecall': sizes})
    summary = summary_rows(rows)
    summary['perSource'] = {}
    for source in sorted({r['source'] for r in rows}):
        subset = [r for r in rows if r['source'] == source]
        sub = summary_rows(subset)
        sub['perSpecies'] = {s: summary_rows([r for r in subset if r['species'] == s])
                             for s in sorted({r['species'] for r in subset})}
        sub['speciesMacroCountMAE'] = sum(x['countMAE'] for x in sub['perSpecies'].values())/len(sub['perSpecies'])
        summary['perSource'][source] = sub
    summary['primaryMacroMAE'] = sum(s['speciesMacroCountMAE'] for s in summary['perSource'].values())/len(summary['perSource'])
    summary.update(confidence=confidence, matchIoU=.5)
    return {'summary': summary, 'images': rows}


def old_nonregression(candidate, baseline):
    """Same zero-deterioration contract as retention old_nonregression."""
    c, b = candidate['summary'], baseline['summary']
    require(c['images'] == b['images'] and c['groundTruthObjects'] == b['groundTruthObjects'], 'Incomparable cohorts')
    gates = []
    def leq(name, value, maximum):
        gates.append({'gate': name, 'pass': bool(value <= maximum+EPS), 'actual': value, 'maximum': maximum})
    for key in ('countMAE', 'primaryMacroMAE', 'FP', 'FN'):
        leq('old63/'+key, c[key], b[key])
    require(set(c['perSource']) == set(b['perSource']), 'Source groups changed')
    for source, sub in c['perSource'].items():
        ref = b['perSource'][source]
        require(set(sub['perSpecies']) == set(ref['perSpecies']), 'Species groups changed')
        for species, metrics in sub['perSpecies'].items():
            for key in ('countMAE', 'FP', 'FN'):
                leq('old63/'+source+'/'+species+'/'+key, metrics[key], ref['perSpecies'][species][key])
        require(set(sub['sizeRecall']) == set(ref['sizeRecall']), 'Size bins changed')
        for size, metrics in sub['sizeRecall'].items():
            require(metrics['groundTruth'] == ref['sizeRecall'][size]['groundTruth'], 'Size-bin GT changed')
            if metrics['groundTruth']:
                leq('old63/'+source+'/'+size+'/FN', metrics['FN'], ref['sizeRecall'][size]['FN'])
    failures = [g['gate'] for g in gates if not g['pass']]
    return {'status': 'PASS' if not failures else 'FAIL', 'failedGates': failures, 'details': gates}


def runtime_versions():
    versions = {'python': sys.version.split()[0]}
    for name in ('numpy', 'scipy', 'Pillow', 'torch', 'torchvision', 'ultralytics'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run(root=ROOT, run_dir=None):
    # This is intentionally the first file access. Do not move imports, binding
    # reads, output-directory probes or any test-path checks above this return.
    run_directory = validate_run_directory(root, run_dir)
    selection_path = run_directory / 'selection.json'
    if not selection_path.is_file():
        return {'status': 'SKIPPED_SELECTION_NOT_READY', 'old63Opened': False, 'new36Accessed': False}
    selection = read_json(selection_path)
    require('selectedConfig' in selection, 'Selection missing selectedConfig')
    if selection['selectedConfig'] is None:
        return {'status': 'SKIPPED_NO_SELECTED_CANDIDATE', 'old63Opened': False, 'new36Accessed': False}
    selection_sha = sha(selection_path)
    eligible = verify_eligibility(root, selection, selection_sha)
    chosen, binding = eligible['chosen'], eligible['binding']
    out = root / OUT_REL
    require(not (out / 'protocol-before-old63.json').exists() and not (out / 'opened-for-inference.json').exists(),
            'OLD63 final comparison is single use; do not rerun or select another candidate')
    protocol = {'status': 'FROZEN_BEFORE_OLD63_PIXELS_GT_AND_CACHE_READ', 'registeredUTC': now(),
                'selectedRunDirectory': str(RUN_REL),
                'operatorPath': DEPENDENCIES[1], 'operatorSHA256': selection['operatorSHA256'],
                'validationEvaluatorPath': DEPENDENCIES[2], 'validationEvaluatorSHA256': selection['evaluatorSHA256'],
                'selectionSHA256': selection_sha, 'independentReviewSHA256': eligible['reviewSHA256'],
                'chosenModelAndConfig': chosen, 'fixedPipeline': FIXED,
                'old63FingerprintBindingSHA256': INPUT_BINDING_SHA, 'expectedOld63Inputs': binding,
                'codeDependencySHA256': eligible['snapshot']['filesSHA256'],
                'trainingAuthorizationSHA256': eligible['authorizationSHA256'],
                'trainingCompletionSHA256': eligible['trainingCompletionSHA256'], 'runtimeVersions': eligible['runtimeVersions'],
                'baseline': 'Existing oldv2 automatic4tile/conf0.3 cached OLD63 predictions only; no new baseline inference.',
                'gates': 'Aggregate MAE/species-macro MAE/FP/FN, every species MAE/FP/FN, every size-bin FN must not increase; EPS1e-9 only.',
                'legacyBoundarySerializationTolerancePx': LEGACY_BOUNDARY_EPS_PX,
                'GTClippedOrRelabeled': False, 'candidateSwitchAfterOpeningAllowed': False,
                'inferencePasses': 'One selected model, one call to automatic4tile per OLD63 image; no rerun/resume/threshold search.',
                'newClinical36AccessAllowed': False, 'appChangesAllowed': False,
                'meaning': 'Already exposed same-study regression, not a fresh blind test or phone acceptance.'}
    write_exclusive(out / 'protocol-before-old63.json', protocol)
    write_exclusive(out / 'opened-for-inference.json', {
        'status': 'OPENING_ONE_FIXED_OLD63_COMPARISON', 'openedUTC': now(),
        'protocolSHA256': sha(out / 'protocol-before-old63.json'), 'noAutomaticResume': True})
    try:
        records, reference = load_old63_after_protocol(root, binding)
        baseline = score_test(reference, records, FIXED['confidence'])
        # No training or clinical-holdout entry point is called. Import only
        # after eligibility + protocol + OLD63 fingerprint verification.
        sys.path.insert(0, str(root / 'ml/colony'))
        import torch
        from ultralytics import YOLO
        from train_isolate_mixture import infer_image
        verify_snapshot(root, eligible['snapshot'])
        require(sha(resolve_repo_path(root, chosen['path'])) == chosen['sha256'], 'Candidate changed before loading')
        torch.set_num_threads(8)
        model = YOLO(str(resolve_repo_path(root, chosen['path'])))
        prediction = {'status': 'IN_PROGRESS_FIXED_CANDIDATE', 'modelSHA256': chosen['sha256'],
                      'selectionSHA256': selection_sha, 'pipeline': FIXED, 'predictions': {}}
        with torch.inference_mode():
            for i, record in enumerate(records):
                path = root / record['imagePath']
                require(sha(path) == record['imageSHA256'], 'Image changed before inference')
                prediction['predictions'][record['id']] = infer_image(model, path, FIXED['pipeline'])
                require(sha(path) == record['imageSHA256'], 'Image changed during inference')
                save_progress(out / 'candidate-predictions.json', prediction)
                print('OLD63 fixed candidate', i+1, '/', len(records), flush=True)
        candidate = score_test(prediction['predictions'], records, FIXED['confidence'])
        gates = old_nonregression(candidate, baseline)
        verify_snapshot(root, eligible['snapshot'])
        require(sha(selection_path) == selection_sha and sha(root / REVIEW_REL) == eligible['reviewSHA256'], 'Selection/audit changed during comparison')
        require(sha(root / BINDING_REL) == INPUT_BINDING_SHA and sha(resolve_repo_path(root, chosen['path'])) == chosen['sha256'], 'Frozen binding/model changed')
        for record in records:
            require(sha(root / record['labelPath']) == record['labelSHA256'], 'GT changed during comparison')
        prediction['status'] = 'COMPLETE_ONE_FIXED_CANDIDATE'
        save_progress(out / 'candidate-predictions.json', prediction)
        report = {'status': 'PASS_FIXED_OLD63_REGRESSION' if gates['status'] == 'PASS' else 'FAIL_FIXED_OLD63_REGRESSION_REJECT_ROUND',
                  'finishedUTC': now(), 'protocolSHA256': sha(out / 'protocol-before-old63.json'),
                  'candidatePredictionSHA256': sha(out / 'candidate-predictions.json'),
                  'selectedConfig': chosen, 'baseline': baseline, 'candidate': candidate, 'gates': gates,
                  'images': 63, 'new36Accessed': False, 'appModelChanged': False, 'deploymentApproved': False,
                  'onFailure': 'Reject this round. Do not select another epoch, alpha or confidence using OLD63.',
                  'limitation': 'Known regression cohort; no fresh blind-test, CoreML or iPhone acceptance.'}
        write_exclusive(out / 'report.json', report)
        return report
    except Exception as exc:
        write_exclusive(out / 'failure.json', {'status': 'FAIL_FINAL_REGRESSION_INTEGRITY_OR_EXECUTION',
                         'errorType': type(exc).__name__, 'error': str(exc), 'old63MayHaveBeenOpened': True,
                         'new36Accessed': False, 'noAutomaticRetryOrModelSwitch': True, 'deploymentApproved': False})
        raise


def self_test(root=ROOT):
    """Only temporary JSON, synthetic boxes, and source AST; no image/model data."""
    import numpy as np
    evidence = {}
    global read_json, sha, verify_eligibility, load_old63_after_protocol
    real_read, real_sha = read_json, sha
    with tempfile.TemporaryDirectory(prefix='countscope-old63-guard-') as directory:
        fake = Path(directory)
        selection_path = fake / RUN_REL / 'selection.json'
        selection_path.parent.mkdir(parents=True)
        selection_path.write_text(json.dumps({'selectedConfig': None, 'status': 'FAIL_NO_CANDIDATE_PASSES'}))
        reads = []
        def guarded_read(path):
            require(Path(path) == selection_path, 'No-selection branch attempted another file read')
            reads.append(str(path)); return real_read(path)
        def forbidden_hash(path):
            raise RuntimeError('No-selection branch attempted hashing ' + str(path))
        try:
            read_json, sha = guarded_read, forbidden_hash
            result = run(fake)
            require(result['status'] == 'SKIPPED_NO_SELECTED_CANDIDATE' and len(reads) == 1, 'Null short-circuit failed')
        finally:
            read_json, sha = real_read, real_sha
        evidence['nullSelectionReadsOnlySelectionNoHashesHeavyImportsOrTestAccess'] = True
        require(not (fake / OUT_REL).exists(), 'Null selection created comparison output')
        configs = [{'epoch': e, 'alpha': a, 'path': str(fake / RUN_REL / 'candidates' / ('epoch%03d-alpha%g.pt' % (e, a))),
                    'sha256': 'a'*64, 'anchorSHA256': ANCHOR_SHA, 'pipeline': 'automatic4tile', 'confidence': .3}
                   for e in (3, 5, 8) for a in (.05, .15, .35, 1.)]
        chosen = configs[0]
        synthetic_selection = {'status': 'PASS_VALIDATION_PENDING_FIXED_OLD63_REGRESSION', 'selectedConfig': chosen,
                               'selectedResult': {'config': chosen, 'gates': {'status': 'PASS'}},
                               'candidateCount': 12, 'candidates': [{'config': c} for c in configs],
                               'old63ReadOrInferred': False, 'new36ReadOrInferred': False, 'appModelChanged': False,
                               'operatorSHA256': 'e'*64, 'evaluatorSHA256': 'f'*64}
        selection_path.write_text(json.dumps(synthetic_selection))
        review_path = fake / REVIEW_REL; review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps({'status': 'PASS_PRETRAINING_ONLY', 'finalCandidateAudit': {'status': 'PENDING'}}))
        audit_reads = []
        def audit_guard_read(path):
            require(Path(path) in (selection_path, review_path), 'Pending audit attempted another file read')
            audit_reads.append(Path(path)); return real_read(path)
        def audit_guard_hash(path):
            require(Path(path) == selection_path, 'Pending audit attempted test/model/dependency hash')
            return real_sha(path)
        try:
            read_json, sha = audit_guard_read, audit_guard_hash
            try:
                run(fake)
            except RuntimeError as exc:
                require(str(exc) == 'Final candidate audit has not passed', 'Unexpected pending-audit failure')
            else:
                raise RuntimeError('Pending final audit was accepted')
            require(audit_reads == [selection_path, review_path], 'Pending-audit read boundary changed')
            require(not (fake / OUT_REL).exists(), 'Pending final audit created opening marker')
            evidence['pendingFinalAuditBlocksBeforeBindingModelAndTestAccess'] = True
        finally:
            read_json, sha = real_read, real_sha

        # Dynamic ordering probe: the test loader is replaced by a stub that
        # never opens data, and demands both exclusive protocol files first.
        real_verify, real_loader = verify_eligibility, load_old63_after_protocol
        loader_calls = []
        def synthetic_eligibility(*_):
            return {'chosen': chosen, 'binding': {'syntheticFingerprintsOnly': True},
                    'reviewSHA256': 'b'*64, 'snapshot': {'filesSHA256': {}},
                    'authorizationSHA256': 'c'*64, 'trainingCompletionSHA256': 'd'*64,
                    'runtimeVersions': {'synthetic': 'no ML imports'}}
        def forbidden_real_loader(probe_root, _):
            require((probe_root / OUT_REL / 'protocol-before-old63.json').is_file(), 'Loader preceded protocol')
            require((probe_root / OUT_REL / 'opened-for-inference.json').is_file(), 'Loader preceded one-use marker')
            loader_calls.append(True)
            raise RuntimeError('SYNTHETIC_STOP_BEFORE_DATA_OPEN')
        try:
            verify_eligibility, load_old63_after_protocol = synthetic_eligibility, forbidden_real_loader
            try:
                run(fake)
            except RuntimeError as exc:
                require(str(exc) == 'SYNTHETIC_STOP_BEFORE_DATA_OPEN', 'Unexpected protocol-order probe failure')
            try:
                run(fake)
            except RuntimeError as exc:
                require('single use' in str(exc), 'Rerun did not fail at single-use lock')
            else:
                raise RuntimeError('One-use lock allowed a rerun')
            require(len(loader_calls) == 1, 'Second opening was attempted')
            evidence['exclusiveProtocolAndMarkerPrecedeTestLoader'] = True
            evidence['secondRunBlockedBeforeTestLoader'] = True
        finally:
            verify_eligibility, load_old63_after_protocol = real_verify, real_loader
        for name in DEPENDENCIES:
            p = fake / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('synthetic source\n')
        snapshot = {'status': 'FROZEN_LOCAL_CODE_NO_TEST_ACCESS', 'allowedRunDirectory': str(RUN_REL),
                    'expectedInputBindingSHA256FromRoot': INPUT_BINDING_SHA,
                    'filesSHA256': local_hashes(fake)}
        verify_snapshot(fake, snapshot)
        (fake / DEPENDENCIES[3]).write_text('tampered synthetic source\n')
        try:
            verify_snapshot(fake, snapshot)
        except RuntimeError:
            evidence['dependencyTamperRejected'] = True
        else:
            raise RuntimeError('Dependency drift was accepted')
        try:
            validate_run_directory(fake, 'output/research-20261025-hardcases/retention')
        except RuntimeError:
            evidence['incompleteOriginalAttemptCannotBeSelected'] = True
        else:
            raise RuntimeError('Incomplete original run was accepted')

    records = [{'id': 'synthetic-a', 'filename': 'synthetic-a', 'source': 'synthetic', 'species': 'a', 'split': 'test',
                'width': 100, 'height': 100, 'xyxy': [[10, 10, 11, 11], [30, 30, 34, 34]]},
               {'id': 'synthetic-b', 'filename': 'synthetic-b', 'source': 'synthetic', 'species': 'b', 'split': 'test',
                'width': 100, 'height': 100, 'xyxy': [[20, 20, 22, 22]]}]
    pred = {'synthetic-a': {'xyxy': [[10, 10, 11, 11]], 'confidence': [.3]},
            'synthetic-b': {'xyxy': [[20, 20, 22, 22]], 'confidence': [.9]}}
    baseline = score_test(pred, records)
    require(baseline['summary']['TP'] == 2 and baseline['summary']['FN'] == 1 and baseline['summary']['countMAE'] == .5,
            'Synthetic score mismatch')
    require(old_nonregression(baseline, baseline)['status'] == 'PASS', 'Baseline cannot pass itself')
    extra = copy.deepcopy(pred); extra['synthetic-a']['xyxy'].append([60, 60, 62, 62]); extra['synthetic-a']['confidence'].append(.9)
    require('old63/FP' in old_nonregression(score_test(extra, records), baseline)['failedGates'], 'Extra FP accepted')
    moved = copy.deepcopy(pred); moved['synthetic-a']['xyxy'] = [[30, 30, 34, 34]]
    require(any('/under16px/FN' in s for s in old_nonregression(score_test(moved, records), baseline)['failedGates']),
            'Size regression hidden by aggregate counts')
    swap = copy.deepcopy(pred); swap['synthetic-a'] = {'xyxy': records[0]['xyxy'], 'confidence': [.9, .9]}
    swap['synthetic-b'] = {'xyxy': [], 'confidence': []}
    swapped = score_test(swap, records)
    require(swapped['summary']['countMAE'] == baseline['summary']['countMAE'], 'Species regression fixture invalid')
    require('old63/synthetic/b/FN' in old_nonregression(swapped, baseline)['failedGates'], 'Species regression hidden by mean')
    bad = copy.deepcopy(records); bad[0]['split'] = 'val'
    try:
        score_test(pred, bad)
    except RuntimeError:
        evidence['splitTestRequiredNoLegacyValRelabeling'] = True
    else:
        raise RuntimeError('Val relabeling accepted')
    require(len(matched_gt(np.array([[.7, .6], [.6, .0]]))) == 2, 'Matching does not maximize valid cardinality')
    rounded = parse_yolo('0 0.5 0.5 1.000000001 1\n', 100, 100)
    require(rounded[0][0] < 0 and rounded[0][2] > 100, 'Legacy GT silently clipped')
    evidence.update(baselinePassesStrictGates=True, oneExtraFPRejected=True,
                    aggregateEqualSpeciesRegressionRejected=True, aggregateEqualSizeRegressionRejected=True,
                    cardinalityBeforeIoUMatching=True, legacyRoundingPreservedWithoutGTClipping=True)
    # Source-only parity with the existing validation gate, not real validation.
    tree = ast.parse((root / DEPENDENCIES[2]).read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('check_leq', 'old_nonregression')]
    require(len(definitions) == 2, 'Cannot locate reference pure gate functions')
    namespace = {'EPS': EPS}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<synthetic-gate-parity>', 'exec'), namespace)
    for candidate in (baseline, score_test(extra, records), score_test(moved, records), swapped):
        reference = namespace['old_nonregression'](candidate, baseline, prefix='old63')
        require(reference == old_nonregression(candidate, baseline)['details'], 'Strict gate differs from frozen retention gate')
    evidence['syntheticGateParityWithCurrentRetentionOldNonregression'] = True
    return {'status': 'PASS_SYNTHETIC_STATIC_ONLY_NO_TEST_ACCESS', 'createdUTC': now(), 'checks': evidence,
            'harnessSHA256': real_sha(Path(__file__)), 'referenceGateSourceSHA256': real_sha(root / DEPENDENCIES[2]),
            'actualOld63OrNew36DataRead': False, 'modelWeightsRead': False, 'modelInferencePerformed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--self-test', action='store_true')
    group.add_argument('--snapshot-code', action='store_true')
    parser.add_argument('--run-dir', default=RUN_REL.name,
                        help='Only retention-savefix-v2 (name, repo-relative or absolute path) is allowed.')
    args = parser.parse_args()
    validate_run_directory(ROOT, args.run_dir)
    if args.self_test:
        result = self_test()
        save_progress(ROOT / RUN_REL / 'final-regression-self-test.json', result)
    elif args.snapshot_code:
        result = snapshot_code(ROOT)
    else:
        result = run(run_dir=args.run_dir)
    print(json.dumps({k: v for k, v in result.items() if k not in ('baseline', 'candidate', 'filesSHA256')}, indent=2))


if __name__ == '__main__':
    main()
