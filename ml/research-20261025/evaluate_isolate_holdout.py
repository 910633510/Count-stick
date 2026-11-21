#!/usr/bin/env python3
"""Open the isolate holdout only after independently reviewed development PASS.

No candidate ranking or threshold search is allowed here. A failed development
selection returns without opening the holdout manifest, images, or labels.
"""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ml/colony'))
import train_isolate_mixture as op
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from evaluate import iou_matrix
from ultralytics import YOLO

RUN = ROOT / 'output/research-20261025/mixed-training'
DATA = ROOT / 'output/datasets/colony-mixed-isolate-v1'
OUT = ROOT / 'output/research-20261025/isolate-holdout'
EXPECTED_MANIFEST = '96069c2c6f0741a21c48e3da13bb3e94fefd29477cdfea1e481cb23b35c43bd0'


def score(predictions, records, confidence):
    rows = []
    for r in records:
        assert r['split'] == 'test'
        p = predictions[r['filename']]
        truth = np.asarray(r['xyxy'], dtype=float).reshape(-1, 4)
        boxes = np.asarray(p['xyxy'], dtype=float).reshape(-1, 4)
        boxes = boxes[np.asarray(p['confidence']) >= confidence]
        overlaps = iou_matrix(truth, boxes)
        matched = set()
        if overlaps.size:
            ii, jj = linear_sum_assignment(-((overlaps >= .5) * 1001 + overlaps))
            matched = {int(i) for i, j in zip(ii, jj) if overlaps[i, j] >= .5}
        sizes = {k: {'groundTruth': 0, 'TP': 0}
                 for k in ['under16px', '16to32px', 'over32px']}
        for i, b in enumerate(truth):
            side = min(b[2] - b[0], b[3] - b[1]) * 1024 / max(r['width'], r['height'])
            key = 'under16px' if side < 16 else '16to32px' if side < 32 else 'over32px'
            sizes[key]['groundTruth'] += 1
            sizes[key]['TP'] += int(i in matched)
        tp = len(matched)
        rows.append({'id': r['filename'], 'source': r['source'], 'species': r['species'],
                     'truth': len(truth), 'predicted': len(boxes),
                     'signedError': len(boxes) - len(truth), 'TP': tp,
                     'FP': len(boxes) - tp, 'FN': len(truth) - tp, 'sizeRecall': sizes})
    result = op.summary_rows(rows)
    species = {s: op.summary_rows([r for r in rows if r['species'] == s])
               for s in sorted({r['species'] for r in rows})}
    result['perSpecies'] = species
    for metric in ['countMAE', 'precision', 'recall', 'F1']:
        result['speciesMacro' + metric[0].upper() + metric[1:]] = float(
            np.mean([s[metric] for s in species.values()]))
    return {'summary': result, 'images': rows}


def predictions(weights, pipeline, tag, records):
    model = YOLO(str(weights))
    result = {'modelSHA256': op.sha(weights), 'pipeline': pipeline, 'predictions': {}}
    for i, r in enumerate(records):
        result['predictions'][r['filename']] = op.infer_image(model, ROOT / r['imagePath'], pipeline)
        op.save(OUT / (tag + '.json'), result)
        if (i + 1) % 6 == 0:
            print(tag, i + 1, '/', len(records), flush=True)
    return result['predictions']


def main():
    selection = op.read(RUN / 'selection.json')
    if selection['selectedConfig'] is None:
        print('SKIPPED: no development candidate passed; holdout remains unopened.')
        return
    assert selection['status'] == 'PASS_DEVELOPMENT_GATES_PENDING_EXTERNAL_TEST'
    review_path = ROOT / 'ml/research-20261025/mixed-independent-review.json'
    review = op.read(review_path)
    assert review.get('status', '').startswith('PASS'), review.get('status')
    assert review.get('finalCandidateAudit', {}).get('status', '').startswith('PASS'), 'Final selection requires independent review'
    assert not (OUT / 'opened-for-inference.json').exists(), 'Holdout has already been opened'
    chosen = selection['selectedConfig']
    weights = Path(chosen['modelPath'])
    assert op.sha(weights) == chosen['modelSHA256']
    baseline = op.read(RUN / 'baseline-binding.json')['selectedBaseline']
    assert op.sha(op.INIT) == baseline['modelSHA256']
    protocol = {
        'registeredBeforeOpeningUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'selectionSHA256': op.sha(RUN / 'selection.json'),
        'independentReviewSHA256': op.sha(review_path),
        'expectedFullManifestSHA256': EXPECTED_MANIFEST,
        'candidate': chosen,
        'baseline': {k: baseline[k] for k in ['pipeline', 'confidence', 'modelSHA256']},
        'comparison': 'Same 36 images, chosen model and configuration only. Old weights whole/tile at frozen baseline confidence.',
        'scope': 'Same clinical study as new training data; isolate-disjoint, not an external-study or smartphone test.',
        'gates': 'At least 5% species-macro MAE gain; image MAE <= baseline+max(0.5,5%); P/R drop <=1pp; F1 no drop; FP <=baseline+max(2,ceil(1%GT)); species P drop<=2pp; species/size TP >= stronger old whole/tile TP-2%GT-1.',
        'noFurtherThresholdOrModelSelection': True,
        'negativeControls': 'Eligibility inherited from the frozen development PASS; not rerun or rescored in this holdout.',
        'appDeploymentApproved': False,
        'evaluatorSHA256': op.sha(__file__),
        'inferenceOperatorSHA256': op.sha(op.__file__),
    }
    op.save(OUT / 'protocol-before-test.json', protocol)
    assert op.sha(DATA / 'manifest.json') == EXPECTED_MANIFEST
    manifest = op.read(DATA / 'manifest.json')
    records = [{**r, 'source': 'zenodo18256864-clinical-originals'}
               for r in manifest['images'] if r['split'] == 'test']
    assert len(records) == 36 and len({r['isolate'] for r in records}) == 19
    for r in records:
        required = {'filename', 'species', 'isolate', 'split', 'imagePath', 'labelPath',
                    'width', 'height', 'sha256', 'labelSHA256', 'xyxy', 'count', 'source'}
        assert required <= r.keys(), required - r.keys()
        assert r['split'] == 'test' and len(r['xyxy']) == r['count']
        assert r['width'] > 0 and r['height'] > 0
        boxes = np.asarray(r['xyxy'], dtype=float).reshape(-1, 4)
        assert boxes.shape == (r['count'], 4)
        assert np.isfinite(boxes).all()
        assert op.sha(ROOT / r['imagePath']) == r['sha256']
        assert op.sha(ROOT / r['labelPath']) == r['labelSHA256']
    op.save(OUT / 'opened-for-inference.json', {
        'openedAtUTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'protocolSHA256': op.sha(OUT / 'protocol-before-test.json'),
        'futureUse': 'Exposed regression cohort after this one fixed comparison.'})
    torch.set_num_threads(8)
    metrics = {}
    for mode in ['whole', 'automatic4tile']:
        pred = predictions(op.INIT, mode, 'baseline-' + mode, records)
        metrics['baseline-' + mode] = score(pred, records, baseline['confidence'])
    pred = predictions(weights, chosen['pipeline'], 'candidate', records)
    metrics['candidate'] = score(pred, records, chosen['confidence'])
    # Apply the already frozen development gates to this single clinical source.
    source = records[0]['source']
    def wrap(result):
        sub = result['summary']
        return {'summary': {'primaryMacroMAE': sub['speciesMacroCountMAE'],
                            'perSource': {source: sub}}, 'negativeCounts': {}}
    b = metrics['baseline-' + baseline['pipeline']]
    gates = op.gates(wrap(metrics['candidate']), wrap(b),
                     [wrap(metrics['baseline-' + m]) for m in ['whole', 'automatic4tile']])
    ap_records = [{**r, 'id': r['filename']} for r in records]
    metrics['candidate']['summary'].update(op.ap_predictions(pred, ap_records))
    report = {'status': 'HOLDOUT_' + gates['status'], 'metrics': metrics, 'gates': gates,
              'protocolSHA256': op.sha(OUT / 'protocol-before-test.json'),
              'testImages': 36, 'testIsolates': 19, 'deploymentApproved': False,
              'limitations': 'Small controlled-image holdout; no iPhone, real-world capture or CoreML acceptance.'}
    op.save(OUT / 'report.json', report)
    op.save(ROOT / 'ml/research-20261025/isolate-holdout-results.json', report)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
