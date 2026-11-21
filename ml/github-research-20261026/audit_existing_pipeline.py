#!/usr/bin/env python3
"""Read-only train/validation diagnosis; never load weights or test images."""
from pathlib import Path
import collections, hashlib, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/research-20261025-hardcases/retention-savefix-v2'
OUT = ROOT / 'ml/github-research-20261026'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = read(RUN / 'records.json')
    assert all(r['split'] in ['train', 'val'] for r in records)
    result = {}
    for role in sorted({r['role'] for r in records}):
        rows = [r for r in records if r['role'] == role]
        sizes, buckets = [], collections.Counter()
        geometry = collections.Counter()
        for r in rows:
            w, h = r['width'], r['height']
            tw, th = math.ceil(.6*w), math.ceil(.6*h)
            for x1, y1, x2, y2 in r['xyxy']:
                side = min(x2-x1, y2-y1)*1024/max(w, h)
                sizes.append(side)
                buckets['under16' if side < 16 else '16to32' if side < 32 else 'over32'] += 1
                accepted = []
                for row in range(2):
                    for col in range(2):
                        left, top = (0 if col == 0 else w-tw), (0 if row == 0 else h-th)
                        a, b, c, d = max(x1,left), max(y1,top), min(x2,left+tw), min(y2,top+th)
                        if a >= c or b >= d:
                            continue
                        cx, cy = (a+c)/2, (b+d)/2
                        if (cx < w/2 if col == 0 else cx >= w/2) and (cy < h/2 if row == 0 else cy >= h/2):
                            accepted.append((c-a)*(d-b)/((x2-x1)*(y2-y1)))
                geometry['noAcceptedVisibleBox'] += len(accepted) == 0
                geometry['multipleAcceptedVisibleBoxes'] += len(accepted) > 1
                geometry['bestAcceptedVisibleBoxAreaBelowHalfGT'] += bool(accepted) and max(accepted) < .5
                geometry['bestAcceptedVisibleBoxIsClipped'] += bool(accepted) and max(accepted) < 1-1e-6
        result[role] = {'images': len(rows), 'positiveImages':sum(bool(r['xyxy']) for r in rows),
                        'emptyImages':sum(not r['xyxy'] for r in rows), 'targets': len(sizes), 'boxesPerImageMean': len(sizes)/len(rows),
                        'sizeAtWholeImage1024': dict(buckets), 'medianMinSidePixels': float(np.median(sizes)) if sizes else None,
                        'hypotheticalExactVisibleBoxGeometry': dict(geometry)}
    exposures = collections.Counter()
    image_exposures = collections.Counter()
    for path in sorted((RUN / 'sampling').glob('epoch*.json')):
        report = read(path)
        for group, count in report['actualPostAugmentationLossBoxCounts'].items():
            bucket, source, *_ = group.split('::')
            key = 'old' if bucket == 'old_anchor' else 'clinical' if source == 'zenodo18256864-clinical-originals' else 'new_hard_positive' if bucket == 'hard_positive' else 'empty'
            exposures[key] += count
        for group, count in report['sourceSpeciesImageCounts'].items():
            bucket, source, *_ = group.split('::')
            key = 'old' if bucket == 'old_anchor' else 'clinical' if source == 'zenodo18256864-clinical-originals' else 'new_hard_positive' if bucket == 'hard_positive' else 'empty'
            image_exposures[key] += count
    output = {'status': 'READ_ONLY_TRAIN_VAL_AUDIT', 'recordsSHA256': sha(RUN/'records.json'), 'roles': result,
              'actualEightEpochPostAugmentationBoxExposures': dict(exposures),
              'actualEightEpochImageExposures': dict(image_exposures),
              'tileMagnificationIgnoringClipping': 1/.6,
              'interpretationLimits': ['Counts of boxes presented to loss are not measured gradient contribution. Zero boxes on empty images does not mean zero background-classification loss.',
                  'Geometry projects exact visible GT rectangles into the current tile-ownership rule; it is not model inference and does not prove causes of actual errors.',
                  'Multiple accepted projections can be duplicates before NMS; NMS and real predictions are not simulated here.'],
              'modelInferencePerformed': False, 'old63OrSealed36Read': False}
    (OUT/'existing-pipeline-audit.json').write_text(json.dumps(output, indent=2)+'\n')

    baseline = read(RUN/'baseline-validation.json')
    candidate = read(RUN/'scores/epoch005-alpha1.json')
    baseline_rows, candidate_rows = {}, {}
    for group in ['old56', 'clinical36', 'hardPositive']:
        baseline_rows.update({r['id']:r for r in baseline[group]['images']})
        candidate_rows.update({r['id']:r for r in candidate[group]['images']})
    lines=[]
    for r in records:
        if r['split'] != 'val':
            continue
        b, c = baseline_rows.get(r['id']), candidate_rows.get(r['id'])
        tags = [r['role'], r['source'], r['species']]
        if b and c:
            if c['FN'] > b['FN']: tags.append('candidate_extra_misses')
            if c['FP'] > b['FP']: tags.append('candidate_extra_false_positives')
            if abs(c['signedError']) < abs(b['signedError']): tags.append('candidate_better_count')
            metrics={key:{k:v[k] for k in ['truth','predicted','TP','FP','FN','signedError']} for key,v in [('baseline',b),('diagnostic_candidate',c)]}
        else:
            n=baseline['realEmptyCounts'][r['id']];m=candidate['realEmptyCounts'][r['id']]
            metrics={'truth':0,'baseline_predicted':n,'diagnostic_candidate_predicted':m}
            tags.append('real_empty')
            if m: tags.append('empty_false_positive')
        lines.append({'id':r['id'],'filepath':str(ROOT/r['imagePath']),'imageSHA256':r['sha256'],
                      'source':r['source'],'species':r['species'],'groupId':r['groupId'],'tags':tags,'metrics':metrics,
                      'metricDefinition':'Imported frozen Hungarian IoU0.5 operating metrics; do not replace with a new UI evaluator and call them identical.',
                      'diagnosticCandidateApproved':False,'trainingUseAllowed':False})
    (OUT/'validation-review-index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in lines))
    print(json.dumps(output,indent=2))
    print('Diagnostic validation index:',len(lines),'rows; no training/GT changes.')


if __name__ == '__main__':
    main()
