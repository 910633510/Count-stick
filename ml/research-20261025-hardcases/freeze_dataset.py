#!/usr/bin/env python3
"""Freeze visual annotations before model inference; preserve old datasets."""
from pathlib import Path
import collections, datetime, hashlib, json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'ml/research-20261025-hardcases'
OUT = ROOT / 'output/datasets/colony-hardcases-retention-v1'
POS = ROOT / 'output/research-20261025-hardcases/positives/annotated/positive-candidates.json'
NEG = ART / 'negatives/negative-candidates.json'
SEED = 'CountScope-hardcases-retention-20261025-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'Preserve existing freeze: ' + str(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    positive, negative = read(POS), read(NEG)
    assert len(positive['images']) == 11 and sum(r['count'] for r in positive['images']) == 59
    assert len(negative['images']) == 40
    assert all(r['annotationQA'] == 'PASS' for r in positive['images'] + negative['images'])
    pos_qa = read(ART / 'positives/qa-receipt.json')
    assert pos_qa['status'].startswith('PASS') and pos_qa['candidateManifestSHA256'] == sha(POS)
    split = {}
    for species in sorted({r['species'] for r in positive['images']}):
        groups = sorted({r['groupId'] for r in positive['images'] if r['species'] == species},
                        key=lambda g: hashlib.sha256((SEED + '|' + species + '|' + g).encode()).hexdigest())
        for i, group in enumerate(groups):
            split[group] = 'val' if len(groups) > 1 and i == 0 else 'train'
    split.update({
        'mendeley_xrbtd74pfj_v3_negative_source_group': 'train',
        'brukeragar_AEBM_plate0000001': 'train',
        'brukeragar_BESBL_plate0000001': 'val',
        'zenodo18703636_session_20230710': 'val',
    })
    rows = []
    protected_hashes = set()
    old = read(ROOT / 'output/datasets/colony-figshare-22022540-v3/yolo-species-v1/manifest.json')
    protected_hashes.update(r['sha256'] for r in old['images'])
    # Only fingerprint strings are extracted for exclusion. No clinical test
    # pixels, boxes, counts or model outputs are inspected or exported.
    clinical = read(ROOT / 'output/datasets/colony-mixed-isolate-v1/manifest.json')
    protected_hashes.update(r['sha256'] for r in clinical['images'])
    del clinical
    hash_split = {}
    for r in positive['images'] + negative['images']:
        row = {**r, 'split': split[r['groupId']]}
        path = ROOT / r['imagePath']
        assert sha(path) == r['sha256']
        assert r['sha256'] not in protected_hashes and r['sourceImageSHA256'] not in protected_hashes
        assert r['sha256'] not in hash_split or hash_split[r['sha256']] == row['split']
        hash_split[r['sha256']] = row['split']
        with Image.open(path) as im:
            assert im.size == (r['width'], r['height']) and im.getexif().get(274, 1) in [None, 1]
            pixels = np.asarray(im.convert('RGB'))
        if r.get('RGBPixelSHA256'):
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == r['RGBPixelSHA256']
        if r.get('cropXYXYInOriginal') is not None:
            with Image.open(ROOT / r['sourceImagePath']) as original:
                crop = np.asarray(original.convert('RGB').crop(r['cropXYXYInOriginal']))
            assert np.array_equal(crop, pixels), 'Training crop intensity/geometry changed'
            x0, y0, _, _ = r['cropXYXYInOriginal']
            expected = [[a-x0, b-y0, c-x0, d-y0] for a,b,c,d in r['ownBoxesXYXYInOriginal']]
            expected = np.maximum(0, np.minimum(expected, [r['width'], r['height'], r['width'], r['height']]))
            assert np.array_equal(expected, r['xyxy'])
        assert r['count'] == len(r['xyxy'])
        assert all(np.isfinite(box).all() and 0 <= box[0] < box[2] <= r['width']
                   and 0 <= box[1] < box[3] <= r['height'] for box in r['xyxy'])
        if r['bucket'] == 'empty_negative':
            assert r['negativeKind'] == 'real_empty_plate' and r['count'] == 0
            assert (ROOT / r['labelPath']).read_bytes() == b''
            assert sha(ROOT / r['labelPath']) == r['labelSHA256']
        rows.append(row)
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({r['groupId'] for r in rows}) == 15
    assert {r['groupId'] for r in rows if r['split'] == 'train'}.isdisjoint(
        {r['groupId'] for r in rows if r['split'] == 'val'})
    counts = {s: dict(collections.Counter(r['bucket'] for r in rows if r['split'] == s))
              for s in ['train', 'val']}
    assert counts == {'train': {'hard_positive': 7, 'empty_negative': 34},
                      'val': {'hard_positive': 4, 'empty_negative': 6}}
    manifest = {
        'status': 'FROZEN_ANNOTATION_QA_PASSED',
        'frozenUTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'testContentIncluded': False, 'images': rows,
        'seed': SEED, 'counts': counts,
        'groupAssignments': split,
        'splitDefinition': 'Positive: one hash-ranked group per species with >=2 groups is validation; singleton species train. Negative: Mendeley source union and AEBM train, BESBL and phone-session validation. All source aliases/crops/views share global groupId.',
        'sourceManifestSHA256': {str(POS.relative_to(ROOT)): sha(POS), str(NEG.relative_to(ROOT)): sha(NEG)},
        'policySHA256': sha(ART / 'annotation-policy.json'),
        'annotationScope': '11 visual instance-box annotations, including9explicit crops, and40real-empty photos. Not a pixel-mask segmentation dataset or clinical-grade ground truth.',
        'independenceLimit': '15 conservative leakage groups; the negative source-union/session groups are not a verified physical-plate count.',
        'clinicalTestAccess': 'Fingerprint metadata exclusion only; no pixels/GT inspection, training, model selection or inference.',
    }
    save(OUT / 'manifest.json', manifest)
    report = {
        'status': 'PASS_FROZEN_ANNOTATIONS_AND_GROUP_BOUNDARY_QA',
        'manifestSHA256': sha(OUT / 'manifest.json'), 'images': 51, 'positiveBoxes': 59,
        'counts': counts, 'conservativeGroups': 15,
        'allSourceByteAndPixelHashesVerified': True,
        'allExplicitCropPixelsAndCoordinateTransformsVerified': True,
        'allDimensionsEXIFAndFiniteBoundsVerified': True,
        'globalGroupTrainValOverlap': 0, 'exactSHAOverlapWithOld369AndClinical185': 0,
        'positiveVisualReview': 'All11accepted samples independently reviewed byroot andcolony_papers; Ecoli5 disagreement quarantined.',
        'negativeVisualReview': 'All64candidates reviewed bynegative agent;40accepted,24quarantined;root independently viewed8accepted images acrossall4conservativegroups.',
        'rootNegativeSampleReceipt': 'output/research-20261025-hardcases/root-negative-review-selection.json',
        'zeroModelInferenceBeforeFreeze': True,
        'testContentUsedForTrainingOrSelection': False,
        'limitations': ['Small research pilot; no microbiologist consensus.', 'Most positives are explicit countable crops of hyperspectral-derived RGB rather than ordinary phone capture.', 'Exact hashes do not exclude all re-encoded/near-duplicate sources.', 'No patient fields imported.'],
    }
    save(ART / 'dataset-qa.json', report)
    print(json.dumps({'manifest': str(OUT / 'manifest.json'), 'sha256': sha(OUT / 'manifest.json'),
                      'counts': counts, 'positiveValIds': [r['id'] for r in rows if r['bucket']=='hard_positive' and r['split']=='val']}, indent=2))


if __name__ == '__main__':
    main()
