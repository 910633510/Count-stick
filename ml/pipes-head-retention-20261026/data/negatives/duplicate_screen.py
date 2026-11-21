"""Identity-only classical exclusion screen; no detector, split changes or GT access."""
from pathlib import Path
import concurrent.futures, hashlib, json, sys
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'ml/pipes-20261026/data'))
from group_round_scenes import features, registration

OUT = ROOT / 'output/pipes-head-retention-20261026/data/negatives'
ART = ROOT / 'ml/pipes-head-retention-20261026/data/negatives'
SELECTED = ['dayang-1', 'hindustan-c45', 'cjm-bar11', 'zhonggongte-reserve']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(row):
    path = ROOT / row['imagePath']
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert('RGB')
        wh = rgb.size
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    return {**row, 'sha256': sha(path), 'RGBPixelSHA256': digest, 'width': wh[0], 'height': wh[1]}

def compare(pair):
    a, b = pair
    h = min(int(np.count_nonzero(a['canonical'] != b['canonical'])),
            int(np.count_nonzero(a['ordinary'] != b['ordinary'])))
    choices = [registration(a, b), registration(a, b, True)]
    choices = [r for r in choices if r]
    r = max(choices, key=lambda q: (q['inliers'], q['ratio'])) if choices else {
        'matches': 0, 'inliers': 0, 'ratio': 0., 'coverageA': 0., 'coverageB': 0., 'flip': False}
    strong = r['inliers'] >= 20 and r['ratio'] >= .6 and min(r['coverageA'], r['coverageB']) >= .04
    suspicious = strong or h <= 8 or (r['inliers'] >= 12 and r['ratio'] >= .4)
    return {'a': a['id'], 'b': b['id'], 'hashDistance': h, **r,
            'strongMatch': strong, 'suspicious': suspicious}

def main():
    protected = ['output/datasets/pipe-ends-maco-v1/manifest.json',
                 'output/datasets/pipe-ends-hardening-v1/manifest.json',
                 'output/datasets/pipe-ends-maco-v1/train-val-manifest.json',
                 'output/datasets/pipe-ends-hardening-v1/train-val-manifest.json']
    before = {p: sha(ROOT / p) for p in protected}
    old = []
    for p in protected[:2]:
        # Only these identity keys are retained/used; counts/boxes are never consulted.
        for r in json.load(open(ROOT / p))['images']:
            old.append({k: r[k] for k in ['id', 'imagePath', 'split', 'groupId']})
    index = json.load(open(ROOT / 'output/pipes-20261026/data/technical-index.json'))
    used = {r['id'] for r in old}
    old += [{'id': r['id'], 'imagePath': r['imagePath'], 'split': 'unused_source_pool', 'groupId': None}
            for r in index['images'] if r['sourceArchive'] == 'round' and r['id'] not in used]
    # IDs from different manifests must not silently overwrite each other.
    assert len({r['id'] for r in old}) == len(old)
    receipt = json.load(open(OUT / 'sources/download-receipts.json'))
    new = [{'id': r['id'], 'imagePath': r['imagePath']} for r in receipt['items'] if r['id'] in SELECTED]
    assert sorted(r['id'] for r in new) == sorted(SELECTED)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(identity, old + new))
        ff = list(pool.map(features, old + new))
    oldids, newids = identities[:len(old)], identities[len(old):]
    oldf, newf = ff[:len(old)], ff[len(old):]
    print('FEATURES', len(oldf), len(newf), flush=True)
    exact = []
    for a in newids:
        for b in oldids:
            equal_byte = a['sha256'] == b['sha256']
            equal_rgb = (a['width'], a['height'], a['RGBPixelSHA256']) == (b['width'], b['height'], b['RGBPixelSHA256'])
            if equal_byte or equal_rgb:
                exact.append({'a': a['id'], 'b': b['id'], 'sameBytes': equal_byte, 'sameRGB': equal_rgb,
                              'oldSplit': b['split'], 'oldGroupId': b['groupId']})
    allresults, summaries = [], []
    for a in newf:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rr = list(pool.map(compare, [(a, b) for b in oldf]))
        ranked = sorted(rr, key=lambda r: (r['strongMatch'], r['inliers'], r['ratio'], -r['hashDistance']), reverse=True)
        allresults.extend(rr)
        summaries.append({'id': a['id'], 'flagged': [r for r in rr if r['suspicious']], 'top3': ranked[:3]})
        print(a['id'], 'flagged', [r for r in rr if r['suspicious']], flush=True)
    newpairs = [compare((a, b)) for i, a in enumerate(newf) for b in newf[i+1:]]
    after = {p: sha(ROOT / p) for p in protected}
    assert before == after
    (OUT / 'duplicate-pairs.json').write_text(json.dumps(allresults, indent=2) + '\n')
    report = {'status': 'CLASSICAL_SCREEN_COMPLETE_REQUIRES_VISUAL_SOURCE_REVIEW',
              'comparisonCounts': {'frozenOld': 241, 'frozenCurrent': 69, 'allRowsIncludingUnusedMACO': len(old),
                                   'new': 4, 'newVsExistingPairs': len(allresults), 'newVsNewPairs': len(newpairs)},
              'method': 'Exact file SHA256 + EXIF-upright RGB SHA256; all-pairs SIFT500/720px RANSAC, normal and mirrored; pHash flags. No learned model.',
              'strongGate': '>=20 inliers, >=0.6 ratio, >=0.04 coverage both images',
              'suspiciousGate': 'strong OR pHash<=8 OR >=12 inliers with ratio>=0.4',
              'exactMatches': exact, 'candidateSummaries': summaries, 'newPairs': newpairs,
              'newIdentities': newids, 'referenceIdentities': oldids,
              'protectedManifestSHA256Before': before, 'protectedManifestSHA256After': after,
              'allPairsPath': str((OUT / 'duplicate-pairs.json').relative_to(ROOT)),
              'allPairsSHA256': sha(OUT / 'duplicate-pairs.json'),
              'testUse': 'Automatic classical identity exclusion only. No test GT inspection, display, learned inference, training, metric selection or label/split edit.',
              'limits': 'Absence of a classical match does not prove different physical stock/pile. Supplier product-page provenance is not author-confirmed scene identity.'}
    (ART / 'duplicate-screen.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DONE', len(exact), sum(r['suspicious'] for r in allresults), flush=True)

if __name__ == '__main__':
    main()
