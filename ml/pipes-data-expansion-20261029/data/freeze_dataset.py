"""Append independently reviewed train-only photos, preserving every frozen prior row."""
from pathlib import Path
import argparse, collections, copy, datetime, hashlib, json, math, re
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'ml/pipes-data-expansion-20261029/data'
OUT = ROOT / 'output/datasets/pipe-ends-data-expansion-v1'
PARENT = ROOT / 'output/datasets/pipe-ends-head-retention-v1/manifest.json'
PARENT_SHA = 'de4c153b89e48d9b4b6fe06ace680070e5a55dd4fab00b66a8180ad836621078'
PARENT_TV_SHA = 'b7185c895ed9c1d695d486a5b0dc3fbd9725510afbbb95c4b74b3eaea1e11393'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def main(proposal_path):
    proposal_path = (ROOT / proposal_path).resolve()
    proposal = read(proposal_path)
    assert proposal['status'] == 'ROOT_AND_INDEPENDENT_QA_PASSED_TRAIN_ONLY'
    assert sha(PARENT) == PARENT_SHA and sha(PARENT.with_name('train-val-manifest.json')) == PARENT_TV_SHA
    prior = read(PARENT)['images']; assert len(prior) == 74
    old_path = ROOT / 'output/datasets/pipe-ends-maco-v1/manifest.json'
    assert sha(old_path) == '960005aa6c7956db8ebf5962693071e0ea01888f0d0380431437894b4fddec41'
    old = read(old_path)['images']; assert len(old) == 241
    extra = copy.deepcopy(proposal['images']); assert extra
    bindings = {str(proposal_path.relative_to(ROOT)): sha(proposal_path)}
    for rel, digest in proposal['reviewBindings'].items():
        assert sha(ROOT / rel) == digest
        bindings[rel] = digest
    ids = {r['id'] for r in old + prior}
    shas = {r['sha256'] for r in old + prior}
    rgbs = {r['RGBPixelSHA256'] for r in old + prior}
    group_splits = {}
    for row in old + prior:
        assert group_splits.get(row['groupId'], row['split']) == row['split']
        group_splits[row['groupId']] = row['split']
    for row in extra:
        assert re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) and row['id'] not in ids
        assert isinstance(row['source'], str) and row['source'].strip()
        assert isinstance(row['groupId'], str) and row['groupId'].strip()
        assert row['split'] == 'train' and row['annotationQA'] == 'PASS'
        assert row['dataNature'] == 'real' and row['isSynthetic'] is False
        assert row['hasAnnotationOverlay'] is False and row.get('sourceHadBlackMask', False) is False
        assert row['modelInferencePerformed'] is False and row['label'] == 'pipe_end'
        assert not any(k in row for k in ['cropRectPixels', 'sourceCropXYXY'])
        assert row['count'] == len(row['xyxy']) and 0 <= row['count'] <= 21504
        assert all(isinstance(box, list) and len(box) == 4 for box in row['xyxy'])
        if row['count']: assert row['subtype'] == 'round' and row['annotationBoundary'] == 'outer_rim'
        else: assert row['negativeKind'] in ['non_pipe', 'empty_scene', 'side_only']
        path = (ROOT / row['imagePath']).resolve()
        assert path.is_relative_to(ROOT) and sha(path) == row['sha256']
        with Image.open(path) as im:
            assert im.size == (row['width'], row['height']) and im.getexif().get(274, 1) in [None, 1]
            rgb = hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        assert row['RGBPixelSHA256'] == rgb and rgb not in rgbs and row['sha256'] not in shas
        assert group_splits.get(row['groupId'], 'train') == 'train'
        assert all(all(math.isfinite(v) for v in box) and 0 <= box[0] < box[2] <= row['width'] and 0 <= box[1] < box[3] <= row['height'] for box in row['xyxy'])
        assert len({tuple(b) for b in row['xyxy']}) == row['count']
        ids.add(row['id']); shas.add(row['sha256']); rgbs.add(rgb); group_splits[row['groupId']] = 'train'
        row['finalTrainingAdmission'] = True
    assert not OUT.exists(), 'Preserve any earlier frozen attempt'
    OUT.mkdir(); labels = OUT / 'added-labels'; labels.mkdir()
    for row in extra:
        w, h = row['width'], row['height']; p = labels / (row['id'] + '.txt')
        text = '\n'.join(f'0 {(a+c)/2/w:.10f} {(b+d)/2/h:.10f} {(c-a)/w:.10f} {(d-b)/h:.10f}' for a,b,c,d in row['xyxy'])
        p.write_text(text + ('\n' if text else '')); row['labelPath'] = str(p.relative_to(ROOT)); row['labelSHA256'] = sha(p)
    rows = copy.deepcopy(prior) + extra
    assert rows[:74] == prior
    for split in ['val', 'test']: assert [r for r in rows if r['split'] == split] == [r for r in prior if r['split'] == split]
    summary = {s: {'images': sum(r['split'] == s for r in rows), 'objects': sum(r['count'] for r in rows if r['split'] == s), 'groups': len({r['groupId'] for r in rows if r['split'] == s})} for s in ['train', 'val', 'test']}
    manifest = {'status': 'FROZEN_ANNOTATION_QA_PASSED', 'createdUTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'testContentIncluded': True, 'images': rows, 'summary': summary, 'parentManifestSHA256': PARENT_SHA, 'parentTrainValManifestSHA256': PARENT_TV_SHA, 'previous74RowsUnchanged': True, 'fresh20RecognitionNeverRun': True, 'addedViews': len(extra), 'sourceReviewBindings': bindings, 'freezerSHA256': sha(__file__), 'historyNote': 'Prior row inference metadata records historical curation. All new rows are train-only. Old47 validation has already been used for development; fresh20 test remains unseen by models. Classical image identity audits do not measure recognition.'}
    write(OUT / 'manifest.json', manifest)
    tv = {**manifest, 'testContentIncluded': False, 'images': [r for r in rows if r['split'] != 'test'], 'summary': {s: summary[s] for s in ['train', 'val']}, 'sealedFullManifestSHA256': sha(OUT / 'manifest.json')}
    assert tv['images'][:54] == read(PARENT.with_name('train-val-manifest.json'))['images']
    write(OUT / 'train-val-manifest.json', tv)
    write(OUT / 'identity-only.json', {'status':'FROZEN_IDENTITY_ONLY_NO_GT_OR_PIXELS', 'images':[{k:r[k] for k in ['id','split','groupId','sha256','RGBPixelSHA256']} for r in rows]})
    qa = {'status':'PASS_ROOT_DATA_EXPANSION_FREEZE', 'createdUTC':manifest['createdUTC'], 'addedViews':len(extra), 'addedPositiveViews':sum(r['count']>0 for r in extra), 'addedNegativeViews':sum(r['count']==0 for r in extra), 'addedObjects':sum(r['count'] for r in extra), 'addedGroups':len({r['groupId'] for r in extra}), 'previous74RowsAnd54DevelopmentRowsUnchanged':True, 'same7FreshValAnd20FreshTestRows':True, 'sourceReviewBindings':bindings, 'manifestSHA256':sha(OUT/'manifest.json'), 'trainValManifestSHA256':sha(OUT/'train-val-manifest.json'), 'summary':summary, 'trainingPerformed':False}
    write(ART / 'root-freeze-qa.json', qa)
    receipt = {'status':'APPROVED_DATA_EXPANSION_PREPARATION', 'manifestPath':str((OUT/'train-val-manifest.json').relative_to(ROOT)), 'manifestSHA256':sha(OUT/'train-val-manifest.json'), 'parentTrainValManifestSHA256':PARENT_TV_SHA, 'sealedFullManifestSHA256':sha(OUT/'manifest.json'), 'qaReportPath':str((ART/'root-freeze-qa.json').relative_to(ROOT)), 'qaReportSHA256':sha(ART/'root-freeze-qa.json')}
    write(ART / 'root-freeze-receipt.json', receipt)
    assert all(sha(ROOT/p) == h for p,h in bindings.items())
    print(json.dumps(qa, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--proposal', type=Path, required=True)
    main(parser.parse_args().proposal)
