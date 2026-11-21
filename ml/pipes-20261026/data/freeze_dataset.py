"""Freeze reviewed pipe scenes and fingerprints before any pipe-model training."""
from pathlib import Path
import collections, datetime, hashlib, json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'output/pipes-20261026/data/proposal'
OUT = ROOT / 'output/datasets/pipe-ends-maco-v1'
ART = ROOT / 'ml/pipes-20261026/data'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    assert not p.exists(), 'Preserve an existing freeze'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    full = read(SOURCE / 'all-splits-manifest.json')
    tv = read(SOURCE / 'train-val-manifest.json')
    assert full['status'] == tv['status'] == 'PROPOSED_QA_PASSED_PENDING_ROOT_FREEZE'
    assert sha(SOURCE/'all-splits-manifest.json') == 'd8509daeb04593cf95db058c99052de1da684e75e761e6f9ee0d5376118a7b43'
    assert sha(SOURCE/'train-val-manifest.json') == '6396529586b548790e4ba7e0dfec84be0b403fdfaafa17dc8f4427f989b56489'
    assert len(full['images']) == 241 and len(tv['images']) == 204
    assert tv['images'] == [r for r in full['images'] if r['split'] in ['train','val']]
    groups, bytes_seen, pixels_seen, ids = {}, {}, {}, set()
    rows=[]
    for source_row in full['images']:
        r=dict(source_row)
        assert r['id'] not in ids
        ids.add(r['id'])
        assert r['split'] in ['train','val','test'] and r['annotationQA'] == 'PASS'
        assert r['dataNature'] == 'real' and r['isSynthetic'] is False and r['hasAnnotationOverlay'] is False
        for seen,key in [(groups,str(r['groupId'])),(bytes_seen,r['sha256']),(pixels_seen,r['RGBPixelSHA256'])]:
            assert key not in seen or seen[key] == r['split'], (r['id'],key)
            seen[key]=r['split']
        image=ROOT/r['imagePath']
        assert image.resolve().is_relative_to(ROOT) and sha(image)==r['sha256']
        with Image.open(image) as im: im.verify()
        with Image.open(image) as im:
            assert im.size==(r['width'],r['height']) and im.getexif().get(274,1) in [None,1]
            assert hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()==r['RGBPixelSHA256']
        assert r['count']==len(r['xyxy']) and r['count']<=21504
        assert all(np.isfinite(box).all() and 0<=box[0]<box[2]<=r['width'] and 0<=box[1]<box[3]<=r['height'] for box in r['xyxy'])
        assert len({tuple(box) for box in r['xyxy']})==len(r['xyxy'])
        label=ROOT/r['labelPath'];assert sha(label)==r['labelSHA256']
        lines=label.read_text().splitlines();assert len(lines)==r['count']
        for line, box in zip(lines,r['xyxy']):
            cls,x,y,w,h=map(float,line.split());assert cls==0
            reconstructed=np.array([(x-w/2)*r['width'],(y-h/2)*r['height'],(x+w/2)*r['width'],(y+h/2)*r['height']])
            assert np.max(np.abs(reconstructed-box))<.01
        if r['count']:
            assert r['subtype']=='round_steel_pipe' and r['annotationBoundary']=='outer_rim'
            assert sha(ROOT/r['authorLabelPath'])==r['authorLabelSHA256']
            r['sourceSubtype']=r['subtype'];r['subtype']='round'
        else:
            assert r['negativeKind'] in ['non_pipe','empty_scene','side_only'] and not lines
        r['sourceBucket']=r['bucket'];r['bucket']='positive' if r['count'] else 'negative'
        rows.append(r)
    by_id={r['id']:r for r in rows}
    # Producer records explicitly bind each background crop to its parent scene.
    for r in rows:
        parent=r.get('parentImageId') or r.get('parentId')
        if parent and parent in by_id:
            assert r['groupId']==by_id[parent]['groupId'] and r['split']==by_id[parent]['split']
    counts={s:{'positive':sum(r['split']==s and r['count']>0 for r in rows),
               'negative':sum(r['split']==s and r['count']==0 for r in rows),
               'groups':len({r['groupId'] for r in rows if r['split']==s}),
               'positiveGroups':len({r['groupId'] for r in rows if r['split']==s and r['count']>0}),
               'targets':sum(r['count'] for r in rows if r['split']==s)} for s in ['train','val','test']}
    assert [(counts[s]['positive'],counts[s]['negative']) for s in ['train','val','test']]==[(156,8),(33,7),(34,3)]
    assert all(counts[s]['positiveGroups']>=10 and counts[s]['negative']>0 for s in ['val','test'])
    frozen={**full,'status':'FROZEN_ANNOTATION_QA_PASSED','frozenUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'images':rows,'counts':counts,'testContentIncluded':True,
            'proposalSHA256':sha(SOURCE/'all-splits-manifest.json'),
            'schemaNormalization':'Positive subtype round_steel_pipe→round; bucket pipe_positive/pipe_negative→positive/negative. Original values retained; no image/box/count/split changes.',
            'learningBeforeFreeze':False}
    save(OUT/'manifest.json',frozen)
    trainval={**tv,'status':'FROZEN_ANNOTATION_QA_PASSED','frozenUTC':frozen['frozenUTC'],
              'images':[r for r in rows if r['split'] in ['train','val']],
              'counts':{s:counts[s] for s in ['train','val']},'testContentIncluded':False,
              'sealedFullManifestSHA256':sha(OUT/'manifest.json'),
              'proposalSHA256':sha(SOURCE/'train-val-manifest.json')}
    save(OUT/'train-val-manifest.json',trainval)
    qa={'status':'PASS_ROOT_FROZEN_PIPE_DATA_QA','fullManifestSHA256':sha(OUT/'manifest.json'),
        'trainValManifestSHA256':sha(OUT/'train-val-manifest.json'),'images':241,'positiveImages':223,
        'objects':sum(r['count'] for r in rows),'maxGT':max(r['count'] for r in rows),'counts':counts,
        'byteAndPixelHashesRecomputed':True,'sourceXMLAndDerivedYOLOHashesVerified':True,
        'coordinatesAndYOLOTransformVerified':True,'globalGroupsAndSourceDerivedSHAExclusive':True,
        'allThreeVisualReviewInputsPreserved':True,'splitChangedFromProposed':False,
        'testLabelsUsedForModelSelectionOrTraining':False,'modelInferenceBeforeFreeze':False,
        'limitations':['Reuse of technically checked authorHBB with scene/coarse-coverage review; not a new exhaustive manual recount.',
                       'Conservative visually/feature-linked scene families are not proven physical-stack IDs.',
                       'Real means visually accepted source photograph, not forensic certification.',
                       'Single public source; ordinary phone field performance remains to be tested.']}
    save(ART/'dataset-qa.json',qa)
    print(json.dumps({'trainVal':str(OUT/'train-val-manifest.json'),'trainValSHA256':sha(OUT/'train-val-manifest.json'),'fullSHA256':sha(OUT/'manifest.json'),'counts':counts},indent=2))


if __name__=='__main__': main()
