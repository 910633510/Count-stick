"""Materialize reviewed negative views; no model inference, old data unchanged."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'output/pipes-hardening-20261026/data/negatives'
SOURCE=Path(__file__).resolve().parent
SEED='CountScope-pipe-hardening-negatives-20261026-v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,j):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(j,indent=2)+'\n')

def main():
    old=read(ROOT/'output/datasets/pipe-ends-maco-v1/manifest.json')['images']
    old_source=read(ROOT/'output/pipes-20261026/data/technical-index.json')['images']
    old_sha={r['sha256'] for r in old+old_source};old_rgb={r['RGBPixelSHA256'] for r in old+old_source}
    web=read(OUT/'sources/web-download-receipts.json');review=read(SOURCE/'web-native-review.json')
    reviewed={r['id']:r for r in review['reviews']};web=[r for r in web if r['id'] in reviewed]
    assert len(web)==12
    web_groups={r['id']:'hardneg_'+r['id'].replace('-','_')+'_scene' for r in web}
    ordered=sorted(web_groups.values(),key=lambda group:hashlib.sha256((SEED+'|'+group).encode()).hexdigest())
    assignments={group:('test' if i<10 else 'val') for i,group in enumerate(ordered)}
    images=[]
    for r in web:
        assert sha(ROOT/r['path'])==r['sha256']==reviewed[r['id']]['sha256']
        with Image.open(ROOT/r['path']) as im:
            assert im.getexif().get(274,1) in [None,1];rgb=im.convert('RGB');rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
        assert r['sha256'] not in old_sha and rgb_sha not in old_rgb
        extension={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}[r['format']]
        target=OUT/'images'/(r['id']+extension);target.parent.mkdir(exist_ok=True)
        if target.exists():assert sha(target)==r['sha256']
        else:shutil.copyfile(ROOT/r['path'],target)
        group=web_groups[r['id']]
        images.append({'id':'hardneg-'+r['id'],'split':assignments[group],'source':'public-industrial-photo:'+r['sourcePage'],
                       'groupId':group,'imagePath':str(target.relative_to(ROOT)),'sha256':sha(target),'RGBPixelSHA256':rgb_sha,
                       'width':rgb.width,'height':rgb.height,'count':0,'xyxy':[],'annotationQA':'PASS','annotationMethod':'Own full-photo visual zero-pipe-end review; physical solid bar ends/machine rollers excluded from pipe class.',
                       'dataNature':'real','isSynthetic':False,'hasAnnotationOverlay':False,'negativeKind':'non_pipe','confuserSubtype':'solid_rod_end' if r['id'] in ['web-lapefer','web-saarstahl','web-vishal','web-allied','web-mmc','web-metalloinvest','web-donswelding'] else 'rebar_end',
                       'sourceImagePath':r['path'],'sourceImageSHA256':r['sha256'],'sourcePage':r['sourcePage'],'sourceImageURL':r['imageURL'],
                       'sourceHadBlackMask':False,'fullSourceFramePreserved':True,'hasPublisherWatermark':r['id']=='web-rivastahl','modelInferencePerformed':False})
    crop_manifest=read(OUT/'crop-proposals/manifest.json');crop_review=read(SOURCE/'crop-review.json')
    approved={r['id'] for r in crop_review['images'] if r['cropReview']=='PASS'}
    assert len(approved)==32
    for r in crop_manifest['images']:
        if r['id'] not in approved:continue
        parent=ROOT/r['sourceImagePath'];path=ROOT/r['imagePath'];assert sha(parent)==r['sourceImageSHA256']
        with Image.open(parent) as im:
            assert im.getexif().get(274,1) in [None,1];parent_rgb=im.convert('RGB');crop=parent_rgb.crop(r['cropRectPixels']);parent_rgb_sha=hashlib.sha256(parent_rgb.tobytes()).hexdigest()
        with Image.open(path) as im:
            assert im.format=='PNG' and im.mode=='RGB' and im.size==(r['width'],r['height']) and np.array_equal(np.asarray(im),np.asarray(crop))
            rgb_sha=hashlib.sha256(im.tobytes()).hexdigest()
        assert sha(parent) not in old_sha and parent_rgb_sha not in old_rgb and sha(path) not in old_sha and rgb_sha not in old_rgb
        diameter=r['sourceId'].rsplit('_',1)[-1]
        # Source has no acquisition IDs. Pool all selected views of a nominal
        # diameter conservatively for training sampling; do not claim each crop
        # is an independent physical scene or assign any crop to val/test.
        group='hardneg_rebardsc_conservative_'+diameter.lower()+'_source_pool'
        images.append({**r,'id':'hardneg-'+r['id'],'status':'VISUALLY_REVIEWED_CANDIDATE','split':'train','source':'IJCAI2021-RebarDSC-real-interior-crop',
                       'groupId':group,'parentGroupId':group,'sha256':sha(path),'RGBPixelSHA256':rgb_sha,'parentRGBPixelSHA256':parent_rgb_sha,
                       'count':0,'xyxy':[],'annotationQA':'PASS','cropReview':'PASS','annotationMethod':'Own visual review of unmasked original RGB rectangular interior; zero visible pipe ends.',
                       'dataNature':'real_crop','isSynthetic':False,'hasAnnotationOverlay':False,'sourceHadBlackMask':True,
                       'negativeKind':'non_pipe','confuserSubtype':'rebar_end','sourcePage':'https://github.com/Armin1337/RebarDSC',
                       'sourceImageURL':'https://drive.usercontent.google.com/download?id=1-rRbIP2ds0zSjcI8j8o1ERm3ethAAiZr&export=download&confirm=t',
                       'sourceArchiveMember':'images/'+Path(r['sourceImagePath']).name,'nominalSourceDiameter':diameter,'groupBasis':'Conservative unresolved source/diameter pool, training only; not asserted physical-pile IDs.','modelInferencePerformed':False})
    assert len(images)==44
    for r in images:
        label=OUT/'labels'/(r['id']+'.txt');label.parent.mkdir(exist_ok=True);label.write_bytes(b'');r['labelPath']=str(label.relative_to(ROOT));r['labelSHA256']=sha(label)
    for key in ['groupId','sha256','RGBPixelSHA256','sourceImageSHA256']:
        split_sets={s:{r[key] for r in images if r['split']==s} for s in ['train','val','test']}
        for a,c in [('train','val'),('train','test'),('val','test')]:assert not split_sets[a]&split_sets[c]
    counts={s:{'images':sum(r['split']==s for r in images),'groups':len({r['groupId'] for r in images if r['split']==s}),'targets':0} for s in ['train','val','test']}
    assert counts['test']['groups']>=10
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'web-native-review.json',SOURCE/'crop-review.json',OUT/'group-screen.json',OUT/'sources/rebardsc-download-receipts.json',ROOT/'output/datasets/pipe-ends-maco-v1/manifest.json']}
    manifest={'status':'NEGATIVE_CANDIDATES_QA_PASSED_PENDING_ROOT_MERGE_FREEZE','seed':SEED,'images':images,'counts':counts,'reviewBindings':inputs,
              'allLabels':'pipe_end class has zero targets in every proposed negative view','modelInferencePerformed':False,
              'groupPolicy':'Fresh full-photo web groups use distinct industrial scene/publisher geometry and no detected matches. All cropped RebarDSC interiors are train only and grouped conservatively by source nominal diameter to avoid per-crop group inflation.',
              'freshSplitPolicy':'SHA256(seed|groupId) order of 12 full natural web scenes: first10 fresh test, remaining2 fresh val. No model predictions/GT count choice.',
              'limitations':['Dataset/industrial public photos, not a prospective phone capture acceptance set.','Classical checks detected no matches to old241 views or1025 round-source corpus; unmatched does not prove independent capture sessions.','Author black-masked parent photos are excluded as full training images. Only reviewed original interior pixels are used as train-only crops.','No source author count labels are reused; zero pipe_end labels were visually assigned for this task.']}
    full=OUT/'negative-candidates.json';save(full,manifest)
    save(OUT/'negative-train-val-candidates.json',{**{k:v for k,v in manifest.items() if k not in ['images','counts']},'images':[r for r in images if r['split']!='test'],'counts':{k:v for k,v in counts.items() if k!='test'},'testContentIncluded':False,'sealedFullManifestSHA256':sha(full)})
    print(json.dumps({'counts':counts,'fullPath':str(full.relative_to(ROOT)),'fullSHA256':sha(full),'trainValSHA256':sha(OUT/'negative-train-val-candidates.json')},indent=2))

if __name__=='__main__':main()
