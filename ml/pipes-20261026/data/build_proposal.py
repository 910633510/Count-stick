#!/usr/bin/env python3
"""Build an audited, group-split proposal; root owns final freezing/authorization."""
from pathlib import Path
import collections,datetime,hashlib,json,math,shutil
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-20261026/data';ART=ROOT/'ml/pipes-20261026/data'
SEED='CountScope-pipe-ends-real-20261026-v1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n')
def num(identity):return int(identity.split('PipeHBB')[-1])

def main():
    merge_path=OUT/'candidate-merge-provisional.json';merge=json.loads(merge_path.read_text());approved={};reviews={}
    for who in ('root','general','colony'):
        path=ART/f'overlay-review-{who}.json';d=json.loads(path.read_text());reviews[str(path.relative_to(ROOT))]=sha(path)
        numbers=d.get('approvedNumbers',d.get('acceptedNumbers'));assert numbers is not None
        for n in numbers:assert n not in approved;approved[n]=who
    assert len(approved)==223
    rows=[];labels=OUT/'derived-labels';labels.mkdir(exist_ok=True)
    for r in merge['images']:
        n=num(r['id'])
        if n not in approved:continue
        p=ROOT/r['imagePath'];assert sha(p)==r['sha256'];assert sha(ROOT/r['labelPath'])==r['labelSHA256']
        row={k:v for k,v in r.items() if k not in ('split','labelPath','labelSHA256','annotationQA','dataNature','isSynthetic','hasAnnotationOverlay')}
        row.update(split=None,bucket='pipe_positive',label='pipe_end',annotationBoundary='outer_rim',annotationQA='PASS',dataNature='real',isSynthetic=False,hasAnnotationOverlay=False,authorLabelPath=r['labelPath'],authorLabelSHA256=r['labelSHA256'],annotationMethod='Author HBB reused after full technical validation and multi-agent scene/coarse-coverage visual review; exact duplicate boxes corrected where documented. Not an independent exhaustive recount.',visualReviewer=approved[n],negativeKind=None)
        row['targetPolicy']='One discernible front end of a round metallic/steel pipe, enclosing visible outer rim and opening; dirt-filled ends retained when tubular wall/body/rim is identifiable. Do not double-count inner hole and rim. No plain rebar/billet, uncertain PVC, multi-opening bent fittings or unverifiable fragments.'
        rows.append(row)
    assert len(rows)==223
    index=json.loads((OUT/'technical-index.json').read_text());by_num={num(r['id']):r for r in index['images'] if r['sourceArchive']=='round'}
    parent_group={num(r['id']):r['groupId'] for r in rows}
    negative_receipts=json.loads((OUT/'sources/negative-preview-receipts.json').read_text());receipt_lookup={r['path']:r for archive in negative_receipts for r in archive['records']}
    negative_groups={
        'RebarHBB00000200':'maco_negative_rebar_rust_hand_family','RebarHBB00000399':'maco_negative_rebar_rust_hand_family',
        'RebarHBB00000598':'maco_negative_rebar_snow_scene','RebarHBB00000797':'maco_negative_rebar_gray_bundle_family','RebarHBB00000996':'maco_negative_rebar_gray_bundle_family',
        'IbeamHBB00000001':'maco_negative_ibeam_rust_family','IbeamHBB00000100':'maco_negative_ibeam_rust_family','IbeamHBB00000499':'maco_negative_ibeam_rust_family',
        'IbeamHBB00000200':'maco_negative_ibeam_gray_family','IbeamHBB00000300':'maco_negative_ibeam_gray_family','IbeamHBB00000400':'maco_negative_ibeam_gray_family'}
    for p in sorted((OUT/'negative-candidates').rglob('*.jpg')):
        if p.stem not in negative_groups:continue
        receipt=receipt_lookup[str(p.relative_to(ROOT))];assert sha(p)==receipt['sha256']
        with Image.open(p) as im:im.load();w,h=im.size;assert im.getexif().get(274,1) in (None,1);pixel=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
        rows.append({'id':'negative_'+p.stem,'split':None,'source':'zenodo17165968-MACO','sourceArchive':'Rebar.zip' if p.stem.startswith('Rebar') else 'I-beam.zip','subtype':'solid_rebar' if p.stem.startswith('Rebar') else 'open_I_profile','label':'pipe_end','bucket':'pipe_negative','groupId':negative_groups[p.stem],'groupingBasis':'Conservative material/capture-family union; public physical pile IDs unavailable.','imagePath':str(p.relative_to(ROOT)),'width':w,'height':h,'sha256':sha(p),'RGBPixelSHA256':pixel,'xyxy':[],'count':0,'negativeKind':'non_pipe','annotationBoundary':None,'annotationQA':'PASS','dataNature':'real','isSynthetic':False,'hasAnnotationOverlay':False,'annotationMethod':'Data-agent full-frame visual zero-pipe-end review, corroborated by author alternative-material category. Zero means no discernible target opening, not no physical pipes anywhere off-frame.','sourceRecord':'https://zenodo.org/records/17165968','sourceImagePath':str(p.relative_to(ROOT)),'sourceImageSHA256':sha(p),'sourceMemberCRCVerified':True,'wholeRemoteArchiveMD5Verified':False})
    background=json.loads((OUT/'review/background-candidate-records.json').read_text())
    for r in background:
        n=r['parentNumber'];assert n in parent_group and r['GTIntersections']==0
        p=ROOT/r['imagePath'];source=ROOT/r['sourceImagePath'];assert sha(source)==r['sourceSHA256']
        with Image.open(source) as im:raw=np.asarray(im.convert('RGB').crop(r['cropXYXY']))
        with Image.open(p) as im:im.load();assert np.array_equal(raw,np.asarray(im));w,h=im.size
        target=OUT/'derived-negatives'/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
        rows.append({'id':f'negative_round_background_{n:08d}','split':None,'source':'zenodo17165968-MACO','sourceArchive':'Steel pipe.zip','subtype':'real_background_crop','label':'pipe_end','bucket':'pipe_negative','groupId':parent_group[n],'groupingBasis':'Exactly the same group as the positive parent photograph and every source alias. Never an independent scene.','imagePath':str(target.relative_to(ROOT)),'width':w,'height':h,'sha256':sha(target),'RGBPixelSHA256':hashlib.sha256(raw.tobytes()).hexdigest(),'xyxy':[],'count':0,'negativeKind':'empty_scene','annotationBoundary':None,'annotationQA':'PASS','dataNature':'real','isSynthetic':False,'hasAnnotationOverlay':False,'annotationMethod':'Explicit real background crop with zero author-GT intersections, then visually confirmed no visible pipe ends. No image synthesis or photometric edits.','sourceRecord':'https://zenodo.org/records/17165968','sourceImagePath':r['sourceImagePath'],'sourceImageSHA256':r['sourceSHA256'],'parentId':by_num[n]['id'],'cropXYXYInOriginal':r['cropXYXY'],'notAnIndependentAdditionalPhoto':True})
    assert len(rows)==241
    # The source-group assignment is frozen before any model inference.
    pos_groups=collections.defaultdict(list)
    for r in rows:
        if r['bucket']=='pipe_positive':pos_groups[r['groupId']].append(r)
    strata=collections.defaultdict(list)
    for g,rr in pos_groups.items():
        median=float(np.median([x['count'] for x in rr]));bin_name='few_under50' if median<50 else 'medium_50to199' if median<200 else 'dense_200plus';strata[bin_name].append(g)
    split={}
    for bin_name,groups in sorted(strata.items()):
        ordered=sorted(groups,key=lambda g:hashlib.sha256((SEED+'|'+bin_name+'|'+g).encode()).hexdigest());quota=max(1,round(.15*len(ordered))) if len(ordered)>=3 else 0
        for i,g in enumerate(ordered):split[g]='test' if i<quota else 'val' if i<2*quota else 'train'
    split.update({'maco_negative_rebar_rust_hand_family':'train','maco_negative_rebar_gray_bundle_family':'test','maco_negative_rebar_snow_scene':'val','maco_negative_ibeam_rust_family':'train','maco_negative_ibeam_gray_family':'val'})
    for r in rows:
        r['split']=split[r['groupId']];path=labels/(r['id']+'.txt');lines=[]
        for box in r['xyxy']:
            assert all(math.isfinite(x) for x in box);x1,y1,x2,y2=box;assert 0<=x1<x2<=r['width'] and 0<=y1<y2<=r['height'];values=((x1+x2)/(2*r['width']),(y1+y2)/(2*r['height']),(x2-x1)/r['width'],(y2-y1)/r['height']);lines.append('0 '+' '.join(f'{v:.12f}' for v in values))
        text='\n'.join(lines)+('\n' if lines else'')
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
        r['labelPath']=str(path.relative_to(ROOT));r['labelSHA256']=sha(path);assert len(lines)==r['count']
    sets={s:{r['groupId'] for r in rows if r['split']==s} for s in ['train','val','test']};assert all(not sets[a]&sets[b] for a,b in [('train','val'),('train','test'),('val','test')]);assert len(sets['val'])>=10 and len(sets['test'])>=10
    hashes={s:{r['sha256'] for r in rows if r['split']==s} for s in sets};pixels={s:{r['RGBPixelSHA256'] for r in rows if r['split']==s} for s in sets}
    assert all(not hashes[a]&hashes[b] and not pixels[a]&pixels[b] for a,b in [('train','val'),('train','test'),('val','test')])
    counts={s:dict(collections.Counter(r['bucket'] for r in rows if r['split']==s)) for s in sets};out=OUT/'proposal';out.mkdir(exist_ok=True)
    full={'status':'PROPOSED_QA_PASSED_PENDING_ROOT_FREEZE','seed':SEED,'createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sourceRecord':'https://zenodo.org/records/17165968','images':rows,'counts':counts,'groupCounts':{s:len(v) for s,v in sets.items()},'groupAssignments':split,'positiveGroupStrata':dict(strata),'reviewInputSHA256':reviews,'mergedSceneEvidenceSHA256':sha(merge_path),'annotationCorrectionsSHA256':sha(ART/'annotation-corrections.json'),'dataNatureDefinition':'Real-photo classification is source+visual screening, not forensic capture authenticity certification. Unknown edited variants and obvious markups are excluded.','annotationScope':'Author HBB geometry and coarse full-image visual coverage screening; not independent exhaustive per-object recount. 11 real alternative-material negatives plus7 reviewed background crops; crops are not independent photos.','splitPolicy':'Positive source-families stratified by median GT count, hash-selected15%test/15%val; fixed conservative negative families; background crops inherit parentgroup. No model used.','squareIncluded':False,'modelInferencePerformed':False,'oldDatasetsChanged':False,'limitations':['Public physical pile IDs unavailable; conservative registration/manual families reduce leakage but do not prove all latent capture independence.','No source self-reportedmAP is a local model result.','Source density/false-negative annotations remain imperfect; recordper-group andper-density results.']}
    save(out/'all-splits-manifest.json',full)
    tv={**full,'images':[r for r in rows if r['split'] in ('train','val')],'testContentIncluded':False,'sealedFullManifestSHA256':sha(out/'all-splits-manifest.json')};tv.pop('groupAssignments');tv.pop('positiveGroupStrata');save(out/'train-val-manifest.json',tv)
    summary={'status':full['status'],'counts':counts,'groupCounts':full['groupCounts'],'images':len(rows),'positiveImages':223,'nonPipeFullPhotos':11,'backgroundCrops':7,'allPositiveGT':sum(r['count'] for r in rows),'maxGTPerImage':max(r['count'] for r in rows),'allSourceGroupsSplitExclusive':True,'sourceAndDerivedPixelByteDuplicateSplitOverlap':0,'testContentKeptOutOfTrainVal':True,'fullProposalSHA256':sha(out/'all-splits-manifest.json'),'trainValProposalSHA256':sha(out/'train-val-manifest.json'),'modelInferencePerformed':False};save(ART/'split-proposal-summary.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
