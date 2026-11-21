#!/usr/bin/env python3
"""Assemble visually annotated positive candidates. Root freezes groups later."""
from pathlib import Path
import json, hashlib, math
import numpy as np
from PIL import Image
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'output/research-20261025-hardcases/positives'
OUT=BASE/'annotated';(OUT/'images').mkdir(parents=True,exist_ok=True)
own_path=ROOT/'ml/research-20261025-hardcases/positives/own-crop-drafts.json'
root_path=ROOT/'ml/research-20261025-hardcases/root-positive-decisions.json'
own=json.loads(own_path.read_text());root=json.loads(root_path.read_text())
review_path=ROOT/'ml/research-20261025-hardcases/positives/second-review-decisions.json'
second=json.loads(review_path.read_text());accepted_own={d['index']:d for d in second['accepted']}
props={x['index']:x for x in json.loads((BASE/'review/author-assisted-proposals.json').read_text())['images']}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
decisions=[]
for d in own['images']:
 if d['index'] not in accepted_own:continue
 assert len(d['boxes'])==accepted_own[d['index']]['count']
 decisions.append({'index':d['index'],'crop':d['crop'],'boxes':d['boxes'],'notes':d['notes'],'sourceProposalIds':d['sourceProposalIds'],'annotator':'/root/colony_papers','secondReview':'PASS_BY_ROOT_VISUAL_REVIEW'})
for d in root['decisions']:
 if d['status'].startswith('ACCEPT'):
  decisions.append({'index':d['index'],'crop':d['cropXYXY'],'boxes':d['xyxy'],'notes':d['method']+' '+d['observation'],'annotator':'/root','secondReview':'PASS_BY_COLONY_PAPERS_VISUAL_REVIEW'})
assert len(decisions)==11 and len({d['index'] for d in decisions})==11
previous=set()
old=ROOT/'output/datasets/colony-figshare-22022540-v3/yolo-species-v1/manifest.json'
previous.update(x['sha256'] for x in json.loads(old.read_text())['images'])
clinical=ROOT/'output/datasets/colony-mixed-isolate-v1/train-val-manifest.json'
previous.update(x['sha256'] for x in json.loads(clinical.read_text())['images'])
rows=[];hashes={};perceptual={}
for d in decisions:
 p=props[d['index']];source=ROOT/p['imagePath'];assert sha(source)==p['sha256']
 assert p['sha256'] not in previous
 with Image.open(source) as image:
  image.load();assert image.mode=='RGB' and image.size==(1800,1224) and image.getexif().get(274,1)==1
  original=np.array(image);crop=d['crop'] or [0,0,image.width,image.height]
  x0,y0,x1,y1=crop;assert 0<=x0<x1<=image.width and 0<=y0<y1<=image.height
  pixels=image.crop(crop);name=p['id']+('-countable-crop' if d['crop'] else '-full')
  destination=OUT/'images'/(name+'.png');pixels.save(destination)
  with Image.open(destination) as check:
   check.load();assert np.array_equal(np.array(check),original[y0:y1,x0:x1]);assert check.getexif().get(274,1)==1
  center=image.crop((360,245,1440,979)).convert('L').resize((32,32),Image.Resampling.LANCZOS)
  coefficients=dctn(np.asarray(center,float),type=2,norm='ortho')[:8,:8];bits=(coefficients>np.median(coefficients[1:])).ravel();perceptual[name]=bits
 boxes=[]
 for b in d['boxes']:
  assert all(math.isfinite(float(v)) for v in b)
  bx1,by1,bx2,by2=b;assert 0<=bx1<bx2<=1800 and 0<=by1<by2<=1224
  translated=[max(bx1,x0)-x0,max(by1,y0)-y0,min(bx2,x1)-x0,min(by2,y1)-y0]
  assert 0<=translated[0]<translated[2]<=x1-x0 and 0<=translated[1]<translated[3]<=y1-y0
  boxes.append(translated)
 assert len({tuple(b) for b in boxes})==len(boxes)
 hs=sha(destination);assert hs not in hashes;hashes[hs]=name
 rows.append({'id':name,'index':d['index'],'split':None,'source':'zenodo20178469-hyperspectral-derived-RGB','species':p['species'],'speciesSemantics':'Source nominal strain label; all visually discernible microbial colony bodies are counted, including differently colored source-contaminant regions. No species diagnosis is inferred.','groupId':p['groupId'],'groupingBasis':'Original CSV id identifies the same sample series; all ages/dates/rotations/crops of that id must stay together.','bucket':'hard_positive','imagePath':str(destination.relative_to(ROOT)),'width':x1-x0,'height':y1-y0,'sha256':hs,'xyxy':boxes,'count':len(boxes),'annotationQA':'PENDING_SECOND_REVIEW' if d['secondReview'].startswith('PENDING') else 'PASS','annotationMethod':'Author-semantic-mask-assisted visual instance boxes; manual splitting/cropping and individually reviewed retained boxes. Not untouched author instance GT and not from-scratch pixel-mask labels.','annotationNotes':d['notes'],'annotator':d['annotator'],'secondReview':d['secondReview'],'negativeKind':None,'sourceImagePath':p['imagePath'],'sourceImageSHA256':p['sha256'],'sourceSemanticMaskPath':p['sourceSemanticMaskPath'],'sourceSemanticMaskSHA256':p['sourceMaskSHA256'],'sourceRecord':'https://zenodo.org/records/20178469','sourceCSV':p['sourceCSV'],'cropXYXYInOriginal':crop,'ownBoxesXYXYInOriginal':d['boxes'],'sourceProposalIds':d.get('sourceProposalIds'),'displayGainUsedForReviewOnly':2,'trainingImageBrightnessOrContrastAltered':False})
 rows[-1]['sourceRGBPixelSHA256']=hashlib.sha256(original.tobytes()).hexdigest()
 rows[-1]['RGBPixelSHA256']=hashlib.sha256(original[y0:y1,x0:x1].tobytes()).hexdigest()
assert len({r['groupId'] for r in rows})==11
pairs=[]
for i,a in enumerate(rows):
 for b in rows[i+1:]:
  distance=int(np.count_nonzero(perceptual[a['id']]!=perceptual[b['id']]))
  if distance<=8:pairs.append({'a':a['id'],'b':b['id'],'hammingDistance':distance,'interpretation':'Flag for visual/source-identity review; perceptual similarity is not proof of a duplicate.'})
quarantine=[{'index':r['index'],'sourceImagePath':props[r['index']]['imagePath'],'groupId':props[r['index']]['groupId'],'status':'FULL_SOURCE_NOT_APPROVED_FOR_DETECTOR','count':None,'reason':'Source outside the accepted explicit crop contains ambiguous merged/clipped/glare regions or is not completely instance-annotated; only the reviewed crop is eligible.'} for r in rows if r['cropXYXYInOriginal']!=[0,0,1800,1224]]
for d in own['quarantinedFullImages']:
 if not any(x['index']==d['index'] for x in quarantine):quarantine.append({'index':d['index'],'sourceImagePath':props[d['index']]['imagePath'],'groupId':props[d['index']]['groupId'],'status':'QUARANTINE_UNRESOLVED_INSTANCE_COUNT','count':None,'reason':d['reason']})
for d in root['decisions']:
 if d['status'].startswith('QUARANTINE'):quarantine.append({'index':d['index'],'sourceImagePath':props[d['index']]['imagePath'],'groupId':props[d['index']]['groupId'],'status':d['status'],'count':None,'reason':d['observation']})
for d in second['rejected']:
 quarantine.append({'index':d['index'],'sourceImagePath':props[d['index']]['imagePath'],'groupId':props[d['index']]['groupId'],'status':d['status'],'count':None,'reason':d['reason']})
 draft=OUT/'images'/(props[d['index']]['id']+'-countable-crop.png')
 if draft.exists():
  q=OUT/'quarantine'/(props[d['index']]['id']+'-DRAFT-REJECTED.png');q.parent.mkdir(exist_ok=True)
  if q.exists():assert q.read_bytes()==draft.read_bytes()
  draft.rename(q)
  quarantine[-1]['rejectedDraftImagePath']=str(q.relative_to(ROOT))
manifest={'status':'ANNOTATION_QA_PASSED_PENDING_ROOT_GROUP_FREEZE','testContentIncluded':False,'images':rows,'imageCount':len(rows),'physicalSampleGroups':len({r['groupId'] for r in rows}),'instanceBoxes':sum(r['count'] for r in rows),'dataDomain':'Real agar-plate RGB derived from a hyperspectral imaging system, not smartphone capture.','originalArchives':'Both downloaded archives passed their publisher MD5; member extraction checked CRC and safe paths. Only Zenodo20178469 is used for these positives.','splitPolicy':'No split assigned. Root must freeze source/sample groups and combine reviewed negatives before model evaluation. Existing sealed clinical test36 not read.','oldTrainValExactImageSHAOverlap':False,'sealedNewTestSHAOverlap':'Root must check hashes without exposing sealed pixels/GT; this preparation never opens that manifest.','perceptualNearDuplicateFlags':pairs,'quarantine':quarantine,'sourceDraftSHA256':{'own':sha(own_path),'root':sha(root_path),'secondReview':sha(review_path)},'noModelPredictionsUsed':True,'newClinicalTest36Read':False}
manifest['perceptualAudit']={'method':'Source-image center crop (360,245,1440,979), grayscale32x32 DCT; low8x8 bits thresholded by median excluding first coefficient row.','flagHammingAtMost':8,'meaning':'Screen only; no flag is not proof of acquisition independence. CSV sample-id grouping remains mandatory.'}
assert {p.name for p in (OUT/'images').glob('*.png')}=={Path(r['imagePath']).name for r in rows}
(OUT/'positive-candidates.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':manifest['status'],'images':len(rows),'groups':manifest['physicalSampleGroups'],'boxes':manifest['instanceBoxes'],'pHashFlags':pairs}))
