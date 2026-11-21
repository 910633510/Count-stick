"""Classical duplicate screen only; no detector inference or GT-driven selection."""
from pathlib import Path
import json,hashlib,collections,concurrent.futures,sys,time
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'ml/pipes-20261026/data'))
from group_round_scenes import features,registration
OUT=ROOT/'output/pipes-hardening-20261026/data/positives';ART=ROOT/'ml/pipes-hardening-20261026/data/positives'
rows=[]
for name in ['root-kaggle-manual-annotations.json','root-commons-manual-annotations.json']:
 for r in json.load(open(ART/name))['images']:
  rows.append({'id':r.get('id','kaggle_'+str(r.get('number')).zfill(3)),'imagePath':r.get('imagePath',r.get('path'))})
for n in [18,19,82,91,116,121,124,125,129,133,145]:rows.append({'id':f'commons_{n:03d}','imagePath':f'output/pipes-hardening-20261026/data/positives/commons-thumbnails/commons_{n:03d}.jpg'})
gh=json.load(open(OUT/'sources/github-image-index.json'));rows.append({'id':'github_005','imagePath':next(r['path'] for r in gh if r['id']=='github_005')})
for n in [21,343,528,780,795]:rows.append({'id':f'new_maco_{n:04d}','imagePath':f'output/pipes-20261026/data/raw/round/Steel pipe/HBB/PipeHBB{n:08d}.jpg'})
# This spent old holdout is read only for duplicate identity screening, never training or selection.
old=json.load(open(ROOT/'output/datasets/pipe-ends-maco-v1/manifest.json'))['images'];used={r['id']:r['groupId'] for r in old};index=json.load(open(ROOT/'output/pipes-20261026/data/technical-index.json'));oldrows=[{'id':r['id'],'imagePath':r['imagePath']} for r in index['images'] if r['sourceArchive']=='round'];known={r['id'] for r in oldrows}
oldrows += [{'id':r['id'],'imagePath':r['imagePath']} for r in old if r['id'] not in known]
allrows=oldrows+rows
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:ff=list(pool.map(features,allrows))
oldf=ff[:len(oldrows)];newf=ff[len(oldrows):];print('features',len(oldf),len(newf),flush=True)
results=[]
def test(pair):
 a,b=pair;h=min(int(np.count_nonzero(a['canonical']!=b['canonical'])),int(np.count_nonzero(a['ordinary']!=b['ordinary'])));r=registration(a,b) or {'matches':0,'inliers':0,'ratio':0.,'coverageA':0.,'coverageB':0.,'flip':False}
 if h<=12:
  q=registration(a,b,True)
  if q and q['inliers']>r['inliers']:r=q
 strong=r['inliers']>=20 and r['ratio']>=.6 and min(r['coverageA'],r['coverageB'])>=.04
 suspicious=strong or h<=8 or (r['inliers']>=12 and r['ratio']>=.4)
 return {'a':a['id'],'b':b['id'],'hashDistance':h,'oldUsed241':b['id'] in used,'oldGroupId':used.get(b['id']),**r,'strongMatch':strong,'suspicious':suspicious}
for i,a in enumerate(newf):
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(test,[(a,b) for b in oldf]))
 ranked=sorted(rr,key=lambda x:(x['strongMatch'],x['inliers'],x['ratio'],-x['hashDistance']),reverse=True);results.extend([r for r in rr if r['suspicious']]);print(a['id'],'matches',[{k:v for k,v in r.items() if k in ['b','oldUsed241','inliers','hashDistance','strongMatch']} for r in ranked[:2]],flush=True)
newpairs=[]
for i,a in enumerate(newf):
 for b in newf[i+1:]:
  r=test((a,b))
  if r['suspicious']:newpairs.append(r)
report={'status':'CLASSICAL_SCREEN_COMPLETE_REQUIRES_MANUAL_MATCH_REVIEW','newCandidates':rows,'oldRoundCount':1025,'oldUsed241Count':241,'oldComparisonRows':len(oldrows),'allOldPairsCompared':len(rows)*len(oldrows),'flaggedOldPairs':results,'flaggedNewPairs':newpairs,'method':'SIFT500/RANSAC allnew xallMACO1025 plusoldnon-round/derivednegative241 extras; canonical/originalpHash; mirroredSIFTforhash<=12. No learned detector.','oldHoldoutUse':'Spent old37 images only read for classical duplicate exclusion; no training, no model metrics, no newGT taken.','limits':'Not a proof of physicalscene independence; manual geography/source-family grouping required. Candidate selfmatches to unused MACO are expected.'}
(ART/'duplicate-screen.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',len(results),len(newpairs),flush=True)
