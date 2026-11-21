"""Classical identity exclusion only; never learned inference or existing test GT use."""
from pathlib import Path
import concurrent.futures, hashlib, json, sys
import numpy as np
import cv2
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'ml/pipes-20261026/data'))
from group_round_scenes import features, registration
B = ROOT/'output/pipes-data-expansion-20261029/data/positives'
A = ROOT/'ml/pipes-data-expansion-20261029/data/positives'
NUMBERS = [29,30,44,47,59,71,72,92,104,112,120,123,125,126,131,134,145,149,
           202,204,211,219,232,237,257,259,261,264,266,277,290,307,310]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def with_identity(row):
    with Image.open(ROOT/row['imagePath']) as im:
        im=ImageOps.exif_transpose(im).convert('RGB'); shape=im.size; rgb=hashlib.sha256(im.tobytes()).hexdigest()
    return {**row,'sha256':sha(ROOT/row['imagePath']),'RGBPixelSHA256':rgb,'width':shape[0],'height':shape[1]}

def mirrored(f):
    gray=np.fliplr(f['gray']).copy()
    kp,desc=cv2.SIFT_create(nfeatures=500,contrastThreshold=.025).detectAndCompute(gray,None)
    return {**f,'gray':gray,'xy':np.array([p.pt for p in kp],dtype=np.float32).reshape(-1,2),'desc':desc}

def pair(a,b,mirror_b):
    candidates=[]
    for flip,right in [(False,b),(True,mirror_b)]:
        r=registration(a,right)
        if r:candidates.append({**r,'flip':flip})
    r=max(candidates,key=lambda r:(r['inliers'],r['ratio'])) if candidates else {'matches':0,'inliers':0,'ratio':0.,'coverageA':0.,'coverageB':0.,'flip':False}
    h=min(int(np.count_nonzero(a['ordinary']!=b['ordinary'])),int(np.count_nonzero(a['canonical']!=b['canonical'])))
    strong=r['inliers']>=20 and r['ratio']>=.6 and min(r['coverageA'],r['coverageB'])>=.04
    return {'a':a['id'],'b':b['id'],'hashDistance':h,**r,'strongMatch':strong,'suspicious':strong or h<=8 or (r['inliers']>=12 and r['ratio']>=.4)}

def main():
    paths=['output/datasets/pipe-ends-maco-v1/manifest.json','output/datasets/pipe-ends-head-retention-v1/manifest.json',
           'output/datasets/pipe-ends-maco-v1/train-val-manifest.json','output/datasets/pipe-ends-head-retention-v1/train-val-manifest.json']
    before={p:sha(ROOT/p) for p in paths}
    refs=[]
    for p in paths[:2]:
        for r in json.load(open(ROOT/p))['images']:
            refs.append({k:r[k] for k in ['id','imagePath','split','groupId']})
    assert len(refs)==315
    used={r['id'] for r in refs}
    for r in json.load(open(ROOT/'output/pipes-20261026/data/technical-index.json'))['images']:
        if r['sourceArchive']=='round' and r['id'] not in used:
            refs.append({'id':r['id'],'imagePath':r['imagePath'],'split':'unused_source_pool','groupId':None})
    source=json.load(open(B/'sources/pcloud-pipes-v2-lowcount-screen-index.json'))
    new=[{'id':r['id'],'imagePath':r['imagePath'],'number':r['number']} for r in source if r['number'] in NUMBERS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ids=list(pool.map(with_identity,refs+new)); fs=list(pool.map(features,refs+new)); mirrors=list(pool.map(mirrored,fs))
    oldids,newids=ids[:len(refs)],ids[len(refs):];old,newf=fs[:len(refs)],fs[len(refs):]
    print('FEATURES',len(old),len(newf),flush=True)
    exact=[]
    for a in newids:
        for b in oldids:
            byte=a['sha256']==b['sha256'];rgb=(a['width'],a['height'],a['RGBPixelSHA256'])==(b['width'],b['height'],b['RGBPixelSHA256'])
            if byte or rgb:exact.append({'a':a['id'],'b':b['id'],'sameBytes':byte,'sameRGB':rgb,'used315':b['id'] in used,'split':b['split']})
    results=[];summaries=[]
    for a in newf:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rr=list(pool.map(lambda k:pair(a,old[k],mirrors[k]),range(len(old))))
        for r in rr:r['existingUsed315']=r['b'] in used
        ranked=sorted(rr,key=lambda r:(r['strongMatch'],r['inliers'],r['ratio'],-r['hashDistance']),reverse=True)
        results.extend(rr);summaries.append({'id':a['id'],'flagged':[r for r in rr if r['suspicious']],'top3':ranked[:3]})
        print(a['id'],'flags',[(r['b'],r['strongMatch'],r['inliers'],r['existingUsed315']) for r in rr if r['suspicious']],flush=True)
    newpairs=[pair(a,b,mirrors[len(refs)+j]) for i,a in enumerate(newf) for j,b in enumerate(newf) if j>i]
    after={p:sha(ROOT/p) for p in paths};assert before==after
    (B/'sources/rfv2-duplicate-pairs.json').write_text(json.dumps(results,indent=2)+'\n')
    report={'status':'CLASSICAL_SCREEN_COMPLETE_REQUIRES_FAMILY_REVIEW','candidateNumbers':NUMBERS,
            'comparisonCounts':{'existingFrozen315':315,'allReferenceRows':len(refs),'newCandidates':len(new),'oldPairs':len(results),'newPairs':len(newpairs)},
            'method':'Exact file and uprightRGB SHA256; all-pairs SIFT500 at720px and mirrored SIFT500, ratio0.72, unique descriptor matches, RANSAC3px; pHash. Mirror features computed once per image, not repeatedly.',
            'strongGate':'>=20 inliers, ratio>=.6, both coverage>=.04','suspiciousGate':'strong OR pHash<=8 OR >=12inliers ratio>=.4',
            'newIdentities':newids,'referenceIdentities':oldids,'exactMatches':exact,'candidateSummaries':summaries,
            'newCandidatePairs':[r for r in newpairs if r['suspicious']],
            'protectedManifestSHA256Before':before,'protectedManifestSHA256After':after,
            'allPairsPath':str((B/'sources/rfv2-duplicate-pairs.json').relative_to(ROOT)),
            'allPairsSHA256':sha(B/'sources/rfv2-duplicate-pairs.json'),
            'testBoundary':'Existing sealed20 used only for automatic identity exclusion. No displayed test pixels, count/bbox access, learned inference, training or selection metrics.',
            'limit':'Classical non-match does not establish physical scene independence; capture/source provenance and visual families still need review.'}
    (A/'rfv2-duplicate-screen.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',flush=True)

if __name__=='__main__':main()
