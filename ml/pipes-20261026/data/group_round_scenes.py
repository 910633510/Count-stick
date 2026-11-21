#!/usr/bin/env python3
"""Provisional leakage groups from classical registration, never model outputs."""
from pathlib import Path
import collections,concurrent.futures,hashlib,json,itertools
import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-20261026/data'
cv2.setNumThreads(1)

def hash_bits(a):
    d=dctn(a.astype(float),type=2,norm='ortho')[:8,:8]
    return (d>np.median(d.ravel()[1:])).ravel()

def features(row):
    with Image.open(ROOT/row['imagePath']) as image:
        image=image.convert('RGB');small=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS));image.thumbnail((720,720));gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    variants=[]
    for rot in range(4):
        a=np.rot90(small,rot);variants.extend([hash_bits(a),hash_bits(np.fliplr(a))])
    integers=[int(''.join('1' if x else '0' for x in bits),2) for bits in variants];canonical=variants[integers.index(min(integers))]
    kp,desc=cv2.SIFT_create(nfeatures=500,contrastThreshold=.025).detectAndCompute(gray,None)
    xy=np.array([p.pt for p in kp],dtype=np.float32).reshape(-1,2)
    return {'id':row['id'],'canonical':canonical,'ordinary':variants[0],'xy':xy,'desc':desc,'gray':gray,'w':gray.shape[1],'h':gray.shape[0]}

def registration(a,b,flip=False):
    xy=b['xy'];desc=b['desc'];shape=(b['h'],b['w'])
    if flip:
        kp,desc=cv2.SIFT_create(nfeatures=500,contrastThreshold=.025).detectAndCompute(np.fliplr(b['gray']).copy(),None)
        xy=np.array([p.pt for p in kp],dtype=np.float32).reshape(-1,2)
    if a['desc'] is None or desc is None or len(a['desc'])<12 or len(desc)<12:return None
    pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(a['desc'],desc,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.72*n.distance]
    # Do not let repeated pipe texture map many features onto the same point.
    good=sorted(good,key=lambda m:m.distance);unique=[];used=set()
    for m in good:
        if m.trainIdx not in used:unique.append(m);used.add(m.trainIdx)
    if len(unique)<12:return {'matches':len(unique),'inliers':0,'ratio':0,'coverageA':0,'coverageB':0,'flip':flip}
    left=np.array([a['xy'][m.queryIdx] for m in unique]);right=np.array([xy[m.trainIdx] for m in unique])
    transform,inliers=cv2.findHomography(left,right,cv2.RANSAC,3.0,maxIters=2500,confidence=.995)
    if transform is None or inliers is None:return None
    keep=inliers.ravel().astype(bool);count=int(keep.sum())
    area=lambda points:float(cv2.contourArea(cv2.convexHull(points))) if len(points)>=3 else 0
    return {'matches':len(unique),'inliers':count,'ratio':count/len(unique),'coverageA':area(left[keep])/(a['w']*a['h']),'coverageB':area(right[keep])/(shape[0]*shape[1]),'flip':flip}

def main():
    source=OUT/'technical-index.json';index=json.loads(source.read_text());rows=[r for r in index['images'] if r['sourceArchive']=='round'];assert len(rows)==1025
    data=[None]*len(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(features,row):i for i,row in enumerate(rows)}
        for n,f in enumerate(concurrent.futures.as_completed(pending),1):
            data[pending[f]]=f.result()
            if n%100==0:print('Features',n,flush=True)
    a=np.stack([r['canonical'] for r in data]);b=np.stack([r['ordinary'] for r in data]);dist=np.minimum(cdist(a,a,metric='hamming'),cdist(b,b,metric='hamming'))*64
    np.fill_diagonal(dist,1000);pairs=set()
    for i in range(len(rows)):
        for j in np.argsort(dist[i])[:10]:
            if dist[i,j]<=22:pairs.add(tuple(sorted((i,int(j)))))
        for j in np.where(dist[i]<=8)[0]:pairs.add(tuple(sorted((i,int(j)))))
        for j in range(i+1,min(i+3,len(rows))):pairs.add((i,j))
    assert len(pairs)<25000
    pairs=sorted(pairs);print('Registration pairs',len(pairs),flush=True);edges=[];suspicious=[]
    def match(pair):
        i,j=pair;r=registration(data[i],data[j]);r=r or {'matches':0,'inliers':0,'ratio':0,'coverageA':0,'coverageB':0,'flip':False}
        def passed(q):return q['inliers']>=20 and q['ratio']>=.6 and min(q['coverageA'],q['coverageB'])>=.04
        if not passed(r) and dist[i,j]<=8:
            flipped=registration(data[i],data[j],True)
            if flipped and flipped['inliers']>r['inliers']:r=flipped
        return i,j,r,passed(r)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for n,res in enumerate(pool.map(match,pairs),1):
            i,j,m,ok=res;row={'a':rows[i]['id'],'b':rows[j]['id'],'hashDistance':float(dist[i,j]),**m}
            if ok:edges.append(row)
            elif dist[i,j]<=8 or (m['inliers']>=12 and m['ratio']>=.4):suspicious.append(row)
            if n%1000==0:print('Matched',n,'/',len(pairs),flush=True)
    parent=list(range(len(rows)));id_idx={r['id']:i for i,r in enumerate(rows)}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for e in edges:
        a,b=find(id_idx[e['a']]),find(id_idx[e['b']]);parent[max(a,b)]=min(a,b)
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):groups[find(i)].append(row['id'])
    grouped=[{'groupId':'maco_round_scene_'+min(v).split('PipeHBB')[-1],'imageIds':v,'images':len(v),'basis':'Classical SIFT/RANSAC connected component; source physical IDs unavailable; visual review still required.'} for _,v in sorted(groups.items())]
    report={'status':'PROVISIONAL_SCENE_GROUPS_REQUIRE_VISUAL_QA','technicalIndexSHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'method':{'descriptor':'SIFT500 on aspect-preserving720px grayscale, no learned weights','candidatePairs':'Top10 canonical/original pHash neighbors <=22bits, all<=8bits, and nearby IDs; mirrored match attempt for close hashes','unionGate':'At least20unique feature matches consistent with homography;>=60%inliers; both spatial hulls cover>=4%frame','groupingLimit':'False unions are conservative. Missing links can remain; no claim of author-confirmed physical pile identity or complete near-duplicate exclusion.'},'imageCount':len(rows),'candidatePairs':len(pairs),'groups':grouped,'groupSizeHistogram':dict(collections.Counter(g['images'] for g in grouped)),'acceptedRegistrationEdges':edges,'suspiciousUnmergedPairs':suspicious,'noLearnedModelInference':True,'noSplitAssigned':True}
    (OUT/'groups-provisional.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'groups':len(grouped),'edges':len(edges),'suspicious':len(suspicious),'largest':sorted([g['images'] for g in grouped],reverse=True)[:12]}))

if __name__=='__main__':main()
