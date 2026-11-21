#!/usr/bin/env python3
"""Audit the original files; freeze species-disjoint splits before model fitting."""
from pathlib import Path
import collections, hashlib, json, math, os, random, time
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import imagehash
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/datasets/colony-figshare-22022540-v3'
RAW=BASE/'raw'; PREP=BASE/'yolo-species-v1'
SEED=20261025

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt=json.loads((BASE/'download-receipt.json').read_text());assert receipt['status']=='COMPLETE_ALL_FILES_MD5_VERIFIED'
    ann=json.loads((RAW/'annot_COCO.json').read_text());imgs={i['id']:i for i in ann['images']}
    assert len(imgs)==369 and len(ann['annotations'])==56865
    byimg=collections.defaultdict(list);invalid=[];clipped=[];duplicates=[];seen=set()
    for a in ann['annotations']:
        im=imgs[a['image_id']];assert int(im['file_name'][2:4])==a['category_id']
        x,y,w,h=a['bbox'];assert all(math.isfinite(v) for v in [x,y,w,h])
        if w<=0 or h<=0:
            invalid.append({'annotationId':a['id'],'imageId':im['id'],'file':im['file_name'],'bbox':a['bbox']});continue
        x2,y2=min(im['width'],x+w),min(im['height'],y+h);x1,y1=max(0,x),max(0,y)
        if x2<=x1 or y2<=y1:
            invalid.append({'annotationId':a['id'],'imageId':im['id'],'file':im['file_name'],'bbox':a['bbox']});continue
        key=(im['id'],x1,y1,x2,y2)
        if key in seen:duplicates.append({'annotationId':a['id'],'imageId':im['id']});continue
        seen.add(key)
        if [x1,y1,x2-x1,y2-y1]!=a['bbox']:clipped.append({'annotationId':a['id'],'file':im['file_name'],'original':a['bbox'],'clipped':[x1,y1,x2-x1,y2-y1]})
        byimg[im['id']].append({'id':a['id'],'bbox':[x1,y1,x2-x1,y2-y1]})
    # Quarantine whole images with degenerate/duplicate labels, rather than inventing ground-truth counts.
    quarantine={i['imageId'] for i in invalid+duplicates}
    exact={};phashes={};image_records=[];orientation_fixes=[]
    for im in ann['images']:
        p=RAW/im['file_name'];assert p.is_file()
        with Image.open(p) as pil:
            pil.load();orientation=pil.getexif().get(274,1);actual=ImageOps.exif_transpose(pil)
            assert actual.size==(im['width'],im['height']),im['file_name']+' remains mismatched after EXIF normalization'
            if orientation not in (1,None):orientation_fixes.append({'file':im['file_name'],'orientation':orientation,'storedSize':list(pil.size),'displaySize':list(actual.size)})
            phashes[im['id']]=imagehash.phash(actual)
        h=sha(p);exact.setdefault(h,[]).append(im['id'])
        image_records.append({'id':im['id'],'file':im['file_name'],'width':im['width'],'height':im['height'],'sha256':h,'species':int(im['file_name'][2:4]),'validBoxCount':len(byimg[im['id']])})
    assert not any(len(v)>1 for v in exact.values()),'Exact image duplicates require grouping before split'
    near=[]
    for i,a in enumerate(ann['images']):
        for b in ann['images'][i+1:]:
            dist=phashes[a['id']]-phashes[b['id']]
            if dist<=4:near.append({'imageA':a['file_name'],'imageB':b['file_name'],'distance':int(dist),'sameSpecies':a['file_name'][:4]==b['file_name'][:4]})
    (BASE/'photo-duplication-audit.json').write_text(json.dumps({'nearPairs':near,'imageRecords':image_records,'orientationFixes':orientation_fixes},indent=2)+'\n')
    # Similar-looking round plates can trigger pHash even across species. Keep entire
    # connected species components together conservatively; do not label these as proven duplicates.
    parent={n:n for n in range(1,25)}
    def find(n):
        while parent[n]!=n:n=parent[n]
        return n
    for pair in near:parent[find(int(pair['imageA'][2:4]))]=find(int(pair['imageB'][2:4]))
    components=collections.defaultdict(list)
    for n in range(1,25):components[find(n)].append(n)
    components=sorted(components.values());random.Random(SEED).shuffle(components)
    target=round(len(image_records)*0.15)
    size=lambda c:sum(i['species'] in c and i['id'] not in quarantine for i in image_records)
    large=[c for c in components if size(c)>target]
    pool=[c for c in components if size(c)<=target]
    chosen={}
    for key in ['test','val']:
        picked=[]
        while pool and (len(picked)<3 or sum(size(c) for c in picked)<target):picked.append(pool.pop(0))
        chosen[key]=picked
    chosen['train']=large+pool
    groups={k:sorted(n for c in v for n in c) for k,v in chosen.items()}
    assert all(len(v)>=3 for v in groups.values())
    membership={n:k for k,v in groups.items() for n in v}
    assert all(membership[int(p['imageA'][2:4])]==membership[int(p['imageB'][2:4])] for p in near)
    split={k:[i['file'] for i in image_records if i['species'] in v and i['id'] not in quarantine] for k,v in groups.items()}
    PREP.mkdir(parents=True,exist_ok=True)
    manifest={'schemaVersion':1,'frozenAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'seed':SEED,'sourceAnnotationSHA256':sha(RAW/'annot_COCO.json'),'labelMapping':'All 24 species collapsed to one class: colony','splitUnit':'Entire bacterial species plus conservative pHash-neighbor connected components. Components too large for a15percent holdout go to training; seeded remaining components fill test/validation to15percent with at least3components each. Validation/test are unseen species; no near-similarity candidate crosses splits.','speciesGroups':groups,'similarityComponents':components,'similarityCaveat':'pHash candidates may be distinct colonies on similar round plates; grouping is conservative, not a claim they are duplicate photos.','splits':split,'images':image_records,'quarantinedImages':[imgs[i]['file_name'] for i in sorted(quarantine)],'invalidAnnotations':invalid,'duplicateBoxes':duplicates,'clippedOutOfImageBoxes':clipped,'nearDuplicatePairs':near,'exifOrientationNormalizations':orientation_fixes,'counts':{k:{'images':len(v),'boxes':sum(len(byimg[i['id']]) for i in ann['images'] if i['file_name'] in v)} for k,v in split.items()}}
    frozen=PREP/'manifest.json'
    if frozen.exists():raise RuntimeError('Frozen manifest already exists; never overwrite an experiment split')
    frozen.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for k,files in split.items():
        idir=PREP/'images'/k;ldir=PREP/'labels'/k;idir.mkdir(parents=True,exist_ok=True);ldir.mkdir(parents=True,exist_ok=True)
        for name in files:
            im=next(i for i in ann['images'] if i['file_name']==name)
            if name in {f['file'] for f in orientation_fixes}:
                with Image.open(RAW/name) as original:ImageOps.exif_transpose(original).convert('RGB').save(idir/name,quality=98,subsampling=0,exif=b'')
            else:os.link(RAW/name,idir/name)
            lines=[]
            for a in byimg[im['id']]:
                x,y,w,h=a['bbox'];lines.append(f'0 {(x+w/2)/im["width"]:.9f} {(y+h/2)/im["height"]:.9f} {w/im["width"]:.9f} {h/im["height"]:.9f}')
            (ldir/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+'\n')
    (PREP/'data.yaml').write_text('path: '+json.dumps(str(PREP))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: colony\n')
    # Audit visualization is a diagnostic annotation plot, not training pixels or fabricated app UI.
    for k in ['train','val','test']:
        names=split[k][:8];canvas=Image.new('RGB',(4*320,2*350),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for idx,name in enumerate(names):
            im=next(i for i in ann['images'] if i['file_name']==name)
            with Image.open(RAW/name) as p:thumb=ImageOps.exif_transpose(p).convert('RGB');thumb.thumbnail((310,310))
            x0=(idx%4)*320;y0=(idx//4)*350;canvas.paste(thumb,(x0,y0));sx=thumb.width/im['width'];sy=thumb.height/im['height']
            for a in byimg[im['id']]:
                x,y,w,h=a['bbox'];draw.rectangle((x0+x*sx,y0+y*sy,x0+(x+w)*sx,y0+(y+h)*sy),outline='#00ff80',width=1)
            draw.text((x0+3,y0+315),f'{name} | {len(byimg[im["id"]])} boxes',fill='#111111')
        canvas.save(PREP/(k+'-audit.jpg'),quality=90)
    report={'status':'PASS_DATA_INTEGRITY_AND_PRETRAIN_SPLIT_FREEZE','manifest':str(frozen),'manifestSHA256':sha(frozen),'counts':manifest['counts'],'quarantinedImages':manifest['quarantinedImages'],'outOfBoundsBoxesClipped':len(clipped),'nearDuplicatePairs':len(near),'exifNormalizedImages':len(orientation_fixes),'dataYaml':str(PREP/'data.yaml')}
    (PREP/'audit-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
