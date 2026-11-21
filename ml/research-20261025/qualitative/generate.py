#!/usr/bin/env python3
"""Render old-val scientific diagnostics from frozen caches, without a model."""
from pathlib import Path
import json, hashlib, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'output/datasets/colony-figshare-22022540-v3/yolo-species-v1'
CACHE = ROOT / 'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/evaluation'
OUT = ROOT / 'output/research-20261025/qualitative'
OUT.mkdir(parents=True, exist_ok=True)
CONFIDENCE = .3
SPECIES = ('sp04', 'sp09', 'sp13')
font_path = '/System/Library/Fonts/Helvetica.ttc'
F = ImageFont.truetype(font_path, 22)
SMALL = ImageFont.truetype(font_path, 17)
TITLE = ImageFont.truetype(font_path, 27)


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def truth(name):
    assert name in val_names
    p = DATA / 'images/val' / name
    with Image.open(p) as im: size = im.size
    w,h = size
    boxes = []
    for line in (DATA / 'labels/val' / Path(name).with_suffix('.txt')).read_text().splitlines():
        cls,x,y,bw,bh = map(float,line.split()); assert cls == 0
        boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    return np.array(boxes).reshape(-1,4), size


def iou(a,b):
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    tl=np.maximum(a[:,None,:2],b[None,:,:2]);br=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(br-tl,0).prod(2)
    return inter / ((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter+1e-9)


def compare(gt,pred):
    b = np.asarray(pred['xyxy'],float).reshape(-1,4)[np.asarray(pred['confidence']) >= CONFIDENCE]
    ov = iou(gt,b); mg=set();mp=set()
    if ov.size:
        rr,cc=linear_sum_assignment(-((ov>=.5).astype(float)*1000+ov))
        pairs=[(int(i),int(j)) for i,j in zip(rr,cc) if ov[i,j]>=.5]
        mg={i for i,j in pairs};mp={j for i,j in pairs}
    centers=(b[:,:2]+b[:,2:])/2
    center_in_gt=((centers[:,None,:] >= gt[None,:,:2]).all(2) & (centers[:,None,:] <= gt[None,:,2:]).all(2)).any(1) if len(b) and len(gt) else np.zeros(len(b),bool)
    return {'boxes':b,'matchedGT':mg,'matchedPred':mp,'TP':len(mg),'FP':len(b)-len(mp),'FN':len(gt)-len(mg),'count':len(b),'countError':len(b)-len(gt),'unmatchedPredictionCenterInsideGT':sum(bool(center_in_gt[j]) for j in range(len(b)) if j not in mp)}


def dashed(draw, points, fill, width=2, dash=9):
    (x1,y1),(x2,y2)=points; length=max(abs(x2-x1),abs(y2-y1))
    if length<1:return
    for start in np.arange(0,length,dash*2):
        end=min(length,start+dash)
        draw.line((x1+(x2-x1)*start/length,y1+(y2-y1)*start/length,x1+(x2-x1)*end/length,y1+(y2-y1)*end/length),fill=fill,width=width)


def render(name, image, gt, comp, bounds, tag):
    w,h=image.size;x0,y0,x1,y1=bounds;cw,ch=x1-x0,y1-y0
    panel_w=820;panel_h=min(820,max(360,round(800*ch/cw)+10));top=104;header=65
    canvas=Image.new('RGB',(2*panel_w,top+2*(panel_h+header)+44),'#f4f5f7');draw=ImageDraw.Draw(canvas)
    draw.text((16,10),f'{name} | {tag} | OLD VALIDATION / CACHED PREDICTIONS',fill='#101720',font=TITLE)
    draw.text((16,46),'GT cyan | matched prediction green | unmatched prediction red | unmatched GT amber',fill='#253047',font=SMALL)
    draw.text((16,73),'IoU match >=0.5, confidence >=0.3. Gray dashed lines: quadrant ownership; not causal proof.',fill='#253047',font=SMALL)
    for idx,mode in enumerate(('raw','gt','whole','tile')):
        px=(idx%2)*panel_w;py=top+(idx//2)*(panel_h+header)
        if mode=='raw': title='Source image (display scaling only)'
        elif mode=='gt':title=f'Frozen GT boxes: {len(gt)} total'
        else:
            c=comp[mode];title=f'{mode.upper()} count {c["count"]} | TP {c["TP"]} FP {c["FP"]} FN {c["FN"]}'
        draw.text((px+12,py+10),title,fill='#101720',font=F)
        scale=min((panel_w-20)/cw,(panel_h-10)/ch)
        view=image.crop(bounds);view=view.resize((round(cw*scale),round(ch*scale)),Image.Resampling.LANCZOS)
        dx=px+(panel_w-view.width)//2;dy=py+header+(panel_h-view.height)//2
        overlay=ImageDraw.Draw(view)
        def box(bb,color,width=2):
            l,t,r,b=bb
            if r<x0 or b<y0 or l>x1 or t>y1:return
            rect=((max(l,x0)-x0)*scale,(max(t,y0)-y0)*scale,(min(r,x1)-x0)*scale,(min(b,y1)-y0)*scale)
            overlay.rectangle(rect,outline=color,width=width)
        if mode=='gt':
            for bb in gt:box(bb,'#00a9cf',2)
        elif mode in comp:
            c=comp[mode]
            for i,bb in enumerate(gt):
                if i not in c['matchedGT']:box(bb,'#ffad00',2)
            for i,bb in enumerate(c['boxes']):box(bb,'#09a74f' if i in c['matchedPred'] else '#ed243f',2)
            if mode=='tile':
                if x0<w/2<x1:dashed(overlay,(((w/2-x0)*scale,0),((w/2-x0)*scale,view.height)),fill='#c7c7c7')
                if y0<h/2<y1:dashed(overlay,((0,(h/2-y0)*scale),(view.width,(h/2-y0)*scale)),fill='#c7c7c7')
        canvas.paste(view,(dx,dy))
    draw.text((16,canvas.height-31),f'Pixel window xyxy={list(bounds)}; counts in headings refer to full image. No labels changed; no inference.',fill='#253047',font=SMALL)
    destination=OUT/(Path(name).stem+'-'+tag+'.png');canvas.save(destination)
    return str(destination.relative_to(ROOT))


manifest_path=DATA/'manifest.json';manifest=json.loads(manifest_path.read_text());val_names=set(manifest['splits']['val'])
paths={'whole':CACHE/'validation-predictions.json','tile':CACHE/'validation-tiled-predictions.json'}
caches={k:json.loads(v.read_text()) for k,v in paths.items()}
selected=[]
for species in SPECIES:
    rows=[]
    for name in sorted(val_names):
        if name.startswith(species+'_'):
            gt,size=truth(name);count=sum(c>=CONFIDENCE for c in caches['tile'][name]['confidence'])
            rows.append({'image':name,'species':species,'GT':len(gt),'tileCount':count,'tileCountError':count-len(gt)})
    selected.append(min(rows,key=lambda r:(-abs(r['tileCountError']),r['image'])))
protocol={'scope':'OLD_VALIDATION_ONLY_NO_NEW_INFERENCE','selectionRule':'For each sp04/sp09/sp13 select maximum absolute tile count error at confidence>=0.3; ties ascending filename. This is diagnostic hard-case selection, not an unbiased evaluation sample.','matchRule':'IoU>=0.5, maximum valid-match cardinality then overlap; same rule as historical evaluator.','selected':selected,'sourceSHA256':{str(manifest_path.relative_to(ROOT)):digest(manifest_path),**{str(p.relative_to(ROOT)):digest(p) for p in paths.values()}},'modelInferencePerformed':False,'testDataRead':False}
protocol_path=OUT/'selection-and-provenance.json'
if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
results=[]
for row in selected:
    name=row['image'];gt,size=truth(name);w,h=size
    comp={k:compare(gt,caches[k][name]) for k in caches}
    with Image.open(DATA/'images/val'/name) as source:image=source.convert('RGB')
    overview=render(name,image,gt,comp,(0,0,w,h),'overview')
    # Deterministic old-val error concentration crop, solely for readable diagnosis.
    tile=comp['tile']
    if row['species']=='sp13':
        focus_boxes=np.array([b for i,b in enumerate(tile['boxes']) if i not in tile['matchedPred']]).reshape(-1,4);focus_kind='metricFP'
    else:
        focus_boxes=np.array([b for i,b in enumerate(gt) if i not in tile['matchedGT']]).reshape(-1,4);focus_kind='metricFN'
    centers=(focus_boxes[:,:2]+focus_boxes[:,2:])/2; side=int(.45*min(w,h)); windows=[]
    for fy in np.linspace(.225,.775,7):
        for fx in np.linspace(.225,.775,7):
            x=int(min(w-side,max(0,fx*w-side/2)));y=int(min(h-side,max(0,fy*h-side/2)))
            inside=((centers>=np.array([x,y])).all(1)&(centers<=np.array([x+side,y+side])).all(1)).sum()
            windows.append((int(inside),x,y,x+side,y+side))
    chosen=sorted(windows,key=lambda q:(-q[0],q[2],q[1]))[0];bounds=chosen[1:]
    detail=render(name,image,gt,comp,bounds,'detail')
    row={**row,'size':size,'figures':{'overview':overview,'detail':detail},'detailWindow':bounds,'detailSelection':f'45%-min-dimension square, 7x7 fixed center grid, largest number of tile {focus_kind} centers, ties y then x.', 'metrics':{k:{q:int(v) for q,v in c.items() if q not in ('boxes','matchedGT','matchedPred')} for k,c in comp.items()},'imageSHA256':digest(DATA/'images/val'/name),'labelSHA256':digest(DATA/'labels/val'/Path(name).with_suffix('.txt'))}
    if row['species']=='sp13':
        rim=(0,1700,1600,h)
        row['figures']['rimDetail']=render(name,image,gt,comp,rim,'rim-detail')
        row['rimDetailWindow']=rim
        row['rimDetailSelection']='Additional bottom-left plate-rim window chosen after overview inspection; descriptive old-val diagnosis only.'
        row['unmatchedTilePredictionsInRimWindow']=int(sum(rim[0]<=c[0]<=rim[2] and rim[1]<=c[1]<=rim[3] for c in centers))
    fn=[i for i in range(len(gt)) if i not in tile['matchedGT']]
    gc=(gt[:,:2]+gt[:,2:])/2
    far=lambda i:min(abs(gc[i,0]-w/2),abs(gc[i,1]-h/2))>.1*min(w,h)
    row['tileMissedGTMoreThan10PercentMinDimensionFromBothOwnershipSeams']=int(sum(far(i) for i in fn))
    row['tileGTMatchedWholeButNotTile']=len(comp['whole']['matchedGT']-tile['matchedGT'])
    results.append(row)
report={'status':'FIGURES_RENDERED_FROM_CACHED_OLD_VALIDATION','protocol':str(protocol_path.relative_to(ROOT)),'selection':results,'interpretationLimits':['FP and FN are metric labels relative to frozen boxes, not proof of biological presence/absence or annotation mistakes.','Final tile caches already underwent ownership filtering and NMS; raw rejected predictions are unavailable, so tile-boundary causation cannot be established.','These are deliberately selected worst old-validation cases, not prevalence estimates or independent test results.'],'modelInferencePerformed':False,'labelsChanged':False,'testDataRead':False}
(OUT/'figure-records.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'selected':[{'image':r['image'],'metrics':r['metrics'],'detailWindow':r['detailWindow']} for r in results]},indent=2))
