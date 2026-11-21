#!/usr/bin/env python3
"""Freeze a validation-chosen threshold, then evaluate the unseen-species test once."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import argparse,collections,hashlib,json,time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image,ImageDraw
from ultralytics import YOLO
DATA=ROOT/'output/datasets/colony-figshare-22022540-v3/yolo-species-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gt(name,split):
    image=DATA/'images'/split/name
    with Image.open(image) as im:w,h=im.size
    boxes=[]
    for line in (DATA/'labels'/split/(Path(name).stem+'.txt')).read_text().splitlines():
        _,xc,yc,bw,bh=map(float,line.split());boxes.append([(xc-bw/2)*w,(yc-bh/2)*h,(xc+bw/2)*w,(yc+bh/2)*h])
    return np.asarray(boxes,dtype=float).reshape(-1,4),(w,h)
def iou_matrix(a,b):
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    tl=np.maximum(a[:,None,:2],b[None,:,:2]);br=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(br-tl,0).prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return inter/(aa[:,None]+bb[None,:]-inter+1e-9)
def metrics(predictions,files,split,confidence,train_median):
    rows=[];totals=collections.Counter();size_counts=collections.Counter()
    for name in files:
        truth,size=gt(name,split);p=predictions[name];keep=np.asarray(p['confidence'])>=confidence;boxes=np.asarray(p['xyxy'],dtype=float).reshape(-1,4)[keep]
        overlaps=iou_matrix(truth,boxes)
        matched=set()
        if overlaps.size:
            # Maximize valid-match cardinality first, then overlap quality.
            score=(overlaps>=0.5).astype(float)*1000+overlaps
            rr,cc=linear_sum_assignment(-score)
            matched={int(r) for r,c in zip(rr,cc) if overlaps[r,c]>=0.5}
        tp=len(matched);fp=len(boxes)-tp;fn=len(truth)-tp
        totals.update(tp=tp,fp=fp,fn=fn)
        for j,b in enumerate(truth):
            min_side=min(b[2]-b[0],b[3]-b[1])*1024/max(size)
            label='under16px' if min_side<16 else '16to32px' if min_side<32 else 'over32px'
            size_counts[label+'GT']+=1;size_counts[label+'TP']+=j in matched
        rows.append({'image':name,'species':name[:4],'truth':len(truth),'predicted':len(boxes),'signedError':len(boxes)-len(truth),'tp':tp,'fp':fp,'fn':fn})
    e=np.array([r['signedError'] for r in rows]);gtc=np.array([r['truth'] for r in rows]);tp,fp,fn=[totals[k] for k in ['tp','fp','fn']]
    summary={'images':len(rows),'groundTruthObjects':int(gtc.sum()),'confidence':confidence,'nmsIoU':0.7,'matchIoU':0.5,'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'countMAE':float(abs(e).mean()),'countRMSE':float(np.sqrt((e**2).mean())),'countBias':float(e.mean()),'exactCountImages':int((e==0).sum()),'exactCountRate':float((e==0).mean()),'within10PercentImages':int((abs(e)<=gtc*.1).sum()),'within10PercentOrOneObjectImages':int((abs(e)<=np.maximum(1,gtc*.1)).sum()),'trainingMedianCountBaselineMAE':float(np.abs(gtc-train_median).mean()),'sizeRecall':{k:{'groundTruth':size_counts[k+'GT'],'truePositives':size_counts[k+'TP'],'recall':size_counts[k+'TP']/max(1,size_counts[k+'GT'])} for k in ['under16px','16to32px','over32px']}}
    return summary,rows

def predict(model,files,split,out):
    preds={}
    source=[str(DATA/'images'/split/name) for name in files]
    for i,r in enumerate(model.predict(source=source,imgsz=1024,conf=0.001,iou=0.7,max_det=1000,rect=False,device='cpu',batch=1,verbose=False,stream=True)):
        name=Path(r.path).name;preds[name]={'xyxy':r.boxes.xyxy.cpu().numpy().round(4).tolist(),'confidence':r.boxes.conf.cpu().numpy().round(7).tolist()}
        if (i+1)%10==0:print(split,'predictions',i+1,'/',len(files),flush=True)
    out.write_text(json.dumps(preds)+'\n');return preds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run.resolve();out=run/'evaluation'
    assert not (out/'test-report.json').exists(),'Test report already exists; do not silently rerun/tune on heldout data'
    out.mkdir(exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text());audit=json.loads((DATA/'audit-report.json').read_text());assert sha(DATA/'manifest.json')==audit['manifestSHA256']
    weights=run/'weights/best.pt';assert weights.is_file() and (run/'training-complete.json').exists()
    torch.set_num_threads(8)
    model=YOLO(str(weights));train_median=float(np.median([len(gt(n,'train')[0]) for n in manifest['splits']['train']]))
    val=predict(model,manifest['splits']['val'],'val',out/'validation-predictions.json')
    grid=[0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50];scores=[metrics(val,manifest['splits']['val'],'val',c,train_median)[0] for c in grid]
    best=min(scores,key=lambda x:(x['countMAE'],-x['confidence']))
    selection={'frozenBeforeTestUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'modelSHA256':sha(weights),'manifestSHA256':audit['manifestSHA256'],'selectedConfidence':best['confidence'],'nmsIoU':0.7,'letterbox':'Fixed1024square, rect=False','validationCandidates':scores,'trainingMedianCountBaseline':train_median}
    (out/'selection-before-test.json').write_text(json.dumps(selection,indent=2)+'\n');print('Validation-selected confidence',best['confidence'],'MAE',best['countMAE'],flush=True)
    test=predict(model,manifest['splits']['test'],'test',out/'test-predictions.json')
    summary,rows=metrics(test,manifest['splits']['test'],'test',best['confidence'],train_median)
    ap=model.val(data=str(DATA/'data.yaml'),split='test',imgsz=1024,batch=2,device='cpu',workers=0,rect=False,conf=0.001,iou=0.7,max_det=1000,plots=True,save_json=True,project=str(out),name='test-ap',exist_ok=False,verbose=False)
    summary.update({'AP50':float(ap.box.map50),'AP50_95':float(ap.box.map),'AP75':float(ap.box.map75),'evaluationDevice':'cpu','status':'COMPLETED_UNSEEN_SPECIES_TEST','modelSHA256':sha(weights),'manifestSHA256':audit['manifestSHA256'],'testSpecies':manifest['speciesGroups']['test'],'scope':'Dataset research evaluation only; no real-device or production acceptance. Test results were not used to retrain or choose threshold.'})
    for species in sorted({r['species'] for r in rows}):
        rr=[r for r in rows if r['species']==species];summary.setdefault('perSpecies',{})[species]={'images':len(rr),'countMAE':sum(abs(r['signedError']) for r in rr)/len(rr),'truth':sum(r['truth'] for r in rr),'TP':sum(r['tp'] for r in rr),'FP':sum(r['fp'] for r in rr),'FN':sum(r['fn'] for r in rr)}
    report={'summary':summary,'images':rows};(out/'test-report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Deterministic diagnostic gallery: four largest absolute count errors, then four smallest.
    ordered=sorted(rows,key=lambda r:(-abs(r['signedError']),r['image']));chosen=ordered[:4]+ordered[-4:]
    canvas=Image.new('RGB',(4*360,2*390),'white');draw=ImageDraw.Draw(canvas)
    for idx,row in enumerate(chosen):
        name=row['image'];truth,size=gt(name,'test');pred=test[name];boxes=np.asarray(pred['xyxy']).reshape(-1,4)[np.asarray(pred['confidence'])>=best['confidence']]
        with Image.open(DATA/'images/test'/name) as im:im=im.convert('RGB');im.thumbnail((350,350))
        x0=idx%4*360;y0=idx//4*390;canvas.paste(im,(x0,y0));sx=im.width/size[0];sy=im.height/size[1]
        for b in boxes:draw.rectangle((x0+b[0]*sx,y0+b[1]*sy,x0+b[2]*sx,y0+b[3]*sy),outline='#ff681e',width=1)
        draw.text((x0+2,y0+351),f'{name} GT{row["truth"]} Pred{row["predicted"]}',fill='black')
    canvas.save(out/'test-count-diagnostics.jpg',quality=92)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));t=[r['truth'] for r in rows];p=[r['predicted'] for r in rows];upper=max(t+p)
    axes[0].scatter(t,p,s=18);axes[0].plot([0,upper],[0,upper],'k--',lw=1);axes[0].set(xlabel='Ground truth colony count',ylabel='Predicted count',title='Frozen unseen-species test')
    axes[1].scatter(t,[r['signedError'] for r in rows],s=18);axes[1].axhline(0,c='k',ls='--',lw=1);axes[1].set(xlabel='Ground truth count',ylabel='Predicted minus truth',title='Count error')
    fig.tight_layout();fig.savefig(out/'count-errors.png',dpi=180);plt.close(fig);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
