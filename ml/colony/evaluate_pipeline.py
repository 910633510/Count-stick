#!/usr/bin/env python3
"""Compare fully automatic inference pipelines on validation only, then freeze before test."""
from evaluate import *
import types
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class
from torchvision.ops import nms

def predict_tiled(model,files,split,out):
    preds={}
    for number,name in enumerate(files):
        with Image.open(DATA/'images'/split/name) as src:
            image=src.convert('RGB');w,h=image.size
        # Four overlapping60%-width/height crops; each owns its central image quadrant.
        tw,th=int(np.ceil(w*.6)),int(np.ceil(h*.6));tiles=[];parts=[]
        for row in range(2):
            for col in range(2):
                x0=0 if col==0 else w-tw;y0=0 if row==0 else h-th
                tiles.append((x0,y0,col,row));parts.append(image.crop((x0,y0,x0+tw,y0+th)))
        boxes=[];scores=[]
        for r,(x0,y0,col,row) in zip(model.predict(parts,imgsz=1024,rect=False,conf=0.001,iou=.7,max_det=1000,device='cpu',verbose=False,stream=True),tiles):
            bb=r.boxes.xyxy.cpu().numpy().copy();ss=r.boxes.conf.cpu().numpy()
            bb[:,[0,2]]+=x0;bb[:,[1,3]]+=y0;center=(bb[:,:2]+bb[:,2:])/2
            keep=((center[:,0]<w/2) if col==0 else (center[:,0]>=w/2)) & ((center[:,1]<h/2) if row==0 else (center[:,1]>=h/2))
            boxes.extend(bb[keep].tolist());scores.extend(ss[keep].tolist())
        if boxes:
            bb=torch.tensor(boxes,dtype=torch.float32);ss=torch.tensor(scores);keep=nms(bb,ss,.7)[:1000];bb=bb[keep];ss=ss[keep]
            preds[name]={'xyxy':bb.tolist(),'confidence':ss.tolist()}
        else:preds[name]={'xyxy':[],'confidence':[]}
        if (number+1)%10==0:print(split,'automatic4tile',number+1,'/',len(files),flush=True)
    out.write_text(json.dumps(preds)+'\n');return preds

def ap_from_predictions(preds,files,split):
    tps=[];scores=[];targets=[]
    matcher=types.SimpleNamespace(iouv=torch.linspace(.5,.95,10))
    for name in files:
        truth,_=gt(name,split);b=np.asarray(preds[name]['xyxy']).reshape(-1,4);c=np.asarray(preds[name]['confidence'])
        matched=BaseValidator.match_predictions(matcher,torch.zeros(len(b)),torch.zeros(len(truth)),torch.from_numpy(iou_matrix(truth,b)).float())
        tps.append(matched.numpy());scores.extend(c);targets.extend([0]*len(truth))
    ans=ap_per_class(np.concatenate(tps),np.asarray(scores),np.zeros(len(scores)),np.asarray(targets),plot=False,names={0:'colony'})
    ap=ans[5];return {'AP50':float(ap[:,0].mean()),'AP50_95':float(ap.mean()),'AP75':float(ap[:,5].mean()),'method':'Ultralytics BaseValidator matching + ap_per_class on saved predictions at IoU0.50:0.95; max1000 detections/image'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run=args.run.resolve();out=run/'evaluation'
    assert not (out/'test-report.json').exists(),'Do not re-tune an already opened holdout'
    out.mkdir(exist_ok=True);torch.set_num_threads(8)
    manifest=json.loads((DATA/'manifest.json').read_text());audit=json.loads((DATA/'audit-report.json').read_text());assert sha(DATA/'manifest.json')==audit['manifestSHA256']
    weights=run/'weights/best.pt';assert (run/'training-complete.json').exists();model=YOLO(str(weights))
    train_median=float(np.median([len(gt(n,'train')[0]) for n in manifest['splits']['train']]))
    rules={'registeredBeforeTestUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'modelSHA256':sha(weights),'manifestSHA256':audit['manifestSHA256'],'modes':['whole','automatic4tile'],'grid':[.05,.1,.15,.2,.25,.3,.4,.5],'geometry':'Four60%extent crops, ownership by original-image quadrant center, finalIoU0.7 NMS. Entire process automatic.','selection':'Whole-mode minimum validation countMAE (ties higher confidence) is baseline. Tiled candidate must lower countMAE by at least5%, keep precision and recall no more than2percentage-points below baseline, and keep F1 at least baseline. Choose minimumMAE eligible tile threshold; otherwise whole.','testPolicy':'No model, threshold or pipeline changes using test outcomes. Primary choice fixed before first test prediction. Also evaluate whole-image reference if tile selected.','supersedes':'First training protocol specified whole-image inference. This validation-only comparison is registered before opening test; the first protocol remains preserved.'}
    (out/'inference-protocol-before-test.json').write_text(json.dumps(rules,indent=2)+'\n')
    def f1(x):return 2*x['TP']/max(1,2*x['TP']+x['FP']+x['FN'])
    val_files=manifest['splits']['val'];whole=predict(model,val_files,'val',out/'validation-predictions.json');whole_scores=[metrics(whole,val_files,'val',c,train_median)[0] for c in rules['grid']];base=min(whole_scores,key=lambda r:(r['countMAE'],-r['confidence']))
    tiled=predict_tiled(model,val_files,'val',out/'validation-tiled-predictions.json');tile_scores=[metrics(tiled,val_files,'val',c,train_median)[0] for c in rules['grid']]
    eligible=[s for s in tile_scores if s['countMAE']<=base['countMAE']*.95 and s['precision']>=base['precision']-.02 and s['recall']>=base['recall']-.02 and f1(s)>=f1(base)]
    chosen=min(eligible,key=lambda r:(r['countMAE'],-r['confidence'])) if eligible else base;mode='automatic4tile' if eligible else 'whole'
    selection={'frozenBeforeTestUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'modelSHA256':sha(weights),'manifestSHA256':audit['manifestSHA256'],'selectedConfidence':chosen['confidence'],'selectedPipeline':mode,'nmsIoU':.7,'letterbox':'Fixed1024square, rect=False','wholeValidationCandidates':whole_scores,'tiledValidationCandidates':tile_scores,'selectedValidation':chosen,'wholeReferenceValidation':base,'trainingMedianCountBaseline':train_median}
    (out/'selection-before-test.json').write_text(json.dumps(selection,indent=2)+'\n');print('Frozen pipeline',mode,'confidence',chosen['confidence'],'validationMAE',chosen['countMAE'],'wholeMAE',base['countMAE'],flush=True)
    test_files=manifest['splits']['test'];test=predict_tiled(model,test_files,'test',out/'test-predictions.json') if mode=='automatic4tile' else predict(model,test_files,'test',out/'test-predictions.json')
    summary,rows=metrics(test,test_files,'test',chosen['confidence'],train_median);summary.update(ap_from_predictions(test,test_files,'test'))
    summary.update({'status':'COMPLETED_UNSEEN_SPECIES_TEST','evaluationDevice':'cpu','pipeline':mode,'modelSHA256':sha(weights),'manifestSHA256':audit['manifestSHA256'],'testSpecies':manifest['speciesGroups']['test'],'F1':f1(summary),'scope':'Frozen same-study unseen-species research test; no physical-device, phone-photo or production acceptance.'})
    if mode=='automatic4tile':
        reference=predict(model,test_files,'test',out/'test-whole-reference-predictions.json');ref,rr=metrics(reference,test_files,'test',base['confidence'],train_median);ref.update(ap_from_predictions(reference,test_files,'test'));(out/'test-whole-reference.json').write_text(json.dumps({'summary':ref,'images':rr},indent=2)+'\n')
    for species in sorted({r['species'] for r in rows}):
        rr=[r for r in rows if r['species']==species];summary.setdefault('perSpecies',{})[species]={'images':len(rr),'countMAE':sum(abs(r['signedError']) for r in rr)/len(rr),'truth':sum(r['truth'] for r in rr),'TP':sum(r['tp'] for r in rr),'FP':sum(r['fp'] for r in rr),'FN':sum(r['fn'] for r in rr)}
    (out/'test-report.json').write_text(json.dumps({'summary':summary,'images':rows},indent=2)+'\n')
    ordered=sorted(rows,key=lambda r:(-abs(r['signedError']),r['image']));chosen_rows=ordered[:4]+ordered[-4:];canvas=Image.new('RGB',(1440,780),'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(chosen_rows):
        name=row['image'];_,size=gt(name,'test');entry=test[name];bb=np.asarray(entry['xyxy']).reshape(-1,4)[np.asarray(entry['confidence'])>=chosen['confidence']]
        with Image.open(DATA/'images/test'/name) as im:im=im.convert('RGB');im.thumbnail((350,350))
        x0=i%4*360;y0=i//4*390;canvas.paste(im,(x0,y0));sx=im.width/size[0];sy=im.height/size[1]
        for b in bb:draw.rectangle((x0+b[0]*sx,y0+b[1]*sy,x0+b[2]*sx,y0+b[3]*sy),outline='#ff681e',width=1)
        draw.text((x0+2,y0+351),f'{name} GT{row["truth"]} Pred{row["predicted"]}',fill='black')
    canvas.save(out/'test-count-diagnostics.jpg',quality=92)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));t=[r['truth'] for r in rows];p=[r['predicted'] for r in rows];upper=max(t+p)
    axes[0].scatter(t,p,s=18);axes[0].plot([0,upper],[0,upper],'k--',lw=1);axes[0].set(xlabel='Ground truth count',ylabel='Predicted count',title='Frozen unseen-species test')
    axes[1].scatter(t,[r['signedError'] for r in rows],s=18);axes[1].axhline(0,c='k',ls='--',lw=1);axes[1].set(xlabel='Ground truth count',ylabel='Count error',title='Predicted minus truth');fig.tight_layout();fig.savefig(out/'count-errors.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
