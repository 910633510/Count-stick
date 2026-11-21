#!/usr/bin/env python3
"""Single pre-registered external comparison; no promotion of internally failed fusion."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
sys.path.insert(0,str(ROOT/'ml/colony'))
from scale_fusion import fuse_predictions
from evaluate import iou_matrix
import collections,hashlib,json,time,types
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class
DATA=ROOT/'output/datasets/colony-external-zenodo18256864-v1'
OUT=ROOT/'output/research-20261025/external-comparison'
ART=ROOT/'ml/research-20261025'
WEIGHTS=ROOT/'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/weights/best.pt'
CONFIG={'min_side_threshold':16.0,'nms_iou':.5,'confidence_floor':.001,'max_detections':1000,'fusion_mode':'global_nms'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prediction_dict(r):return {'xyxy':r.boxes.xyxy.cpu().numpy().tolist(),'confidence':r.boxes.conf.cpu().numpy().tolist()}
def predict(model,records):
 whole={};tile={};start=time.time()
 for number,entry in enumerate(records):
  name=entry['filename'];path=DATA/'images'/name
  with Image.open(path) as source:image=source.convert('RGB')
  params=dict(imgsz=1024,rect=False,conf=.001,iou=.7,max_det=1000,device='cpu',verbose=False)
  whole[name]=prediction_dict(model.predict(image,**params)[0])
  w,h=image.size;tw,th=int(np.ceil(w*.6)),int(np.ceil(h*.6));parts=[];placements=[]
  for row in range(2):
   for col in range(2):
    x0=0 if col==0 else w-tw;y0=0 if row==0 else h-th;parts.append(image.crop((x0,y0,x0+tw,y0+th)));placements.append((x0,y0,col,row))
  boxes=[];scores=[]
  for r,(x0,y0,col,row) in zip(model.predict(parts,stream=True,**params),placements):
   b=r.boxes.xyxy.cpu().numpy().copy();s=r.boxes.conf.cpu().numpy();b[:,[0,2]]+=x0;b[:,[1,3]]+=y0;c=(b[:,:2]+b[:,2:])/2
   keep=((c[:,0]<w/2) if col==0 else (c[:,0]>=w/2))&((c[:,1]<h/2) if row==0 else (c[:,1]>=h/2));boxes.extend(b[keep].tolist());scores.extend(s[keep].tolist())
  if boxes:
   b=torch.tensor(boxes,dtype=torch.float32);s=torch.tensor(scores,dtype=torch.float32);keep=nms(b,s,.7)[:1000];tile[name]={'xyxy':b[keep].tolist(),'confidence':s[keep].tolist()}
  else:tile[name]={'xyxy':[],'confidence':[]}
  if (number+1)%5==0:print('External predictions',number+1,'/',len(records),'seconds',round(time.time()-start),flush=True)
 (OUT/'whole-predictions.json').write_text(json.dumps(whole)+'\n');(OUT/'tile-predictions.json').write_text(json.dumps(tile)+'\n');return whole,tile

def score(preds,records):
 rows=[];size=collections.Counter();correct=[];all_scores=[];target_classes=[];matcher=types.SimpleNamespace(iouv=torch.linspace(.5,.95,10))
 for entry in records:
  name=entry['filename'];truth=np.array(entry['xyxy'],dtype=float).reshape(-1,4);p=preds[name];all_b=np.array(p['xyxy'],dtype=float).reshape(-1,4);conf=np.array(p['confidence']);ap_iou=iou_matrix(truth,all_b)
  correct.append(BaseValidator.match_predictions(matcher,torch.zeros(len(all_b)),torch.zeros(len(truth)),torch.from_numpy(ap_iou).float()).numpy());all_scores.extend(conf);target_classes.extend([0]*len(truth))
  b=all_b[conf>=.3];ious=iou_matrix(truth,b);matched=set()
  if ious.size:
   rr,cc=linear_sum_assignment(-((ious>=.5).astype(float)*1000+ious));matched={int(r) for r,c in zip(rr,cc) if ious[r,c]>=.5}
  tp=len(matched);fp=len(b)-tp;fn=len(truth)-tp
  for j,box in enumerate(truth):
   side=min(box[2]-box[0],box[3]-box[1])*1024/max(entry['width'],entry['height']);key='under16px' if side<16 else '16to32px' if side<32 else 'over32px';size[key+'GT']+=1;size[key+'TP']+=j in matched
  rows.append({'image':name,'species':entry['species'],'truth':len(truth),'predicted':len(b),'error':len(b)-len(truth),'TP':tp,'FP':fp,'FN':fn})
 species={}
 for name in sorted({r['species'] for r in rows}):
  rr=[r for r in rows if r['species']==name];tp=sum(r['TP'] for r in rr);fp=sum(r['FP'] for r in rr);fn=sum(r['FN'] for r in rr);species[name]={'images':len(rr),'GT':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'countMAE':sum(abs(r['error']) for r in rr)/len(rr)}
 tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows);errors=np.array([r['error'] for r in rows]);ap=ap_per_class(np.concatenate(correct),np.array(all_scores),np.zeros(len(all_scores)),np.array(target_classes),plot=False,names={0:'colony'})[5]
 result={'images':len(rows),'species':len(species),'GT':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'F1':2*tp/max(1,2*tp+fp+fn),'countMAE':float(abs(errors).mean()),'countRMSE':float(np.sqrt((errors**2).mean())),'countBias':float(errors.mean()),'exactCountImages':int((errors==0).sum()),'macroSpeciesRecall':float(np.mean([x['recall'] for x in species.values()])),'macroSpeciesPrecision':float(np.mean([x['precision'] for x in species.values()])),'macroSpeciesMAE':float(np.mean([x['countMAE'] for x in species.values()])),'AP50':float(ap[:,0].mean()),'AP50_95':float(ap.mean()),'perSpecies':species,'sizeRecall':{k:{'GT':size[k+'GT'],'TP':size[k+'TP'],'recall':size[k+'TP']/max(1,size[k+'GT'])} for k in ['under16px','16to32px','over32px']}}
 return result,rows

def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'report.json').exists(),'External cohort has already been opened for this comparison'
 manifest=json.loads((DATA/'manifest.json').read_text());assert manifest['status']=='FROZEN_EXTERNAL_TEST_BEFORE_INFERENCE' and sha(DATA/'manifest.json')=='b148c2c632486adc725acff7f923704c25f15fefedf038a148bd87c7c3524827'
 quality=json.loads((ART/'external-data-review.json').read_text());assert 'PASS' in quality.get('status',''),quality.get('status')
 assert sha(WEIGHTS)=='91d70c3067f2fdb77d2ba2541e6105bf3caf730da7bbc32e5f4d2c9310b0b337'
 protocol={'frozenBeforeInferenceUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'datasetManifestSHA256':sha(DATA/'manifest.json'),'weightsSHA256':sha(WEIGHTS),'candidate':CONFIG,'candidateSelectionReason':'Single best aggregate old-validation candidate, fixed before external inference. It FAILED internal subgroup gates and remains non-promotable.','references':['whole image','original automatic4tile'],'confidence':.3,'predictionFloorForAP':.001,'maxDetections':1000,'newTrainingPerformed':False,'externalThresholdTuningAllowed':False,'promotionRequiresInternalPass':True,'internalPass':False,'externalRelativeGates':'Count and macroMAE improve at least5% vs tile; TP/FN/FP no worse than tile; no species/size recall drop>2percentage-points vs stronger of whole/tile. This cannot override failed internal gates or missing phone validation.','dataScope':'Separate-study76originalimages,19species; controlled imaging, not smartphone acceptance.','fusionSourceSHA256':sha(ROOT/'ml/colony/scale_fusion.py')}
 (OUT/'protocol-before-test.json').write_text(json.dumps(protocol,indent=2)+'\n');(ART/'external-protocol-before-test.json').write_text(json.dumps(protocol,indent=2)+'\n')
 (DATA/'opened-for-inference.json').write_text(json.dumps({'openedAtUTC':protocol['frozenBeforeInferenceUTC'],'protocolSHA256':sha(OUT/'protocol-before-test.json'),'futureUse':'Exposed regression cohort after this evaluation; do not call future tests on it blind.'},indent=2)+'\n')
 torch.set_num_threads(8);model=YOLO(str(WEIGHTS));whole,tile=predict(model,manifest['images']);fusion={r['filename']:fuse_predictions(whole[r['filename']],tile[r['filename']],(r['width'],r['height']),**CONFIG) for r in manifest['images']};(OUT/'fusion-predictions.json').write_text(json.dumps(fusion)+'\n')
 results={};rows={}
 for name,pred in [('whole',whole),('tile',tile),('fusion',fusion)]:results[name],rows[name]=score(pred,manifest['images']);print(name,'MAE',results[name]['countMAE'],'TP',results[name]['TP'],'FP',results[name]['FP'],'macroRecall',results[name]['macroSpeciesRecall'],flush=True)
 w,t,f=[results[k] for k in ['whole','tile','fusion']];checks={'countMAE':f['countMAE']<=t['countMAE']*.95,'macroMAE':f['macroSpeciesMAE']<=t['macroSpeciesMAE']*.95,'TP':f['TP']>=t['TP'],'FP':f['FP']<=t['FP'],'FN':f['FN']<=t['FN']}
 for s in f['perSpecies']:checks['recall.'+s]=f['perSpecies'][s]['recall']>=max(w['perSpecies'][s]['recall'],t['perSpecies'][s]['recall'])-.02
 for s in f['sizeRecall']:checks['sizeRecall.'+s]=f['sizeRecall'][s]['recall']>=max(w['sizeRecall'][s]['recall'],t['sizeRecall'][s]['recall'])-.02
 report={'status':'EXTERNAL_COMPARISON_COMPLETED_NO_PROMOTION','finishedAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocolSHA256':sha(OUT/'protocol-before-test.json'),'datasetManifestSHA256':sha(DATA/'manifest.json'),'metrics':results,'checks':checks,'externalRelativeGatePassed':all(checks.values()),'internalGatePassed':False,'deploymentApproved':False,'futureUseOfThisCohort':'Regression only; now exposed','images':rows}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');(ART/'external-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',report['status'],'external gate',report['externalRelativeGatePassed'],flush=True)
if __name__=='__main__':main()
