"""Pipe validation on hash-bound actual App-rendered inputs; no test reader."""
import argparse,collections,json,math,time
from pathlib import Path
import pipe_ml as p
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class
from types import SimpleNamespace

def verify_authorization():
 a=p.read(p.OUT/'training-authorization.json');assert a['codeSHA256']==p.code_hashes() and a['protocolSHA256']==p.sha(p.OUT/'protocol.json') and a['dataBindingSHA256']==p.sha(p.OUT/'data-binding.json') and a['rootReceiptSHA256']==p.sha(a['rootReceiptPath'])
 p.official_init_check();p.root_training_approval(a['rootReceiptPath']);return a

def app_inputs(path):
 manifest=p.read(path);assert manifest['status']=='VALIDATED_APP_PREPROCESSING_INPUTS'
 assert manifest['recordsSHA256']==p.sha(p.OUT/'records.json') and manifest['testContentIncluded'] is False
 assert manifest['approvedAppSourceCodeSHA256']==p.read(p.OUT/'protocol.json')['appSourceCodeSHA256']==p.app_source_hashes()
 assert manifest['appExportManifests']
 for receipt in manifest['appExportManifests']:assert p.sha(receipt['path'])==receipt['sha256']
 records={r['id']:r for r in p.read(p.OUT/'records.json') if r['split']=='val'};variants=manifest['variants'];expected=set(p.read(p.OUT/'protocol.json')['evaluationVariants']);assert set(variants)==expected and len(variants)==len(expected)
 assert len(manifest['images'])==len(records)*len(variants) and all(r['variant'] in expected for r in manifest['images'])
 assert len({(r['variant'],r['id']) for r in manifest['images']})==len(manifest['images'])
 for variant in variants:
  rows=[r for r in manifest['images'] if r['variant']==variant];assert {r['id'] for r in rows}==set(records) and len(rows)==len(records)
 for r in manifest['images']:
  assert r['id'] in records and r['sourceImageSHA256']==records[r['id']]['sha256'];assert p.sha(r['networkImagePath'])==r['networkImageSHA256']
  with Image.open(r['networkImagePath']) as im:assert im.size==(1024,1024)
  assert p.sha(r['photoImagePath'])==r['photoImageSHA256']
  with Image.open(r['photoImagePath']) as im:assert im.size==(r['photoWidth'],r['photoHeight'])
  expected_geometry=p.pipeline_geometry(r['photoWidth'],r['photoHeight']);assert all(abs(r['geometry'][k]-value)<1e-8 for k,value in expected_geometry.items())
 return manifest,records

def raw_predictions(epoch,inputs_path):
 verify_authorization();manifest,records=app_inputs(inputs_path);weights=p.OUT/'checkpoints'/f'epoch{epoch:03d}.pt';meta=p.read(weights.with_suffix('.json'));assert p.sha(weights)==meta['sha256']
 root=p.OUT/'predictions'/f'epoch{epoch:03d}';root.mkdir(parents=True,exist_ok=True);binding={'modelSHA256':meta['sha256'],'inputsManifestSHA256':p.sha(inputs_path),'recordsSHA256':p.sha(p.OUT/'records.json'),'codeSHA256':p.code_hashes(),'confidenceFloor':.001,'preNMS':True}
 if (root/'binding.json').exists():assert p.read(root/'binding.json')==binding
 else:p.save(root/'binding.json',binding)
 model=YOLO(str(weights)).model.float().cpu().eval();assert model.names=={0:'pipe_end'} and model.model[-1].nc==1 and not model.model[-1].end2end;model.fuse();torch.set_num_threads(8)
 for r in manifest['images']:
  dest=root/r['variant']/(r['id']+'.npz');side=dest.with_suffix('.json')
  if dest.exists() and side.exists():
   s=p.read(side);assert s['npzSHA256']==p.sha(dest) and s['id']==r['id'] and s['variant']==r['variant'] and s['modelSHA256']==meta['sha256'] and s['networkImageSHA256']==r['networkImageSHA256'] and s['rawLocations']==21504 and s['NMSApplied'] is False and s['outputCapApplied'] is False
   with np.load(dest,allow_pickle=False) as arrays:
    b,c=arrays['xyxy'],arrays['confidence'];assert b.dtype==c.dtype==np.float32 and b.ndim==2 and b.shape[1]==4 and c.ndim==1 and len(b)==len(c)==s['retainedAfterFloor'] and len(c)<=21504;assert np.isfinite(b).all() and np.isfinite(c).all() and np.all((c>=.001)&(c<=1))
   continue
  with Image.open(r['networkImagePath']) as image:rgb=np.asarray(image.convert('RGB')).copy()
  tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).float()/255
  assert tensor.is_contiguous() and tensor.dtype==torch.float32
  with torch.inference_mode():output=model(tensor);output=output[0] if isinstance(output,tuple) else output
  assert output.shape==(1,5,21504) and torch.isfinite(output).all()
  raw=output[0].T;xywh,scores=raw[:,:4],raw[:,4];boxes=torch.cat([xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2],1);keep=scores>=.001
  dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,xyxy=boxes[keep].numpy(),confidence=scores[keep].numpy())
  p.save(side,{'id':r['id'],'variant':r['variant'],'npzSHA256':p.sha(dest),'modelSHA256':meta['sha256'],'networkImageSHA256':r['networkImageSHA256'],'rawLocations':21504,'retainedAfterFloor':int(keep.sum()),'NMSApplied':False,'outputCapApplied':False});print('raw',epoch,r['variant'],r['id'],int(keep.sum()),flush=True)

def iou_matrix(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)))
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(hi-lo,0).prod(2);aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1);return inter/(aa[:,None]+bb[None,:]-inter+1e-9)

def transformed_truth(record,input_row):
 sx=input_row['photoWidth']/record['originalWidth'];sy=input_row['photoHeight']/record['originalHeight'];return np.asarray([[a*sx,b*sy,c*sx,d*sy] for a,b,c,d in record['originalXYXY']],dtype=float).reshape(-1,4)

def postprocess(raw,geometry,confidence,iou):
 scores=torch.from_numpy(raw['confidence']);boxes=torch.from_numpy(raw['xyxy']);mask=scores>=confidence;boxes,scores=boxes[mask],scores[mask]
 keep=nms(boxes,scores,iou);assert len(keep)<=21504
 mapped,valid=p.unletterbox(boxes[keep].numpy(),geometry)
 return {'xyxy':mapped[valid].tolist(),'confidence':scores[keep].numpy()[valid].tolist(),'nmsBeforeUnpadding':True,'invalidPaddingOnlyBoxesDropped':int((~valid).sum()),'arbitraryMarkerCapApplied':False,'rawCapacitySaturated':len(keep)==21504}

def count_metrics(predictions,records,input_rows,confidence):
 rows=[]
 for inp in input_rows:
  r=records[inp['id']];truth=transformed_truth(r,inp);entry=predictions[r['id']];boxes=np.asarray(entry['xyxy'],dtype=float).reshape(-1,4)[np.asarray(entry['confidence'])>=confidence];overlap=iou_matrix(truth,boxes);matched=set()
  if overlap.size:
   rr,cc=linear_sum_assignment(-((overlap>=.5).astype(float)*(max(len(truth),len(boxes))+1)+overlap));matched={int(a) for a,b in zip(rr,cc) if overlap[a,b]>=.5}
  sizes={k:{'GT':0,'TP':0} for k in ['under16','16to32','over32']}
  for i,b in enumerate(truth):
   side=min(b[2]-b[0],b[3]-b[1])*1024/max(inp['photoWidth'],inp['photoHeight']);key='under16' if side<16 else '16to32' if side<32 else 'over32';sizes[key]['GT']+=1;sizes[key]['TP']+=int(i in matched)
  tp=len(matched);rows.append({'id':r['id'],'source':r['source'],'groupId':r['groupId'],'negativeKind':r.get('negativeKind'),'GT':len(truth),'predicted':len(boxes),'TP':tp,'FP':len(boxes)-tp,'FN':len(truth)-tp,'absoluteError':abs(len(boxes)-len(truth)),'signedError':len(boxes)-len(truth),'sizeCounts':sizes})
 positives=[r for r in rows if r['GT']>0];negatives=[r for r in rows if r['GT']==0];groups=collections.defaultdict(list)
 for r in positives:groups[str(r['groupId'])].append(r)
 group_metrics={g:{'images':len(rr),'GT':sum(r['GT'] for r in rr),'WAPE':sum(r['absoluteError'] for r in rr)/sum(r['GT'] for r in rr)} for g,rr in groups.items()}
 tp,fp,fn=[sum(r[k] for r in rows) for k in ['TP','FP','FN']]
 summary={'images':len(rows),'positiveImages':len(positives),'negativeImages':len(negatives),'GT':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'positiveCountMAE':float(np.mean([r['absoluteError'] for r in positives])),'positiveCountWAPE':sum(r['absoluteError'] for r in positives)/max(1,sum(r['GT'] for r in positives)),'sceneMacroWAPE':float(np.mean([g['WAPE'] for g in group_metrics.values()])),'negativeFalsePositiveImages':sum(r['predicted']>0 for r in negatives),'smallCountFailures':[r['id'] for r in positives if r['GT']<=20 and r['absoluteError']>1],'perScene':group_metrics,'sizeRecall':{s:{'GT':sum(r['sizeCounts'][s]['GT'] for r in rows),'TP':sum(r['sizeCounts'][s]['TP'] for r in rows)} for s in ['under16','16to32','over32']}}
 summary['negativeByKind']={kind:{'images':sum(r['negativeKind']==kind for r in negatives),'falsePositiveImages':sum(r['negativeKind']==kind and r['predicted']>0 for r in negatives),'FP':sum(r['FP'] for r in negatives if r['negativeKind']==kind)} for kind in sorted({r['negativeKind'] for r in negatives})}
 summary['perSource']={source:{'images':len(rr),'GT':sum(r['GT'] for r in rr),'TP':sum(r['TP'] for r in rr),'FP':sum(r['FP'] for r in rr),'FN':sum(r['FN'] for r in rr),'countMAE':float(np.mean([r['absoluteError'] for r in rr]))} for source in sorted({r['source'] for r in rows}) for rr in [[r for r in rows if r['source']==source]]}
 summary['perCountDensity']={name:{'images':len(rr),'GT':sum(r['GT'] for r in rr),'TP':sum(r['TP'] for r in rr),'FP':sum(r['FP'] for r in rr),'FN':sum(r['FN'] for r in rr),'countMAE':float(np.mean([r['absoluteError'] for r in rr]))} for name,lo,hi in [('1to20',1,20),('21to100',21,100),('101to500',101,500),('over500',501,10**9)] for rr in [[r for r in rows if lo<=r['GT']<=hi]] if rr}
 for size in summary['sizeRecall'].values():size['FN']=size['GT']-size['TP'];size['recall']=size['TP']/size['GT'] if size['GT'] else None
 return {'summary':summary,'images':rows}

def ap_metrics(predictions,records,input_rows):
 tps=[];scores=[];targets=[];matcher=SimpleNamespace(iouv=torch.linspace(.5,.95,10))
 for inp in input_rows:
  r=records[inp['id']];truth=transformed_truth(r,inp);entry=predictions[r['id']];boxes=np.asarray(entry['xyxy'],dtype=float).reshape(-1,4);match=BaseValidator.match_predictions(matcher,torch.zeros(len(boxes)),torch.zeros(len(truth)),torch.tensor(iou_matrix(truth,boxes),dtype=torch.float32));tps.append(match.numpy());scores.extend(entry['confidence']);targets.extend([0]*len(truth))
 if not scores:return {'AP50':0.,'AP50_95':0.}
 values=ap_per_class(np.concatenate(tps),np.asarray(scores),np.zeros(len(scores)),np.asarray(targets),plot=False,names={0:'pipe_end'})[5];return {'AP50':float(values[:,0].mean()),'AP50_95':float(values.mean())}

def eligible(result):
 s=result['summary'];checks={'precision>=.95':s['precision']>=.95,'recall>=.95':s['recall']>=.95,'positiveWAPE<=.05':s['positiveCountWAPE']<=.05,'sceneMacroWAPE<=.10':s['sceneMacroWAPE']<=.10,'smallCountsGT<=20Error<=1':not s['smallCountFailures'],'allNegativeImagesZero':s['negativeFalsePositiveImages']==0};return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}

def select(inputs_path):
 authorization=verify_authorization();assert not (p.OUT/'selection.json').exists()
 completion=p.read(p.OUT/'training-complete.json');assert completion['status']=='COMPLETE_20_EPOCHS_WITH_ACTUAL_REGISTERED_CHECKPOINTS' and completion['rootReceiptSHA256']==authorization['rootReceiptSHA256']
 from train_pipe_ends import checkpoint_check
 assert set(completion['checkpoints'])=={'8','14','20'}
 for epoch in [8,14,20]:assert completion['checkpoints'][str(epoch)]==checkpoint_check(epoch)
 manifest,records=app_inputs(inputs_path);protocol=p.read(p.OUT/'protocol.json');results=[]
 for epoch in protocol['checkpointEpochs']:
  raw_predictions(epoch,inputs_path)
  for iou in protocol['nmsIoUGrid']:
   ap_predictions={};by_conf={c:{} for c in protocol['confidenceGrid']}
   for inp in manifest['images']:
    raw=dict(np.load(p.OUT/'predictions'/f'epoch{epoch:03d}'/inp['variant']/(inp['id']+'.npz')));key=(inp['variant'],inp['id']);ap_predictions[key]=postprocess(raw,inp['geometry'],.001,iou)
    for conf in protocol['confidenceGrid']:by_conf[conf][key]=postprocess(raw,inp['geometry'],conf,iou)
   ap={variant:ap_metrics({r['id']:ap_predictions[(variant,r['id'])] for r in manifest['images'] if r['variant']==variant},records,[r for r in manifest['images'] if r['variant']==variant]) for variant in manifest['variants']}
   for conf in protocol['confidenceGrid']:
    variants={}
    for variant in manifest['variants']:
     rows=[r for r in manifest['images'] if r['variant']==variant];result=count_metrics({r['id']:by_conf[conf][(variant,r['id'])] for r in rows},records,rows,conf);result['summary'].update(ap[variant]);result['gates']=eligible(result);variants[variant]=result
    item={'epoch':epoch,'confidence':conf,'nmsIoU':iou,'variants':variants,'status':'PASS' if all(v['gates']['status']=='PASS' for v in variants.values()) else 'FAIL','modelSHA256':p.sha(p.OUT/'checkpoints'/f'epoch{epoch:03d}.pt')};results.append(item);p.save(p.OUT/'scores'/f'epoch{epoch:03d}-conf{conf}-iou{iou}.json',item);print('score',epoch,conf,iou,item['status'],flush=True)
 passing=[r for r in results if r['status']=='PASS'];rank=lambda r:(max(v['summary']['sceneMacroWAPE'] for v in r['variants'].values()),max(v['summary']['positiveCountMAE'] for v in r['variants'].values()),-r['confidence'],r['epoch'],r['nmsIoU']);chosen=min(passing,key=rank) if passing else None;research=min(results,key=rank)
 p.save(p.OUT/'selection.json',{'status':'VAL_QUALIFIED_PENDING_FROZEN_SCENE_TEST' if chosen else 'FAIL_VALIDATION_QUALIFICATION','selected':chosen,'bestResearchCandidateNotAutomaticallyUsable':research,'candidates':results,'inputsManifestSHA256':p.sha(inputs_path),'codeSHA256':p.code_hashes(),'protocolSHA256':p.sha(p.OUT/'protocol.json'),'testAccessed':False,'VALIDATED_COREML_EXPORTIssued':False})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select']);parser.add_argument('--inputs',type=Path,required=True);args=parser.parse_args();select(args.inputs)
