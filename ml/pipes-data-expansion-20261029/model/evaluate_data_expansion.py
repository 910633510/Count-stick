"""Exactly three checkpoints, fixed threshold; old40+freshval only; no test reader."""
import argparse,hashlib,json
from pathlib import Path
import pipe_data_expansion as p
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
# Read-only pure scoring/postprocessing functions; original eight files hash-bound.
from evaluate_pipe_ends import postprocess,count_metrics,ap_metrics,eligible

def authorization():
 a=p.read(p.OUT/'training-authorization.json');assert a['codeSHA256']==p.code_hashes() and a['protocolSHA256']==p.sha(p.OUT/'protocol.json') and a['dataBindingSHA256']==p.sha(p.OUT/'data-binding.json') and a['rootReceiptSHA256']==p.sha(a['rootReceiptPath']);p.verify(a['rootReceiptPath']);return a

def inputs():
 from pipe_ml import pipeline_geometry
 from bind_app_inputs import inspect_manifest
 d=p.read(p.OUT/'evaluation-inputs.json');assert p.sha(d['previousInputsPath'])==d['previousInputsSHA256']==p.PREVIOUS_INPUTS_SHA;previous=p.read(d['previousInputsPath']);assert d['images']==previous['images']
 assert d['status']=='VALIDATED_DATA_EXPANSION_APP_INPUTS' and d['testContentIncluded'] is False and d['recordsSHA256']==p.sha(p.OUT/'records.json') and d['approvedAppSourceCodeSHA256']==p.app_hashes()
 assert p.sha(p.OLD/'selection.json')==p.OLD_SELECTION_SHA
 assert p.sha(d['oldInputManifestPath'])==d['oldInputManifestSHA256']==p.read(p.OLD/'selection.json')['inputsManifestSHA256'] and p.sha(d['oldRuntimeReviewPath'])==d['oldRuntimeReviewSHA256']
 original=p.read(d['oldInputManifestPath']);old_rows={(r['id'],r['variant']):r for r in original['images']};fresh_rows={};approved={k:v for k,v in d['approvedAppSourceCodeSHA256'].items() if k!='NMS'}
 for f in d['freshAppExportManifests']:
  assert p.sha(f['path'])==f['sha256'];manifest,rows=inspect_manifest(f['path']);assert manifest['kind']=='frozen_train_val_photo' and manifest['sourceCodeSHA256']==approved
  for row in rows:
   key=(row['id'],row['variant']);assert key not in fresh_rows and row['split']=='val' and row['sourceManifestSHA256']==p.read(p.OUT/'data-binding.json')['validationSourceManifestSHA256'];fresh_rows[key]=row
 records={r['id']:r for r in p.read(p.OUT/'records.json') if r['split']=='val'};assert len(d['images'])==2*len(records) and {(r['id'],r['variant']) for r in d['images']}=={(i,v) for i in records for v in ['gallery','camera']}
 assert set(old_rows)=={(i,v) for i,r in records.items() if r['dataRole']=='old' for v in ['gallery','camera']}
 assert set(fresh_rows)=={(i,v) for i,r in records.items() if r['dataRole']=='fresh' for v in ['gallery','camera']}
 for r in d['images']:
  assert r['split']=='val' and r['sourceImageSHA256']==records[r['id']]['sha256'] and r['dataRole']==records[r['id']]['dataRole'];assert p.sha(r['networkImagePath'])==r['networkImageSHA256'] and p.sha(r['photoImagePath'])==r['photoImageSHA256']
  key=(r['id'],r['variant']);origin=old_rows[key] if r['dataRole']=='old' else fresh_rows[key];assert r=={**origin,'dataRole':r['dataRole']},'Bound input row differs from its frozen original/export receipt'
  with Image.open(r['networkImagePath']) as image:assert image.size==(1024,1024)
  with Image.open(r['photoImagePath']) as image:assert image.size==(r['photoWidth'],r['photoHeight'])
  geometry=pipeline_geometry(r['photoWidth'],r['photoHeight']);assert set(r['geometry'])==set(geometry) and all(abs(r['geometry'][k]-v)<1e-8 for k,v in geometry.items())
 return d,records

def raw_predictions(epoch):
 authorization();data,records=inputs();weights=p.OUT/'checkpoints'/f'epoch{epoch:03d}.pt';meta=p.read(weights.with_suffix('.json'));assert p.sha(weights)==meta['sha256']
 folder=p.OUT/'predictions'/f'epoch{epoch:03d}';folder.mkdir(parents=True,exist_ok=True);binding={'modelSHA256':meta['sha256'],'inputsManifestSHA256':p.sha(p.OUT/'evaluation-inputs.json'),'recordsSHA256':p.sha(p.OUT/'records.json'),'codeSHA256':p.code_hashes(),'confidenceFloor':.001,'preNMS':True}
 if (folder/'binding.json').exists():assert p.read(folder/'binding.json')==binding
 else:p.save(folder/'binding.json',binding)
 model=YOLO(str(weights)).model.float().cpu().eval();assert model.names=={0:'pipe_end'} and model.model[-1].nc==1 and not model.model[-1].end2end;model.fuse();torch.set_num_threads(8)
 for r in data['images']:
  dest=folder/r['variant']/(r['id']+'.npz');side=dest.with_suffix('.json')
  if dest.exists() and side.exists():
   s=p.read(side);assert s['npzSHA256']==p.sha(dest) and s['id']==r['id'] and s['variant']==r['variant'] and s['modelSHA256']==meta['sha256'] and s['networkImageSHA256']==r['networkImageSHA256'] and s['rawLocations']==21504 and s['NMSApplied'] is False and s['outputCapApplied'] is False
   with np.load(dest,allow_pickle=False) as f:b,c=f['xyxy'],f['confidence'];assert b.dtype==c.dtype==np.float32 and b.shape==(len(c),4) and len(c)==s['retainedAfterFloor']<=21504 and np.isfinite(b).all() and np.isfinite(c).all() and np.all((c>=.001)&(c<=1))
   continue
  with Image.open(r['networkImagePath']) as im:rgb=np.asarray(im.convert('RGB')).copy();assert im.size==(1024,1024)
  tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).float()/255
  with torch.inference_mode():out=model(tensor);out=out[0] if isinstance(out,tuple) else out
  assert out.shape==(1,5,21504) and torch.isfinite(out).all();raw=out[0].T;xywh,c=raw[:,:4],raw[:,4];b=torch.cat((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2),1);keep=c>=.001;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,xyxy=b[keep].numpy(),confidence=c[keep].numpy())
  p.save(side,{'id':r['id'],'variant':r['variant'],'npzSHA256':p.sha(dest),'modelSHA256':meta['sha256'],'networkImageSHA256':r['networkImageSHA256'],'rawLocations':21504,'retainedAfterFloor':int(keep.sum()),'NMSApplied':False,'outputCapApplied':False});print('raw',epoch,r['dataRole'],r['variant'],r['id'],flush=True)

def old_nonregression(candidate,baseline):
 c,b=candidate['summary'],baseline['summary'];checks={}
 for key in ['TP']:checks[key]=c[key]>=b[key]
 for key in ['FP','FN','positiveCountWAPE','sceneMacroWAPE']:checks[key]=c[key]<=b[key]+(1e-12 if isinstance(b[key],float) else 0)
 for size in b['sizeRecall']:
  assert c['sizeRecall'][size]['GT']==b['sizeRecall'][size]['GT'];checks['sizeTP/'+size]=c['sizeRecall'][size]['TP']>=b['sizeRecall'][size]['TP']
 assert c['perScene'].keys()==b['perScene'].keys()
 checks['allNegativeImagesZero']=c['negativeFalsePositiveImages']==0
 return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}

def select():
 a=authorization();assert not (p.OUT/'selection.json').exists();done=p.read(p.OUT/'training-complete.json');assert done['status']=='COMPLETE_DATA_EXPANSION_12_EPOCHS_WITH_REGISTERED_CHECKPOINTS' and done['rootReceiptSHA256']==a['rootReceiptSHA256']
 from train_data_expansion import checkpoint_check
 protocol=p.read(p.OUT/'protocol.json');assert protocol['checkpointEpochs']==[4,8,12] and protocol['confidence']==.3 and protocol['nmsIoU']==.7;assert set(done['checkpoints'])=={'4','8','12'}
 for e in protocol['checkpointEpochs']:assert done['checkpoints'][str(e)]==checkpoint_check(e)
 data,records=inputs();baseline=p.read(p.OLD/'selection.json')['selected'];assert p.sha(p.OLD/'selection.json')==p.OLD_SELECTION_SHA and baseline['confidence']==.3 and baseline['nmsIoU']==.7
 # Baseline must pass its own exact old-scene gates.
 for variant in ['gallery','camera']:assert old_nonregression(baseline['variants'][variant],baseline['variants'][variant])['status']=='PASS'
 candidates=[]
 for epoch in protocol['checkpointEpochs']:
  raw_predictions(epoch);result={'epoch':epoch,'confidence':.3,'nmsIoU':.7,'modelSHA256':p.sha(p.OUT/'checkpoints'/f'epoch{epoch:03d}.pt'),'variants':{}}
  for variant in ['gallery','camera']:
   result['variants'][variant]={}
   for role in ['old','fresh']:
    rows=[r for r in data['images'] if r['variant']==variant and r['dataRole']==role];preds={};aps={}
    for r in rows:
     with np.load(p.OUT/'predictions'/f'epoch{epoch:03d}'/variant/(r['id']+'.npz'),allow_pickle=False) as f:raw={k:f[k].copy() for k in ['xyxy','confidence']}
     preds[r['id']]=postprocess(raw,r['geometry'],.3,.7);aps[r['id']]=postprocess(raw,r['geometry'],.001,.7)
    score=count_metrics(preds,records,rows,.3);score['summary'].update(ap_metrics(aps,records,rows));score['qualification']=eligible(score)
    score['gates']=old_nonregression(score,baseline['variants'][variant]) if role=='old' else eligible(score);result['variants'][variant][role]=score
  result['status']='PASS' if all(result['variants'][v][role]['gates']['status']=='PASS' for v in ['gallery','camera'] for role in ['old','fresh']) else 'FAIL';candidates.append(result);p.save(p.OUT/'scores'/f'epoch{epoch:03d}.json',result);print('score',epoch,result['status'],flush=True)
 rank=lambda r:(max(r['variants'][v]['fresh']['summary']['positiveCountMAE'] for v in ['gallery','camera']),max(r['variants'][v]['old']['summary']['positiveCountMAE'] for v in ['gallery','camera']),r['epoch']);passing=[c for c in candidates if c['status']=='PASS'];chosen=min(passing,key=rank) if passing else None
 p.save(p.OUT/'selection.json',{'status':'DATA_EXPANSION_VAL_QUALIFIED_PENDING_FRESH_HOLDOUT' if chosen else 'FAIL_DATA_EXPANSION_VALIDATION','selected':chosen,'bestResearchCandidateNotAutomaticallyUsable':min(candidates,key=rank),'candidates':candidates,'protocolSHA256':p.sha(p.OUT/'protocol.json'),'codeSHA256':p.code_hashes(),'inputsManifestSHA256':p.sha(p.OUT/'evaluation-inputs.json'),'oldBaselineSelectionSHA256':p.OLD_SELECTION_SHA,'old37TestAccessed':False,'freshTestAccessed':False,'VALIDATED_COREML_EXPORTIssued':False})
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['select']);args=parser.parse_args();select()
