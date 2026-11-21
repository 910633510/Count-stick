"""One fixed hardening candidate on the new group holdout; no former-test reader."""
from pathlib import Path
import argparse,datetime,json,math
import pipe_hardening as p
import release_pipe as release

def gate(receipt_path):
 binding=release.selected_binding()
 if binding is None:return None  # Never open a root receipt or heldout manifest for failed development.
 receipt=p.read(receipt_path);assert receipt['status']=='APPROVED_HARDENING_FRESH_TEST_SINGLE_CANDIDATE'
 for key in ['selectionSHA256','protocolSHA256','dataBindingSHA256','modelSHA256','epoch','operatingPoint','inputVariants','codeSHA256','releaseCodeSHA256','appSourceCodeSHA256']:assert receipt[key]==binding[key]
 assert receipt['evaluatorSHA256']==p.sha(__file__) and receipt['testExportHelperSHA256']==p.sha(release.HELPER)
 data=p.read(p.OUT/'data-binding.json');assert receipt['sealedManifestSHA256']==data['sealedFullManifestSHA256'];assert type(receipt['expectedTestImages']) is int and 0<receipt['expectedTestImages']<=100
 assert receipt['sealedManifestSHA256']!=p.read(p.OLD/'root-training-receipt.json')['sealedFullManifestSHA256'],'Former failed holdout cannot become the new test'
 parity_path=p.resolve(receipt['parityReportPath']);assert p.sha(parity_path)==receipt['parityReportSHA256'];parity=p.read(parity_path);assert parity['status']=='PASS_HARDENING_TORCH_COREML_VISION_PARITY'
 for key in ['selectionSHA256','protocolSHA256','dataBindingSHA256','modelSHA256','operatingPoint','releaseCodeSHA256','appSourceCodeSHA256']:assert parity[key]==binding[key]
 export=p.read(p.OUT/'export/export-status.json');assert parity['exportStatusSHA256']==p.sha(p.OUT/'export/export-status.json') and parity['exportID']==export['exportID'] and parity['packageTreeSHA256']==export['packageTreeSHA256']==release.tree_sha(export['packagePath'])
 return {**binding,'rootApprovalPath':str(p.resolve(receipt_path).relative_to(p.ROOT)),'rootApprovalSHA256':p.sha(receipt_path),'parityReportPath':str(parity_path.relative_to(p.ROOT)),'parityReportSHA256':p.sha(parity_path),'exportID':export['exportID'],'packageTreeSHA256':export['packageTreeSHA256'],'sealedManifestPath':receipt['sealedManifestPath'],'sealedManifestSHA256':receipt['sealedManifestSHA256'],'expectedTestImages':receipt['expectedTestImages'],'testExportHelperPath':str(release.HELPER.relative_to(p.ROOT)),'testExportHelperSHA256':p.sha(release.HELPER),'evaluatorSHA256':p.sha(__file__)}
def authorize(receipt_path):
 binding=gate(receipt_path)
 if binding is None:return {'status':'SKIP_NO_QUALIFIED_CANDIDATE','freshTestAccessed':False}
 protocol={'status':'FROZEN_SINGLE_HARDENING_TEST_BEFORE_ACCESS','createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':binding,'freshTestAccessed':False,'policy':'One selected checkpoint, fixed.3/.7, two App pixel paths; no test-based model switching, retraining or threshold adjustment.'};release.exclusive(p.OUT/'test/pre-test-protocol.json',protocol);return protocol

def load_inputs(binding,app_export_path):
 import numpy as np
 from PIL import Image
 from pipe_ml import pipeline_geometry
 registration_sha=p.sha(p.OUT/'test/pre-test-protocol.json');export=p.read(app_export_path);assert export['status']=='APP_PREPROCESSING_PIXELS_EXPORTED_NO_MODEL_INFERENCE' and export['kind']=='frozen_test_photo' and export['preTestProtocolSHA256']==registration_sha and export['sealedManifestSHA256']==binding['sealedManifestSHA256'];assert export['sourceCodeSHA256']==binding['appSourceCodeSHA256'] and export['testExportHelperSHA256']==binding['testExportHelperSHA256']
 sealed=p.resolve(binding['sealedManifestPath']);assert p.sha(sealed)==binding['sealedManifestSHA256'];manifest=p.read(sealed);assert manifest['status']=='FROZEN_ANNOTATION_QA_PASSED';test=[r for r in manifest['images'] if r['split']=='test'];other=[r for r in manifest['images'] if r['split']!='test'];assert len(test)==binding['expectedTestImages'] and len({r['id'] for r in test})==len(test)
 for key in ['groupId','sha256']:
  assert not {r[key] for r in test}&{r[key] for r in other}
 records={}
 for r in test:
  assert r['annotationQA']=='PASS' and r['dataNature']=='real' and r['isSynthetic'] is False and r['hasAnnotationOverlay'] is False and not r.get('sourceHadBlackMask',False)
  assert r['count']==len(r['xyxy']) and 0<=r['count']<=21504 and p.sha(p.resolve(r['imagePath']))==r['sha256'];assert all(all(math.isfinite(v) for v in [a,b,c,d]) and 0<=a<c<=r['width'] and 0<=b<d<=r['height'] for a,b,c,d in r['xyxy'])
  if r['count']:assert r['subtype']=='round' and r['annotationBoundary']=='outer_rim'
  else:assert r['negativeKind'] in ['non_pipe','empty_scene','side_only']
  records[r['id']]={**r,'originalWidth':r['width'],'originalHeight':r['height'],'originalXYXY':r['xyxy']}
 rows=[];seen=set();variant_map={'imageio-import':'gallery','camera-normalized':'camera'}
 for x in export['records']:
  assert x['id'] in records and x['split']=='test' and x['appPath'] in variant_map;r=records[x['id']];key=(x['id'],variant_map[x['appPath']]);assert key not in seen;seen.add(key);assert x['sourceSHA256']==r['sha256'] and x['sourceManifestSHA256']==binding['sealedManifestSHA256'] and x['sourceEXIFOrientation'] in [1,None] and x['outputOrientation']=='up'
  geometry=pipeline_geometry(x['preparedWidth'],x['preparedHeight']);assert 0<max(x['preparedWidth'],x['preparedHeight'])<=2048
  for k in ['gain','resizedWidth','resizedHeight','left','top']:assert abs(geometry[k]-x[k])<1e-8
  assert x['networkSize']==1024 and x['paddingRGB']==[114,114,114] and x['right']==1024-x['resizedWidth']-x['left'] and x['bottom']==1024-x['resizedHeight']-x['top']
  for path_key,hash_key,size in [('preparedPNG','preparedSHA256',(x['preparedWidth'],x['preparedHeight'])),('networkPNG','networkSHA256',(1024,1024))]:
   path=p.resolve(x[path_key]);assert path.is_relative_to(p.ROOT) and p.sha(path)==x[hash_key]
   with Image.open(path) as im:
    assert im.format=='PNG' and im.size==size
    if path_key=='networkPNG':
     pixels=np.asarray(im.convert('RGB'));mask=np.ones((1024,1024),dtype=bool);mask[x['top']:x['top']+x['resizedHeight'],x['left']:x['left']+x['resizedWidth']]=False;assert bool(np.all(pixels[mask]==114))
  rows.append({'id':x['id'],'variant':key[1],'split':'test','photoWidth':x['preparedWidth'],'photoHeight':x['preparedHeight'],'geometry':geometry,'sourceImageSHA256':r['sha256'],'networkImagePath':str(p.resolve(x['networkPNG'])),'networkImageSHA256':x['networkSHA256'],'photoImagePath':str(p.resolve(x['preparedPNG'])),'photoImageSHA256':x['preparedSHA256']})
 assert seen=={(id,v) for id in records for v in ['gallery','camera']} and len(rows)==2*len(records)
 return {'status':'VALIDATED_HARDENING_FRESH_TEST_INPUTS','images':rows,'variants':['gallery','camera'],'testContentIncluded':True,'sealedManifestSHA256':binding['sealedManifestSHA256'],'preTestProtocolSHA256':registration_sha,'appExportPath':str(p.resolve(app_export_path)),'appExportSHA256':p.sha(app_export_path)},records

def evaluate(receipt_path,app_export_path):
 binding=gate(receipt_path)
 if binding is None:return {'status':'SKIP_NO_QUALIFIED_CANDIDATE','freshTestAccessed':False}
 registration=p.read(p.OUT/'test/pre-test-protocol.json');assert registration['bindings']==binding and registration['status']=='FROZEN_SINGLE_HARDENING_TEST_BEFORE_ACCESS'
 release.exclusive(p.OUT/'test/test-opened.json',{'status':'OPENED_ONCE_FIXED_HARDENING_CANDIDATE','bindings':binding,'preTestProtocolSHA256':p.sha(p.OUT/'test/pre-test-protocol.json')})
 try:
  import numpy as np
  import torch
  from PIL import Image
  from ultralytics import YOLO
  from evaluate_pipe_ends import postprocess,count_metrics,ap_metrics,eligible
  inputs,records=load_inputs(binding,app_export_path);release.exclusive(p.OUT/'test/evaluation-inputs.json',inputs)
  model=YOLO(str(p.resolve(binding['weightsPath']))).model.float().cpu().eval();assert model.names=={0:'pipe_end'} and model.model[-1].nc==1 and not model.model[-1].end2end;model.fuse();torch.set_num_threads(8);operating={};ap_preds={}
  for x in inputs['images']:
   assert p.sha(x['networkImagePath'])==x['networkImageSHA256']
   with Image.open(x['networkImagePath']) as im:rgb=np.asarray(im.convert('RGB')).copy()
   tensor=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).unsqueeze(0).float()/255
   with torch.inference_mode():out=model(tensor);out=out[0] if isinstance(out,tuple) else out
   assert out.shape==(1,5,21504) and torch.isfinite(out).all();raw=out[0].T;xywh,c=raw[:,:4],raw[:,4];assert ((c>=0)&(c<=1)).all();b=torch.cat((xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2),1);keep=c>=.001;arrays={'xyxy':b[keep].numpy(),'confidence':c[keep].numpy()};path=p.OUT/'test/raw'/x['variant']/(x['id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as f:np.savez_compressed(f,**arrays)
   release.exclusive(path.with_suffix('.json'),{'id':x['id'],'variant':x['variant'],'split':'test','npzSHA256':p.sha(path),'modelSHA256':binding['modelSHA256'],'networkImageSHA256':x['networkImageSHA256'],'rawLocations':21504,'retainedAfterFloor':int(keep.sum()),'NMSApplied':False,'outputCapApplied':False});key=(x['variant'],x['id']);operating[key]=postprocess(arrays,x['geometry'],.3,.7);ap_preds[key]=postprocess(arrays,x['geometry'],.001,.7);print('fresh test raw',*key,flush=True)
  results={}
  for variant in ['gallery','camera']:
   rows=[x for x in inputs['images'] if x['variant']==variant];score=count_metrics({x['id']:operating[(variant,x['id'])] for x in rows},records,rows,.3);score['summary'].update(ap_metrics({x['id']:ap_preds[(variant,x['id'])] for x in rows},records,rows));score['gates']=eligible(score);score['predictions']={x['id']:operating[(variant,x['id'])] for x in rows};results[variant]=score
  assert gate(receipt_path)==binding and p.sha(app_export_path)==inputs['appExportSHA256'] and p.sha(p.resolve(binding['sealedManifestPath']))==binding['sealedManifestSHA256']
  report={**binding,'status':'PASS_HARDENING_FRESH_GROUP_HOLDOUT' if all(v['gates']['status']=='PASS' for v in results.values()) else 'FAIL_HARDENING_FRESH_GROUP_HOLDOUT','testImages':len(records),'split':'test','variants':results,'inputManifestPath':str(p.OUT/'test/evaluation-inputs.json'),'inputManifestSHA256':p.sha(p.OUT/'test/evaluation-inputs.json'),'preTestProtocolSHA256':p.sha(p.OUT/'test/pre-test-protocol.json'),'old37TestAccessed':False,'selectionBasedOnTest':False,'newTrainingPerformed':False};release.exclusive(p.OUT/'test/test-report.json',report);return report
 except BaseException as error:release.exclusive(p.OUT/'test/test-failure.json',{'status':'ERROR_FRESH_TEST_OPENED_NO_QUALIFICATION','errorType':type(error).__name__,'error':str(error),'bindings':binding});raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['authorize','evaluate']);parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--app-export',type=Path);args=parser.parse_args();result=authorize(args.receipt) if args.phase=='authorize' else evaluate(args.receipt,args.app_export);print(json.dumps({k:v for k,v in result.items() if k in ['status','testImages','freshTestAccessed']},indent=2))
