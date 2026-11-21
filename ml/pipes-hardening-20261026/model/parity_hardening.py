"""Frozen raw-v3 CoreML/Vision parity on development PNGs only; no test reader."""
from pathlib import Path
import argparse,hashlib,json,math,subprocess
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import pipe_hardening as p
import evaluate_hardening as e
import release_pipe as r

def mapped(objects,g):
 boxes=[];scores=[]
 for o in objects:
  assert o['label']=='pipe_end';b=np.asarray(o['networkXYXY'],dtype=float);assert b.shape==(4,) and np.isfinite(b).all() and math.isfinite(o['confidence'])
  b[[0,2]]=(b[[0,2]]-g['left'])/g['gain'];b[[1,3]]=(b[[1,3]]-g['top'])/g['gain'];b[[0,2]]=b[[0,2]].clip(0,g['sourceWidth']);b[[1,3]]=b[[1,3]].clip(0,g['sourceHeight'])
  if b[2]>b[0] and b[3]>b[1]:boxes.append(b.tolist());scores.append(o['confidence'])
 return {'xyxy':boxes,'confidence':scores}
def comparison(a,b,g,limits):
 from evaluate_pipe_ends import iou_matrix
 aa=np.asarray(a['xyxy'],dtype=float).reshape(-1,4);bb=np.asarray(b['xyxy'],dtype=float).reshape(-1,4);same=len(aa)==len(bb);edge=score=0.;minimum=1.
 if len(aa) and len(bb):
  overlaps=iou_matrix(aa,bb);ia,ib=linear_sum_assignment(-overlaps);minimum=float(overlaps[ia,ib].min());edge=float(np.abs(aa[ia]-bb[ib]).max()*g['gain']);score=float(np.abs(np.asarray(a['confidence'])[ia]-np.asarray(b['confidence'])[ib]).max())
 elif not same:minimum=0.
 checks={'exactCounts':same,'minimumIoU':minimum>=limits['minimumMatchedIoU'],'maximumEdge':edge<=limits['maximumEdgeErrorNetworkPixels'],'maximumConfidence':score<=limits['maximumConfidenceError']}
 return {'status':'PASS' if all(checks.values()) else 'FAIL','referenceCount':len(aa),'candidateCount':len(bb),'minimumIoU':minimum,'maximumEdgeNetworkPixels':edge,'maximumConfidenceDifference':score,'checks':checks}
def main():
 binding=r.selected_binding()
 if binding is None:return {'status':'SKIP_NO_QUALIFIED_CANDIDATE','freshTestAccessed':False}
 protocol=p.read(p.OUT/'parity-protocol.json');assert protocol['status']=='FROZEN_HARDENING_PARITY_BEFORE_PREDICTION' and protocol['bindings']==binding
 old=p.read(r.PARITY_SOURCE);assert protocol['numericPolicy']==old['numericPolicy'] and protocol['qualificationPolicy']==old['qualificationPolicy']
 export=p.read(p.OUT/'export/export-status.json');assert protocol['exportStatusSHA256']==p.sha(p.OUT/'export/export-status.json') and export['modelSHA256']==binding['modelSHA256'] and export['packageTreeSHA256']==r.tree_sha(export['packagePath'])
 data,records=e.inputs();selected=p.read(p.OUT/'selection.json')['selected'];run=p.OUT/'parity';assert not run.exists();run.mkdir();modes=['direct_cpu','direct_cpu_ne','vision_cpu','vision_cpu_ne']
 rows=[]
 for x in data['images']:
  with Image.open(x['networkImagePath']) as im:rgb=np.asarray(im.convert('RGB'))
  rows.append({k:x[k] for k in ['id','variant','networkImagePath','networkImageSHA256']}|{'decodedRGBSHA256':hashlib.sha256(rgb.tobytes()).hexdigest()})
 request={'status':'APPROVED_NATIVE_PIPE_PARITY_RUN','packagePath':export['packagePath'],'confidenceThreshold':.3,'iouThreshold':.7,'expectedModelMetadata':{'countscope.export_id':export['exportID'],'countscope.checkpoint_sha256':binding['modelSHA256'],'countscope.confidence_threshold':'0.3','countscope.iou_threshold':'0.7','countscope.class_label':'pipe_end','countscope.input_size':'1024'},'images':rows,'bindings':binding};r.exclusive(run/'request.json',request)
 native=run/'native_runner';subprocess.run(['xcrun','swiftc','-O',str(r.NATIVE),str(r.NMS),'-o',str(native)],check=True)
 try:
  with (run/'native.log').open('w') as log:subprocess.run([str(native),'--run',str(run/'request.json'),str(run/'native')],stdout=log,stderr=subprocess.STDOUT,check=True)
  outputs=p.read(run/'native/native-results.json');assert outputs['requestSHA256']==p.sha(run/'request.json');indexed={(v['variant'],v['id']):v for v in outputs['records']};assert len(indexed)==len(rows)==len(outputs['records']) and set(indexed)=={(v['variant'],v['id']) for v in rows};perimage=[];predictions={m:{} for m in modes};passed=True
  for x in data['images']:
   key=(x['variant'],x['id']);out=indexed[key];rawfile=p.OUT/'predictions'/f'epoch{binding["epoch"]:03d}'/x['variant']/(x['id']+'.npz');side=p.read(rawfile.with_suffix('.json'));assert side['npzSHA256']==p.sha(rawfile) and side['modelSHA256']==binding['modelSHA256'] and side['networkImageSHA256']==x['networkImageSHA256']
   with np.load(rawfile,allow_pickle=False) as arrays:raw={k:arrays[k].copy() for k in ['xyxy','confidence']}
   reference=e.postprocess(raw,x['geometry'],.3,.7);nativepred={m:mapped(out['modes'][m]['objects'],x['geometry']) for m in modes};rgb=all(v==out['decodedRGBSHA256'] for v in out['actualInputRGBSHA256'].values());shape=all(v['rawShape']==[1,5,21504] and v['rawDataType']==65568 for v in out['modes'].values())
   tests={'cpuVsTorch':comparison(reference,nativepred['direct_cpu'],x['geometry'],protocol['numericPolicy']['CPUCoreML_vs_Torch']),'cpuNEVsCPU':comparison(nativepred['direct_cpu'],nativepred['direct_cpu_ne'],x['geometry'],protocol['numericPolicy']['allCoreML_vs_CPUCoreML']),'visionCPUVsDirectCPU':comparison(nativepred['direct_cpu'],nativepred['vision_cpu'],x['geometry'],protocol['numericPolicy']['Vision_vs_directCoreML_sameComputeUnits']),'visionCPU_NEVsDirectCPU_NE':comparison(nativepred['direct_cpu_ne'],nativepred['vision_cpu_ne'],x['geometry'],protocol['numericPolicy']['Vision_vs_directCoreML_sameComputeUnits'])}
   ok=rgb and shape and all(v['status']=='PASS' for v in tests.values());passed=passed and ok;perimage.append({'id':x['id'],'variant':x['variant'],'dataRole':x['dataRole'],'exactRGB':rgb,'rawSingleClassContract':shape,'comparisons':tests,'status':'PASS' if ok else 'FAIL'})
   for m in modes:predictions[m][key]=nativepred[m]
  qualification={};anchor=p.read(p.OLD/'selection.json')['selected']
  for mode in modes:
   qualification[mode]={}
   for variant in ['gallery','camera']:
    qualification[mode][variant]={}
    for role in ['old','fresh']:
     subset=[x for x in data['images'] if x['variant']==variant and x['dataRole']==role];score=e.count_metrics({x['id']:predictions[mode][(variant,x['id'])] for x in subset},records,subset,.3);s=score['summary'];base=selected['variants'][variant][role]['summary'];score['gates']=e.old_nonregression(score,anchor['variants'][variant]) if role=='old' else e.eligible(score);score['nonRegressionAgainstSelectedTorch']={k:s[k]>=base[k] if k=='TP' else s[k]<=base[k]+1e-12 for k in ['TP','FP','FN','positiveCountWAPE','sceneMacroWAPE']};score['nonRegressionAgainstSelectedTorch']['negativeFP']=s['negativeFalsePositiveImages']==0
     passed=passed and score['gates']['status']=='PASS' and all(score['nonRegressionAgainstSelectedTorch'].values());qualification[mode][variant][role]=score
  assert r.selected_binding()==binding and r.tree_sha(export['packagePath'])==export['packageTreeSHA256']
  report={**binding,'status':'PASS_HARDENING_TORCH_COREML_VISION_PARITY' if passed else 'FAIL_HARDENING_TORCH_COREML_VISION_PARITY','exportID':export['exportID'],'packageTreeSHA256':export['packageTreeSHA256'],'exportStatusSHA256':p.sha(p.OUT/'export/export-status.json'),'parityProtocolSHA256':p.sha(p.OUT/'parity-protocol.json'),'inputManifestPath':str(p.OUT/'evaluation-inputs.json'),'inputManifestSHA256':p.sha(p.OUT/'evaluation-inputs.json'),'images':len(rows),'perImage':perimage,'qualification':qualification,'nativeResultsPath':str(run/'native/native-results.json'),'nativeResultsSHA256':p.sha(run/'native/native-results.json'),'nativeRunnerBinarySHA256':p.sha(native),'freshTestAccessed':False,'modelRetrained':False,'thresholdsChanged':False};r.exclusive(run/'parity-report.json',report);return report
 except BaseException as error:r.exclusive(run/'parity-error.json',{'status':'ERROR_HARDENING_PARITY_NO_PASS','bindings':binding,'errorType':type(error).__name__,'error':str(error),'freshTestAccessed':False});raise
if __name__=='__main__':print(json.dumps({k:v for k,v in main().items() if k in ['status','images','freshTestAccessed']},indent=2))
