"""Frozen same-input native CoreML/Vision parity; validation only, no model training."""
import argparse,datetime,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[3];MODEL=ROOT/'output/pipes-20261026/model';ART=Path(__file__).parent;OUT=ROOT/'output/pipes-20261026/parity'
sys.path.insert(0,str(ROOT/'ml/pipes-20261026/model'))
import evaluate_pipe_ends as e
import pipe_ml as p
import runtime_v3_root as rt

def tree_sha(root):
 root=Path(root);d=hashlib.sha256()
 for file in sorted(x for x in root.rglob('*') if x.is_file()):d.update(str(file.relative_to(root)).encode()+b'\0'+file.read_bytes())
 return d.hexdigest()
def emit(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def mapped(objects,g):
 boxes=[];scores=[];labels=[]
 for o in objects:
  b=np.asarray(o['networkXYXY'],dtype=float);assert b.shape==(4,) and np.isfinite(b).all() and math.isfinite(o['confidence'])
  b[[0,2]]=(b[[0,2]]-g['left'])/g['gain'];b[[1,3]]=(b[[1,3]]-g['top'])/g['gain'];b[[0,2]]=b[[0,2]].clip(0,g['sourceWidth']);b[[1,3]]=b[[1,3]].clip(0,g['sourceHeight'])
  if b[2]>b[0] and b[3]>b[1]:boxes.append(b.tolist());scores.append(o['confidence']);labels.append(o['label'])
 return {'xyxy':boxes,'confidence':scores,'labels':labels}
def comparison(a,b,g,limits):
 aa=np.asarray(a['xyxy'],dtype=float).reshape(-1,4);bb=np.asarray(b['xyxy'],dtype=float).reshape(-1,4);same=len(aa)==len(bb);edge=0.;score=0.;minimum=1.
 if len(aa) and len(bb):
  iou=e.iou_matrix(aa,bb);ra,rb=linear_sum_assignment(-iou);minimum=float(iou[ra,rb].min());edge=float(np.abs(aa[ra]-bb[rb]).max()*g['gain']);score=float(np.abs(np.asarray(a['confidence'])[ra]-np.asarray(b['confidence'])[rb]).max())
 elif not same:minimum=0.
 checks={'exactCounts':same,'minimumIoU':minimum>=limits['minimumMatchedIoU'],'maximumEdge':edge<=limits['maximumEdgeErrorNetworkPixels'],'maximumConfidence':score<=limits['maximumConfidenceError']}
 return {'status':'PASS' if all(checks.values()) else 'FAIL','referenceCount':len(aa),'candidateCount':len(bb),'minimumIoU':minimum,'maximumEdgeNetworkPixels':edge,'maximumConfidenceDifference':score,'checks':checks}

def main(receipt):
 protocol=p.read(ART/'protocol-v3.json');assert protocol['status']=='FROZEN_RUNTIME_V3_RAW_NMS_TECHNICAL_REPLACEMENT_SAME_TOLERANCES';rt.verify_authorization()
 selected=p.read(MODEL/'selection.json');chosen=selected['selected'];assert chosen and chosen['status']=='PASS' and p.sha(MODEL/'selection.json')==protocol['selectionSHA256']
 authorization=p.read(receipt);assert authorization['status']=='APPROVED_PIPE_PARITY_EXPORT' and authorization['selectionSHA256']==protocol['selectionSHA256'] and authorization['modelSHA256']==protocol['modelSHA256']
 exported=p.read(MODEL/'export-raw-v3/export-status.json');assert exported['status']=='EXPORTED_RAW_V3_PENDING_PARITY_AND_FROZEN_TEST' and exported['selectionSHA256']==protocol['selectionSHA256'] and exported['modelSHA256']==protocol['modelSHA256'];assert tree_sha(exported['packagePath'])==exported['packageTreeSHA256']
 assert {'confidence':exported['confidenceThreshold'],'nmsIoU':exported['iouThreshold']}==protocol['operatingPoint']
 manifest,records=rt.app_inputs(MODEL/'evaluation-inputs.json');assert len(records)==40 and len(manifest['images'])==80 and manifest['variants']==['gallery','camera'] and all(r['split']=='val' for r in records.values());assert selected['inputsManifestSHA256']==p.sha(MODEL/'evaluation-inputs.json')
 run=OUT/'validated-candidate-run-v3-metadata';run.mkdir(parents=True,exist_ok=False)
 binding={'selectionSHA256':p.sha(MODEL/'selection.json'),'modelSHA256':exported['modelSHA256'],'protocolSHA256':p.sha(MODEL/'protocol.json'),'operatingPoint':protocol['operatingPoint'],'inputVariants':manifest['variants'],'exportID':exported['exportID'],'packageTreeSHA256':exported['packageTreeSHA256'],'inputManifestBindings':{v:{'path':str(MODEL/'evaluation-inputs.json'),'sha256':p.sha(MODEL/'evaluation-inputs.json')} for v in manifest['variants']},'parityProtocolSHA256':p.sha(ART/'protocol-v3.json'),'runnerSHA256':p.sha(ART/'native_parity_v3.swift'),'comparisonScriptSHA256':p.sha(__file__),'rootExportReceiptSHA256':p.sha(receipt),'runtimeVersion':3,'nmsSourceSHA256':p.read(rt.AMEND)['nmsSourceSHA256'],'runtimeAmendmentPath':str(rt.AMEND),'runtimeAmendmentSHA256':p.sha(rt.AMEND),'runtimeVerifierSHA256':p.sha(Path(rt.__file__)),'originalALLStatus':'FAIL_UNSUPPORTED_MACOS_MPSGRAPH_MLIR_ABORT'}
 emit(run/'opened.json',{'status':'FIXED_VAL_PARITY_OPENED','bindings':binding,'createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'testAccessed':False})
 try:
  rows=[]
  for x in manifest['images']:
   with Image.open(x['networkImagePath']) as im:rgb=np.asarray(im.convert('RGB'));rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest()
   rows.append({'id':x['id'],'variant':x['variant'],'networkImagePath':x['networkImagePath'],'networkImageSHA256':x['networkImageSHA256'],'decodedRGBSHA256':rgbsha})
  request={'status':'APPROVED_NATIVE_PIPE_PARITY_RUN','packagePath':exported['packagePath'],'confidenceThreshold':exported['confidenceThreshold'],'iouThreshold':exported['iouThreshold'],'expectedModelMetadata':{'countscope.export_id':exported['exportID'],'countscope.checkpoint_sha256':exported['modelSHA256'],'countscope.confidence_threshold':str(exported['confidenceThreshold']).removeprefix('0'),'countscope.iou_threshold':str(exported['iouThreshold']).removeprefix('0'),'countscope.class_label':'pipe_end','countscope.input_size':'1024'},'images':rows,'bindings':binding};emit(run/'request.json',request)
  native=OUT/'native_parity_v3';subprocess.run(['xcrun','swiftc','-O',str(ART/'native_parity_v3.swift'),str(ROOT/'Count stick/PipeRawPostprocessor.swift'),'-o',str(native)],check=True)
  with (run/'native.log').open('w') as log:subprocess.run([str(native),'--run',str(run/'request.json'),str(run/'native')],stdout=log,stderr=subprocess.STDOUT,check=True)
  result=p.read(run/'native/native-results.json');assert result['status']=='NATIVE_OUTPUTS_CAPTURED_NOT_YET_QUALIFIED' and result['requestSHA256']==p.sha(run/'request.json')
  output={(r['variant'],r['id']):r for r in result['records']};assert len(output)==80
  expected={(r['variant'],r['id']) for r in manifest['images']};assert set(output)==expected
  per_image=[];predictions={k:{} for k in ['direct_cpu','direct_cpu_ne','vision_cpu','vision_cpu_ne']};allpass=True
  for x in manifest['images']:
   key=(x['variant'],x['id']);r=output[key];rawfile=MODEL/'predictions'/f'epoch{chosen["epoch"]:03d}'/x['variant']/(x['id']+'.npz');side=p.read(rawfile.with_suffix('.json'));assert side['npzSHA256']==p.sha(rawfile) and side['modelSHA256']==exported['modelSHA256'] and side['networkImageSHA256']==x['networkImageSHA256']
   with np.load(rawfile,allow_pickle=False) as arrays:raw={k:arrays[k].copy() for k in ['xyxy','confidence']}
   reference=e.postprocess(raw,x['geometry'],chosen['confidence'],chosen['nmsIoU'])
   nativepred={mode:mapped(v['objects'],x['geometry']) for mode,v in r['modes'].items()}
   rgb_ok=all(v==r['decodedRGBSHA256'] for v in r['actualInputRGBSHA256'].values());labels_ok=all(label=='pipe_end' for v in nativepred.values() for label in v['labels']);padding_ok=all(mode['rawShape']==[1,5,21504] and mode['rawDataType'] in [65568,'float32'] for mode in r['modes'].values())
   checks={'cpuVsTorch':comparison(reference,nativepred['direct_cpu'],x['geometry'],protocol['numericPolicy']['CPUCoreML_vs_Torch']),'allVsCPU':comparison(nativepred['direct_cpu'],nativepred['direct_cpu_ne'],x['geometry'],protocol['numericPolicy']['allCoreML_vs_CPUCoreML']),'visionCPUVsDirectCPU':comparison(nativepred['direct_cpu'],nativepred['vision_cpu'],x['geometry'],protocol['numericPolicy']['Vision_vs_directCoreML_sameComputeUnits']),'visionAllVsDirectAll':comparison(nativepred['direct_cpu_ne'],nativepred['vision_cpu_ne'],x['geometry'],protocol['numericPolicy']['Vision_vs_directCoreML_sameComputeUnits'])}
   ok=rgb_ok and labels_ok and padding_ok and all(v['status']=='PASS' for v in checks.values());allpass=allpass and ok
   per_image.append({'id':x['id'],'variant':x['variant'],'status':'PASS' if ok else 'FAIL','exactRGB':rgb_ok,'labels':labels_ok,'rawSingleClassContract':padding_ok,'comparisons':checks})
   for mode in predictions:predictions[mode][key]={k:nativepred[mode][k] for k in ['xyxy','confidence']}
  qualification={}
  for mode in predictions:
   qualification[mode]={}
   for variant in manifest['variants']:
    inputs=[x for x in manifest['images'] if x['variant']==variant];score=e.count_metrics({x['id']:predictions[mode][(variant,x['id'])] for x in inputs},records,inputs,chosen['confidence']);score['gates']=e.eligible(score);s=score['summary'];base=chosen['variants'][variant]['summary'];nonreg={'TP':s['TP']>=base['TP'],'FP':s['FP']<=base['FP'],'FN':s['FN']<=base['FN'],'positiveCountWAPE':s['positiveCountWAPE']<=base['positiveCountWAPE']+1e-12,'sceneMacroWAPE':s['sceneMacroWAPE']<=base['sceneMacroWAPE']+1e-12,'everyNegativeZero':s['negativeFalsePositiveImages']==0};score['nonRegressionAgainstTorch']=nonreg;qualification[mode][variant]=score;allpass=allpass and score['gates']['status']=='PASS' and all(nonreg.values())
  rt.verify_authorization();assert tree_sha(exported['packagePath'])==binding['packageTreeSHA256'] and p.sha(MODEL/'selection.json')==binding['selectionSHA256'] and p.sha(ART/'protocol-v3.json')==binding['parityProtocolSHA256']
  report={**binding,'status':'PASS_PIPE_TORCH_COREML_VISION_PARITY' if allpass else 'FAIL_PIPE_TORCH_COREML_VISION_PARITY','createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':result['macOSVersion'],'images':80,'photographs':40,'modes':['direct_cpu','direct_cpu_ne','vision_cpu','vision_cpu_ne'],'perImage':per_image,'qualification':qualification,'nativeResultsPath':str(run/'native/native-results.json'),'nativeResultsSHA256':p.sha(run/'native/native-results.json'),'nativeRunnerBinarySHA256':p.sha(native),'testAccessed':False,'thresholdsChanged':False,'modelRetrained':False,'limitations':protocol['limits']}
  emit(run/'parity-report.json',report);print(report['status'],p.sha(run/'parity-report.json'))
 except BaseException as error:
  emit(run/'parity-error.json',{**binding,'status':'ERROR_NATIVE_PIPE_PARITY_NO_PASS','errorType':type(error).__name__,'error':str(error),'testAccessed':False,'retryPolicy':'Retainallartifacts; anytechnicalresume requiresreviewand unchangedcandidate/thresholds/tolerances.'});raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();main(args.receipt)
