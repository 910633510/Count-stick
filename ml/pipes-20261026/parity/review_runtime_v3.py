"""Independent cached-output audit. No model execution, optimizer, or test-data reader."""
import collections,datetime,hashlib,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.ops import nms
ROOT=Path(__file__).resolve().parents[3]
M=ROOT/'output/pipes-20261026/model'; P=ROOT/'ml/pipes-20261026/parity'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tree(p):
 p=Path(p); h=hashlib.sha256()
 for f in sorted(x for x in p.rglob('*') if x.is_file()):h.update(str(f.relative_to(p)).encode()+b'\0'+f.read_bytes())
 return h.hexdigest()
def equal(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();[equal(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
 else:assert a==b,(a,b)
def iou(a,b):
 a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
 inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return inter/((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter+1e-9)
def mapped(boxes,scores,g,dtype=float):
 b=np.asarray(boxes,dtype=dtype).reshape(-1,4).copy(); c=np.asarray(scores,dtype=float)
 b[:,[0,2]]=(b[:,[0,2]]-g['left'])/g['gain'];b[:,[1,3]]=(b[:,[1,3]]-g['top'])/g['gain']
 b[:,[0,2]]=b[:,[0,2]].clip(0,g['sourceWidth']);b[:,[1,3]]=b[:,[1,3]].clip(0,g['sourceHeight'])
 good=(b[:,2]>b[:,0])&(b[:,3]>b[:,1]);return b[good].astype(float),c[good]
def compare(a,b,g,policy):
 ba,ca=a;bb,cb=b;edge=score=0.;minimum=1.;same=len(ba)==len(bb)
 if len(ba) and len(bb):
  overlaps=iou(ba,bb);ia,ib=linear_sum_assignment(-overlaps);minimum=float(overlaps[ia,ib].min());edge=float(np.max(abs(ba[ia]-bb[ib]))*g['gain']);score=float(np.max(abs(ca[ia]-cb[ib])))
 elif not same:minimum=0.
 checks={'exactCounts':same,'minimumIoU':minimum>=policy['minimumMatchedIoU'],'maximumEdge':edge<=policy['maximumEdgeErrorNetworkPixels'],'maximumConfidence':score<=policy['maximumConfidenceError']}
 assert all(checks.values())
 return dict(status='PASS',referenceCount=len(ba),candidateCount=len(bb),minimumIoU=minimum,maximumEdgeNetworkPixels=edge,maximumConfidenceDifference=score,checks=checks)
A=read(P/'runtime-v3-amendment.json');report_path=ROOT/'output/pipes-20261026/parity/validated-candidate-run-v3-metadata/parity-report.json';R=read(report_path)
assert sha(P/'runtime-v3-amendment.json')=='b29a29bb547a500bd8596b829bba252abb44688fd3d1ad82b5b6399de8c6e253'
assert sha(report_path)=='99dcfe1253af853087998894d1eba7d390c12af76aba7a715f0c06b9ae4459c7'
assert R['status']=='PASS_PIPE_TORCH_COREML_VISION_PARITY' and R['runtimeVersion']==3
for path,digest in A['codeSHA256'].items():assert sha(ROOT/path)==digest
for key,path in [('protocolSHA256',M/'protocol.json'),('selectionSHA256',M/'selection.json'),('modelSHA256',M/'checkpoints/epoch020.pt'),('trainingRootReceiptSHA256',M/'root-training-receipt.json'),('exportStatusSHA256',ROOT/A['exportStatusPath']),('priorExportStatusSHA256',M/'export/export-status.json'),('nmsSourceSHA256',ROOT/A['nmsSourcePath'])]:assert sha(path)==A[key]
for key in ['protocolSHA256','selectionSHA256','modelSHA256','operatingPoint','exportID','packageTreeSHA256','nmsSourceSHA256']:assert R[key]==A[key]
assert A['allowedChanges']==['computeUnits_all_to_cpuAndNeuralEngine','coreml_raw_nms_false','shared_swift_greedy_nms']
for name,path in {'PhotoProcessing':'Count stick/StickDetector.swift','PipeLetterbox':'Count stick/PipeDetector.swift','ExportHelper':'Count stickTests/PipePreprocessingExportTests.swift'}.items():assert sha(ROOT/path)==A['newAppSourceCodeSHA256'][name]
old=(ROOT/A['oldPipeDetectorSnapshotPath']).read_text();new=(ROOT/'Count stick/PipeDetector.swift').read_text();assert sha(ROOT/A['oldPipeDetectorSnapshotPath'])==A['oldPipeDetectorSHA256']
def block(text):return text[text.index('struct PipeLetterbox {'):text.index('enum PipeCountRegion')]
assert block(old)==block(new) and hashlib.sha256(block(old).encode()).hexdigest()==A['pipeLetterboxBlockSHA256']
assert old[:old.index('final class PipeDetector:')]==new[:new.index('final class PipeDetector:')]
for name in ['PhotoProcessing','ExportHelper']:assert A['oldAppSourceCodeSHA256'][name]==A['newAppSourceCodeSHA256'][name]
for sub,key in [('export-raw-v3','packageTreeSHA256'),('export','priorPackageTreeSHA256')]:
 e=read(M/sub/'export-status.json');assert tree(e['packagePath'])==A[key]
rawexport=read(ROOT/A['exportStatusPath']);assert rawexport['nms'] is False and rawexport['quantize']==32 and rawexport['inputSize']==1024 and rawexport['outputShape']==[1,5,21504] and rawexport['outputDataType']==65568
v1=read(P/'protocol.json');v3=read(P/'protocol-v3.json');assert v1['numericPolicy']==v3['numericPolicy'] and v1['qualificationPolicy']==v3['qualificationPolicy'];assert sha(P/'protocol-v3.json')==R['parityProtocolSHA256']==A['v3parityProtocolSHA256']
for file,key in [('native_parity_v3.swift','runnerSHA256'),('run_parity_v3_root.py','comparisonScriptSHA256'),('runtime_v3_root.py','runtimeVerifierSHA256')]:assert sha(P/file)==R[key]
assert sha(R['nativeResultsPath'])==R['nativeResultsSHA256'];N=read(R['nativeResultsPath']);request_path=report_path.parent/'request.json';Q=read(request_path);assert sha(request_path)==N['requestSHA256'];assert Q['confidenceThreshold']==.3 and Q['iouThreshold']==.7
assert sha(ROOT/'output/pipes-20261026/parity/native_parity_v3')==R['nativeRunnerBinarySHA256']
inputs=read(M/'evaluation-inputs.json');records={x['id']:x for x in read(M/'records.json') if x['split']=='val'};assert len(records)==40 and len(inputs['images'])==80 and sha(M/'records.json')==inputs['recordsSHA256']
assert inputs['testContentIncluded'] is False and inputs['approvedAppSourceCodeSHA256']==A['oldAppSourceCodeSHA256']
for v,b in R['inputManifestBindings'].items():assert v in ['gallery','camera'] and sha(b['path'])==b['sha256']==sha(M/'evaluation-inputs.json')
output={(x['id'],x['variant']):x for x in N['records']};ri={(x['id'],x['variant']):x for x in R['perImage']};assert len(output)==len(ri)==80
modes=['direct_cpu','direct_cpu_ne','vision_cpu','vision_cpu_ne'];allpred={m:{} for m in modes};perimage=[];cachehashes={}
for x in inputs['images']:
 key=(x['id'],x['variant']);n=output[key];g=x['geometry'];assert x['split']=='val' and x['id'] in records
 assert sha(x['networkImagePath'])==x['networkImageSHA256']==n['networkImageSHA256']
 with Image.open(x['networkImagePath']) as im:rgb=np.asarray(im.convert('RGB'));assert rgb.shape==(1024,1024,3)
 rgbsha=hashlib.sha256(rgb.tobytes()).hexdigest();assert n['decodedRGBSHA256']==rgbsha and set(n['actualInputRGBSHA256'].values())=={rgbsha}
 npz=M/'predictions/epoch020'/x['variant']/(x['id']+'.npz');side=read(npz.with_suffix('.json'));assert sha(npz)==side['npzSHA256'] and side['networkImageSHA256']==x['networkImageSHA256'] and side['modelSHA256']==A['modelSHA256'] and side['id']==x['id'] and side['variant']==x['variant'];cachehashes[str(npz.relative_to(ROOT))]=sha(npz)
 with np.load(npz,allow_pickle=False) as f:b=f['xyxy'];c=f['confidence'];assert b.dtype==c.dtype==np.float32 and np.isfinite(b).all() and np.isfinite(c).all()
 keep=c>=np.float32(.3);b,c=b[keep],c[keep];ids=nms(torch.from_numpy(b),torch.from_numpy(c),.7).numpy();reference=mapped(b[ids],c[ids],g,np.float32)
 native={}
 for mode in modes:
  o=n['modes'][mode];assert o['rawShape']==[1,5,21504] and o['rawDataType']==65568 and not o['paddingClassesPresent']
  assert all(v['label']=='pipe_end' and .3<=v['confidence']<=1 and 0<=v['anchor']<21504 for v in o['objects'])
  bb=np.asarray([v['networkXYXY'] for v in o['objects']],dtype=float).reshape(-1,4);ss=np.asarray([v['confidence'] for v in o['objects']]);assert np.isfinite(bb).all() and np.isfinite(ss).all()
  native[mode]=mapped(bb,ss,g);allpred[mode][key]=native[mode]
 comparisons={name:compare(a,b,g,v3['numericPolicy'][policy]) for name,a,b,policy in [('cpuVsTorch',reference,native['direct_cpu'],'CPUCoreML_vs_Torch'),('allVsCPU',native['direct_cpu'],native['direct_cpu_ne'],'allCoreML_vs_CPUCoreML'),('visionCPUVsDirectCPU',native['direct_cpu'],native['vision_cpu'],'Vision_vs_directCoreML_sameComputeUnits'),('visionAllVsDirectAll',native['direct_cpu_ne'],native['vision_cpu_ne'],'Vision_vs_directCoreML_sameComputeUnits')]}
 equal(comparisons,ri[key]['comparisons']);assert all(ri[key][q] is True for q in ['exactRGB','labels','rawSingleClassContract']);perimage.append(comparisons)
# Independent GT matching, size, scene and qualification recomputation for every mode/variant.
selected=read(M/'selection.json')['selected'];summaries={}
for mode in modes:
 summaries[mode]={}
 for variant in ['gallery','camera']:
  rows=[];claimed=R['qualification'][mode][variant];byid={r['id']:r for r in claimed['images']}
  for x in [z for z in inputs['images'] if z['variant']==variant]:
   r=records[x['id']];truth=np.asarray(r['originalXYXY'],dtype=float).reshape(-1,4)*np.asarray([x['photoWidth']/r['originalWidth'],x['photoHeight']/r['originalHeight']]*2);b,c=allpred[mode][(x['id'],variant)];overlap=iou(truth,b);matched=set()
   if overlap.size:
    rr,cc=linear_sum_assignment(-((overlap>=.5)*(max(len(truth),len(b))+1)+overlap));matched={int(a) for a,z in zip(rr,cc) if overlap[a,z]>=.5}
   size={s:{'GT':0,'TP':0} for s in ['under16','16to32','over32']}
   for idx,t in enumerate(truth):
    side=min(t[2]-t[0],t[3]-t[1])*1024/max(x['photoWidth'],x['photoHeight']);s='under16' if side<16 else '16to32' if side<32 else 'over32';size[s]['GT']+=1;size[s]['TP']+=int(idx in matched)
   row=dict(id=x['id'],source=r['source'],groupId=r['groupId'],negativeKind=r.get('negativeKind'),GT=len(truth),predicted=len(b),TP=len(matched),FP=len(b)-len(matched),FN=len(truth)-len(matched),absoluteError=abs(len(b)-len(truth)),signedError=len(b)-len(truth),sizeCounts=size);equal(row,byid[x['id']]);rows.append(row)
  pos=[r for r in rows if r['GT']];neg=[r for r in rows if not r['GT']];groups=collections.defaultdict(list)
  for r in pos:groups[r['groupId']].append(r)
  tp,fp,fn=[sum(r[k] for r in rows) for k in ['TP','FP','FN']];summary=dict(TP=tp,FP=fp,FN=fn,precision=tp/(tp+fp),recall=tp/(tp+fn),positiveCountMAE=sum(r['absoluteError'] for r in pos)/len(pos),positiveCountWAPE=sum(r['absoluteError'] for r in pos)/sum(r['GT'] for r in pos),sceneMacroWAPE=float(np.mean([sum(r['absoluteError'] for r in rr)/sum(r['GT'] for r in rr) for rr in groups.values()])),negativeFalsePositiveImages=sum(bool(r['predicted']) for r in neg))
  for k,v in summary.items():equal(v,claimed['summary'][k])
  base=selected['variants'][variant]['summary'];assert tp>=base['TP'] and fp<=base['FP'] and fn<=base['FN'];assert summary['positiveCountWAPE']<=base['positiveCountWAPE']+1e-12 and summary['sceneMacroWAPE']<=base['sceneMacroWAPE']+1e-12
  assert summary['precision']>=.95 and summary['recall']>=.95 and summary['positiveCountWAPE']<=.05 and summary['sceneMacroWAPE']<=.1 and all(r['predicted']==0 for r in neg) and all(r['absoluteError']<=1 for r in pos if r['GT']<=20)
  assert claimed['gates']['status']=='PASS' and all(claimed['gates']['checks'].values()) and all(claimed['nonRegressionAgainstTorch'].values());summaries[mode][variant]=summary
assert not R['testAccessed'] and not R['modelRetrained'] and not R['thresholdsChanged']
review={k:A[k] for k in ['protocolSHA256','selectionSHA256','modelSHA256','codeSHA256','operatingPoint','oldAppSourceCodeSHA256','newAppSourceCodeSHA256','allowedChanges','pixelGeometryUnchanged','trainingRootReceiptSHA256','exportStatusPath','exportStatusSHA256','packageTreeSHA256','exportID','priorExportStatusSHA256','priorPackageTreeSHA256','nmsSourcePath','nmsSourceSHA256']}
review.update(status='PASS_PIPE_RUNTIME_V3_SOURCE_AMENDMENT',createdUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),amendmentSHA256=sha(P/'runtime-v3-amendment.json'),parityReportPath=str(report_path.relative_to(ROOT)),parityReportSHA256=sha(report_path),parityProtocolPath=str((P/'protocol-v3.json').relative_to(ROOT)),parityProtocolSHA256=sha(P/'protocol-v3.json'),independentAuditScriptSHA256=sha(__file__),nativeResultsSHA256=R['nativeResultsSHA256'],recomputedAll80Images=True,recomputedAll8ModeVariantQualifications=True,originalNumericAndQualificationPolicyUnchanged=True,pipeLetterboxBlockSHA256=A['pipeLetterboxBlockSHA256'],originalEightCodeFilesUnchanged=True,torchRawCacheSHA256=cachehashes,qualificationSummaries=summaries,maximumCPUvsTorchEdgePixels=max(c['cpuVsTorch']['maximumEdgeNetworkPixels'] for c in perimage),maximumCPUvsTorchScoreDifference=max(c['cpuVsTorch']['maximumConfidenceDifference'] for c in perimage),minimumCPUvsTorchIoU=min(c['cpuVsTorch']['minimumIoU'] for c in perimage),sameBackendVisionExactValues=True,earlierFailuresPreserved=['ALL_MLIR_ABORT','V2_APPLE_NMS_COUNT_MISMATCH','V3_METADATA_STRING_FORMAT_PRE_PREDICTION_STOP'],scope='Read-only cached val audit. No model inference, no optimizer, no sealed test content read.',limitations=['Mac CPU and CPU+NeuralEngine parity does not prove physical iPhone runtime/performance.','Positive validation has no GT<=20 images; under16px target recall remains33/69.','Annotation source labels were visually reviewed, not an independent exhaustive recount.'])
dest=P/'runtime-v3-independent-review.json';assert not dest.exists();dest.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');print(str(dest),sha(dest),review['status'])
