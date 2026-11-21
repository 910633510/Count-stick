"""Independent cached fixed-holdout review. Invoke only after the report exists.
Does not load or execute a model, train, modify predictions or select a candidate.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,importlib.util,json,math
import numpy as np
import torch
from torchvision.ops import nms
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-20261026/model'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def resolve(p):return Path(p) if Path(p).is_absolute() else ROOT/p
def eq(a,b):
 if isinstance(a,dict):assert set(a)==set(b);[eq(a[k],b[k]) for k in a]
 elif isinstance(a,list):assert len(a)==len(b);[eq(x,y) for x,y in zip(a,b)]
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
 else:assert a==b,(a,b)
def iou(a,b):
 a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
 overlap=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return overlap/((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-overlap+1e-9)
def pp(boxes,scores,g,floor,threshold):
 take=scores>=np.float32(floor);b=boxes[take];c=scores[take];keep=nms(torch.from_numpy(b),torch.from_numpy(c),threshold).numpy();b=b[keep].copy();c=c[keep]
 b[:,[0,2]]=(b[:,[0,2]]-g['left'])/g['gain'];b[:,[1,3]]=(b[:,[1,3]]-g['top'])/g['gain'];b[:,[0,2]]=b[:,[0,2]].clip(0,g['sourceWidth']);b[:,[1,3]]=b[:,[1,3]].clip(0,g['sourceHeight']);valid=(b[:,2]>b[:,0])&(b[:,3]>b[:,1])
 return {'xyxy':b[valid].tolist(),'confidence':c[valid].tolist(),'nmsBeforeUnpadding':True,'invalidPaddingOnlyBoxesDropped':int((~valid).sum()),'arbitraryMarkerCapApplied':False,'rawCapacitySaturated':len(keep)==21504}
def ap_match(truth,pred):
 overlaps=iou(truth,pred).astype(np.float32);result=np.zeros((len(pred),10),dtype=bool)
 for j,t in enumerate(torch.linspace(.5,.95,10).tolist()):
  pairs=np.asarray(np.nonzero(overlaps>=t)).T
  if len(pairs)>1:
   pairs=pairs[overlaps[pairs[:,0],pairs[:,1]].argsort()[::-1]];pairs=pairs[np.unique(pairs[:,1],return_index=True)[1]];pairs=pairs[np.unique(pairs[:,0],return_index=True)[1]]
  if len(pairs):result[pairs[:,1].astype(int),j]=True
 return result
def ap_from(tp,scores,gt):
 if not scores:return {'AP50':0.,'AP50_95':0.}
 order=np.argsort(-np.asarray(scores));tp=np.concatenate(tp)[order];fp=(1-tp).cumsum(0);tp=tp.cumsum(0);recall=tp/(gt+1e-16);precision=tp/(tp+fp);aps=[]
 for j in range(10):
  r=recall[:,j];p=precision[:,j];mr=np.concatenate(([0.],r,[r[-1]],[1.]));mp=np.concatenate(([1.],p,[0.],[0.]));mp=np.maximum.accumulate(mp[::-1])[::-1];grid=np.linspace(0,1,101);aps.append(float(np.trapezoid(np.interp(grid,mr,mp),grid)))
 return {'AP50':aps[0],'AP50_95':float(np.mean(aps))}
def metric_row(record,x,pred):
 truth=np.asarray(record['xyxy'],dtype=float).reshape(-1,4)*np.asarray([x['photoWidth']/record['width'],x['photoHeight']/record['height']]*2);boxes=np.asarray(pred['xyxy'],dtype=float).reshape(-1,4);overlaps=iou(truth,boxes);matches=set()
 if overlaps.size:
  rr,cc=linear_sum_assignment(-((overlaps>=.5)*(max(len(truth),len(boxes))+1)+overlaps));matches={int(a) for a,b in zip(rr,cc) if overlaps[a,b]>=.5}
 size={k:{'GT':0,'TP':0} for k in ['under16','16to32','over32']}
 for i,t in enumerate(truth):
  m=min(t[2]-t[0],t[3]-t[1])*1024/max(x['photoWidth'],x['photoHeight']);s='under16' if m<16 else '16to32' if m<32 else 'over32';size[s]['GT']+=1;size[s]['TP']+=int(i in matches)
 n,m,tp=len(truth),len(boxes),len(matches)
 return dict(id=record['id'],source=record['source'],groupId=record['groupId'],negativeKind=record.get('negativeKind'),GT=n,predicted=m,TP=tp,FP=m-tp,FN=n-tp,absoluteError=abs(m-n),signedError=m-n,sizeCounts=size),truth
def summary(rows):
 pos=[r for r in rows if r['GT']];neg=[r for r in rows if not r['GT']];groups=collections.defaultdict(list)
 for r in pos:groups[str(r['groupId'])].append(r)
 pergroup={g:{'images':len(v),'GT':sum(r['GT'] for r in v),'WAPE':sum(r['absoluteError'] for r in v)/sum(r['GT'] for r in v)} for g,v in groups.items()};tp,fp,fn=[sum(r[k] for r in rows) for k in ['TP','FP','FN']]
 s=dict(images=len(rows),positiveImages=len(pos),negativeImages=len(neg),GT=tp+fn,TP=tp,FP=fp,FN=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),positiveCountMAE=float(np.mean([r['absoluteError'] for r in pos])),positiveCountWAPE=sum(r['absoluteError'] for r in pos)/max(1,sum(r['GT'] for r in pos)),sceneMacroWAPE=float(np.mean([v['WAPE'] for v in pergroup.values()])),negativeFalsePositiveImages=sum(r['predicted']>0 for r in neg),smallCountFailures=[r['id'] for r in pos if r['GT']<=20 and r['absoluteError']>1],perScene=pergroup)
 s['sizeRecall']={k:{'GT':sum(r['sizeCounts'][k]['GT'] for r in rows),'TP':sum(r['sizeCounts'][k]['TP'] for r in rows)} for k in ['under16','16to32','over32']}
 for v in s['sizeRecall'].values():v['FN']=v['GT']-v['TP'];v['recall']=v['TP']/v['GT'] if v['GT'] else None
 s['negativeByKind']={k:{'images':sum(r['negativeKind']==k for r in neg),'falsePositiveImages':sum(r['negativeKind']==k and r['predicted']>0 for r in neg),'FP':sum(r['FP'] for r in neg if r['negativeKind']==k)} for k in sorted({r['negativeKind'] for r in neg})}
 def aggregate(v):return {'images':len(v),'GT':sum(r['GT'] for r in v),'TP':sum(r['TP'] for r in v),'FP':sum(r['FP'] for r in v),'FN':sum(r['FN'] for r in v),'countMAE':float(np.mean([r['absoluteError'] for r in v]))}
 s['perSource']={k:aggregate([r for r in rows if r['source']==k]) for k in sorted({r['source'] for r in rows})}
 s['perCountDensity']={name:aggregate(v) for name,lo,hi in [('1to20',1,20),('21to100',21,100),('101to500',101,500),('over500',501,10**9)] for v in [[r for r in rows if lo<=r['GT']<=hi]] if v}
 return s
def gates(s):
 checks={'precision>=.95':s['precision']>=.95,'recall>=.95':s['recall']>=.95,'positiveWAPE<=.05':s['positiveCountWAPE']<=.05,'sceneMacroWAPE<=.10':s['sceneMacroWAPE']<=.1,'smallCountsGT<=20Error<=1':not s['smallCountFailures'],'allNegativeImagesZero':s['negativeFalsePositiveImages']==0};return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}
def main(receipt_path):
 # No heldout contents are opened until an actual completed report is present.
 report_path=OUT/'test/test-report.json';R=read(report_path);assert R['status'] in ['PASS_PIPE_FROZEN_GROUP_HOLDOUT','FAIL_PIPE_FROZEN_GROUP_HOLDOUT']
 gate_path=Path(__file__).with_name('evaluate_frozen_test.py');assert sha(gate_path)==R['evaluatorSHA256'];spec=importlib.util.spec_from_file_location('frozen_test_gate_review',gate_path);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate);binding=gate.gate(receipt_path)
 registration=read(OUT/'test/pre-test-protocol.json');assert registration['bindings']==binding and sha(OUT/'test/pre-test-protocol.json')==R['preTestProtocolSHA256'];assert read(OUT/'test/test-opened.json')['bindings']==binding
 for key in ['modelSHA256','selectionSHA256','protocolSHA256','operatingPoint','inputVariants','codeSHA256','evaluatorSHA256','sealedManifestSHA256']:assert R[key]==binding[key]
 assert R['runtimeSourceBinding']=={k:v for k,v in binding.items() if k.startswith('runtime') or k in ['trainingAppSourceCodeSHA256','appSourceCodeSHA256','PipeLetterboxBlockSHA256']}
 assert R['testImages']==37 and R['split']=='test' and not R['selectionBasedOnTest'] and not R['newTrainingPerformed'];assert R['operatingPoint']=={'confidence':.3,'nmsIoU':.7}
 manifests=list(R['inputManifestBindings'].values());assert len(manifests)==2 and manifests[0]==manifests[1]
 inputs_path=resolve(manifests[0]['path']);assert sha(inputs_path)==manifests[0]['sha256'];inputs=read(inputs_path);assert len(inputs['images'])==74 and inputs['status']=='VALIDATED_APP_FROZEN_TEST_INPUTS' and inputs['preTestProtocolSHA256']==R['preTestProtocolSHA256']
 audit_path=resolve(R['inputAuditBinding']['path']);assert sha(audit_path)==R['inputAuditBinding']['sha256'];audit=read(audit_path);assert audit['status'].startswith('PASS') and audit['inputsManifestSHA256']==sha(inputs_path)
 sealed=resolve(binding['sealedManifestPath']);assert sha(sealed)==binding['sealedManifestSHA256'];allrows=read(sealed)['images'];records={r['id']:r for r in allrows if r['split']=='test'};other=[r for r in allrows if r['split']!='test'];assert len(records)==37
 for k in ['groupId','sha256','RGBPixelSHA256']:assert not {r[k] for r in records.values()}&{r[k] for r in other}
 assert {(x['id'],x['variant']) for x in inputs['images']}=={(i,v) for i in records for v in ['gallery','camera']}
 results={};cache_hashes={}
 for variant in ['gallery','camera']:
  rows=[];aptp=[];apscores=[];totalgt=0
  for x in [z for z in inputs['images'] if z['variant']==variant]:
   r=records[x['id']];assert x['split']=='test' and x['sourceImageSHA256']==r['sha256'];assert r['count']==len(r['xyxy']) and r['annotationQA']=='PASS';assert sha(resolve(x['networkImagePath']))==x['networkImageSHA256'] and sha(resolve(x['photoImagePath']))==x['photoImageSHA256']
   path=OUT/'test/raw'/variant/(x['id']+'.npz');side=read(path.with_suffix('.json'));assert sha(path)==side['npzSHA256'] and side['id']==x['id'] and side['split']=='test' and side['variant']==variant and side['modelSHA256']==binding['modelSHA256'] and side['networkImageSHA256']==x['networkImageSHA256'] and side['preTestProtocolSHA256']==R['preTestProtocolSHA256'];assert side['rawLocations']==21504 and not side['NMSApplied'] and not side['outputCapApplied']
   with np.load(path,allow_pickle=False) as data:b=data['xyxy'];c=data['confidence'];assert b.dtype==c.dtype==np.float32 and b.shape==(len(c),4) and len(c)==side['retainedAfterFloor']<=21504 and np.isfinite(b).all() and np.isfinite(c).all() and np.all((c>=.001)&(c<=1))
   cache_hashes[str(path.relative_to(ROOT))]=sha(path);operating=pp(b,c,x['geometry'],.3,.7);eq(operating,R['variants'][variant]['predictions'][x['id']]);row,truth=metric_row(r,x,operating);rows.append(row);ap=pp(b,c,x['geometry'],.001,.7);aptp.append(ap_match(truth,ap['xyxy']));apscores.extend(ap['confidence']);totalgt+=len(truth)
  score=summary(rows);score.update(ap_from(aptp,apscores,totalgt));eq(rows,R['variants'][variant]['images']);eq(score,R['variants'][variant]['summary']);elig=gates(score);eq(elig,R['variants'][variant]['gates']);results[variant]={'summary':score,'gates':elig}
 qualified=all(v['gates']['status']=='PASS' for v in results.values());assert R['status']==('PASS_PIPE_FROZEN_GROUP_HOLDOUT' if qualified else 'FAIL_PIPE_FROZEN_GROUP_HOLDOUT')
 review={'status':'PASS_INDEPENDENT_AUDIT_OF_TEST_PASS' if qualified else 'PASS_INDEPENDENT_AUDIT_OF_TEST_FAIL','createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'testReportPath':str(report_path.relative_to(ROOT)),'testReportSHA256':sha(report_path),'auditScriptSHA256':sha(__file__),'modelSHA256':binding['modelSHA256'],'selectionSHA256':binding['selectionSHA256'],'protocolSHA256':binding['protocolSHA256'],'codeSHA256':binding['codeSHA256'],'operatingPoint':binding['operatingPoint'],'runtimeSourceBinding':R['runtimeSourceBinding'],'inputManifestSHA256':sha(inputs_path),'inputAuditSHA256':sha(audit_path),'sealedManifestSHA256':sha(sealed),'testImages':37,'variants':results,'rawCacheSHA256':cache_hashes,'all74CachesRecomputed':True,'allPredictionsRowsSummaryAPAndGatesMatch':True,'testCandidateQualified':qualified,'newModelInference':False,'newOptimizerSteps':0,'otherCandidatesEvaluated':False,'limitations':R['limitations']}
 dest=Path(__file__).with_name('independent-test-review.json');assert not dest.exists();dest.write_text(json.dumps(review,indent=2,allow_nan=False)+'\n');print(dest,sha(dest),review['status'])
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();main(args.receipt)
