"""Independent artifact/cache audit. No training, image reads or model inference."""
import collections,csv,datetime,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import nms
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-20261026/model';REPORT=Path(__file__).with_name('independent-validation-review.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_equal(a,b,path=''):
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),(path,'keys',set(a)^set(b))
  for k in a:check_equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),(path,'list size')
  for i,(x,y) in enumerate(zip(a,b)):check_equal(x,y,path+'/'+str(i))
 elif isinstance(a,(int,float)) and not isinstance(a,bool):assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9,(path,a,b)
 else:assert a==b,(path,a,b)

def overlap(a,b):
 a=np.asarray(a,dtype=float).reshape(-1,4);b=np.asarray(b,dtype=float).reshape(-1,4)
 if not len(a) or not len(b):return np.empty((len(a),len(b)))
 w=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]));inter=w.prod(-1)
 return inter/((a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter+1e-9)

def post(raw,g,conf,iou):
 idx=np.flatnonzero(raw['confidence']>=conf);b=torch.from_numpy(raw['xyxy'][idx]);s=torch.from_numpy(raw['confidence'][idx]);keep=nms(b,s,iou).numpy();bb=b.numpy()[keep].copy();ss=s.numpy()[keep]
 bb[:,[0,2]]=(bb[:,[0,2]]-g['left'])/g['gain'];bb[:,[1,3]]=(bb[:,[1,3]]-g['top'])/g['gain'];bb[:,[0,2]]=bb[:,[0,2]].clip(0,g['sourceWidth']);bb[:,[1,3]]=bb[:,[1,3]].clip(0,g['sourceHeight']);valid=(bb[:,2]>bb[:,0])&(bb[:,3]>bb[:,1]);return bb[valid],ss[valid]

def truth(r,x):
 a=np.asarray(r['originalXYXY'],dtype=float).reshape(-1,4).copy();a[:,[0,2]]*=x['photoWidth']/r['originalWidth'];a[:,[1,3]]*=x['photoHeight']/r['originalHeight'];return a

def metrics(pred,records,inputs):
 rows=[]
 for x in inputs:
  r=records[x['id']];gt=truth(r,x);boxes,scores=pred[x['id']];ov=overlap(gt,boxes);matched=set()
  if ov.size:
   rr,cc=linear_sum_assignment(-((ov>=.5).astype(float)*(max(len(gt),len(boxes))+1)+ov));matched={int(a) for a,b in zip(rr,cc) if ov[a,b]>=.5}
  sizes={k:{'GT':0,'TP':0} for k in ['under16','16to32','over32']}
  for i,box in enumerate(gt):
   side=min(box[2]-box[0],box[3]-box[1])*1024/max(x['photoWidth'],x['photoHeight']);k='under16' if side<16 else '16to32' if side<32 else 'over32';sizes[k]['GT']+=1;sizes[k]['TP']+=int(i in matched)
  tp=len(matched);rows.append({'id':r['id'],'source':r['source'],'groupId':r['groupId'],'negativeKind':r.get('negativeKind'),'GT':len(gt),'predicted':len(boxes),'TP':tp,'FP':len(boxes)-tp,'FN':len(gt)-tp,'absoluteError':abs(len(boxes)-len(gt)),'signedError':len(boxes)-len(gt),'sizeCounts':sizes})
 positives=[r for r in rows if r['GT']];neg=[r for r in rows if not r['GT']];groups=collections.defaultdict(list)
 for r in positives:groups[str(r['groupId'])].append(r)
 gm={g:{'images':len(v),'GT':sum(r['GT'] for r in v),'WAPE':sum(r['absoluteError'] for r in v)/sum(r['GT'] for r in v)} for g,v in groups.items()}
 tp,fp,fn=[sum(r[k] for r in rows) for k in ['TP','FP','FN']]
 s={'images':len(rows),'positiveImages':len(positives),'negativeImages':len(neg),'GT':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'positiveCountMAE':float(np.mean([r['absoluteError'] for r in positives])),'positiveCountWAPE':sum(r['absoluteError'] for r in positives)/max(1,sum(r['GT'] for r in positives)),'sceneMacroWAPE':float(np.mean([v['WAPE'] for v in gm.values()])),'negativeFalsePositiveImages':sum(r['predicted']>0 for r in neg),'smallCountFailures':[r['id'] for r in positives if r['GT']<=20 and r['absoluteError']>1],'perScene':gm,'sizeRecall':{k:{'GT':sum(r['sizeCounts'][k]['GT'] for r in rows),'TP':sum(r['sizeCounts'][k]['TP'] for r in rows)} for k in ['under16','16to32','over32']}}
 for v in s['sizeRecall'].values():v['FN']=v['GT']-v['TP'];v['recall']=v['TP']/v['GT'] if v['GT'] else None
 s['negativeByKind']={k:{'images':sum(r['negativeKind']==k for r in neg),'falsePositiveImages':sum(r['negativeKind']==k and r['predicted']>0 for r in neg),'FP':sum(r['FP'] for r in neg if r['negativeKind']==k)} for k in sorted({r['negativeKind'] for r in neg})}
 s['perSource']={k:{'images':len(v),'GT':sum(r['GT'] for r in v),'TP':sum(r['TP'] for r in v),'FP':sum(r['FP'] for r in v),'FN':sum(r['FN'] for r in v),'countMAE':float(np.mean([r['absoluteError'] for r in v]))} for k in sorted({r['source'] for r in rows}) for v in [[r for r in rows if r['source']==k]]}
 s['perCountDensity']={k:{'images':len(v),'GT':sum(r['GT'] for r in v),'TP':sum(r['TP'] for r in v),'FP':sum(r['FP'] for r in v),'FN':sum(r['FN'] for r in v),'countMAE':float(np.mean([r['absoluteError'] for r in v]))} for k,lo,hi in [('1to20',1,20),('21to100',21,100),('101to500',101,500),('over500',501,10**9)] for v in [[r for r in rows if lo<=r['GT']<=hi]] if v}
 return {'summary':s,'images':rows}

def ap(pred,records,inputs):
 tps=[];scores=[];targets=[];matcher=SimpleNamespace(iouv=torch.linspace(.5,.95,10))
 for x in inputs:
  gt=truth(records[x['id']],x);boxes,sc=pred[x['id']];tps.append(BaseValidator.match_predictions(matcher,torch.zeros(len(boxes)),torch.zeros(len(gt)),torch.tensor(overlap(gt,boxes),dtype=torch.float32)).numpy());scores.extend(sc);targets.extend([0]*len(gt))
 if not scores:return {'AP50':0.,'AP50_95':0.}
 values=ap_per_class(np.concatenate(tps),np.asarray(scores),np.zeros(len(scores)),np.asarray(targets),plot=False,names={0:'pipe_end'})[5];return {'AP50':float(values[:,0].mean()),'AP50_95':float(values.mean())}

def gates(r):
 s=r['summary'];c={'precision>=.95':s['precision']>=.95,'recall>=.95':s['recall']>=.95,'positiveWAPE<=.05':s['positiveCountWAPE']<=.05,'sceneMacroWAPE<=.10':s['sceneMacroWAPE']<=.10,'smallCountsGT<=20Error<=1':not s['smallCountFailures'],'allNegativeImagesZero':s['negativeFalsePositiveImages']==0};return {'status':'PASS' if all(c.values()) else 'FAIL','checks':c}

def main():
 torch.set_num_threads(4);authorization=read(OUT/'training-authorization.json');protocol=read(OUT/'protocol.json');binding=read(OUT/'data-binding.json');selection=read(OUT/'selection.json');records=read(OUT/'records.json');rb={r['id']:r for r in records};loader=read(OUT/'dataloader-binding.json');completion=read(OUT/'training-complete.json')
 for name,digest in authorization['codeSHA256'].items():assert sha(ROOT/name)==digest
 assert sha(OUT/'protocol.json')==authorization['protocolSHA256'] and sha(OUT/'data-binding.json')==authorization['dataBindingSHA256'];assert sha(OUT/'records.json')==binding['recordsSHA256'];assert sha(authorization['rootReceiptPath'])==authorization['rootReceiptSHA256']
 assert completion['status']=='COMPLETE_20_EPOCHS_WITH_ACTUAL_REGISTERED_CHECKPOINTS' and completion['rootReceiptSHA256']==authorization['rootReceiptSHA256'];assert len(records)==204
 for r in records:assert sha(r['preparedImage'])==r['preparedImageSHA256'] and sha(r['preparedLabel'])==r['preparedLabelSHA256']
 order=loader['orderedIds'];assert set(order)=={r['id'] for r in records if r['split']=='train'} and len(order)==164
 gc=collections.Counter((rb[id]['bucket'],rb[id]['groupId']) for id in order);gs={b:{rb[id]['groupId'] for id in order if rb[id]['bucket']==b} for b in ['positive','negative']};m={'positive':.85,'negative':.15};weights=torch.tensor([m[rb[id]['bucket']]/len(gs[rb[id]['bucket']])/gc[(rb[id]['bucket'],rb[id]['groupId'])] for id in order],dtype=torch.double);assert np.allclose(weights.numpy(),loader['weights'],rtol=0,atol=1e-15)
 gen=torch.Generator().manual_seed(protocol['training']['seed']+71);totals=collections.Counter();sample_hashes={};allactual=[]
 for epoch in range(1,21):
  path=OUT/'sampling'/f'epoch{epoch:03d}.json';s=read(path);assert s['epoch']==epoch and len(s['actualImageIds'])==164 and s['samplerPrefixMatched'] is True
  expected=[order[i] for i in torch.multinomial(weights,164,True,generator=gen).tolist()];assert s['actualImageIds']==expected
  assert all(rb[id]['split']=='train' for id in expected)
  for field,key in [('sourceImageCounts','source'),('groupImageCounts','groupId'),('bucketImageCounts','bucket')]:assert dict(collections.Counter(rb[id][key] for id in expected))==s[field]
  original=collections.Counter()
  for id in expected:original[rb[id]['source']]+=rb[id]['count'];totals[rb[id]['bucket']]+=1
  assert dict(original)==s['sampledOriginalGTBoxes'];assert all(0<=v<=original[k] for k,v in s['postAugmentationGTBoxes'].items());assert all(math.isfinite(v) for v in s['lossItems'].values());sample_hashes[str(epoch)]=sha(path);allactual+=expected
 assert len(allactual)==3280
 csvrows=list(csv.DictReader((OUT/'run/results.csv').open()));assert len(csvrows)==20 and [int(r['epoch']) for r in csvrows]==list(range(1,21))
 snapshots=[]
 for epoch in [8,14,20]:
  path=OUT/'checkpoints'/f'epoch{epoch:03d}.pt';meta=read(path.with_suffix('.json'));assert completion['checkpoints'][str(epoch)]==meta and sha(path)==meta['sha256'] and meta['protocolSHA256']==authorization['protocolSHA256'] and meta['dataBindingSHA256']==authorization['dataBindingSHA256']
  ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==epoch-1 and ck['ema'].names=={0:'pipe_end'} and ck['ema'].model[-1].nc==1
  state=ck['ema'].state_dict();assert all(torch.isfinite(t).all() for t in state.values() if t.is_floating_point());steps=[float(s['step']) for s in ck['optimizer']['state'].values() if 'step' in s];assert steps and min(steps)>0
  snapshots.append({'epoch':epoch,'internalEpoch':ck['epoch'],'sha256':meta['sha256'],'optimizerStepMin':min(steps),'optimizerStepMax':max(steps),'optimizerParameterStates':len(steps),'EMAUpdates':ck['updates'],'EMATensorsFinite':True})
 assert all(snapshots[i]['optimizerStepMax']<snapshots[i+1]['optimizerStepMin'] for i in range(2))
 print('Training audit passed:3280 actual draws,3 real optimizer checkpoints',flush=True)
 manifest=read(OUT/'evaluation-inputs.json');assert sha(OUT/'evaluation-inputs.json')==selection['inputsManifestSHA256'];assert len(manifest['images'])==80;val={r['id']:r for r in records if r['split']=='val'};variants=manifest['variants'];assert variants==['gallery','camera'];inputrows={v:[x for x in manifest['images'] if x['variant']==v] for v in variants};results=[];cachehashes={}
 for epoch in [8,14,20]:
  cache={};weightsha=completion['checkpoints'][str(epoch)]['sha256'];cb=read(OUT/'predictions'/f'epoch{epoch:03d}'/'binding.json');assert cb['modelSHA256']==weightsha and cb['inputsManifestSHA256']==selection['inputsManifestSHA256'] and cb['recordsSHA256']==binding['recordsSHA256'] and cb['codeSHA256']==authorization['codeSHA256'] and cb['preNMS'] is True
  for x in manifest['images']:
   path=OUT/'predictions'/f'epoch{epoch:03d}'/x['variant']/(x['id']+'.npz');s=read(path.with_suffix('.json'));assert s['id']==x['id'] and s['variant']==x['variant'] and s['modelSHA256']==weightsha and s['networkImageSHA256']==x['networkImageSHA256'] and s['npzSHA256']==sha(path);assert s['NMSApplied'] is False and s['outputCapApplied'] is False and s['rawLocations']==21504
   with np.load(path,allow_pickle=False) as z:raw={k:z[k].copy() for k in ['xyxy','confidence']}
   b,c=raw['xyxy'],raw['confidence'];assert b.dtype==c.dtype==np.float32 and b.shape==(len(c),4) and len(c)==s['retainedAfterFloor'] and len(c)<=21504 and np.isfinite(b).all() and np.isfinite(c).all() and np.all((c>=.001)&(c<=1));cache[(x['variant'],x['id'])]=raw;cachehashes[str(path.relative_to(ROOT))]=s['npzSHA256']
  for iou in [.5,.7]:
   aps={v:ap({x['id']:post(cache[(v,x['id'])],x['geometry'],.001,iou) for x in inputrows[v]},val,inputrows[v]) for v in variants}
   for conf in [.2,.3,.4]:
    vv={}
    for v in variants:
     r=metrics({x['id']:post(cache[(v,x['id'])],x['geometry'],conf,iou) for x in inputrows[v]},val,inputrows[v]);r['summary'].update(aps[v]);r['gates']=gates(r);vv[v]=r
    item={'epoch':epoch,'confidence':conf,'nmsIoU':iou,'variants':vv,'status':'PASS' if all(x['gates']['status']=='PASS' for x in vv.values()) else 'FAIL','modelSHA256':weightsha};saved=read(OUT/'scores'/f'epoch{epoch:03d}-conf{conf}-iou{iou}.json');check_equal(item,saved);results.append(item)
    print('Recomputed',epoch,conf,iou,item['status'],flush=True)
 check_equal(results,selection['candidates']);rank=lambda r:(max(v['summary']['sceneMacroWAPE'] for v in r['variants'].values()),max(v['summary']['positiveCountMAE'] for v in r['variants'].values()),-r['confidence'],r['epoch'],r['nmsIoU']);passing=[r for r in results if r['status']=='PASS'];chosen=min(passing,key=rank) if passing else None;check_equal(chosen,selection['selected']);check_equal(min(results,key=rank),selection['bestResearchCandidateNotAutomaticallyUsable'])
 assert sha(OUT/'selection.json')=='9eda69afbaa72634270f86f910e7f87719bbd55a5880d6dbb000e6d4a62209f8'
 report={'status':'PASS_INDEPENDENT_PIPE_TRAINING_AND_VALIDATION_AUDIT','createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditScriptSHA256':sha(__file__),'trainingAudit':{'status':'PASS','actualImageDraws':3280,'actualBuckets':dict(totals),'epochs':20,'deterministicSamplerReplayMatchedEveryID':True,'onlyTrainingIDsConsumed':True,'csvEpochRows':20,'samplingSHA256':sample_hashes,'checkpoints':snapshots,'sourceAndPreparedHashesRechecked':True},'validationAudit':{'status':'PASS_RECOMPUTED_ALL18_CONFIGURATIONS_AND_SELECTION','selectionSHA256':sha(OUT/'selection.json'),'selected':selection['selected'],'modelSHA256':chosen['modelSHA256'] if chosen else None,'protocolSHA256':authorization['protocolSHA256'],'codeSHA256':authorization['codeSHA256'],'recordsSHA256':binding['recordsSHA256'],'dataBindingSHA256':authorization['dataBindingSHA256'],'inputsManifestSHA256':selection['inputsManifestSHA256'],'candidateCount':18,'qualifiedCount':len(passing),'rawCacheFiles':len(cachehashes),'rawCacheSHA256':cachehashes,'operatingCountsFPFNMAEAllGroupsRecomputed':True,'AP50AndAP50_95Recomputed':True,'allGatesAndRankingMatched':True,'metricsUseSameAuthorGT':'Independent recomputation is not independent reannotation.','coverageLimits':['No positivevalidationimage withGT<=20; that guard hasno empiricalcoverage.','under16network-pixel targets: selectedTP33/GT69 perpath, recall~47.8%, despiteaggregatequalification.','40validationphotos are33positive+7negative,35conservative groups; onlysinglepublicsource, notprospectivephonefielddata.']},'sealedTestAccessed':False,'modelInferencePerformed':False,'optimizerStepsPerformedByAuditor':0,'CoreMLModelEnabled':False}
 REPORT.write_text(json.dumps(report,indent=2)+'\n');print('PASS independent validation audit',sha(REPORT),flush=True)
if __name__=='__main__':main()
