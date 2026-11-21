#!/usr/bin/env python3
"""Independent read-only verification from saved detections; no model inference."""
from pathlib import Path
import collections,json,hashlib,math,time,types
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'output/research-20261025-hardcases/retention-savefix-v2'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
EPS=1e-9

def overlap(a,b):
 if not len(a) or not len(b):return np.zeros((len(a),len(b)))
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(hi-lo,0).prod(2)
 return inter/(np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None,:]-inter+1e-9)

def aggregate(rows):
 tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows);errors=np.array([r['signedError'] for r in rows]);s={'images':len(rows),'groundTruthObjects':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'F1':2*tp/max(1,2*tp+fp+fn),'countMAE':float(abs(errors).mean()),'countRMSE':float(np.sqrt((errors**2).mean())),'countBias':float(errors.mean()),'sizeRecall':{}}
 for size in ('under16px','16to32px','over32px'):
  gt=sum(r['sizeRecall'][size]['groundTruth'] for r in rows);n=sum(r['sizeRecall'][size]['TP'] for r in rows);s['sizeRecall'][size]={'groundTruth':gt,'TP':n,'FN':gt-n,'recall':n/gt if gt else None}
 return s

def detection_ap(predictions,records):
 tp=[];scores=[];targets=[];matcher=types.SimpleNamespace(iouv=torch.linspace(.5,.95,10))
 for r in records:
  p=predictions[r['id']];gt=np.asarray(r['xyxy'],float).reshape(-1,4);boxes=np.asarray(p['xyxy'],float).reshape(-1,4)
  matches=BaseValidator.match_predictions(matcher,torch.zeros(len(boxes)),torch.zeros(len(gt)),torch.from_numpy(overlap(gt,boxes)).float());tp.append(matches.numpy());scores.extend(p['confidence']);targets.extend([0]*len(gt))
 if not scores:return {'AP50':0.,'AP50_95':0.,'AP75':0.}
 ap=ap_per_class(np.concatenate(tp),np.asarray(scores),np.zeros(len(scores)),np.asarray(targets),plot=False,names={0:'colony'})[5]
 return {'AP50':float(ap[:,0].mean()),'AP50_95':float(ap.mean()),'AP75':float(ap[:,5].mean())}

def score_positive(predictions,records,physical=False):
 rows=[]
 for r in records:
  assert r['split']=='val';p=predictions[r['id']];gt=np.asarray(r['xyxy'],float).reshape(-1,4);boxes=np.asarray(p['xyxy'],float).reshape(-1,4)[np.asarray(p['confidence'])>=.3];iou=overlap(gt,boxes);matched=set()
  if iou.size:
   ii,jj=linear_sum_assignment(-((iou>=.5).astype(float)*1001+iou));matched={int(i) for i,j in zip(ii,jj) if iou[i,j]>=.5}
  sizes={k:{'groundTruth':0,'TP':0} for k in ('under16px','16to32px','over32px')}
  for i,b in enumerate(gt):
   side=min(b[2]-b[0],b[3]-b[1])*1024/max(r['width'],r['height']);key='under16px' if side<16 else '16to32px' if side<32 else 'over32px';sizes[key]['groundTruth']+=1;sizes[key]['TP']+=int(i in matched)
  n=len(matched);rows.append({'id':r['id'],'source':r['source'],'species':r['species'],'truth':len(gt),'predicted':len(boxes),'signedError':len(boxes)-len(gt),'TP':n,'FP':len(boxes)-n,'FN':len(gt)-n,'sizeRecall':sizes})
 s=aggregate(rows);s['perSource']={}
 for source in sorted({r['source'] for r in rows}):
  subrows=[r for r in rows if r['source']==source];sub=aggregate(subrows);sub['perSpecies']={sp:aggregate([r for r in subrows if r['species']==sp]) for sp in sorted({r['species'] for r in subrows})}
  for metric in ('countMAE','precision','recall','F1'):sub['speciesMacro'+metric[0].upper()+metric[1:]]=float(np.mean([v[metric] for v in sub['perSpecies'].values()]))
  s['perSource'][source]=sub
 for metric in ('countMAE','precision','recall','F1'):s['equalSourceSpeciesMacro'+metric[0].upper()+metric[1:]]=float(np.mean([v['speciesMacro'+metric[0].upper()+metric[1:]] for v in s['perSource'].values()]))
 s['primaryMacroMAE']=s['equalSourceSpeciesMacroCountMAE'];s.update(confidence=.3,matchIoU=.5);s.update(detection_ap(predictions,records));report={'summary':s,'images':rows,'physicalGroups':len({str(r['groupId']) for r in records})}
 if physical:
  byid={r['id']:r for r in records};grouped=collections.defaultdict(list)
  for r in rows:q=byid[r['id']];grouped[q['source'],q['species'],str(q['groupId'])].append(r)
  groups=[{'source':src,'species':sp,'groupId':g,**aggregate(v)} for (src,sp,g),v in sorted(grouped.items())];sourcevalues=[]
  for src in sorted({r['source'] for r in groups}):
   speciesvalues=[]
   for sp in sorted({r['species'] for r in groups if r['source']==src}):
    val=float(np.mean([r['countMAE'] for r in groups if r['source']==src and r['species']==sp]));speciesvalues.append(val);s['perSource'][src]['perSpecies'][sp]['physicalGroupMacroCountMAE']=val
   val=float(np.mean(speciesvalues));sourcevalues.append(val);s['perSource'][src]['physicalGroupSpeciesMacroCountMAE']=val
  s['imageWeightedPrimaryMacroMAE']=s['primaryMacroMAE'];s['primaryMacroMAE']=float(np.mean(sourcevalues));s['physicalGroupSpeciesSourceMacroMAE']=s['primaryMacroMAE'];report['perPhysicalGroup']=groups
 return report

def compare(expected,reported,path=''):
 if isinstance(expected,dict):
  for k,v in expected.items():assert k in reported,(path,k);compare(v,reported[k],path+'/'+k)
 elif isinstance(expected,list):
  assert len(expected)==len(reported),path
  for i,(x,y) in enumerate(zip(expected,reported)):compare(x,y,path+'/'+str(i))
 elif isinstance(expected,float):assert math.isclose(expected,reported,rel_tol=1e-11,abs_tol=1e-9),(path,expected,reported)
 else:assert expected==reported,(path,expected,reported)

def gates(c,b):
 rows=[]
 def leq(name,value,limit):rows.append({'gate':name,'pass':bool(value<=limit+EPS),'actual':value,'maximum':limit})
 x,y=c['old56']['summary'],b['old56']['summary']
 for k in ('countMAE','primaryMacroMAE','FP','FN'):leq('old56/'+k,x[k],y[k])
 for source,sub in x['perSource'].items():
  ref=y['perSource'][source]
  for sp,v in sub['perSpecies'].items():
   for k in ('countMAE','FP','FN'):leq('old56/'+source+'/'+sp+'/'+k,v[k],ref['perSpecies'][sp][k])
  for size,v in sub['sizeRecall'].items():
   if v['groundTruth']:leq('old56/'+source+'/'+size+'/FN',v['FN'],ref['sizeRecall'][size]['FN'])
 if b['clinical36'] is not None:
  for k in ('countMAE','FP','FN'):leq('clinical36/'+k,c['clinical36']['summary'][k],b['clinical36']['summary'][k])
 x,y=c['hardPositive']['summary'],b['hardPositive']['summary'];limit=y['primaryMacroMAE']*.95;rows.append({'gate':'hardPositive/primaryMacroMAE_5percentGain','pass':bool(y['primaryMacroMAE']>0 and x['primaryMacroMAE']<=limit+EPS),'actual':x['primaryMacroMAE'],'maximum':limit,'baselineMustBeNonzeroForRelativeGain':True})
 for k in ('FP','FN'):leq('hardPositive/'+k,x[k],y[k])
 for name,count in c['realEmptyCounts'].items():leq('realEmpty/'+name,count,0)
 for name,count in c['pureControlCounts'].items():leq('pureControl/'+name,count,0)
 failed=[r['gate'] for r in rows if not r['pass']];return {'status':'FAIL' if failed else 'PASS','failedGates':failed,'details':rows}

def main():
 selection=read(RUN/'selection.json');baseline=read(RUN/'baseline-validation.json');records=[r for r in read(RUN/'records.json') if r['split']=='val'];assert len(records)==102
 sets={'old56':[r for r in records if r['role']=='old_validation'],'clinical36':[r for r in records if r['role']=='clinical_validation'],'hardPositive':[r for r in records if r['role']=='hardcase_val' and r['bucket']=='hard_positive']};assert {k:len(v) for k,v in sets.items()}=={'old56':56,'clinical36':36,'hardPositive':4};empty=[r for r in records if r['role']=='hardcase_val' and r['bucket']=='empty_negative'];assert len(empty)==6
 assert selection['candidateCount']==len(selection['candidates'])==12;assert {(c['config']['epoch'],c['config']['alpha']) for c in selection['candidates']}=={(e,a) for e in [3,5,8] for a in [.05,.15,.35,1.]}
 calculations={};cacheaudit=[];configurations=[('baseline',None,baseline)]+[(f'epoch{r["config"]["epoch"]:03d}-alpha{r["config"]["alpha"]:g}',r['config'],r) for r in selection['candidates']]
 for tag,config,reported in configurations:
  path=RUN/'predictions'/(tag+'.json');cache=read(path);assert cache['recordsSHA256']==sha(RUN/'records.json') and cache['protocolSHA256']==sha(RUN/'protocol.json');assert cache['pipeline']=='automatic4tile' and cache['confidenceFloor']==.001 and cache['evaluationDevice']=='cpu';assert cache['evaluatorSHA256']==sha(ROOT/'ml/research-20261025-hardcases/evaluate_retention_savefix.py');assert set(cache['predictions'])=={r['id'] for r in records} and set(cache['pureControls'])=={'white','gray','black'}
  if config:assert cache['modelSHA256']==config['sha256']==sha(config['path'])
  else:assert cache['modelSHA256']==baseline['modelSHA256']=='91d70c3067f2fdb77d2ba2541e6105bf3caf730da7bbc32e5f4d2c9310b0b337'
  for pred in list(cache['predictions'].values())+list(cache['pureControls'].values()):
   boxes=np.asarray(pred['xyxy'],float).reshape(-1,4);conf=np.asarray(pred['confidence'],float);assert conf.shape==(len(boxes),) and len(boxes)<=1000 and np.isfinite(boxes).all() and np.isfinite(conf).all() and ((conf>=0)&(conf<=1)).all()
  result={name:score_positive(cache['predictions'],rr,physical=name=='hardPositive') for name,rr in sets.items()};result['realEmptyCounts']={r['id']:sum(c>=.3 for c in cache['predictions'][r['id']]['confidence']) for r in empty};result['pureControlCounts']={k:sum(c>=.3 for c in p['confidence']) for k,p in cache['pureControls'].items()};compare(result,reported,tag);calculations[tag]=result;cacheaudit.append({'tag':tag,'cacheSHA256':sha(path),'modelSHA256':cache['modelSHA256'],'all102PredictionMembershipAndFiniteArraysVerified':True,'allOperatingMetricsAndAPIndependentlyRecomputed':True})
  print('AUDIT',tag,flush=True)
 base=calculations['baseline'];audit=[];passing=[]
 for tag,config,reported in configurations[1:]:
  result=calculations[tag];g=gates(result,base);compare(g,reported['gates'],tag+'/gates');compare(reported,read(RUN/'scores'/(tag+'.json')),tag+'/scoreFile');audit.append({'epoch':config['epoch'],'alpha':config['alpha'],'hardPositiveMacroMAE':result['hardPositive']['summary']['primaryMacroMAE'],'old56MAE':result['old56']['summary']['countMAE'],'failedGateCount':len(g['failedGates']),'gateStatus':g['status'],'realEmptyCounts':result['realEmptyCounts'],'pureControlCounts':result['pureControlCounts'],'failedGates':g['failedGates']})
  if g['status']=='PASS':passing.append((tag,config,result))
 chosen=min(passing,key=lambda x:(x[2]['hardPositive']['summary']['primaryMacroMAE'],x[2]['hardPositive']['summary']['countMAE'],x[1]['alpha'],x[1]['epoch'])) if passing else None;assert len(passing)==selection['passingCandidateCount'];assert selection['selectedConfig']==(chosen[1] if chosen else None);assert selection['status']==('PASS_VALIDATION_PENDING_FIXED_OLD63_REGRESSION' if chosen else 'FAIL_NO_CANDIDATE_PASSES');assert not passing and selection['selectedResult'] is None
 bundle={'protocolSHA256':RUN/'protocol.json','recordsSHA256':RUN/'records.json','operatorSHA256':ROOT/'ml/research-20261025-hardcases/train_retention_savefix.py','evaluatorSHA256':ROOT/'ml/research-20261025-hardcases/evaluate_retention_savefix.py'}
 for k,p in bundle.items():assert selection[k]==sha(p)
 assert selection['baselineSHA256']==sha(RUN/'baseline-validation.json');assert selection['old63ReadOrInferred'] is False and selection['new36ReadOrInferred'] is False and selection['appModelChanged'] is False
 log=(RUN/'evaluate.log').read_text();assert not any(s in log for s in ['NMS time limit','RuntimeWarning','Traceback']);best=min(audit,key=lambda x:x['hardPositiveMacroMAE']);report_path=ROOT/'ml/research-20261025-hardcases/independent-retention-review.json';r=read(report_path);r['status']='PASS_FINAL_AUDIT_SAVEFIX_CANDIDATES_ALL_FAILED';r['finalCandidateAudit']={'status':'PASS_AUDIT_OF_FAIL_NO_CANDIDATE_PASSES','runDirectory':str(RUN.relative_to(ROOT)),'selectionSHA256':sha(RUN/'selection.json'),'candidateCount':12,'passingCandidateCount':0,'selectedConfig':None,**{k:sha(p) for k,p in bundle.items()},'candidateIndexSHA256':sha(RUN/'candidates/index.json'),'baselineSHA256':sha(RUN/'baseline-validation.json'),'all13CachesMetricsMatchingCountsSizeRecallAndAPRecomputed':True,'all12GateDetailsAndRankingRecomputed':True,'individualScoreFilesMatchSelection':True,'cacheAudit':cacheaudit,'candidates':audit,'bestHardPositiveDiagnosticCandidate':best,'CPUScoringLogNoWarningOrTraceback':True,'noOld63OrNew36InferencePerformed':True,'modelInferencePerformedByReviewer':False,'interpretation':'The experiment audit passed; everycandidatefailed. No model is approved and finalold63modelinference should be skipped.'};r['reviewedUTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());r['finalAuditVerifierSHA256']=sha(__file__);report_path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'finalCandidateAudit':r['finalCandidateAudit']['status'],'passingCandidates':len(passing),'selectionSHA256':sha(RUN/'selection.json'),'bestHardPositiveDiagnostic':best,'reviewSHA256':sha(report_path)},indent=2),flush=True)
if __name__=='__main__':main()
