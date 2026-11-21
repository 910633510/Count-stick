#!/usr/bin/env python3
"""Fixed CPU/4tile/conf0.3 validation of exactly12 pre-registered weight blends.

There is deliberately no old63 or new36 test reader here. The root performs the
single old63 final regression only after this selection freezes one candidate.
"""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import train_retention as train
from train_isolate_mixture import infer_image,score_predictions,ap_predictions,summary_rows
ROOT,OUT=train.ROOT,train.OUT
read,save,sha=train.read,train.save,train.sha
EPS=1e-9

def verify_frozen_code_binding():
 authorization=read(OUT/'training-authorization.json')
 assert authorization['operatorSHA256']==sha(train.__file__),'Operator changed after root-approved training authorization'
 assert authorization['evaluatorSHA256']==sha(__file__),'Evaluator changed after root-approved training authorization'
 assert authorization['localDependencySHA256']==train.local_dependency_hashes(),'Inference/scoring helper dependency changed after authorization'
 assert authorization['protocolSHA256']==sha(OUT/'protocol.json') and authorization['dataBindingSHA256']==sha(OUT/'data-binding.json')
 assert authorization['rootReceiptSHA256']==sha(authorization['rootReceiptPath']),'Root receipt changed after authorization was frozen'
 return authorization

def check_leq(name,value,reference,rows):
 rows.append({'gate':name,'pass':bool(value<=reference+EPS),'actual':value,'maximum':reference})

def old_nonregression(candidate,baseline,prefix='old56'):
 """A baseline-passable check with no object/count deterioration tolerance."""
 rows=[];c,b=candidate['summary'],baseline['summary']
 assert c['images']==b['images'] and c['groundTruthObjects']==b['groundTruthObjects']
 for key in ['countMAE','primaryMacroMAE','FP','FN']:check_leq(prefix+'/'+key,c[key],b[key],rows)
 assert set(c['perSource'])==set(b['perSource'])
 for source,cs in c['perSource'].items():
  bs=b['perSource'][source];assert set(cs['perSpecies'])==set(bs['perSpecies'])
  for species,sub in cs['perSpecies'].items():
   for key in ['countMAE','FP','FN']:check_leq(prefix+'/'+source+'/'+species+'/'+key,sub[key],bs['perSpecies'][species][key],rows)
  for size,sub in cs['sizeRecall'].items():
   ref=bs['sizeRecall'][size];assert sub['groundTruth']==ref['groundTruth']
   if sub['groundTruth']:check_leq(prefix+'/'+source+'/'+size+'/FN',sub['FN'],ref['FN'],rows)
 return rows

def gate_all(candidate,baseline):
 rows=old_nonregression(candidate['old56'],baseline['old56'])
 if baseline['clinical36'] is not None:
  assert candidate['clinical36'] is not None
  for key in ['countMAE','FP','FN']:check_leq('clinical36/'+key,candidate['clinical36']['summary'][key],baseline['clinical36']['summary'][key],rows)
 c,b=candidate['hardPositive']['summary'],baseline['hardPositive']['summary']
 maximum=b['primaryMacroMAE']*.95;rows.append({'gate':'hardPositive/primaryMacroMAE_5percentGain','pass':bool(b['primaryMacroMAE']>0 and c['primaryMacroMAE']<=maximum+EPS),'actual':c['primaryMacroMAE'],'maximum':maximum,'baselineMustBeNonzeroForRelativeGain':True})
 for key in ['FP','FN']:check_leq('hardPositive/'+key,c[key],b[key],rows)
 for image,count in candidate['realEmptyCounts'].items():check_leq('realEmpty/'+image,count,0,rows)
 for color,count in candidate['pureControlCounts'].items():check_leq('pureControl/'+color,count,0,rows)
 failed=[r['gate'] for r in rows if not r['pass']];return {'status':'PASS' if not failed else 'FAIL','failedGates':failed,'details':rows}

def physical_group_macro(report,records):
 """New hard-positive objective only; preserve old/clinical metrics unchanged."""
 by_id={r['id']:r for r in records};grouped=collections.defaultdict(list)
 for image in report['images']:
  r=by_id[image['id']];grouped[(r['source'],r['species'],str(r['groupId']))].append(image)
 groups=[{'source':source,'species':species,'groupId':group,**summary_rows(images)} for (source,species,group),images in sorted(grouped.items())]
 sources={}
 for source in sorted({g['source'] for g in groups}):
  spp={}
  for species in sorted({g['species'] for g in groups if g['source']==source}):
   selected=[g for g in groups if g['source']==source and g['species']==species];spp[species]=float(np.mean([g['countMAE'] for g in selected]));report['summary']['perSource'][source]['perSpecies'][species]['physicalGroupMacroCountMAE']=spp[species]
  sources[source]=float(np.mean(list(spp.values())));report['summary']['perSource'][source]['physicalGroupSpeciesMacroCountMAE']=sources[source]
 summary=report['summary'];summary['imageWeightedPrimaryMacroMAE']=summary['primaryMacroMAE'];summary['primaryMacroMAE']=float(np.mean(list(sources.values())));summary['physicalGroupSpeciesSourceMacroMAE']=summary['primaryMacroMAE'];summary['primaryMetricDefinition']='Mean image count error within physical group, then unweighted group mean within species, species mean within source, and source mean.';report['perPhysicalGroup']=groups;return report

def val_records():
 binding=read(OUT/'data-binding.json');assert binding['recordsSHA256']==sha(OUT/'records.json');assert binding['protocolSHA256']==sha(OUT/'protocol.json')
 records=[r for r in read(OUT/'records.json') if r['split']=='val'];assert any(r['role']=='old_validation' for r in records)
 assert any(r['role']=='hardcase_val' and r['bucket']=='hard_positive' for r in records)
 assert any(r['role']=='hardcase_val' and r['bucket']=='empty_negative' for r in records)
 return records

def predictions(weights,tag):
 records=val_records();path=OUT/'predictions'/(tag+'.json');model_sha=sha(weights);record_sha=sha(OUT/'records.json')
 if path.exists():
  cache=read(path);assert cache['modelSHA256']==model_sha and cache['recordsSHA256']==record_sha and cache['protocolSHA256']==sha(OUT/'protocol.json')
 else:cache={'modelSHA256':model_sha,'recordsSHA256':record_sha,'protocolSHA256':sha(OUT/'protocol.json'),'pipeline':'automatic4tile','confidenceFloor':.001,'evaluationDevice':'cpu','predictions':{},'pureControls':{},'evaluatorSHA256':sha(__file__),'inferenceSourceSHA256':sha(ROOT/'ml/colony/train_isolate_mixture.py')}
 if tag=='baseline' and not cache['predictions']:
  prior=ROOT/'output/research-20261025/mixed-training';old_records=read(prior/'records.json');old_cache=read(prior/'predictions/baseline-automatic4tile.json')
  assert old_cache['modelSHA256']==model_sha and old_cache['recordsSHA256']==sha(prior/'records.json');by_sha={r['sha256']:r['id'] for r in old_records if r['split']=='val'}
  for r in records:
   if r['sha256'] in by_sha:cache['predictions'][r['id']]=old_cache['predictions'][by_sha[r['sha256']]]
  cache['baselineCacheReuse']={'cacheSHA256':sha(prior/'predictions/baseline-automatic4tile.json'),'recordsSHA256':sha(prior/'records.json'),'reusedImages':len(cache['predictions']),'compatibility':'Same oldv2 SHA/CPU1024/rectFalse/conf.001/IoU.7/max1000; identical infer_image function.'};save(path,cache)
 pending=[r for r in records if r['id'] not in cache['predictions']];missing=[c for c in ['white','gray','black'] if c not in cache['pureControls']]
 if pending or missing:
  torch.set_num_threads(8);model=train.YOLO(str(weights))
  for r in pending:
   assert sha(r['preparedImage'])==r['sha256'];cache['predictions'][r['id']]=infer_image(model,r['preparedImage'],'automatic4tile');save(path,cache);print(tag,'val',len(cache['predictions']),'/',len(records),flush=True)
  for color in missing:
   value={'white':255,'gray':127,'black':0}[color];p=OUT/'pure-controls'/(color+'.png');p.parent.mkdir(exist_ok=True);Image.new('RGB',(1024,1024),(value,)*3).save(p);cache['pureControls'][color]=infer_image(model,p,'automatic4tile');save(path,cache)
 assert set(cache['predictions'])=={r['id'] for r in records};return cache

def score(cache):
 records=val_records();sets={'old56':[r for r in records if r['role']=='old_validation'],'clinical36':[r for r in records if r['role']=='clinical_validation'],'hardPositive':[r for r in records if r['role']=='hardcase_val' and r['bucket']=='hard_positive']};report={}
 for name,rows in sets.items():
  if not rows:report[name]=None;continue
  report[name]=score_predictions(cache['predictions'],rows,.3);report[name]['summary'].update(ap_predictions(cache['predictions'],rows));report[name]['physicalGroups']=len({str(r['groupId']) for r in rows})
  if name=='hardPositive':physical_group_macro(report[name],rows)
 report['realEmptyCounts']={r['id']:int(sum(c>=.3 for c in cache['predictions'][r['id']]['confidence'])) for r in records if r['role']=='hardcase_val' and r['bucket']=='empty_negative'}
 report['pureControlCounts']={color:int(sum(c>=.3 for c in entry['confidence'])) for color,entry in cache['pureControls'].items()};return report

def aspirational_old56():
 records=train.old_records('val');prior=train.ANCHOR.parent.parent/'evaluation';summaries={}
 for mode,file in [('whole','validation-predictions.json'),('automatic4tile','validation-tiled-predictions.json')]:
  pred=read(prior/file);mapped={r['id']:pred[Path(r['imagePath']).name] for r in records};summaries[mode]=score_predictions(mapped,records,.3)['summary']
 source='figshare22022540v3';pair=[s['perSource'][source] for s in summaries.values()];reference={'nonBinding':True,'reason':'Stronger per-subgroup whole/tile recall can exceed actual frozen pipeline; it is an aspirational diagnostic, not a nonregression acceptance baseline.','confidence':.3,'perSpecies':{},'perGTSizeBin':{},'wholeSummary':summaries['whole'],'tileSummary':summaries['automatic4tile']}
 for species in pair[0]['perSpecies']:
  gt=pair[0]['perSpecies'][species]['groundTruthObjects'];tp=max(s['perSpecies'][species]['TP'] for s in pair);reference['perSpecies'][species]={'GT':gt,'referenceTP':tp,'referenceRecall':tp/gt}
 for size in pair[0]['sizeRecall']:
  gt=pair[0]['sizeRecall'][size]['groundTruth'];tp=max(s['sizeRecall'][size]['TP'] for s in pair);reference['perGTSizeBin'][size]={'GT':gt,'referenceTP':tp,'referenceRecall':tp/gt if gt else None}
 return reference

def aspirational_comparison(old56,reference):
 sub=old56['summary']['perSource']['figshare22022540v3'];rows=[]
 for species,ref in reference['perSpecies'].items():rows.append({'group':species,'actualTP':sub['perSpecies'][species]['TP'],'aspirationalTP':ref['referenceTP'],'shortfall':max(0,ref['referenceTP']-sub['perSpecies'][species]['TP'])})
 for size,ref in reference['perGTSizeBin'].items():rows.append({'group':size,'actualTP':sub['sizeRecall'][size]['TP'],'aspirationalTP':ref['referenceTP'],'shortfall':max(0,ref['referenceTP']-sub['sizeRecall'][size]['TP'])})
 return {'acceptanceGate':False,'subgroups':rows}

def baseline():
 verify_frozen_code_binding()
 p=OUT/'baseline-validation.json'
 if p.exists():
  result=read(p);assert result['protocolSHA256']==sha(OUT/'protocol.json') and result['recordsSHA256']==sha(OUT/'records.json');return result
 cache=predictions(train.ANCHOR,'baseline');result=score(cache);check=old_nonregression(result['old56'],result['old56']);assert all(g['pass'] for g in check)
 reference=aspirational_old56();result.update({'alpha':0,'eligibleTrainedCandidate':False,'modelSHA256':train.ANCHOR_SHA,'protocolSHA256':sha(OUT/'protocol.json'),'recordsSHA256':sha(OUT/'records.json'),'evaluatorSHA256':sha(__file__),'old56BaselinePassableUnitCheck':check,'baselineCacheSHA256':sha(OUT/'predictions/baseline.json'),'aspirationalOld56Reference':reference,'aspirationalOld56Comparison':aspirational_comparison(result['old56'],reference)});save(p,result);return result

def select():
 verify_frozen_code_binding()
 assert (OUT/'training-complete.json').exists();assert not (OUT/'selection.json').exists(),'Selection already frozen; do not rescore using final regression outcomes'
 b=baseline();index=read(OUT/'candidates/index.json');assert index['protocolSHA256']==sha(OUT/'protocol.json')
 assert {(r['epoch'],r['alpha']) for r in index['candidates']}=={(e,a) for e in train.EPOCHS for a in train.ALPHAS} and len(index['candidates'])==12
 results=[]
 for r in index['candidates']:
  assert sha(r['path'])==r['sha256'];tag=f'epoch{r["epoch"]:03d}-alpha{r["alpha"]:g}';cache=predictions(r['path'],tag);candidate=score(cache);candidate.update({'config':r,'gates':gate_all(candidate,b),'aspirationalOld56Comparison':aspirational_comparison(candidate['old56'],b['aspirationalOld56Reference'])});results.append(candidate);save(OUT/'scores'/(tag+'.json'),candidate);print(tag,candidate['gates']['status'],len(candidate['gates']['failedGates']),flush=True)
 passing=[c for c in results if c['gates']['status']=='PASS'];selected=min(passing,key=lambda c:(c['hardPositive']['summary']['primaryMacroMAE'],c['hardPositive']['summary']['countMAE'],c['config']['alpha'],c['config']['epoch'])) if passing else None
 save(OUT/'selection.json',{'status':'PASS_VALIDATION_PENDING_FIXED_OLD63_REGRESSION' if selected else 'FAIL_NO_CANDIDATE_PASSES','frozenUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selectedConfig':None if selected is None else selected['config'],'selectedResult':selected,'candidateCount':12,'passingCandidateCount':len(passing),'candidates':results,'baselineSHA256':sha(OUT/'baseline-validation.json'),'protocolSHA256':sha(OUT/'protocol.json'),'operatorSHA256':sha(train.__file__),'evaluatorSHA256':sha(__file__),'recordsSHA256':sha(OUT/'records.json'),'old63ReadOrInferred':False,'new36ReadOrInferred':False,'appModelChanged':False})

def gates_smoke():
 # Reuse exposed old56 predictions solely to verify a real baseline passes itself.
 records=train.old_records('val');prior=train.ANCHOR.parent.parent/'evaluation';pred=read(prior/'validation-tiled-predictions.json');mapped={r['id']:pred[Path(r['imagePath']).name] for r in records};base=score_predictions(mapped,records,.3)
 rows=old_nonregression(base,base);assert all(r['pass'] for r in rows)
 import copy
 bad=copy.deepcopy(base);bad['summary']['FP']+=1;assert any(not g['pass'] for g in old_nonregression(bad,base))
 bad=copy.deepcopy(base);source=next(iter(bad['summary']['perSource']));species=next(iter(bad['summary']['perSource'][source]['perSpecies']));bad['summary']['perSource'][source]['perSpecies'][species]['FN']+=1;assert any(not g['pass'] for g in old_nonregression(bad,base))
 toy=[];toy_pred={}
 for source,group,views,error in [('a','one',1,0),('a','two',10,10),('b','three',1,0)]:
  for i in range(views):
   name=source+group+str(i);boxes=[[j*20,0,j*20+10,10] for j in range(max(1,error))];toy.append({'id':name,'source':source,'species':'same','groupId':group,'split':'val','width':1024,'height':1024,'xyxy':boxes});toy_pred[name]={'xyxy':[] if error else boxes,'confidence':[] if error else [.9]*len(boxes)}
 group_report=physical_group_macro(score_predictions(toy_pred,toy,.3),toy);assert group_report['summary']['primaryMacroMAE']==2.5;assert group_report['summary']['imageWeightedPrimaryMacroMAE']>4.5
 save(OUT/'gates-smoke.json',{'status':'PASS','old56BaselinePassesOwnNonregressionGates':True,'oneExtraFPRejected':True,'oneExtraSpeciesFNRejected':True,'hardPositivePhysicalGroupMacroUnequalViewCountsChecked':True,'old56BaselineSummary':base['summary'],'evaluatorSHA256':sha(__file__),'noTrainingOrCandidateInference':True,'old63ReadOrInferred':False,'new36ReadOrInferred':False});print('PASS baseline-passable zero-object-tolerance gates and group-balanced hardcase objective')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['gates-smoke','baseline','select']);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if args.phase=='gates-smoke':gates_smoke()
 elif args.phase=='baseline':baseline()
 else:select()
