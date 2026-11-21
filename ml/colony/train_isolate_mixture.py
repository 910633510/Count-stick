#!/usr/bin/env python3
"""Isolate-separated mixed-source fine-tune; no new-test content is accessed.

Actual WeightedRandomSampler is passed into the training InfiniteDataLoader.
Consumed filenames are audited in preprocess_batch, independently of prefetch.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/research-20261025/mixed-training'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import argparse,collections,hashlib,json,math,shutil,time
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler,Dataset
from PIL import Image
from ultralytics import YOLO,settings
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader,seed_worker
from mps_compat import install_cpu_assigner_for_mps
SEED=20261025
OLD=ROOT/'output/datasets/colony-figshare-22022540-v3/yolo-species-v1'
CLINICAL=ROOT/'output/datasets/colony-mixed-isolate-v1/train-val-manifest.json'
INIT=ROOT/'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/weights/best.pt'
EXPECTED_CLINICAL_SHA='3c55eb76f092f0bda29e14248da2d99be09b9189700cbd44e100a71fc1f097a5'
CHECKPOINTS=(3,5,8)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text())
def group(r):return r['source']+'::'+r['species']

def weights_for(records):
 counts=collections.Counter(group(r) for r in records);sources=sorted({r['source'] for r in records});species={s:{r['species'] for r in records if r['source']==s} for s in sources}
 weights=[1/(len(sources)*len(species[r['source']])*counts[group(r)]) for r in records]
 assert abs(sum(weights)-1)<1e-9
 return torch.tensor(weights,dtype=torch.double),counts,species

class AuditedWeightedSampler(WeightedRandomSampler):
 def __init__(self,records,num_samples,seed=SEED):
  self.records=records;weights,self.group_sizes,self.source_species=weights_for(records);g=torch.Generator().manual_seed(seed+71)
  super().__init__(weights,num_samples,replacement=True,generator=g);self.generated_rounds=[]
 def __iter__(self):
  indices=list(super().__iter__());self.generated_rounds.append(indices);return iter(indices)

class IndexDataset(Dataset):
 def __init__(self,n):self.n=n
 def __len__(self):return self.n
 def __getitem__(self,i):return i

def sampler_smoke():
 # Deliberately skewed source/species group sizes; observe DataLoader batches, not just weights.
 records=[]
 for source,species,n in [('a','one',1),('a','two',5),('b','three',2),('b','four',10)]:records.extend([{'source':source,'species':species} for _ in range(n)])
 sampler=AuditedWeightedSampler(records,100000);loader=InfiniteDataLoader(IndexDataset(len(records)),batch_size=64,sampler=sampler,shuffle=False,num_workers=0)
 observed=[]
 for batch in loader:observed.extend(int(i) for i in batch)
 loader.close();assert len(observed)==100000
 groups=collections.Counter(group(records[i]) for i in observed);source=collections.Counter(records[i]['source'] for i in observed)
 assert all(abs(n/100000-.25)<.01 for n in groups.values()) and all(abs(n/100000-.5)<.01 for n in source.values())
 assert observed==sampler.generated_rounds[0]
 again=AuditedWeightedSampler(records,100000);assert list(iter(again))==observed
 result={'status':'PASS_ACTUAL_DATALOADER_WEIGHTED_SAMPLER','samplesObserved':len(observed),'sourceCounts':dict(source),'groupCounts':dict(groups),'samplerClass':type(sampler).__name__,'loaderClass':type(loader).__name__,'sameSeedReproducible':True,'maxAbsoluteProbabilityTolerance':.01};save(OUT/'sampler-smoke.json',result);return result

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'data-binding.json').exists():
  binding=read(OUT/'data-binding.json');assert sha(CLINICAL)==binding['clinicalTrainValSHA256'];return binding
 assert sha(CLINICAL)==EXPECTED_CLINICAL_SHA
 clinical=read(CLINICAL);assert clinical['testContentIncluded'] is False
 assert set(r['split'] for r in clinical['images'])=={'train','val'}
 old=read(OLD/'manifest.json');by={r['file']:r for r in old['images']};rows=[]
 for split in ['train','val']:
  for n in old['splits'][split]:
   im=by[n];w,h=im['width'],im['height'];label=OLD/'labels'/split/(Path(n).stem+'.txt');xy=[]
   for line in label.read_text().splitlines():
    cls,x,y,bw,bh=map(float,line.split());assert cls==0;xy.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
   rows.append({'split':split,'source':'figshare22022540v3','species':f'sp{im["species"]:02d}','isolate':None,'originalName':n,'imagePath':str((OLD/'images'/split/n).relative_to(ROOT)),'labelPath':str(label.relative_to(ROOT)),'width':w,'height':h,'sha256':sha(OLD/'images'/split/n),'labelSHA256':sha(label),'xyxy':xy,'count':len(xy)})
 for r in clinical['images']:
  assert r['split'] in ['train','val'];rows.append(dict(r))
 assert collections.Counter(r['split'] for r in rows)=={'train':360,'val':92}
 clinical_train={ (r['species'],str(r['isolate'])) for r in clinical['images'] if r['split']=='train'}
 clinical_val={ (r['species'],str(r['isolate'])) for r in clinical['images'] if r['split']=='val'};assert not clinical_train&clinical_val
 output_rows=[]
 for i,r in enumerate(rows):
  src=ROOT/r['imagePath'];label=ROOT/r['labelPath'];assert sha(src)==r['sha256'] and sha(label)==r['labelSHA256'];assert r['count']==len(r['xyxy'])
  with Image.open(src) as im:assert im.size==(r['width'],r['height']) and im.getexif().get(274,1) in [1,None];im.verify()
  if src.suffix.lower() in ['.jpg','.jpeg']:
   with src.open('rb') as f:f.seek(-2,2);assert f.read()==b'\xff\xd9','Do not allow loader JPEG repair on source hardlinks'
  assert r['count']<=1000,'Dataset count exceeds frozen max_det; diagnose before training'
  lines=label.read_text().splitlines();assert len(lines)==r['count'] and len(set(lines))==len(lines)
  for line,box in zip(lines,r['xyxy']):
   cls,x,y,bw,bh=map(float,line.split());assert cls==0 and 0<x<1 and 0<y<1 and 0<bw<=1 and 0<bh<=1
   converted=np.array([(x-bw/2)*r['width'],(y-bh/2)*r['height'],(x+bw/2)*r['width'],(y+bh/2)*r['height']]);assert np.max(np.abs(converted-np.array(box)))<.01
  stem=f'{r["source"]}__{i:04d}__{src.stem}';dest=OUT/'dataset/images'/r['split']/(stem+src.suffix.lower());ldest=OUT/'dataset/labels'/r['split']/(stem+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);ldest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():os.link(src,dest)
  shutil.copyfile(label,ldest)
  output_rows.append({**r,'id':stem,'preparedImage':str(dest.resolve()),'preparedLabel':str(ldest.resolve())})
 for color,value in [('white',255),('gray',127),('black',0)]:
  p=OUT/'negative-controls'/(color+'.png');p.parent.mkdir(exist_ok=True);Image.new('RGB',(1024,1024),(value,value,value)).save(p)
 yaml=OUT/'dataset/data.yaml';yaml.write_text('path: '+json.dumps(str((OUT/'dataset').resolve()))+'\ntrain: images/train\nval: images/val\nnames:\n  0: colony\n')
 save(OUT/'records.json',output_rows)
 bind={'status':'PASS_TRAIN_VAL_ONLY_BINDING','clinicalTrainValSHA256':sha(CLINICAL),'sealedFullManifestSHA256':clinical['sealedFullManifestSHA256'],'oldManifestSHA256':sha(OLD/'manifest.json'),'recordsSHA256':sha(OUT/'records.json'),'counts':dict(collections.Counter(r['split'] for r in rows)),'sourceSpeciesCounts':{split:{s:len({r['species'] for r in rows if r['split']==split and r['source']==s}) for s in {r['source'] for r in rows}} for split in ['train','val']},'newTestContentRead':False};save(OUT/'data-binding.json',bind);return bind

def protocol():
 p=OUT/'protocol.json'
 if p.exists():return read(p)
 cfg={'epochs':8,'imgsz':1024,'batch':4,'device':'mps','workers':2,'seed':SEED,'deterministic':True,'amp':False,'optimizer':'AdamW','nbs':16,'lr0':.0001,'lrf':.1,'weight_decay':.0005,'warmup_epochs':1.,'warmup_bias_lr':.0001,'patience':100,'mosaic':0.,'close_mosaic':0,'mixup':0.,'copy_paste':0.,'degrees':0.,'translate':.08,'scale':.25,'fliplr':.5,'flipud':.5,'hsv_h':.01,'hsv_s':.2,'hsv_v':.2,'cache':False,'max_det':1000,'plots':False,'save':True,'save_period':-1,'verbose':False}
 r={'registeredBeforeTrainingUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'methodMotivation':'Data/appearance diversity and source/species-balanced exposure; not a literal Cellpose2 reproduction.','initialModel':str(INIT),'initialModelSHA256':sha(INIT),'dataRole':'360train(247old+113clinical),92val(56old+36clinical); newtest remains inaccessible to this operator.','sampler':'Actual WeightedRandomSampler: source uniform, species uniform within source, then image uniform within group; replacement=True,360draws/epoch,continuous seeded generator.','samplingCaveat':'Equal image/species exposure is not equal per-object loss; actual box exposures are recorded too.','trainingConfig':cfg,'checkpointsForSelection':[3,5,8],'selectionGrid':{'pipeline':['whole','automatic4tile'],'confidence':[.2,.3,.4],'nmsIoU':.7},'baseline':'Initialv2 weights on same mixed val,6pipeline/conf configurations; select smallest equal-source mean of within-source species-macro countMAE.','primaryMetric':'Mean across sources of mean per-species countMAE; sources have equal weight regardless19versus6species.','frozenGates':{'primaryMacroMAERelativeMax':.95,'sourceMacroMAERelativeMax':1.05,'sourceImageMAETolerance':'max(0.5,5%baseline)','sourcePrecisionRecallMaxDrop':.01,'sourceF1MustNotDrop':True,'sourceFPAdditionalBudget':'max(2,ceil(1%sourceGT))','sourceSpeciesRecallMaxDrop':.02,'sourceSpeciesPrecisionMaxDrop':.02,'sourceSpeciesRecallTPRoundingAllowance':1,'sourceSizeRecallMaxDrop':.02,'sourceSizeRecallTPRoundingAllowance':1,'protectedRecallReference':'Stronger whole/tile baseline at the selected baseline confidence, per source/species/size.','allNegativeControlsCountsMustBeZero':True},'negativeControls':{'images':['white','gray','black'],'RGBValues':[255,127,0],'shape':[1024,1024],'treatment':'Normal model and normalwhole/tile inference; no uniform-image special case and no threshold adjustment.'},'rankingAmongPass':'Minimum primaryMacroMAE, then microcountMAE, then fewer passes, then earlier checkpoint; if none pass, selectedConfig=null and FAIL is preserved.','newTestTrainingInferenceOrTuningAllowed':False,'oldTestsUsedForSelection':False,'appOrASCChangesAllowed':False}
 save(p,r);return r

class MixtureTrainer(DetectionTrainer):
 def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
  if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
  assert rank in [-1,0] and self.world_size<=1,'Single-processMPS experiment only'
  dataset=self.build_dataset(dataset_path,mode,batch_size);assert not dataset.rect
  all_rows=read(OUT/'records.json');by={str(Path(r['preparedImage']).resolve()):r for r in all_rows if r['split']=='train'}
  records=[by[str(Path(p).resolve())] for p in dataset.im_files];assert len(records)==360
  self._image_to_index={str(Path(p).resolve()):i for i,p in enumerate(dataset.im_files)};self._records=records;self._observed={};self._reported=set()
  sampler=AuditedWeightedSampler(records,len(records));self._audited_sampler=sampler
  gen=torch.Generator().manual_seed(SEED+313)
  loader=InfiniteDataLoader(dataset,batch_size=batch_size,shuffle=False,sampler=sampler,num_workers=2,prefetch_factor=4,pin_memory=False,collate_fn=dataset.collate_fn,worker_init_fn=seed_worker,generator=gen,drop_last=False)
  assert loader.sampler is sampler and isinstance(loader.sampler,WeightedRandomSampler)
  self._weighted_loader=loader
  save(OUT/'dataloader-binding.json',{'status':'ACTUAL_TRAINING_DATALOADER_BOUND','loaderClass':type(loader).__name__,'samplerClass':type(loader.sampler).__name__,'numSamples':len(sampler),'batches':len(loader),'sourceProbability':{s:float(sum(sampler.weights[i] for i,r in enumerate(records) if r['source']==s)) for s in sampler.source_species},'speciesPerSource':{s:len(v) for s,v in sampler.source_species.items()},'perImageWeights':{r['id']:float(sampler.weights[i]) for i,r in enumerate(records)},'orderedImageIds':[r['id'] for r in records],'newTestAccess':False})
  return loader
 def preprocess_batch(self,batch):
  # This records consumed training filenames, not prefetched sampler indices.
  epoch=int(self.epoch);ids=[self._image_to_index[str(Path(p).resolve())] for p in batch['im_file']];self._observed.setdefault(epoch,[]).extend(ids)
  return super().preprocess_batch(batch)

def train():
 cfg=protocol()['trainingConfig'];binding=prepare();assert read(OUT/'sampler-smoke.json')['status'].startswith('PASS');assert torch.backends.mps.is_available();install_cpu_assigner_for_mps();torch.set_num_threads(8)
 settings.update({'sync':False,'wandb':False,'mlflow':False,'comet':False,'clearml':False,'neptune':False,'tensorboard':False})
 if (OUT/'run').exists():raise RuntimeError('Existing experiment run; preserve it before diagnosing/resuming')
 model=YOLO(str(INIT))
 def save_checkpoint(t):
  epoch=t.epoch+1
  if epoch in CHECKPOINTS:
   dest=OUT/'checkpoints'/f'epoch{epoch:03d}.pt';dest.parent.mkdir(exist_ok=True);shutil.copy2(t.last,dest);save(OUT/'checkpoints'/f'epoch{epoch:03d}.json',{'epoch':epoch,'sha256':sha(dest),'dataBindingSHA256':sha(OUT/'data-binding.json'),'protocolSHA256':sha(OUT/'protocol.json')})
 def audit_epoch(t):
  e=t.epoch
  if e in t._reported:return
  # Ultralytics final_eval invokes on_fit_epoch_end after advancing epoch to epochs.
  # It is a terminal validation callback, not an additional training epoch.
  if e==t.epochs and set(t._reported)==set(range(t.epochs)) and not t._observed.get(e):return
  observed=t._observed.get(e,[]);assert len(observed)==360,(e,len(observed))
  generated=[i for block in t._audited_sampler.generated_rounds for i in block]
  all_observed=[i for ep in sorted(t._observed) for i in t._observed[ep]];assert all_observed==generated[:len(all_observed)],'Consumed sampler sequence mismatch; do not claim weighting after recovery/reset'
  sc=collections.Counter();gc=collections.Counter();bc=collections.Counter()
  for i in observed:r=t._records[i];sc[r['source']]+=1;gc[group(r)]+=1;bc[group(r)]+=r['count']
  report={'epoch':e+1,'consumedImages':len(observed),'sourceImageCounts':dict(sc),'sourceSpeciesImageCounts':dict(gc),'sourceSpeciesBoxExposures':dict(bc),'actualConsumedImageIds':[t._records[i]['id'] for i in observed],'generatedPrefixMatched':True,'prefetchNote':'Continuous seeded generator; generated rounds may run ahead. Only consumed filenames are credited to this epoch.','metrics':{k:float(v) for k,v in t.metrics.items()}};save(OUT/'sampling'/f'epoch{e+1:03d}.json',report);save(OUT/'progress.json',{'stage':'training','epoch':e+1,'epochs':8,'sourceCounts':dict(sc),'metrics':report['metrics']});t._reported.add(e)
 model.add_callback('on_model_save',save_checkpoint);model.add_callback('on_fit_epoch_end',audit_epoch)
 started=time.time();result=model.train(trainer=MixtureTrainer,data=str(OUT/'dataset/data.yaml'),project=str(OUT),name='run',exist_ok=False,**cfg)
 for epoch in CHECKPOINTS:assert (OUT/'checkpoints'/f'epoch{epoch:03d}.pt').exists()
 save(OUT/'training-complete.json',{'status':'COMPLETE_8_EPOCHS_WITH_ACTUAL_WEIGHTED_DATALOADER_AUDIT','epochs':8,'seconds':time.time()-started,'sourceSHA256':sha(__file__),'dataBindingSHA256':sha(OUT/'data-binding.json'),'protocolSHA256':sha(OUT/'protocol.json'),'checkpointEpochs':list(CHECKPOINTS),'newTestAccessed':False,'modelChangesInApp':False});print('TRAINING COMPLETE',flush=True)

def evaluation_definitions():
 """Bind details left implicit in the protocol, before any model candidate is scored."""
 p=OUT/'evaluation-definitions.json'
 if p.exists():return read(p)
 r={'registeredUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocolSHA256':sha(OUT/'protocol.json'),'matching':'IoU>=0.5 maximum-cardinality Hungarian matching, with IoU tie breaking.','size':'Original GT min-side *1024/max(width,height): <16, [16,32), >=32.','baselineTie':'primaryMacroMAE, then overall countMAE, then whole before automatic4tile, then higher confidence.','candidateTie':'primaryMacroMAE, then overall countMAE, then whole before automatic4tile, then earlier epoch, then higher confidence.','precisionReference':'Selected baseline configuration for every source and source/species precision.','recallReference':'Per source/species and source/size use max TP across whole/tile at selected baseline confidence. Source overall P/R/F1 and FP compare selected baseline.','roundingAllowance':'Candidate subgroup TP >= referenceTP - 0.02*GT - 1 (skip size group only if GT=0).','primaryMeaning':'For each source, unweighted mean of per-species image countMAE; then unweighted mean of the two source values.','negativeControls':'Each normal whole/tile configuration must produce exactly zero detections at its operating confidence for white/gray/black. They do not enter positive-image countMAE or AP.','AP':'Full saved conf>=0.001 detections, Ultralytics match_predictions and ap_per_class; AP does not depend on candidate operating confidence.','evaluationDevice':'CPU; train MPS validation logs are diagnostics, never checkpoint selection.','changesToFrozenGates':None,'newTestAccess':False}
 save(p,r);return r

def infer_image(model,path,mode):
 from torchvision.ops import nms
 kwargs=dict(imgsz=1024,rect=False,conf=.001,iou=.7,max_det=1000,device='cpu',verbose=False)
 if mode=='whole':
  pred=model.predict(str(path),**kwargs)[0];return {'xyxy':pred.boxes.xyxy.cpu().numpy().round(4).tolist(),'confidence':pred.boxes.conf.cpu().numpy().round(7).tolist()}
 assert mode=='automatic4tile'
 with Image.open(path) as im:image=im.convert('RGB');w,h=image.size
 tw,th=int(math.ceil(w*.6)),int(math.ceil(h*.6));parts=[];tiles=[]
 for row in range(2):
  for col in range(2):
   x0=0 if col==0 else w-tw;y0=0 if row==0 else h-th;tiles.append((x0,y0,col,row));parts.append(image.crop((x0,y0,x0+tw,y0+th)))
 boxes=[];scores=[]
 for pred,(x0,y0,col,row) in zip(model.predict(parts,stream=True,**kwargs),tiles):
  bb=pred.boxes.xyxy.cpu().numpy().copy();ss=pred.boxes.conf.cpu().numpy();bb[:,[0,2]]+=x0;bb[:,[1,3]]+=y0;center=(bb[:,:2]+bb[:,2:])/2
  keep=((center[:,0]<w/2) if col==0 else (center[:,0]>=w/2)) & ((center[:,1]<h/2) if row==0 else (center[:,1]>=h/2));boxes.extend(bb[keep].tolist());scores.extend(ss[keep].tolist())
 if not boxes:return {'xyxy':[],'confidence':[]}
 bb=torch.tensor(boxes,dtype=torch.float32);ss=torch.tensor(scores);keep=nms(bb,ss,.7)[:1000];return {'xyxy':bb[keep].tolist(),'confidence':ss[keep].tolist()}

def cached_predictions(weights,tag,mode):
 records=[r for r in read(OUT/'records.json') if r['split']=='val'];path=OUT/'predictions'/f'{tag}-{mode}.json';model_sha=sha(weights);record_sha=sha(OUT/'records.json')
 if path.exists():
  cache=read(path);assert cache['modelSHA256']==model_sha and cache['recordsSHA256']==record_sha and cache['pipeline']==mode
 else:cache={'modelSHA256':model_sha,'recordsSHA256':record_sha,'pipeline':mode,'confidenceFloor':.001,'evaluationDevice':'cpu','predictions':{},'negativeControls':{}}
 if tag=='baseline' and not cache['predictions']:
  # Reuse only the old validation cache produced by exactly the frozen initial weights.
  prior=INIT.parent.parent/'evaluation';selection=read(prior/'selection-before-test.json');assert selection['modelSHA256']==model_sha
  oldpred=read(prior/('validation-predictions.json' if mode=='whole' else 'validation-tiled-predictions.json'))
  for r in records:
   if r['source']=='figshare22022540v3':cache['predictions'][r['id']]=oldpred[r['originalName']]
  cache['oldValidationCacheReuse']={'images':56,'selectionSHA256':sha(prior/'selection-before-test.json'),'sourceCacheSHA256':sha(prior/('validation-predictions.json' if mode=='whole' else 'validation-tiled-predictions.json'))}
  save(path,cache)
 pending=[r for r in records if r['id'] not in cache['predictions']]
 missing_neg=[name for name in ['white','gray','black'] if name not in cache['negativeControls']]
 if pending or missing_neg:
  model=YOLO(str(weights));torch.set_num_threads(8)
  for r in pending:
   cache['predictions'][r['id']]=infer_image(model,r['preparedImage'],mode);save(path,cache)
   print(tag,mode,'validation',len(cache['predictions']),'/',len(records),flush=True)
  for name in missing_neg:cache['negativeControls'][name]=infer_image(model,OUT/'negative-controls'/(name+'.png'),mode);save(path,cache)
 assert set(cache['predictions'])=={r['id'] for r in records};return cache

def summary_rows(rows):
 tp=sum(r['TP'] for r in rows);fp=sum(r['FP'] for r in rows);fn=sum(r['FN'] for r in rows);errors=np.array([r['signedError'] for r in rows]);out={'images':len(rows),'groundTruthObjects':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'F1':2*tp/max(1,2*tp+fp+fn),'countMAE':float(abs(errors).mean()),'countRMSE':float(np.sqrt((errors**2).mean())),'countBias':float(errors.mean()),'sizeRecall':{}}
 for name in ['under16px','16to32px','over32px']:
  gt=sum(r['sizeRecall'][name]['groundTruth'] for r in rows);stp=sum(r['sizeRecall'][name]['TP'] for r in rows);out['sizeRecall'][name]={'groundTruth':gt,'TP':stp,'FN':gt-stp,'recall':stp/gt if gt else None}
 return out

def score_predictions(predictions,records,confidence):
 from evaluate import iou_matrix
 from scipy.optimize import linear_sum_assignment
 rows=[]
 for r in records:
  assert r['split']=='val';entry=predictions[r['id']];truth=np.asarray(r['xyxy'],dtype=float).reshape(-1,4);bb=np.asarray(entry['xyxy'],dtype=float).reshape(-1,4)[np.asarray(entry['confidence'])>=confidence];overlaps=iou_matrix(truth,bb);matched=set()
  if overlaps.size:
   rr,cc=linear_sum_assignment(-((overlaps>=.5).astype(float)*1001+overlaps));matched={int(i) for i,j in zip(rr,cc) if overlaps[i,j]>=.5}
  sizes={k:{'groundTruth':0,'TP':0} for k in ['under16px','16to32px','over32px']}
  for i,b in enumerate(truth):
   side=min(b[2]-b[0],b[3]-b[1])*1024/max(r['width'],r['height']);label='under16px' if side<16 else '16to32px' if side<32 else 'over32px';sizes[label]['groundTruth']+=1;sizes[label]['TP']+=int(i in matched)
  tp=len(matched);rows.append({'id':r['id'],'source':r['source'],'species':r['species'],'truth':len(truth),'predicted':len(bb),'signedError':len(bb)-len(truth),'TP':tp,'FP':len(bb)-tp,'FN':len(truth)-tp,'sizeRecall':sizes})
 summary=summary_rows(rows);summary['perSource']={}
 for source in sorted({r['source'] for r in rows}):
  selected=[r for r in rows if r['source']==source];sub=summary_rows(selected);sub['perSpecies']={species:summary_rows([r for r in selected if r['species']==species]) for species in sorted({r['species'] for r in selected})}
  for metric in ['countMAE','precision','recall','F1']:sub['speciesMacro'+metric[0].upper()+metric[1:]]=float(np.mean([x[metric] for x in sub['perSpecies'].values()]))
  summary['perSource'][source]=sub
 for metric in ['countMAE','precision','recall','F1']:summary['equalSourceSpeciesMacro'+metric[0].upper()+metric[1:]]=float(np.mean([s['speciesMacro'+metric[0].upper()+metric[1:]] for s in summary['perSource'].values()]))
 summary['primaryMacroMAE']=summary['equalSourceSpeciesMacroCountMAE'];summary['confidence']=confidence;summary['matchIoU']=.5;return {'summary':summary,'images':rows}

def ap_predictions(predictions,records):
 import types
 from evaluate import iou_matrix
 from ultralytics.engine.validator import BaseValidator
 from ultralytics.utils.metrics import ap_per_class
 tps=[];scores=[];targets=[];matcher=types.SimpleNamespace(iouv=torch.linspace(.5,.95,10))
 for r in records:
  p=predictions[r['id']];truth=np.asarray(r['xyxy'],dtype=float).reshape(-1,4);b=np.asarray(p['xyxy'],dtype=float).reshape(-1,4);matched=BaseValidator.match_predictions(matcher,torch.zeros(len(b)),torch.zeros(len(truth)),torch.from_numpy(iou_matrix(truth,b)).float());tps.append(matched.numpy());scores.extend(p['confidence']);targets.extend([0]*len(truth))
 if not scores:return {'AP50':0.,'AP50_95':0.,'AP75':0.}
 answer=ap_per_class(np.concatenate(tps),np.asarray(scores),np.zeros(len(scores)),np.asarray(targets),plot=False,names={0:'colony'});ap=answer[5];return {'AP50':float(ap[:,0].mean()),'AP50_95':float(ap.mean()),'AP75':float(ap[:,5].mean())}

def score_cache(cache,epoch,mode,confidence):
 records=[r for r in read(OUT/'records.json') if r['split']=='val'];result=score_predictions(cache['predictions'],records,confidence);result.update({'epoch':epoch,'pipeline':mode,'confidence':confidence,'modelSHA256':cache['modelSHA256'],'negativeCounts':{name:int(sum(c>=confidence for c in p['confidence'])) for name,p in cache['negativeControls'].items()}});return result

def gates(candidate,baseline,protected):
 c,b=candidate['summary'],baseline['summary'];fail=[];details=[]
 def gate(name,passed,actual,limit):
  details.append({'gate':name,'pass':bool(passed),'actual':actual,'limit':limit})
  if not passed:fail.append(name)
 gate('primaryMacroMAE_5percentImprovement',c['primaryMacroMAE']<=b['primaryMacroMAE']*.95,c['primaryMacroMAE'],b['primaryMacroMAE']*.95)
 for source,cs in c['perSource'].items():
  bs=b['perSource'][source];prefix=source+'/'
  gate(prefix+'speciesMacroCountMAE',cs['speciesMacroCountMAE']<=bs['speciesMacroCountMAE']*1.05,cs['speciesMacroCountMAE'],bs['speciesMacroCountMAE']*1.05)
  limit=bs['countMAE']+max(.5,bs['countMAE']*.05);gate(prefix+'imageCountMAE',cs['countMAE']<=limit,cs['countMAE'],limit)
  for metric in ['precision','recall']:gate(prefix+metric,cs[metric]>=bs[metric]-.01,cs[metric],bs[metric]-.01)
  gate(prefix+'F1',cs['F1']>=bs['F1'],cs['F1'],bs['F1'])
  limit=bs['FP']+max(2,math.ceil(bs['groundTruthObjects']*.01));gate(prefix+'FP',cs['FP']<=limit,cs['FP'],limit)
  for species,ss in cs['perSpecies'].items():
   ref=bs['perSpecies'][species];strong=max(p['summary']['perSource'][source]['perSpecies'][species]['TP'] for p in protected);limit=strong-.02*ss['groundTruthObjects']-1;gate(prefix+species+'/recallTP',ss['TP']>=limit,ss['TP'],limit);gate(prefix+species+'/precision',ss['precision']>=ref['precision']-.02,ss['precision'],ref['precision']-.02)
  for size,ss in cs['sizeRecall'].items():
   if not ss['groundTruth']:continue
   strong=max(p['summary']['perSource'][source]['sizeRecall'][size]['TP'] for p in protected);limit=strong-.02*ss['groundTruth']-1;gate(prefix+size+'/recallTP',ss['TP']>=limit,ss['TP'],limit)
 for name,count in candidate['negativeCounts'].items():gate('negative/'+name,count==0,count,0)
 return {'status':'PASS' if not fail else 'FAIL','failures':fail,'details':details}

def baseline():
 evaluation_definitions();result_path=OUT/'baseline-binding.json'
 if result_path.exists():return read(result_path)
 candidates=[]
 for mode in ['whole','automatic4tile']:
  cache=cached_predictions(INIT,'baseline',mode);ap=ap_predictions(cache['predictions'],[r for r in read(OUT/'records.json') if r['split']=='val'])
  for conf in [.2,.3,.4]:
   score=score_cache(cache,0,mode,conf);score['summary'].update(ap);candidates.append(score)
 chosen=min(candidates,key=lambda c:(c['summary']['primaryMacroMAE'],c['summary']['countMAE'],c['pipeline']!='whole',-c['confidence']));protected=[c for c in candidates if c['confidence']==chosen['confidence']]
 result={'boundBeforeCandidateScoringUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocolSHA256':sha(OUT/'protocol.json'),'definitionsSHA256':sha(OUT/'evaluation-definitions.json'),'candidates':candidates,'selectedBaseline':chosen,'protectedRecallReferences':protected,'newTestAccessed':False};save(result_path,result);print('BASELINE FROZEN',chosen['pipeline'],chosen['confidence'],chosen['summary']['primaryMacroMAE'],flush=True);return result

def evaluate():
 assert (OUT/'training-complete.json').exists(),'Evaluate checkpoints only after bounded training completes'
 baseline_result=baseline();base=baseline_result['selectedBaseline'];protected=baseline_result['protectedRecallReferences'];candidates=[]
 for epoch in CHECKPOINTS:
  weights=OUT/'checkpoints'/f'epoch{epoch:03d}.pt'
  for mode in ['whole','automatic4tile']:
   cache=cached_predictions(weights,f'epoch{epoch:03d}',mode);ap=ap_predictions(cache['predictions'],[r for r in read(OUT/'records.json') if r['split']=='val'])
   for conf in [.2,.3,.4]:
    result=score_cache(cache,epoch,mode,conf);result['summary'].update(ap);result['gates']=gates(result,base,protected);candidates.append(result);save(OUT/'scores'/f'epoch{epoch:03d}-{mode}-conf{conf}.json',result);print('CANDIDATE',epoch,mode,conf,result['gates']['status'],result['summary']['primaryMacroMAE'],'failures',len(result['gates']['failures']),flush=True)
   save(OUT/'evaluation-progress.json',{'checkpointEpoch':epoch,'pipeline':mode,'completedCandidates':len(candidates),'totalCandidates':18})
 passing=[c for c in candidates if c['gates']['status']=='PASS'];selected=min(passing,key=lambda c:(c['summary']['primaryMacroMAE'],c['summary']['countMAE'],c['pipeline']!='whole',c['epoch'],-c['confidence'])) if passing else None
 chosen=None if selected is None else {'epoch':selected['epoch'],'modelPath':str(OUT/'checkpoints'/f'epoch{selected["epoch"]:03d}.pt'),'modelSHA256':selected['modelSHA256'],'pipeline':selected['pipeline'],'confidence':selected['confidence'],'nmsIoU':.7,'imgsz':1024}
 save(OUT/'selection.json',{'status':'PASS_DEVELOPMENT_GATES_PENDING_EXTERNAL_TEST' if selected else 'FAIL_NO_CANDIDATE_PASSES','frozenUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selectedConfig':chosen,'baselineBindingSHA256':sha(OUT/'baseline-binding.json'),'baselineCodeBindingSHA256':sha(OUT/'baseline-binding.provenance.json'),'operatorSHA256':sha(__file__),'trainingCompletionSHA256':sha(OUT/'training-complete.json'),'trainingStartSnapshotSHA256':sha(OUT/'operator-training-start.py'),'protocolSHA256':sha(OUT/'protocol.json'),'definitionsSHA256':sha(OUT/'evaluation-definitions.json'),'samplingFieldSemanticsSHA256':sha(OUT/'sampling-field-semantics.json'),'candidateCount':len(candidates),'passingCandidateCount':len(passing),'candidates':candidates,'selectedCandidate':selected,'newTestAccessed':False,'appModelReplaced':False});print('FINAL SELECTION',chosen,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','baseline','evaluate']);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if a.phase=='prepare':protocol();print(prepare());print(sampler_smoke())
 elif a.phase=='train':train()
 elif a.phase=='baseline':baseline()
 else:evaluate()
