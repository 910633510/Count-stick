#!/usr/bin/env python3
"""Bounded hardcase adaptation with an old-training-only frozen teacher.

No optimizer step is allowed without a hash-bound root data/QA/protocol receipt.
Old test data never enter this operator. Old56/hardval selection and a subsequent
single frozen-candidate old63 regression are separate from training.
"""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/research-20261025-hardcases/retention-savefix-v2'
sys.path.insert(0,str(ROOT/'ml/colony'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import argparse,ast,collections,copy,hashlib,json,math,re,shutil,time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler,Dataset
from PIL import Image
from ultralytics import YOLO,settings
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader,seed_worker
from ultralytics.utils.loss import v8DetectionLoss
from mps_compat import install_cpu_assigner_for_mps
from train_isolate_mixture import sha,save,read

SEED=20261026
OLD=ROOT/'output/datasets/colony-figshare-22022540-v3/yolo-species-v1'
CLINICAL=ROOT/'output/datasets/colony-mixed-isolate-v1/train-val-manifest.json'
ANCHOR=ROOT/'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/weights/best.pt'
ANCHOR_SHA='91d70c3067f2fdb77d2ba2541e6105bf3caf730da7bbc32e5f4d2c9310b0b337'
CLINICAL_SOURCE='zenodo18256864-clinical-originals'
BUCKET_MASS={'old_anchor':.65,'hard_positive':.25,'empty_negative':.10}
EPOCHS=(3,5,8)
ALPHAS=(.05,.15,.35,1.)

def local_dependency_hashes():
 return {name:sha(ROOT/name) for name in ['ml/colony/train_isolate_mixture.py','ml/colony/evaluate.py','ml/colony/mps_compat.py']}

def make_protocol():
 p=OUT/'protocol.json'
 if p.exists():return read(p)
 config={'epochs':8,'imgsz':1024,'batch':4,'device':'mps','workers':2,'seed':SEED,'deterministic':True,'amp':False,'optimizer':'AdamW','nbs':16,'lr0':.0001,'lrf':.1,'weight_decay':.0005,'warmup_epochs':1.,'warmup_bias_lr':.0001,'freeze':list(range(11)),'patience':100,'mosaic':0.,'close_mosaic':0,'mixup':0.,'copy_paste':0.,'degrees':0.,'translate':.08,'scale':.25,'fliplr':.5,'flipud':.5,'hsv_h':.01,'hsv_s':.2,'hsv_v':.2,'cache':False,'max_det':1000,'plots':False,'save':True,'save_period':-1,'verbose':False,'val':False}
 protocol={'status':'METHOD_FIXED_DATA_AND_ROOT_RECEIPT_REQUIRED','registeredUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'anchorSHA256':ANCHOR_SHA,'trainingConfig':config,'bucketSampling':BUCKET_MASS,'sampler':'Actual weighted replacement sampler;360consumed images per epoch; bucket then source then species then physical group then image balance. Continuous seeded generator; audit filenames after prefetch.','clinicalPositiveWithinBucketMax':.2,'clinicalSource':CLINICAL_SOURCE,'frozen':'Backbone layers0..10, all BatchNorm affine parameters and running buffers, DFL fixed projection; EMA frozen entries reset to anchor before save.','distillation':{'images':'ONLY old_anchor train rows; teacher sees the exact already-augmented student tensor. No val/test or new hardcase/empty labels in the teacher loss.','teacher':'Frozen oldv2 eval/no_grad; no new model weights saved inside checkpoint.','classKLWeight':5.,'boxDFLKLWeight':1.,'positiveClassAnchorMinTeacherProbability':.1,'boxAnchorMinTeacherProbability':.25,'backgroundAnchorMaxTeacherProbability':.05,'backgroundTopStudentAnchors':256,'classNormalization':'Per-image mean teacher-positive KL and top-student teacher-background KL, then equal mean of the two; old images averaged.','boxNormalization':'Confidence-weighted DFL KL across teacher-positive anchors; mean over four sides and old images.','KDScale':'Add each weighted KD scalar times complete batch size, matching base criterion batch scaling.'},'frameworkMPSValidation':'Disabled explicitly; fixed checkpoint selection only through later CPU evaluator; no automatic best-model selection.','checkpoints':list(EPOCHS),'weightInterpolationAlphas':list(ALPHAS),'candidateCount':12,'frozenOperationalPipeline':{'pipeline':'automatic4tile','confidence':.3,'nmsIoU':.7,'imgsz':1024,'maxDetections':1000},'old56SelectionGates':{'overall':['countMAE<=baseline','speciesMacroCountMAE<=baseline','FP<=baseline','FN<=baseline'],'eachSpecies':['countMAE<=baseline','FP<=baseline','FN<=baseline'],'eachGTSizeBin':['FN<=baseline'],'tolerance':'0 objects / 0 MAE deterioration;1e-9 comparison epsilon only','baseline':'Actual oldv2 automatic4tile/conf0.3; alpha0 must pass these gates but is never an eligible trained candidate.','aspirationalReference':'Prior stronger-of-whole/tile metrics reported separately, not mislabeled as baseline-passable nonregression.'},'clinicalVal36AdditionalGate':'Against oldv2 same pipeline/confidence: aggregate countMAE/FP/FN do not increase. These images are validation only.','hardPositiveValGates':['physical-group-balanced then species-balanced then source-balanced macro countMAE improves at least5%','FP does not increase','FN does not increase'],'negativeValGate':'Every real empty validation image and each pure white/gray/black control has exactly zero predictions.','selectionOrder':'Among all-gate PASS: minimum hard-positive macroMAE, then hard-positive imageMAE, then smaller alpha, then earlier epoch.','old63FinalRegression':'One unique old56+hardval-selected frozen candidate only, after selection. If it fails, reject round; never choose another alpha/checkpoint using old63. Not run by training phases.','new36TestAccessAllowed':False,'trainingDataGate':'Require frozen train-val-only annotation manifest and independent QA PASS bound to root receipt and this protocol SHA before any optimizer step.','appChangesAllowed':False,'furtherRoundsAllowed':False}
 protocol['clinicalCapMeaning']='20% within hard_positive /5% total is a sampling-probability cap, not a deterministic per-epoch realized count quota.'
 protocol['trainingDataGate']+=' Root receipt also binds operatorSHA256.'
 save(p,protocol);return protocol

def record_group(r):return r['bucket']+'::'+r['source']+'::'+r['species']

def sampling_weights(records):
 assert set(r['bucket'] for r in records)==set(BUCKET_MASS)
 counts=collections.Counter((record_group(r),str(r['groupId'])) for r in records);source_mass={}
 for bucket,mass in BUCKET_MASS.items():
  sources=sorted({r['source'] for r in records if r['bucket']==bucket})
  if bucket=='hard_positive' and CLINICAL_SOURCE in sources:
   others=[s for s in sources if s!=CLINICAL_SOURCE];assert others,'Clinical data cannot replace new hardcase positives'
   source_mass.update({(bucket,CLINICAL_SOURCE):mass*.2});source_mass.update({(bucket,s):mass*.8/len(others) for s in others})
  else:source_mass.update({(bucket,s):mass/len(sources) for s in sources})
 species={(b,s):{r['species'] for r in records if r['bucket']==b and r['source']==s} for b,s in source_mass}
 groups={key:{str(r['groupId']) for r in records if record_group(r)==key} for key in {record_group(r) for r in records}}
 weights=torch.tensor([source_mass[(r['bucket'],r['source'])]/len(species[(r['bucket'],r['source'])])/len(groups[record_group(r)])/counts[(record_group(r),str(r['groupId']))] for r in records],dtype=torch.double)
 assert abs(float(weights.sum())-1)<1e-9
 return weights

class RetentionSampler(WeightedRandomSampler):
 def __init__(self,records,n=360):
  super().__init__(sampling_weights(records),n,replacement=True,generator=torch.Generator().manual_seed(SEED+71));self.generated=[]
 def __iter__(self):
  indices=list(super().__iter__());self.generated.extend(indices);return iter(indices)

def old_records(split):
 manifest=read(OLD/'manifest.json');by={r['file']:r for r in manifest['images']};rows=[]
 assert split in ['train','val'],'No old test reader in training operator'
 for name in manifest['splits'][split]:
  r=by[name];w,h=r['width'],r['height'];xy=[]
  for line in (OLD/'labels'/split/(Path(name).stem+'.txt')).read_text().splitlines():
   cls,x,y,bw,bh=map(float,line.split());assert cls==0;xy.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
  image=OLD/'images'/split/name;rows.append({'id':'old_'+split+'_'+Path(name).stem,'split':split,'source':'figshare22022540v3','species':f'sp{r["species"]:02d}','groupId':'figshare22022540v3_'+name,'bucket':'old_anchor','role':'old_train' if split=='train' else 'old_validation','imagePath':str(image.relative_to(ROOT)),'width':w,'height':h,'sha256':sha(image),'xyxy':xy,'count':len(xy),'annotationQA':'PASS','annotationMethod':'Previously frozen original study boxes','negativeKind':None})
 return rows

def prepare(manifest_path,include_clinical=True):
 assert sha(ANCHOR)==ANCHOR_SHA
 manifest_path=Path(manifest_path).resolve()
 if (OUT/'data-binding.json').exists():
  prior=read(OUT/'data-binding.json');assert prior['manifestSHA256']==sha(manifest_path) and prior['clinicalIncluded']==include_clinical and prior['protocolSHA256']==sha(OUT/'protocol.json'),'Never replace a prepared frozen dataset in place';return prior
 assert not (OUT/'run').exists(),'Do not alter training data after a run starts'
 manifest=read(manifest_path)
 assert manifest['status']=='FROZEN_ANNOTATION_QA_PASSED' and manifest['testContentIncluded'] is False
 rows=old_records('train')+old_records('val')
 new=manifest['images'];assert new and set(r['split'] for r in new)=={'train','val'}
 for r in new:
  assert r['annotationQA']=='PASS' and r['bucket'] in ['hard_positive','empty_negative'];assert r['count']==len(r['xyxy'])
  if r['bucket']=='empty_negative':assert r['count']==0 and r['negativeKind']=='real_empty_plate'
  else:assert r['count']>0
  rows.append({**r,'role':'hardcase_'+r['split']})
 if include_clinical:
  clinical=read(CLINICAL);assert clinical['testContentIncluded'] is False
  for r in clinical['images']:
   assert r['split'] in ['train','val'];rows.append({**r,'id':'clinical_'+r['split']+'_'+Path(r['imagePath']).stem,'groupId':CLINICAL_SOURCE+'_'+r['species']+'_'+str(r['isolate']),'bucket':'hard_positive','role':'clinical_train' if r['split']=='train' else 'clinical_validation','annotationQA':'PASS','annotationMethod':r['coordinateSource'],'negativeKind':None})
 ids=[r['id'] for r in rows];assert len(set(ids))==len(ids)
 groups={};hash_roles={}
 for r in rows:
  assert re.fullmatch(r'[A-Za-z0-9_.-]+',r['id']) and r['id'] not in ['.','..'],'Unsafe image id'
  key=str(r['groupId']);assert key not in groups or groups[key]==r['split'];groups[key]=r['split']
  assert r['sha256'] not in hash_roles or hash_roles[r['sha256']]==r['split'],'Exact image duplicate across train/val';hash_roles[r['sha256']]=r['split']
  src=(ROOT/r['imagePath']).resolve();assert src.is_relative_to(ROOT.resolve()) and src.is_file() and sha(src)==r['sha256'];assert 0<=r['count']<=1000
  if r.get('labelPath'):
   original_label=ROOT/r['labelPath']
   if r.get('labelSHA256'):assert sha(original_label)==r['labelSHA256']
   original_lines=original_label.read_text().splitlines();assert len(original_lines)==r['count']
   for line,box in zip(original_lines,r['xyxy']):
    cls,x,y,bw,bh=map(float,line.split());assert cls==0
    converted=np.array([(x-bw/2)*r['width'],(y-bh/2)*r['height'],(x+bw/2)*r['width'],(y+bh/2)*r['height']]);assert np.max(np.abs(converted-np.array(box)))<.01
  # PNG getexif may load pixel data, after which Pillow.verify is invalid.
  with Image.open(src) as im:im.verify()
  with Image.open(src) as im:assert im.size==(r['width'],r['height']) and im.getexif().get(274,1) in [None,1]
  # Use copies, not hardlinks: a loader repair can never change source evidence.
  dest=OUT/'dataset/images'/r['split']/(r['id']+src.suffix.lower());label=OUT/'dataset/labels'/r['split']/(r['id']+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(src,dest)
  assert sha(dest)==r['sha256'];lines=[]
  for b in r['xyxy']:
   # Existing Figshare normalized labels were decimal-rounded. Reconstructed
   # edges can exceed the image by~1e-6pixels. Preserve GT; never clip or relabel.
   eps=1e-5 if r['source']=='figshare22022540v3' else 0.
   x1,y1,x2,y2=map(float,b);assert -eps<=x1<x2<=r['width']+eps and -eps<=y1<y2<=r['height']+eps;lines.append(f'0 {(x1+x2)/2/r["width"]:.10f} {(y1+y2)/2/r["height"]:.10f} {(x2-x1)/r["width"]:.10f} {(y2-y1)/r["height"]:.10f}')
  assert len(set(lines))==len(lines);label.write_text('\n'.join(lines)+('\n' if lines else ''));r['preparedImage']=str(dest.resolve());r['preparedLabel']=str(label.resolve());r['preparedLabelSHA256']=sha(label)
 save(OUT/'records.json',rows)
 (OUT/'dataset/data.yaml').write_text('path: '+json.dumps(str(OUT/'dataset'))+'\ntrain: images/train\nval: images/val\nnames:\n  0: colony\n')
 train=[r for r in rows if r['split']=='train'];weights=sampling_weights(train)
 for bucket in BUCKET_MASS:assert any(r['bucket']==bucket and r['split']=='val' and r['role']!='clinical_validation' for r in rows)
 binding={'status':'PREPARED_PENDING_ROOT_RECEIPT','manifestPath':str(manifest_path),'manifestSHA256':sha(manifest_path),'recordsSHA256':sha(OUT/'records.json'),'dataYAMLSHA256':sha(OUT/'dataset/data.yaml'),'protocolSHA256':sha(OUT/'protocol.json'),'trainImages':len(train),'valImages':len(rows)-len(train),'physicalGroupsByRole':{role:len({str(r['groupId']) for r in rows if r['role']==role}) for role in {r['role'] for r in rows}},'clinicalIncluded':include_clinical,'sourceRoles':dict(collections.Counter(r['role'] for r in rows)),'samplingBucketProbability':{b:sum(float(weights[i]) for i,r in enumerate(train) if r['bucket']==b) for b in BUCKET_MASS},'new36TestAccessed':False};save(OUT/'data-binding.json',binding);return binding

def raw(pred):return pred[1] if isinstance(pred,tuple) else pred

def retention_terms(student,teacher):
 """Differentiable class and DFL KL for aligned raw anchors; no GT matching needed."""
 s,t=student['scores'],teacher['scores'].detach();assert s.shape==t.shape and s.shape[1]==1
 q=t.sigmoid().clamp(1e-6,1-1e-6);kl=F.binary_cross_entropy_with_logits(s,q,reduction='none')+(q*q.log()+(1-q)*(1-q).log());kl=kl.squeeze(1)
 probability=q.squeeze(1);pos=(probability>=.1).to(kl.dtype);pos_loss=(kl*pos).sum(1)/pos.sum(1).clamp_min(1)
 bg=(probability<.05);rank=s.detach().sigmoid().squeeze(1).masked_fill(~bg,-1);k=min(256,rank.shape[1]);indices=rank.topk(k,dim=1).indices;valid=bg.gather(1,indices).to(kl.dtype);bg_loss=(kl.gather(1,indices)*valid).sum(1)/valid.sum(1).clamp_min(1)
 cls=((pos_loss+bg_loss)*.5).mean().clamp_min(0)
 sb,tb=student['boxes'],teacher['boxes'].detach();assert sb.shape==tb.shape and sb.shape[1]%4==0
 n,c,a=sb.shape;reg=c//4;log_p=sb.reshape(n,4,reg,a).log_softmax(2);p_t=tb.reshape(n,4,reg,a).softmax(2);dkl=F.kl_div(log_p,p_t,reduction='none').sum(2).mean(1)
 w=(probability>=.25).to(probability.dtype)*probability;box=((dkl*w).sum(1)/w.sum(1).clamp_min(1)).mean().clamp_min(0)
 return cls,box

class AnchorCriterion:
 def __init__(self,model,teacher,train_by_path):
  self.base=v8DetectionLoss(model);self.teacher=teacher;self.train_by_path=train_by_path;self.teacher_consumed=[]
 def __call__(self,predictions,batch):
  pred=raw(predictions);base_loss,items=self.base(predictions,batch);indices=[]
  for i,path in enumerate(batch['im_file']):
   r=self.train_by_path[str(Path(path).resolve())];assert r['split']=='train'
   if r['bucket']=='old_anchor':indices.append(i);self.teacher_consumed.append(r['id'])
  cls=pred['scores'].sum()*0;box=pred['boxes'].sum()*0
  if indices:
   index=torch.tensor(indices,device=batch['img'].device,dtype=torch.long)
   with torch.no_grad():target=raw(self.teacher(batch['img'].index_select(0,index)))
   selected={k:pred[k].index_select(0,index) for k in ['scores','boxes']};cls,box=retention_terms(selected,target)
  cls=cls*5.;box=box*1.;size=batch['img'].shape[0];loss=torch.cat((base_loss.reshape(-1),torch.stack([cls,box])*size));items={**items,'anchor_cls':cls.detach(),'anchor_dfl':box.detach()};return loss,items

def freeze_bn(model):
 for module in model.modules():
  if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
   module.eval()
   for param in module.parameters():param.requires_grad_(False)

class RetentionTrainer(DetectionTrainer):
 def _model_train(self):
  super()._model_train();freeze_bn(self.model)
 def _setup_train(self):
  super()._setup_train();self._model_train()
  teacher=YOLO(str(ANCHOR)).model.to(self.device).float().eval()
  for p in teacher.parameters():p.requires_grad_(False)
  assert not teacher.model[-1].end2end
  self._teacher=teacher;self.model.criterion=AnchorCriterion(self.model,teacher,self._train_by_path)
  self.loss_names=tuple(self.loss_names)+('anchor_cls','anchor_dfl')
  parameters=dict(self.model.named_parameters());state=self.model.state_dict();teacher_state=teacher.state_dict()
  self._frozen_keys=[k for k in state if k not in parameters or not parameters[k].requires_grad]
  self._anchor_frozen={k:teacher_state[k].detach().clone() for k in self._frozen_keys}
  for k in self._frozen_keys:assert torch.equal(state[k],teacher_state[k]),k
  assert not hasattr(self.ema.ema,'criterion') or self.ema.ema.criterion is None,'Teacher must not be copied into EMA checkpoint'
  save(OUT/'model-binding.json',{'anchorSHA256':sha(ANCHOR),'frozenKeys':self._frozen_keys,'trainableParameterNames':[n for n,p in self.model.named_parameters() if p.requires_grad],'teacherRegisteredInStudentStateDict':False,'teacherOnlyOldTrain':True})
 def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
  if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
  assert rank in [-1,0] and self.world_size<=1
  dataset=self.build_dataset(dataset_path,mode,batch_size);assert not dataset.rect
  self._train_by_path={str(Path(r['preparedImage']).resolve()):r for r in read(OUT/'records.json') if r['split']=='train'}
  self._ordered=[self._train_by_path[str(Path(p).resolve())] for p in dataset.im_files];self._path_index={str(Path(r['preparedImage']).resolve()):i for i,r in enumerate(self._ordered)}
  self._sampler=RetentionSampler(self._ordered);self._observed={};self._aug_boxes={};self._reported=set()
  loader=InfiniteDataLoader(dataset,batch_size=batch_size,shuffle=False,sampler=self._sampler,num_workers=2,prefetch_factor=4,pin_memory=False,collate_fn=dataset.collate_fn,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(SEED+313),drop_last=False)
  assert loader.sampler is self._sampler
  save(OUT/'dataloader-binding.json',{'orderedImageIds':[r['id'] for r in self._ordered],'perImageWeights':[float(w) for w in self._sampler.weights],'samplesPerEpoch':360,'batchesPerEpoch':len(loader),'samplerClass':type(loader.sampler).__name__});return loader
 def preprocess_batch(self,batch):
  batch=super().preprocess_batch(batch);e=int(self.epoch);ids=[self._path_index[str(Path(p).resolve())] for p in batch['im_file']];self._observed.setdefault(e,[]).extend(ids)
  # These are actual labels surviving data augmentation and passed to criterion.
  box_counts=torch.bincount(batch['batch_idx'].long().detach().cpu(),minlength=len(ids)).tolist();self._aug_boxes.setdefault(e,[]).extend(box_counts);return batch
 def validate(self):return {},0.0
 def final_eval(self):
  save(OUT/'framework-validation-skipped.json',{'reason':'Fixed checkpoint/alpha selection uses separate CPU predictions; no MPS NMS or automatic best-model validation.','epochsCompleted':len(self._reported)})
 def save_model(self):
  # Assertion-only guards: never accept the framework's automatic repair of
  # non-finite weights or fp16-overflow clamping as a clean checkpoint.
  for label,model in [('live',self.model),('ema',self.ema.ema)]:
   for key,value in model.state_dict().items():
    if value.is_floating_point():
     assert bool(torch.isfinite(value).all()),f'{label} nonfinite state: {key}'
     if label=='ema':assert bool((value.abs()<=torch.finfo(torch.float16).max).all()),f'fp16 serialization overflow: {key}'
  if hasattr(self,'_anchor_frozen'):
   live=self.model.state_dict()
   for key,value in self._anchor_frozen.items():assert torch.equal(live[key],value),'Frozen live state drift: '+key
  # EMA arithmetic may drift even constant buffers by rounding. Restore frozen
  # anchor entries before serialization, making the interpolation invariant exact.
  if hasattr(self,'_anchor_frozen'):
   state=self.ema.ema.state_dict()
   with torch.no_grad():
    for k,v in self._anchor_frozen.items():state[k].copy_(v)
  return super().save_model()

def assert_registered_checkpoint(epoch):
 p=OUT/'checkpoints'/f'epoch{epoch:03d}.pt';metadata=p.with_suffix('.json')
 assert p.is_file() and metadata.is_file(),f'Missing registered checkpoint at completed epoch{epoch}'
 record=read(metadata);assert record['epoch']==epoch and record['sha256']==sha(p)
 assert record['protocolSHA256']==sha(OUT/'protocol.json') and record['dataBindingSHA256']==sha(OUT/'data-binding.json')
 checkpoint=torch.load(p,map_location='cpu',weights_only=False);assert checkpoint['epoch']==epoch-1,'Checkpoint internal epoch does not match registration'
 return record

def save_registered_checkpoint(t):
 epoch=t.epoch+1
 if epoch in EPOCHS:
  p=OUT/'checkpoints'/f'epoch{epoch:03d}.pt';p.parent.mkdir(exist_ok=True);shutil.copy2(t.last,p);save(p.with_suffix('.json'),{'epoch':epoch,'sha256':sha(p),'protocolSHA256':sha(OUT/'protocol.json'),'dataBindingSHA256':sha(OUT/'data-binding.json')})
  assert_registered_checkpoint(epoch)

def audit_epoch(t):
 e=t.epoch
 if e in t._reported:return
 if e==t.epochs and set(t._reported)==set(range(t.epochs)) and not t._observed.get(e):return
 ids=t._observed.get(e,[]);assert len(ids)==360 and len(t._aug_boxes[e])==360
 if e+1 in EPOCHS:assert_registered_checkpoint(e+1)
 all_ids=[i for epoch in sorted(t._observed) for i in t._observed[epoch]];assert all_ids==t._sampler.generated[:len(all_ids)]
 expected_teacher=[t._ordered[i]['id'] for i in all_ids if t._ordered[i]['bucket']=='old_anchor'];assert t.model.criterion.teacher_consumed==expected_teacher
 original=collections.Counter();augmented=collections.Counter();buckets=collections.Counter();groups=collections.Counter();physical=collections.Counter()
 for i,count in zip(ids,t._aug_boxes[e]):
  r=t._ordered[i];key=record_group(r);original[key]+=r['count'];augmented[key]+=count;buckets[r['bucket']]+=1;groups[key]+=1;physical[r['source']+'::'+str(r['groupId'])]+=1
 save(OUT/'sampling'/f'epoch{e+1:03d}.json',{'epoch':e+1,'actualImageIds':[t._ordered[i]['id'] for i in ids],'bucketImageCounts':dict(buckets),'sourceSpeciesImageCounts':dict(groups),'physicalGroupImageCounts':dict(physical),'sampledOriginalGTBoxCounts':dict(original),'actualPostAugmentationLossBoxCounts':dict(augmented),'samplerPrefixMatch':True,'teacherOldTrainOnlyExactMatch':True,'lossItems':{k:float(v) for k,v in t.tloss.items()}});t._reported.add(e)

def check_root_receipt(path):
 receipt=read(path);binding=read(OUT/'data-binding.json');assert receipt['status']=='APPROVED_FOR_TRAINING_AFTER_ANNOTATION_QA'
 assert receipt['operatorSHA256']==sha(__file__),'Root receipt must bind final reviewed operator code'
 assert receipt['evaluatorSHA256']==sha(Path(__file__).with_name('evaluate_retention_savefix.py')),'Root receipt must bind final reviewed scoring code'
 assert receipt['localDependencySHA256']==local_dependency_hashes(),'Root receipt must bind inference/scoring/MPS helper dependencies'
 assert receipt['manifestSHA256']==binding['manifestSHA256']==sha(binding['manifestPath'])
 assert receipt['protocolSHA256']==sha(OUT/'protocol.json') and receipt['dataBindingSHA256']==sha(OUT/'data-binding.json')
 qa=Path(receipt['qaReportPath']);assert sha(qa)==receipt['qaReportSHA256'];qa_result=read(qa);assert qa_result['status'].startswith('PASS')
 assert binding['recordsSHA256']==sha(OUT/'records.json') and binding['dataYAMLSHA256']==sha(OUT/'dataset/data.yaml')
 for r in read(OUT/'records.json'):assert sha(r['preparedImage'])==r['sha256'] and sha(r['preparedLabel'])==r['preparedLabelSHA256']
 return receipt

def train(receipt):
 check_root_receipt(receipt);assert sha(ANCHOR)==ANCHOR_SHA and torch.backends.mps.is_available();assert not (OUT/'run').exists(),'Never overwrite a previous training attempt'
 install_cpu_assigner_for_mps();torch.set_num_threads(8);settings.update({'sync':False,'wandb':False,'mlflow':False,'comet':False,'clearml':False,'neptune':False,'tensorboard':False})
 save(OUT/'training-authorization.json',{'rootReceiptPath':str(Path(receipt).resolve()),'rootReceiptSHA256':sha(receipt),'operatorSHA256':sha(__file__),'evaluatorSHA256':sha(Path(__file__).with_name('evaluate_retention_savefix.py')),'localDependencySHA256':local_dependency_hashes(),'protocolSHA256':sha(OUT/'protocol.json'),'dataBindingSHA256':sha(OUT/'data-binding.json'),'recordedBeforeAnyOptimizerStep':True})
 shutil.copy2(__file__,OUT/'operator-training-start.py');model=YOLO(str(ANCHOR))
 model.add_callback('on_model_save',save_registered_checkpoint);model.add_callback('on_fit_epoch_end',audit_epoch)
 started=time.time();model.train(trainer=RetentionTrainer,data=str(OUT/'dataset/data.yaml'),project=str(OUT),name='run',exist_ok=False,**read(OUT/'protocol.json')['trainingConfig'])
 assert len(list((OUT/'sampling').glob('epoch*.json')))==8
 checkpoint_records={str(epoch):assert_registered_checkpoint(epoch) for epoch in EPOCHS}
 save(OUT/'training-complete.json',{'status':'COMPLETE_8_EPOCHS_CLEAN_EXIT_WITH_REGISTERED_CHECKPOINTS','seconds':time.time()-started,'operatorSHA256':sha(OUT/'operator-training-start.py'),'protocolSHA256':sha(OUT/'protocol.json'),'rootReceiptSHA256':sha(receipt),'checkpoints':checkpoint_records,'new36TestAccessed':False,'appModified':False})

def blend_model(anchor_model,trained_model,alpha,frozen_keys):
 """Convex merge only learned floating parameters; every other entry is invariant."""
 assert 0<=alpha<=1
 base=anchor_model.float().cpu();trained=trained_model.float().cpu();a,b=base.state_dict(),trained.state_dict();assert set(a)==set(b)
 params=dict(base.named_parameters());frozen=set(frozen_keys);merged={}
 for k,av in a.items():
  bv=b[k];assert av.shape==bv.shape and av.dtype==bv.dtype,k
  if av.is_floating_point():assert torch.isfinite(av).all() and torch.isfinite(bv).all(),k
  if k in frozen or k not in params or not av.is_floating_point():
   assert torch.equal(av,bv),'Frozen/buffer drift: '+k;merged[k]=av.clone()
  else:merged[k]=av.clone() if alpha==0 else bv.clone() if alpha==1 else av*(1-alpha)+bv*alpha
 base.load_state_dict(merged,strict=True);return base

def make_candidates():
 assert (OUT/'training-complete.json').exists();frozen=read(OUT/'model-binding.json')['frozenKeys'];index=[]
 if (OUT/'candidates/index.json').exists():
  existing=read(OUT/'candidates/index.json');assert existing['protocolSHA256']==sha(OUT/'protocol.json')
  for r in existing['candidates']:assert sha(r['path'])==r['sha256']
  return existing
 for epoch in EPOCHS:
  checkpoint=OUT/'checkpoints'/f'epoch{epoch:03d}.pt';assert sha(checkpoint)==read(checkpoint.with_suffix('.json'))['sha256']
  for alpha in ALPHAS:
   anchor=YOLO(str(ANCHOR));student=YOLO(str(checkpoint));model=blend_model(anchor.model,student.model,alpha,frozen);path=OUT/'candidates'/f'epoch{epoch:03d}-alpha{alpha:g}.pt';path.parent.mkdir(exist_ok=True)
   payload={'model':model,'ema':None,'optimizer':None,'epoch':epoch-1,'train_args':anchor.ckpt.get('train_args',{}),'date':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'version':'8.4.123'};torch.save(payload,path)
   loaded=YOLO(str(path)).model.float().cpu();assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
   index.append({'epoch':epoch,'alpha':alpha,'path':str(path),'sha256':sha(path),'trainedCheckpointSHA256':sha(checkpoint),'anchorSHA256':ANCHOR_SHA,'pipeline':'automatic4tile','confidence':.3})
 save(OUT/'candidates/index.json',{'candidates':index,'alphaZeroEligible':False,'count':len(index),'protocolSHA256':sha(OUT/'protocol.json')})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['init','prepare','train','candidates']);parser.add_argument('--manifest',type=Path);parser.add_argument('--receipt',type=Path);parser.add_argument('--without-clinical',action='store_true');args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if args.phase=='init':print(json.dumps(make_protocol(),indent=2))
 elif args.phase=='prepare':assert args.manifest;make_protocol();print(prepare(args.manifest,not args.without_clinical))
 elif args.phase=='train':assert args.receipt;train(args.receipt)
 else:make_candidates()
