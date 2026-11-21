"""Independent pipe baseline; data/protocol/root approval required before training."""
import argparse,collections,shutil,time
from pathlib import Path
import pipe_ml as p
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO,settings
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader,seed_worker
from ultralytics.data.augment import RandomPerspective
from mps_compat import install_cpu_assigner_for_mps

def install_checked_affine():
 if getattr(RandomPerspective.apply_bboxes,'_pipe_checked',False):return
 original=RandomPerspective.apply_bboxes
 def checked(self,bboxes,matrix):
  assert np.isfinite(bboxes).all() and np.isfinite(matrix).all()
  result=original(self,bboxes,matrix)
  assert np.isfinite(result).all(),'Nonfinite transformed GT before candidate filtering'
  # Independent elementwise multiply/add avoids the macOS BLAS matmul route.
  if len(bboxes):
   xy=np.ones((len(bboxes)*4,3),dtype=bboxes.dtype);xy[:,:2]=bboxes[:,[0,1,2,3,0,3,2,1]].reshape(-1,2)
   oracle=np.stack([xy[:,0]*matrix[j,0]+xy[:,1]*matrix[j,1]+xy[:,2]*matrix[j,2] for j in range(3)],axis=1)
   oracle=(oracle[:,:2]/oracle[:,2:3] if self.perspective else oracle[:,:2]).reshape(-1,8)
   xs,ys=oracle[:,[0,2,4,6]],oracle[:,[1,3,5,7]];expected=np.concatenate((xs.min(1),ys.min(1),xs.max(1),ys.max(1)),dtype=bboxes.dtype).reshape(4,len(bboxes)).T
   assert np.isfinite(expected).all() and np.allclose(result,expected,rtol=1e-6,atol=1e-3),'Affine BLAS/scalar oracle mismatch'
  return result
 checked._pipe_checked=True;RandomPerspective.apply_bboxes=checked

def seed_and_guard(worker_id):seed_worker(worker_id);install_checked_affine()

def sampling_weights(records,negative_mass):
 buckets={r['bucket'] for r in records};assert buckets=={'positive','negative'}
 group_counts=collections.Counter((r['bucket'],str(r['groupId'])) for r in records);group_sets={b:{str(r['groupId']) for r in records if r['bucket']==b} for b in buckets};masses={'positive':1-negative_mass,'negative':negative_mass}
 weights=torch.tensor([masses[r['bucket']]/len(group_sets[r['bucket']])/group_counts[(r['bucket'],str(r['groupId']))] for r in records],dtype=torch.double);assert abs(float(weights.sum())-1)<1e-9;return weights

class SceneSampler(WeightedRandomSampler):
 def __init__(self,records,draws,seed,negative_mass):
  super().__init__(sampling_weights(records,negative_mass),draws,replacement=True,generator=torch.Generator().manual_seed(seed+71));self.generated=[]
 def __iter__(self):
  sequence=list(super().__iter__());self.generated.extend(sequence);return iter(sequence)

class PipeTrainer(DetectionTrainer):
 def _setup_train(self):
  super()._setup_train();assert self.model.model[-1].nc==1 and self.model.names=={0:'pipe_end'}
  strides=[int(s) for s in self.model.stride.tolist()];raw_locations=sum((self.args.imgsz//s)**2 for s in strides);assert raw_locations==21504 and self.args.max_det>=raw_locations
  p.save(p.OUT/'model-binding.json',{'initSHA256':p.WEIGHTS_SHA,'classes':self.model.names,'strides':strides,'rawLocations':raw_locations,'parameterCount':sum(x.numel() for x in self.model.parameters()),'noColonyInitialization':True})
 def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
  if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
  assert rank in [-1,0] and self.world_size<=1
  assert batch_size==p.read(p.OUT/'protocol.json')['training']['batch'],'Do not silently change frozen batch size during OOM recovery'
  dataset=self.build_dataset(dataset_path,mode,batch_size);assert not dataset.rect
  records={str(Path(r['preparedImage']).resolve()):r for r in p.read(p.OUT/'records.json') if r['split']=='train'};self._rows=[records[str(Path(x).resolve())] for x in dataset.im_files];self._index={str(Path(x).resolve()):i for i,x in enumerate(dataset.im_files)}
  protocol=p.read(p.OUT/'protocol.json');draws=protocol['samplesPerEpoch'];assert draws%batch_size==0
  self._sampler=SceneSampler(self._rows,draws,self.args.seed,protocol['negativeSamplingMass']);self._observed={};self._loss_boxes={};self._reported=set()
  loader=InfiniteDataLoader(dataset,batch_size=batch_size,sampler=self._sampler,shuffle=False,num_workers=2,prefetch_factor=4,pin_memory=False,collate_fn=dataset.collate_fn,worker_init_fn=seed_and_guard,generator=torch.Generator().manual_seed(self.args.seed+313),drop_last=False)
  p.save(p.OUT/'dataloader-binding.json',{'orderedIds':[r['id'] for r in self._rows],'weights':[float(w) for w in self._sampler.weights],'drawsPerEpoch':draws,'batchSize':batch_size,'groups':len({r['groupId'] for r in self._rows}),'samplerClass':type(loader.sampler).__name__});return loader
 def preprocess_batch(self,batch):
  batch=super().preprocess_batch(batch)
  assert all(torch.isfinite(batch[k]).all() for k in ['img','bboxes','cls','batch_idx'])
  ids=[self._index[str(Path(x).resolve())] for x in batch['im_file']];self._observed.setdefault(int(self.epoch),[]).extend(ids);self._loss_boxes.setdefault(int(self.epoch),[]).extend(torch.bincount(batch['batch_idx'].long().cpu(),minlength=len(ids)).tolist());return batch
 def validate(self):
  # Keep placeholder metric columns stable; they are never accuracy evidence.
  if self.best_fitness is None:self.best_fitness=0.
  return dict(self.metrics),0.
 def final_eval(self):p.save(p.OUT/'framework-validation-skipped.json',{'status':'CPU_APP_PIXEL_EVALUATION_SEPARATE','frameworkMetricColumnsArePlaceholders':True,'epochsAudited':len(self._reported)})
 def save_model(self):
  for tag,model in [('live',self.model),('ema',self.ema.ema)]:
   for name,value in model.state_dict().items():
    if value.is_floating_point():
     assert torch.isfinite(value).all(),f'{tag} nonfinite: {name}'
     if tag=='ema':assert (value.abs()<=torch.finfo(torch.float16).max).all(),f'fp16 overflow: {name}'
  return super().save_model()

def checkpoint_check(epoch):
 file=p.OUT/'checkpoints'/f'epoch{epoch:03d}.pt';meta=file.with_suffix('.json');assert file.is_file() and meta.is_file()
 record=p.read(meta);assert record['epoch']==epoch and record['sha256']==p.sha(file) and record['protocolSHA256']==p.sha(p.OUT/'protocol.json') and record['dataBindingSHA256']==p.sha(p.OUT/'data-binding.json')
 loaded=torch.load(file,map_location='cpu',weights_only=False);assert loaded['epoch']==epoch-1;model=loaded['ema'];assert model.model[-1].nc==1 and model.names=={0:'pipe_end'};return record

def on_save(trainer):
 epoch=trainer.epoch+1
 if epoch in p.read(p.OUT/'protocol.json')['checkpointEpochs']:
  file=p.OUT/'checkpoints'/f'epoch{epoch:03d}.pt';file.parent.mkdir(exist_ok=True);shutil.copy2(trainer.last,file);p.save(file.with_suffix('.json'),{'epoch':epoch,'sha256':p.sha(file),'protocolSHA256':p.sha(p.OUT/'protocol.json'),'dataBindingSHA256':p.sha(p.OUT/'data-binding.json')});checkpoint_check(epoch)

def audit_epoch(trainer):
 e=trainer.epoch
 if e in trainer._reported:return
 if e==trainer.epochs and trainer._reported==set(range(trainer.epochs)) and not trainer._observed.get(e):return
 expected=p.read(p.OUT/'protocol.json')['samplesPerEpoch'];ids=trainer._observed.get(e,[]);assert len(ids)==expected and len(trainer._loss_boxes[e])==expected
 all_ids=[i for epoch in sorted(trainer._observed) for i in trainer._observed[epoch]];assert all_ids==trainer._sampler.generated[:len(all_ids)]
 if e+1 in p.read(p.OUT/'protocol.json')['checkpointEpochs']:checkpoint_check(e+1)
 source=collections.Counter();groups=collections.Counter();buckets=collections.Counter();original=collections.Counter();actual=collections.Counter()
 for i,n in zip(ids,trainer._loss_boxes[e]):
  r=trainer._rows[i];source[r['source']]+=1;groups[str(r['groupId'])]+=1;buckets[r['bucket']]+=1;original[r['source']]+=r['count'];actual[r['source']]+=n
 report={'epoch':e+1,'actualImageIds':[trainer._rows[i]['id'] for i in ids],'sourceImageCounts':dict(source),'groupImageCounts':dict(groups),'bucketImageCounts':dict(buckets),'sampledOriginalGTBoxes':dict(original),'postAugmentationGTBoxes':dict(actual),'samplerPrefixMatched':True,'lossItems':{k:float(v) for k,v in trainer.tloss.items()}};assert all(np.isfinite(v) for v in report['lossItems'].values());p.save(p.OUT/'sampling'/f'epoch{e+1:03d}.json',report);trainer._reported.add(e)

def train(receipt):
 p.official_init_check();p.root_training_approval(receipt);assert not (p.OUT/'run').exists(),'Preserve previous attempts'
 protocol=p.read(p.OUT/'protocol.json');assert protocol['training']['epochs']==20 and protocol['checkpointEpochs']==[8,14,20]
 install_cpu_assigner_for_mps();install_checked_affine();torch.set_num_threads(8);settings.update({'sync':False,'wandb':False,'mlflow':False,'comet':False,'clearml':False,'neptune':False,'tensorboard':False})
 p.save(p.OUT/'training-authorization.json',{'rootReceiptPath':str(Path(receipt).resolve()),'rootReceiptSHA256':p.sha(receipt),'codeSHA256':p.code_hashes(),'protocolSHA256':p.sha(p.OUT/'protocol.json'),'dataBindingSHA256':p.sha(p.OUT/'data-binding.json'),'officialWeightsSHA256':p.WEIGHTS_SHA})
 model=YOLO(str(p.WEIGHTS));model.add_callback('on_model_save',on_save);model.add_callback('on_fit_epoch_end',audit_epoch);start=time.time();model.train(trainer=PipeTrainer,data=str(p.OUT/'dataset/data.yaml'),project=str(p.OUT),name='run',exist_ok=False,**protocol['training'])
 records={str(e):checkpoint_check(e) for e in protocol['checkpointEpochs']};assert len(list((p.OUT/'sampling').glob('epoch*.json')))==20
 p.save(p.OUT/'training-complete.json',{'status':'COMPLETE_20_EPOCHS_WITH_ACTUAL_REGISTERED_CHECKPOINTS','seconds':time.time()-start,'checkpoints':records,'rootReceiptSHA256':p.sha(receipt),'testAccessed':False,'appModelEnabled':False})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','train']);parser.add_argument('--manifest',type=Path);parser.add_argument('--receipt',type=Path);args=parser.parse_args();p.OUT.mkdir(parents=True,exist_ok=True)
 if args.phase=='prepare':assert args.manifest;print(p.prepare(args.manifest))
 else:assert args.receipt;train(args.receipt)
