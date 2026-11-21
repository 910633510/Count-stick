"""Bounded real sampler/save lifecycle checks; zero optimizer steps, no new data."""
import collections,copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset
import pipe_hardening as p
import train_hardening as t
from ultralytics import YOLO
class SyntheticDataset(Dataset):
 rect=False
 def __init__(self,rows):self.im_files=[r['preparedImage'] for r in rows]
 def __len__(self):return len(self.im_files)
 def __getitem__(self,i):return {'im_file':self.im_files[i]}
 @staticmethod
 def collate_fn(batch):return {'im_file':[r['im_file'] for r in batch]}

def run():
 outer=p.OUT;folder=outer/'smoke-v2';assert not folder.exists();folder.mkdir(parents=True);p.OUT=folder
 protocol=p.proposal();p.save(folder/'protocol.json',protocol);rows=[]
 for bucket in ['old_positive','new_positive','negative']:
  for source,groups,views in [('sourceA',2,2),('sourceB',1,1)]:
   for g in range(groups):
    for v in range(views):
     id=f'{bucket}_{source}_{g}_{v}';rows.append(dict(id=id,split='train',bucket=bucket,source=source,groupId=f'{bucket}_{source}_{g}',preparedImage=str(folder/(id+'.not-a-real-image')),count=0 if bucket=='negative' else 3))
 p.save(folder/'records.json',rows);p.save(folder/'data-binding.json',{'status':'SYNTHETIC_MECHANICAL_SMOKE_ONLY'})
 trainer=object.__new__(t.HardeningTrainer);trainer.args=SimpleNamespace(seed=20261026);trainer.world_size=0;dataset=SyntheticDataset(rows);trainer.build_dataset=lambda *args:dataset
 loader=trainer.get_dataloader('synthetic',batch_size=4,rank=-1,mode='train');observed=[]
 for batch in loader:observed.extend(Path(x).stem for x in batch['im_file'])
 expected=[trainer._rows[i]['id'] for i in trainer._sampler.generated[:len(observed)]];assert observed==expected and len(observed)==320
 masses={b:sum(float(w) for r,w in zip(trainer._rows,trainer._sampler.weights) if r['bucket']==b) for b in protocol['bucketMasses']};assert all(abs(masses[b]-m)<1e-12 for b,m in protocol['bucketMasses'].items())
 if hasattr(loader.iterator,'_shutdown_workers'):loader.iterator._shutdown_workers()
 model=YOLO(str(p.INIT)).model.float().cpu();trainer.model=model;trainer.args=SimpleNamespace(imgsz=1024,freeze=4,save_period=-1,epochs=12)
 trainer.freeze_layer_names=[f'model.{i}.' for i in range(4)]+['.dfl']
 for name,value in model.named_parameters():
  value.requires_grad=not any(x in name for x in trainer.freeze_layer_names)
 trainer._frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(tuple(f'model.{i}.' for i in range(4)))};trainer._model_train();trainer._check_frozen()
 trainer.ema=SimpleNamespace(ema=copy.deepcopy(model).eval(),updates=0);trainer.optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=.0001);trainer.scaler=torch.amp.GradScaler('cpu',enabled=False);trainer.metrics={};trainer.fitness=trainer.best_fitness=0.;trainer.read_results_csv=lambda:{};trainer.wdir=folder/'weights';trainer.last=trainer.wdir/'last.pt';trainer.best=trainer.wdir/'best.pt';trainer.save_period=-1;trainer.callbacks={'on_model_save':[t.on_save]}
 checkpoints={}
 for epoch in [4,8,12]:
  trainer.epoch=epoch-1;saved=trainer.save_model();assert saved is True;trainer.run_callbacks('on_model_save');meta=t.checkpoint_check(epoch);checkpoint=torch.load(folder/'checkpoints'/f'epoch{epoch:03d}.pt',map_location='cpu',weights_only=False);assert checkpoint['epoch']==epoch-1 and checkpoint['optimizer']['state']=={};assert all(torch.isfinite(v).all() for v in checkpoint['ema'].state_dict().values() if v.is_floating_point());checkpoints[str(epoch)]=meta
 report={'status':'PASS_ACTUAL_WEIGHTED_DATALOADER_AND_TRUE_CALLBACK_CHECKPOINT_LIFECYCLE','operatorSHA256':p.sha(t.__file__),'commonSHA256':p.sha(p.__file__),'initializationSHA256':p.sha(p.INIT),'actualConsumedImages':len(observed),'actualImageIds':observed,'samplerPrefixMatched':True,'expectedBucketMasses':masses,'actualBucketCounts':dict(collections.Counter(next(r['bucket'] for r in rows if r['id']==id) for id in observed)),'snapshots':checkpoints,'frozenLiveStateExact':True,'frozenBNEvalDuringTraining':True,'optimizerSteps':0,'modelForwardCalls':0,'trainingDataLoaded':False,'testAccessed':False};p.save(folder/'smoke-report.json',report);print(report['status'],p.sha(folder/'smoke-report.json'));p.OUT=outer
if __name__=='__main__':run()
