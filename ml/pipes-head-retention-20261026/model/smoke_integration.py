"""Synthetic full trainer setup, one actual mixed batch/backward, and save callbacks; zero steps."""
from pathlib import Path
import copy,json
import pipe_head_retention as p
import numpy as np
import torch
from PIL import Image
from train_head_retention import HeadRetentionTrainer,on_save,checkpoint_check,install_checked_affine
from mps_compat import install_cpu_assigner_for_mps

def main():
 outer=p.OUT;folder=outer/'smoke';assert not folder.exists();folder.mkdir(parents=True);p.OUT=folder
 old=[r['id'] for r in p.read(p.OLD/'records.json') if r['split']=='train' and r['count']>0];assert len(old)==156
 rows=[]
 for i,(sample_id,bucket) in enumerate([(old[0],'old_positive'),(old[1],'old_positive'),('synthetic_new_positive','new_positive'),('synthetic_negative','negative')]):
  image=folder/'dataset/images/train'/(sample_id+'.png');label=folder/'dataset/labels/train'/(sample_id+'.txt');image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True);pixels=np.full((96,128,3),100+i*20,dtype=np.uint8);pixels[25:60,35:90]=[180,130,90];Image.fromarray(pixels).save(image);label.write_text('0 .5 .5 .3 .3\n' if bucket!='negative' else '')
  rows.append({'id':sample_id,'split':'train','bucket':bucket,'source':'synthetic_fixture_only','groupId':sample_id,'preparedImage':str(image),'count':1 if bucket!='negative' else 0})
 p.save(folder/'records.json',rows);protocol=p.proposal();protocol['training']['device']='cpu';protocol['training']['workers']=0;p.save(folder/'protocol.json',protocol);p.save(folder/'data-binding.json',{'status':'SYNTHETIC_FIXTURE_ONLY_NO_REAL_TRAINING'})
 data=folder/'dataset/data.yaml';data.write_text('path: '+json.dumps(str(folder/'dataset'))+'\ntrain: images/train\nval: images/train\nnames:\n  0: pipe_end\n')
 install_cpu_assigner_for_mps();install_checked_affine();torch.set_num_threads(4)
 trainer=HeadRetentionTrainer(overrides={**protocol['training'],'model':str(p.INIT),'data':str(data),'project':str(folder),'name':'setup','exist_ok':False});trainer._setup_train();trainer.epoch=0
 old_found=False
 for batch in trainer.train_loader:
  batch=trainer.preprocess_batch(batch)
  if batch['old_positive_mask'].any():old_found=True;break
 assert old_found
 # One bounded synthetic128 network pass, separate from declared1024 production recipe.
 batch['img']=torch.nn.functional.interpolate(batch['img'],size=(128,128),mode='bilinear',align_corners=False)
 loss,items=trainer.model(batch);assert torch.isfinite(loss).all();loss.sum().backward();assert all(torch.isfinite(parameter.grad).all() for parameter in trainer.model.parameters() if parameter.grad is not None);trainer._check_frozen()
 events=trainer.model.criterion.events;assert events and events[-1]['actualTeacherImageIds']==[id for id in batch['kd_image_ids'] if id in trainer._old_ids]
 assert 'synthetic_new_positive' not in events[-1]['actualTeacherImageIds'] and 'synthetic_negative' not in events[-1]['actualTeacherImageIds']
 optimizer_ids={id(x) for group in trainer.optimizer.param_groups for x in group['params']};assert optimizer_ids=={id(x) for x in trainer.model.parameters() if x.requires_grad} and not optimizer_ids & {id(x) for x in trainer._teacher.parameters()}
 assert trainer.optimizer.state_dict()['state']=={} and getattr(trainer.ema.ema,'criterion',None) is None
 trainer.best_fitness=trainer.fitness=0.;trainer.add_callback('on_model_save',on_save);snapshots={}
 for epoch in [4,8,12]:
  trainer.epoch=epoch-1;assert trainer.save_model() is True;trainer.run_callbacks('on_model_save');meta=checkpoint_check(epoch);loaded=torch.load(folder/'checkpoints'/f'epoch{epoch:03d}.pt',map_location='cpu',weights_only=False);assert getattr(loaded['ema'],'criterion',None) is None and not any('teacher' in name for name,_ in loaded['ema'].named_modules()) and loaded['optimizer']['state']=={};snapshots[str(epoch)]=meta
 for loader in [trainer.train_loader,trainer.test_loader]:
  if hasattr(loader,'iterator') and hasattr(loader.iterator,'_shutdown_workers'):loader.iterator._shutdown_workers()
 report={'status':'PASS_HEAD_RETENTION_TRAINER_INTEGRATION_NO_OPTIMIZER','trainerSHA256':p.sha(Path(__file__).with_name('train_head_retention.py')),'criterionSHA256':p.sha(Path(__file__).with_name('head_kd.py')),'commonSHA256':p.sha(p.__file__),'scriptSHA256':p.sha(__file__),'actualBatchIds':batch['kd_image_ids'],'actualTeacherIds':events[-1]['actualTeacherImageIds'],'originalOldPositiveIdsCount':len(trainer._old_ids),'previousNewPositiveNeverTeacherOld':True,'teacherNotInOptimizer':True,'optimizerContainsTrainableHeadOnly':True,'allBNAndBodyStateUnchanged':True,'teacherStateUnchanged':True,'allGradientsFinite':True,'KDWeights':{'classification':8.,'DFL':1.},'lossItems':{k:float(v) for k,v in items.items()},'snapshots':snapshots,'savedStandardModelNoTeacherOrCriterion':True,'optimizerSteps':0,'networkInputUsedInSyntheticSmoke':128,'declaredTrainingInput':1024,'realPhotographsRead':False,'freshTestRead':False,'trainingStarted':False};p.save(folder/'report.json',report);print(report['status'],p.sha(folder/'report.json'));p.OUT=outer
if __name__=='__main__':main()
