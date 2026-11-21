"""Exercise the installed save_model boolean/callback/copy/reload lifecycle.

Temporary synthetic input and unchanged anchor weights only;zero optimizer steps.
"""
import copy,tempfile,torch
from pathlib import Path
from PIL import Image
import train_retention_savefix as m

def must_reject(function,label):
 try:function()
 except AssertionError:return
 raise AssertionError('Expected rejection: '+label)

def main():
 actual=m.OUT;reports=[]
 with tempfile.TemporaryDirectory(prefix='countscope-savefix-lifecycle-') as temp:
  m.OUT=Path(temp);records=[];trainer=None
  try:
   m.save(m.OUT/'protocol.json',m.read(actual/'protocol.json'));m.save(m.OUT/'data-binding.json',{'syntheticOnly':True,'notATrainingAuthorization':True})
   for split in ['train','val']:
    for i,bucket in enumerate(['old_anchor','hard_positive','empty_negative']):
     p=m.OUT/'dataset/images'/split/(bucket+'.png');label=m.OUT/'dataset/labels'/split/(bucket+'.txt');p.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(256,256),(40+i*60,)*3).save(p);label.write_text('0 .5 .5 .1 .1\n' if bucket!='empty_negative' else '')
     records.append({'id':split+'_'+bucket,'split':split,'bucket':bucket,'source':bucket,'species':bucket,'groupId':split+'_'+bucket,'count':int(bucket!='empty_negative'),'preparedImage':str(p),'preparedLabel':str(label)})
   m.save(m.OUT/'records.json',records);yaml=m.OUT/'dataset/data.yaml';yaml.write_text('path: '+str(m.OUT/'dataset')+'\ntrain: images/train\nval: images/val\nnames:\n  0: colony\n')
   config=copy.deepcopy(m.read(actual/'protocol.json')['trainingConfig']);config.update({'model':str(m.ANCHOR),'data':str(yaml),'project':str(m.OUT),'name':'synthetic-save-lifecycle','imgsz':256})
   m.install_cpu_assigner_for_mps();trainer=m.RetentionTrainer(overrides=config);trainer._setup_train();trainer.metrics={};trainer.fitness=0.;trainer.best_fitness=0.
   trainer.callbacks['on_model_save'].append(m.save_registered_checkpoint)
   for epoch in [3,5,8]:
    trainer.epoch=epoch-1;trainer.csv.write_text('epoch,train/box_loss\n'+str(epoch)+',0.0\n')
    returned=trainer.save_model();assert returned is True
    # Same conditional callback contract as the installed BaseTrainer loop.
    if returned:trainer.run_callbacks('on_model_save')
    metadata=m.assert_registered_checkpoint(epoch);checkpoint=torch.load(m.OUT/'checkpoints'/f'epoch{epoch:03d}.pt',map_location='cpu',weights_only=False);assert checkpoint['epoch']==epoch-1
    assert checkpoint.get('ema') is not None and (not hasattr(checkpoint['ema'],'criterion') or checkpoint['ema'].criterion is None)
    reports.append({'completedEpoch':epoch,'saveModelReturnedTrue':True,'registeredCopyReloaded':True,'internalEpochZeroBased':checkpoint['epoch'],'snapshotSHA256':metadata['sha256']})
   before=m.sha(trainer.last);live=trainer.model.state_dict();ema=trainer.ema.ema.state_dict();frozen=next(k for k in trainer._frozen_keys if live[k].is_floating_point())
   for tensor,value,label in [(live[frozen],float('nan'),'nonfinite live'),(ema[frozen],float('inf'),'nonfinite EMA'),(ema[frozen],70000.,'fp16 serialization overflow')]:
    old=tensor.clone()
    with torch.no_grad():tensor.reshape(-1)[0]=value
    must_reject(trainer.save_model,label)
    with torch.no_grad():tensor.copy_(old)
    assert m.sha(trainer.last)==before,'Guard must fail before modifying checkpoint bytes'
   original=live[frozen].clone()
   with torch.no_grad():live[frozen].add_(.01)
   must_reject(trainer.save_model,'frozen live drift')
   with torch.no_grad():live[frozen].copy_(original)
   assert m.sha(trainer.last)==before
   missing=m.OUT/'checkpoints/epoch003.pt';parked=missing.with_suffix('.parked');missing.rename(parked);must_reject(lambda:m.assert_registered_checkpoint(3),'missing required epoch3');parked.rename(missing)
   assert len(trainer.optimizer.state)==0 and trainer.ema.updates==0
  finally:
   if trainer is not None:
    for name in ['train_loader','test_loader']:
     loader=getattr(trainer,name,None)
     if loader is not None:loader.close()
   m.OUT=actual
 report={'status':'PASS_REAL_SAVE_CALLBACK_COPY_RELOAD_LIFECYCLE','checkpoints':reports,'sameConditionalAsInstalledTrainer':True,'nonfiniteLiveEMAAndFP16OverflowRejectedBeforeSave':True,'frozenLiveDriftRejectedBeforeSave':True,'missingRegisteredCheckpointRejected':True,'optimizerSteps':0,'emaUpdates':0,'syntheticOnly':True,'realTrainValTestDataUsed':False,'savedSyntheticCheckpointsRemovedWithTemporaryDirectory':True,'operatorSHA256':m.sha(m.__file__),'smokeSourceSHA256':m.sha(__file__)}
 m.save(actual/'save-lifecycle-smoke.json',report);print(report)

if __name__=='__main__':main()
