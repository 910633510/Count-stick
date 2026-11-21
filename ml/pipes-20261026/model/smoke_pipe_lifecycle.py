"""Real save-return/callback/reload smoke, synthetic data, no optimizer step."""
import copy,json,tempfile,collections
from pathlib import Path
import pipe_ml as p
import torch
from PIL import Image
from train_pipe_ends import PipeTrainer,on_save,checkpoint_check,SceneSampler,install_checked_affine
from mps_compat import install_cpu_assigner_for_mps

def main():
 original=p.OUT;trainer=None;reports=[]
 with tempfile.TemporaryDirectory(prefix='pipe-lifecycle-') as temporary:
  p.OUT=Path(temporary)
  try:
   config={'epochs':20,'imgsz':1024,'batch':4,'device':'mps','workers':2,'seed':20261026,'amp':False,'optimizer':'AdamW','lr0':.0005,'warmup_bias_lr':.0005,'warmup_epochs':1.,'nbs':16,'lrf':.1,'mosaic':0.,'close_mosaic':0,'mixup':0.,'cache':False,'max_det':21504,'val':False,'plots':False,'save':True,'save_period':-1,'verbose':False}
   p.save(p.OUT/'protocol.json',{'training':config,'checkpointEpochs':[8,14,20],'samplesPerEpoch':8,'negativeSamplingMass':.15});p.save(p.OUT/'data-binding.json',{'syntheticOnly':True});records=[]
   for split in ['train','val']:
    for i,bucket in enumerate(['positive','positive','negative']):
     file=p.OUT/'dataset/images'/split/f'{bucket}{i}.png';label=p.OUT/'dataset/labels'/split/f'{bucket}{i}.txt';file.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(256,256),(50+i*50,)*3).save(file);label.write_text('0 .5 .5 .1 .1\n' if bucket=='positive' else '')
     records.append({'id':split+str(i),'split':split,'source':'synthetic','groupId':split+str(i),'bucket':bucket,'count':int(bucket=='positive'),'preparedImage':str(file)})
   p.save(p.OUT/'records.json',records);yaml=p.OUT/'dataset/data.yaml';yaml.write_text('path: '+str(p.OUT/'dataset')+'\ntrain: images/train\nval: images/val\nnames:\n  0: pipe_end\n')
   install_cpu_assigner_for_mps();install_checked_affine();overrides={**config,'model':str(p.WEIGHTS),'data':str(yaml),'project':str(p.OUT),'name':'synthetic'};trainer=PipeTrainer(overrides=overrides);trainer._setup_train();assert trainer.model.names=={0:'pipe_end'}
   old_keys=set(trainer.metrics);placeholder,fitness=trainer.validate();assert set(placeholder)==old_keys
   trainer.metrics=placeholder;trainer.fitness=fitness;trainer.best_fitness=0.;trainer.callbacks['on_model_save'].append(on_save)
   for epoch in [8,14,20]:
    trainer.epoch=epoch-1;trainer.csv.write_text('epoch,time,train/box_loss\n'+str(epoch)+',0,0\n');returned=trainer.save_model();assert returned is True
    if returned:trainer.run_callbacks('on_model_save')
    meta=checkpoint_check(epoch);ckpt=torch.load(p.OUT/'checkpoints'/f'epoch{epoch:03d}.pt',map_location='cpu',weights_only=False);assert ckpt['epoch']==epoch-1;reports.append({'epoch':epoch,'internalEpoch':ckpt['epoch'],'sha256':meta['sha256'],'names':ckpt['ema'].names})
   before=p.sha(trainer.last);state=trainer.ema.ema.state_dict();key=next(k for k,v in state.items() if v.is_floating_point());old=state[key].clone()
   with torch.no_grad():state[key].reshape(-1)[0]=float('inf')
   try:trainer.save_model()
   except AssertionError:pass
   else:raise AssertionError('Nonfinite EMA must fail before serializer repair')
   with torch.no_grad():state[key].copy_(old)
   assert p.sha(trainer.last)==before and len(trainer.optimizer.state)==0
  finally:
   if trainer:
    for name in ['train_loader','test_loader']:
     loader=getattr(trainer,name,None)
     if loader:loader.close()
   p.OUT=original
 p.save(original/'save-lifecycle-smoke.json',{'status':'PASS_PIPE_REAL_SAVE_CALLBACK_AND_RELOAD_NO_OPTIMIZER','snapshots':reports,'optimizerSteps':0,'syntheticOnly':True,'realDataUsed':False,'officialCOCOInitRebuiltSinglePipeClass':True,'EMAInvalidStateRefusedBeforeSave':True,'placeholderMetricKeysRemainStable':True,'temporarySnapshotsRemoved':True,'operatorSHA256':p.sha(Path(__file__).with_name('train_pipe_ends.py'))});print('PASS pipe save-return/callback/reload lifecycle')

if __name__=='__main__':main()
