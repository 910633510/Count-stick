"""Full trainer wiring test with temporary synthetic files and zero optimizer steps."""
import copy,tempfile,torch
from pathlib import Path
from PIL import Image
import train_retention as m

def main():
 original=m.OUT
 with tempfile.TemporaryDirectory(prefix='countscope-retention-wiring-') as temp:
  m.OUT=Path(temp);records=[]
  try:
   for split in ['train','val']:
    for j,bucket in enumerate(['old_anchor','hard_positive','empty_negative']):
     p=m.OUT/'dataset/images'/split/(bucket+'.png');label=m.OUT/'dataset/labels'/split/(bucket+'.txt');p.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
     Image.new('RGB',(256,256),(40+j*60,)*3).save(p);label.write_text('0 .5 .5 .1 .1\n' if bucket!='empty_negative' else '')
     records.append({'id':split+'_'+bucket,'split':split,'bucket':bucket,'source':bucket,'species':bucket,'groupId':split+'_'+bucket,'count':int(bucket!='empty_negative'),'preparedImage':str(p),'preparedLabel':str(label)})
   m.save(m.OUT/'records.json',records);yaml=m.OUT/'dataset/data.yaml';yaml.write_text('path: '+str(m.OUT/'dataset')+'\ntrain: images/train\nval: images/val\nnames:\n  0: colony\n')
   config=copy.deepcopy(m.read(original/'protocol.json')['trainingConfig']);config.update({'model':str(m.ANCHOR),'data':str(yaml),'project':str(m.OUT),'name':'synthetic-only','imgsz':256,'plots':False,'verbose':False})
   m.install_cpu_assigner_for_mps();trainer=m.RetentionTrainer(overrides=config);trainer._setup_train();trainer.epoch=0
   assert len(trainer._ordered)==3 and isinstance(trainer.train_loader.sampler,m.RetentionSampler)
   expected_teacher=[];batch=next(iter(trainer.train_loader));prepared=trainer.preprocess_batch(batch)
   for path in prepared['im_file']:
    row=trainer._train_by_path[str(Path(path).resolve())]
    if row['bucket']=='old_anchor':expected_teacher.append(row['id'])
   loss,items=trainer.model(prepared);assert torch.isfinite(loss).all();loss.sum().backward()
   assert trainer.model.criterion.teacher_consumed==expected_teacher
   assert len(trainer.optimizer.state)==0,'No optimizer step is permitted in this smoke'
   assert all(p.grad is None for p in trainer._teacher.parameters())
   assert all(torch.equal(trainer.model.state_dict()[k],value) for k,value in trainer._anchor_frozen.items())
   assert not hasattr(trainer.ema.ema,'criterion') or trainer.ema.ema.criterion is None
   report={'status':'PASS_FULL_TRAINER_WIRING_NO_OPTIMIZER_STEP','actualLoaderSampler':type(trainer.train_loader.sampler).__name__,'actualBatchImages':len(prepared['im_file']),'teacherOldOnlyBatchImages':len(expected_teacher),'lossItems':{k:float(v) for k,v in items.items()},'optimizerStateEntries':0,'frozenParametersAndBuffersUnchanged':True,'teacherNotInEMA':True,'operatorSHA256':m.sha(m.__file__),'syntheticImagesOnly':True,'realTrainValTestDataUsed':False}
   for loader in [trainer.train_loader,trainer.test_loader]:loader.close()
  finally:m.OUT=original
 m.save(original/'trainer-wiring-smoke.json',report);print(report)

if __name__=='__main__':main()
