"""Runtime checks using synthetic tensors only; never takes an optimizer step."""
import collections,tempfile
import torch
import train_retention as m
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import ModelEMA

class IdDataset(torch.utils.data.Dataset):
 def __init__(self,n):self.n=n
 def __len__(self):return self.n
 def __getitem__(self,i):return i

records=[]
for bucket,source,species,group,n in [('old_anchor','old','a','a1',1),('old_anchor','old','a','a2',8),('old_anchor','old','b','b1',8),('hard_positive','new','c','c1',1),('hard_positive','new','c','c2',9),('hard_positive','new','d','d1',11),('hard_positive',m.CLINICAL_SOURCE,'e','e1',1),('hard_positive',m.CLINICAL_SOURCE,'e','e2',4),('empty_negative','empty','empty','plate1',1),('empty_negative','empty','empty','plate2',12)]:records.extend([dict(bucket=bucket,source=source,species=species,groupId=group) for _ in range(n)])
sampler=m.RetentionSampler(records,100000);loader=m.InfiniteDataLoader(IdDataset(len(records)),batch_size=64,sampler=sampler,num_workers=0,shuffle=False);seen=[]
for batch in loader:seen.extend(batch.tolist())
loader.close();assert seen==sampler.generated[:len(seen)]
buckets=collections.Counter(records[i]['bucket'] for i in seen);assert all(abs(buckets[k]/len(seen)-v)<.01 for k,v in m.BUCKET_MASS.items());clinical=sum(records[i]['source']==m.CLINICAL_SOURCE for i in seen);assert abs(clinical/len(seen)-.05)<.01
physical=collections.Counter(records[i]['groupId'] for i in seen)
expected_group_mass={'a1':.1625,'a2':.1625,'b1':.325,'c1':.05,'c2':.05,'d1':.1,'e1':.025,'e2':.025,'plate1':.05,'plate2':.05}
assert all(abs(physical[g]/len(seen)-prob)<.01 for g,prob in expected_group_mass.items())
m.install_cpu_assigner_for_mps();device='mps' if torch.backends.mps.is_available() else 'cpu';torch.manual_seed(123)
t={'scores':torch.zeros(2,1,64,device=device),'boxes':torch.randn(2,64,64,device=device)}
s={'scores':(t['scores']+.5).requires_grad_(),'boxes':(t['boxes']+torch.randn_like(t['boxes'])*.1).requires_grad_()}
cls,box=m.retention_terms(s,t);assert cls.item()>0 and box.item()>0;(cls+box).backward();assert s['scores'].grad.abs().sum().item()>0 and s['boxes'].grad.abs().sum().item()>0
same=m.retention_terms(t,t);assert all(abs(x.item())<1e-6 for x in same)
student=m.YOLO(str(m.ANCHOR)).model.float().to(device).train();teacher=m.YOLO(str(m.ANCHOR)).model.float().to(device).eval()
for p in teacher.parameters():p.requires_grad_(False)
for n,p in student.named_parameters():
 # The real trainer explicitly re-enables unfrozen loaded inference parameters.
 p.requires_grad_(not(any(n.startswith(f'model.{i}.') for i in range(11)) or '.dfl.' in n))
m.freeze_bn(student);student.args=get_cfg();before={n:v.clone() for n,v in student.named_buffers()};ema=ModelEMA(student)
paths=['/synthetic/old','/synthetic/new','/synthetic/empty'];rows={p:dict(split='train',bucket=b,id=p) for p,b in zip(paths,['old_anchor','hard_positive','empty_negative'])}
criterion=m.AnchorCriterion(student,teacher,rows);student.criterion=criterion
assert not hasattr(ema.ema,'criterion') or ema.ema.criterion is None
batch={'img':torch.zeros(3,3,256,256,device=device),'im_file':paths,'batch_idx':torch.empty(0,device=device),'cls':torch.empty(0,1,device=device),'bboxes':torch.empty(0,4,device=device)}
pred=m.raw(student(batch['img']));pred['scores']=pred['scores']+4.;loss,items=criterion(pred,batch);assert torch.isfinite(loss).all() and items['anchor_cls'].item()>0
loss.sum().backward();assert criterion.teacher_consumed==['/synthetic/old'];assert all(p.grad is None for p in teacher.parameters());assert all(p.grad is None for p in student.parameters() if not p.requires_grad);assert any(p.grad is not None and p.grad.abs().sum().item()>0 for p in student.parameters() if p.requires_grad);assert all(torch.equal(v,dict(student.named_buffers())[n]) for n,v in before.items())
with tempfile.NamedTemporaryFile(suffix='.pt') as file:
 torch.save({'model':ema.ema.cpu(),'ema':None,'train_args':{},'optimizer':None},file.name)
 loaded=m.YOLO(file.name).model;assert not hasattr(loaded,'criterion') or loaded.criterion is None
anchor=m.YOLO(str(m.ANCHOR)).model.float().cpu();modified=m.YOLO(str(m.ANCHOR)).model.float().cpu();parameters=dict(anchor.named_parameters());frozen=[k for k in anchor.state_dict() if k not in parameters or k.startswith('model.0.')]
name=next(n for n,p in modified.named_parameters() if n not in frozen and p.numel()>1)
with torch.no_grad():dict(modified.named_parameters())[name].add_(.01)
a0=m.blend_model(m.YOLO(str(m.ANCHOR)).model,modified,0,frozen);assert all(torch.equal(v,a0.state_dict()[k]) for k,v in anchor.state_dict().items())
a=m.blend_model(m.YOLO(str(m.ANCHOR)).model,modified,.15,frozen);assert torch.equal(a.state_dict()[name],anchor.state_dict()[name]*.85+modified.state_dict()[name]*.15)
try:
 bad=m.YOLO(str(m.ANCHOR)).model.float();bad.state_dict()[frozen[0]].add_(.01);m.blend_model(m.YOLO(str(m.ANCHOR)).model,bad,.15,frozen)
except AssertionError:pass
else:raise AssertionError('Frozen drift not rejected')
report={'status':'PASS_NO_OPTIMIZER_STEPS','device':device,'actualDataloaderSamples':len(seen),'bucketCounts':dict(buckets),'clinicalSamples':clinical,'samplerPrefixMatch':True,'KDIdenticalPredictionLossZero':True,'KDClassAndDFLGradientsNonzero':True,'actualModelCriterionFinite':True,'teacherOnlySyntheticOldTrainRows':True,'teacherNoGradient':True,'frozenBackboneBNNoGradientOrBufferChange':True,'EMACheckpointHasNoTeacherOrCustomCriterion':True,'alphaZeroExactlyAnchor':True,'alpha15ConvexMathExact':True,'frozenDriftRejected':True,'operatorSHA256':m.sha(m.__file__),'optimizerSteps':0,'realDataReadForTraining':False,'persistedTrainedWeights':False,'firstSmokeCorrection':'Loaded YOLO inference parameters start requires_grad=False; test now mirrors trainer re-enabling of trainable neck/head. No real training attempt occurred.'}
report.update({'physicalGroupCounts':dict(physical),'expectedPhysicalGroupProbability':expected_group_mass,'unequalRepeatedViewGroupBalanceVerified':True,'preTrainingSamplerRevision':2})
m.save(m.OUT/'smoke.json',report);print(report)
