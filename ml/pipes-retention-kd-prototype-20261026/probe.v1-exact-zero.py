"""One synthetic forward/backward per device and official save lifecycle. No optimizer step."""
from pathlib import Path
import copy,hashlib,inspect,json,sys
from types import SimpleNamespace
import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import ModelEMA
from ultralytics.engine.trainer import BaseTrainer
from head_kd import head_only,train_with_all_bn_frozen,raw_dict,direct_kl,OldPositiveKDLoss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/pipes-retention-kd-prototype-20261026';ANCHOR=ROOT/'output/pipes-20261026/model/checkpoints/epoch020.pt';SHA='48a77fd9198b23695f470b172397e918c9113238bfef2d421d8bbd431f69af12'
sys.path.insert(0,str(ROOT/'ml/colony'));from mps_compat import install_cpu_assigner_for_mps

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert digest(ANCHOR)==SHA;assert not (OUT/'probe-report.json').exists();OUT.mkdir(parents=True,exist_ok=True);install_cpu_assigner_for_mps();torch.set_num_threads(4);torch.manual_seed(20261026)
 results={};saved=None
 for device in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
  teacher=YOLO(str(ANCHOR)).model.float().to(device).eval()
  for parameter in teacher.parameters():parameter.requires_grad_(False)
  student=copy.deepcopy(teacher);prefix=head_only(student);student.args=SimpleNamespace(**student.args) if isinstance(student.args,dict) else student.args
  frozen={k:v.detach().cpu().clone() for k,v in student.state_dict().items() if not k.startswith(prefix) or 'running_' in k or 'num_batches_tracked' in k or '.dfl.' in k}
  # EMA is initialized before attaching the training-only custom criterion.
  ema=ModelEMA(student);assert getattr(ema.ema,'criterion',None) is None
  params=[p for p in student.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.0001);assert not {id(p) for p in teacher.parameters()} & {id(p) for g in optimizer.param_groups for p in g['params']}
  image=torch.rand(4,3,128,128,device=device);before={k:v.detach().cpu().clone() for k,v in teacher.state_dict().items()}
  raw=raw_dict(student(image));
  with torch.no_grad():target=raw_dict(teacher(image))
  initial={k:float((raw[k]-target[k]).abs().max()) for k in ['scores','boxes']};assert max(initial.values())==0
  cls,dfl,_=direct_kl(raw,target,student.model[-1].reg_max);assert float(cls)==float(dfl)==0
  # A mathematical perturbation of logits, not a parameter update/candidate.
  ps=raw['scores'].detach().clone();pb=raw['boxes'].detach().clone();ps[0:2]+=.25;pb[0:2,0:16:2,:]+=.25;ps.requires_grad_();pb.requires_grad_();subset={'scores':ps[:2],'boxes':pb[:2],'feats':[]};tt={'scores':target['scores'][:2],'boxes':target['boxes'][:2],'feats':[]};kc,kd,_=direct_kl(subset,tt,student.model[-1].reg_max);assert kc>0 and kd>0;(kc+kd).backward();assert torch.isfinite(ps.grad).all() and torch.isfinite(pb.grad).all() and torch.count_nonzero(ps.grad[2:])==0 and torch.count_nonzero(pb.grad[2:])==0
  batch={'img':image,'batch_idx':torch.tensor([0.,1.,2.],device=device),'cls':torch.zeros(3,1,device=device),'bboxes':torch.tensor([[.5,.5,.3,.3],[.3,.4,.2,.2],[.6,.6,.2,.3]],device=device),'old_positive_mask':torch.tensor([True,True,False,False],device=device)}
  criterion=OldPositiveKDLoss(v8DetectionLoss(student),teacher,student.model[-1].reg_max,1.,1.);student.criterion=criterion;train_with_all_bn_frozen(student);assert not teacher.training
  loss,items=criterion(raw,batch);assert torch.isfinite(loss).all();loss.sum().backward();trainable_grads=[p.grad for p in params if p.grad is not None];assert trainable_grads and all(torch.isfinite(g).all() for g in trainable_grads);assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in student.parameters() if not p.requires_grad)
  assert all(torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in frozen.items());assert all(torch.equal(v,teacher.state_dict()[k].detach().cpu()) for k,v in before.items());assert not any('teacher' in k for k in student.state_dict())
  ema.update(student);ema.update_attr(student,include=['yaml','nc','args','names','stride','class_weights']);assert getattr(ema.ema,'criterion',None) is None
  results[device]={'status':'PASS_SYNTHETIC_FORWARD_BACKWARD_NO_OPTIMIZER_STEP','headPrefix':prefix,'trainableParameters':sum(p.numel() for p in params),'totalParameters':sum(p.numel() for p in student.parameters()),'rawShapes':{k:list(raw[k].shape) for k in ['scores','boxes']},'initialTeacherStudentRawMaximumDifference':initial,'initialDirectKL':[float(cls),float(dfl)],'perturbedKL':[float(kc.detach()),float(kd.detach())],'newPositiveNegativeKDGradientExactlyZero':True,'teacherNoGradAndStateUnchanged':True,'backboneNeckAndAllBNStateUnchanged':True,'allTrainableGradientsFinite':True,'teacherNotRegisteredInStudent':True,'emaHasNoCustomCriterion':True,'criterionObservation':criterion.last,'optimizerSteps':0,'supervisedAndKDTerms':{k:float(v) for k,v in items.items()}}
  if device=='cpu':
   fake=object.__new__(BaseTrainer);fake.model=student;fake.ema=ema;fake.optimizer=optimizer;fake.scaler=torch.amp.GradScaler('cpu',enabled=False);fake.args=SimpleNamespace(epochs=12);fake.epoch=0;fake.best_fitness=fake.fitness=0.;fake.metrics={};fake.read_results_csv=lambda:{};fake.wdir=OUT/'save-smoke';fake.last=fake.wdir/'last.pt';fake.best=fake.wdir/'best.pt';fake.save_period=-1
   assert fake.save_model() is True;checkpoint=torch.load(fake.last,map_location='cpu',weights_only=False);network=checkpoint['ema'];assert getattr(network,'criterion',None) is None and not any('teacher' in k for k in network.state_dict()) and checkpoint['optimizer']['state']=={}
   saved={'path':str(fake.last.relative_to(ROOT)),'sha256':digest(fake.last),'standardModelType':type(network).__name__,'noTeacherModule':not any('teacher' in name for name,_ in network.named_modules()),'criterionIsNone':network.criterion is None,'optimizerStateEmpty':True,'saveReturnedTrue':True}
  del raw,target,teacher,student,ema,optimizer,criterion
  if device=='mps':torch.mps.empty_cache()
 report={'status':'PASS_HEAD_ONLY_OLD_POSITIVE_DIRECT_KD_PROTOTYPE','sourceSHA256':{f.name:digest(f) for f in [Path(__file__),Path(__file__).with_name('head_kd.py')]},'initializationSHA256':SHA,'devices':results,'saveLifecycle':saved,'realDataRead':False,'old37OrFresh20Read':False,'trainingRunStarted':False,'optimizerSteps':0,'limits':['Synthetic128 pixels only; no evidence of1024 accuracy or nonregression.','No real teacher-quality or KD coefficient calibration performed.','No downscale augmentation adopted; await development-cache diagnosis.']};(OUT/'probe-report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],digest(OUT/'probe-report.json'))
if __name__=='__main__':main()
