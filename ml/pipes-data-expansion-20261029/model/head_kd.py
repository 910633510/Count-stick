"""Training-only head/old-positive direct KL. Teacher is external; no deployment dependency."""
import weakref
import torch
import torch.nn.functional as F
from ultralytics.nn.modules.head import Detect

def head_only(model):
 head=model.model[-1];assert isinstance(head,Detect) and not head.end2end and head.nc==1
 prefix=f'model.{len(model.model)-1}.'
 for name,parameter in model.named_parameters():parameter.requires_grad_(name.startswith(prefix) and '.dfl.' not in name)
 train_with_all_bn_frozen(model)
 return prefix

def train_with_all_bn_frozen(model):
 model.train()
 for module in model.modules():
  if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
 assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))

def raw_dict(prediction):
 if isinstance(prediction,tuple):prediction=prediction[1]
 assert isinstance(prediction,dict) and set(prediction)=={'boxes','scores','feats'}
 return prediction

def direct_kl(student_raw,teacher_raw,reg_max):
 """Per-image class foreground/background balanced Bernoulli KL + weighted DFL KL.

 Classification weights are soft teacher probability p and 1-p, each normalized
 within image. No hard pseudo-label threshold or random feature projection.
 DFL uses the same soft teacher probability over anchors. Only old-positive
 images reach this function. T=1; both KLs are normalized per image then averaged.
 """
 s=raw_dict(student_raw);t=raw_dict(teacher_raw);ss,ts=s['scores'],t['scores'].detach();sb,tb=s['boxes'],t['boxes'].detach()
 assert ss.shape==ts.shape and sb.shape==tb.shape and ss.ndim==3 and ss.shape[1]==1 and sb.shape[1]==4*reg_max
 assert ss.shape[0]>0 and all(torch.isfinite(x).all() for x in [ss,ts,sb,tb])
 prob=ts.sigmoid();eps=1e-6
 class_kl=prob*(F.logsigmoid(ts)-F.logsigmoid(ss))+(1-prob)*(F.logsigmoid(-ts)-F.logsigmoid(-ss))
 foreground=(class_kl*prob).sum((1,2))/prob.sum((1,2)).clamp_min(eps)
 background=(class_kl*(1-prob)).sum((1,2))/(1-prob).sum((1,2)).clamp_min(eps)
 cls=.5*(foreground+background).mean()
 b,_,a=sb.shape;sl=F.log_softmax(sb.reshape(b,4,reg_max,a),dim=2);tl=F.log_softmax(tb.reshape(b,4,reg_max,a),dim=2)
 dfl_kl=(tl.exp()*(tl-sl)).sum(2).mean(1);weight=prob.squeeze(1);dfl=((dfl_kl*weight).sum(1)/weight.sum(1).clamp_min(eps)).mean()
 assert torch.isfinite(cls) and torch.isfinite(dfl)
 # Small signed roundoff is reported, never hidden through a loss clamp.
 return cls,dfl,{'teacherProbabilityMass':prob.sum((1,2)).detach(),'classForegroundKL':foreground.detach(),'classBackgroundKL':background.detach()}

class OldPositiveKDLoss:
 """Plain callable, not nn.Module. Teacher is external, weakly referenced."""
 def __init__(self,supervised,teacher,reg_max,allowed_old_positive_ids,lambda_cls=8.,lambda_dfl=1.):
  self.supervised=supervised;self.teacher_ref=weakref.ref(teacher);self.reg_max=reg_max;self.lambda_cls=lambda_cls;self.lambda_dfl=lambda_dfl;self.last=None;self.allowed_old_positive_ids=frozenset(allowed_old_positive_ids);self.events=[]
 def __call__(self,prediction,batch):
  s=raw_dict(prediction);loss,parts=self.supervised(s,batch);mask=batch['old_positive_mask'];assert mask.dtype==torch.bool and mask.shape==(batch['img'].shape[0],)
  image_ids=batch['kd_image_ids'];assert len(image_ids)==len(mask);expected=torch.tensor([i in self.allowed_old_positive_ids for i in image_ids],dtype=torch.bool,device=mask.device);assert torch.equal(mask,expected),'Teacher-old mask does not match original train-positive ID set'
  indices=mask.nonzero(as_tuple=False).flatten();teacher=self.teacher_ref();assert teacher is not None and not teacher.training and all(not x.requires_grad for x in teacher.parameters())
  if len(indices):
   with torch.no_grad():target=raw_dict(teacher(batch['img'].index_select(0,indices)))
   subset={'scores':s['scores'].index_select(0,indices),'boxes':s['boxes'].index_select(0,indices),'feats':s['feats']}
   cls,dfl,stats=direct_kl(subset,target,self.reg_max)
  else:cls=s['scores'].sum()*0;dfl=s['boxes'].sum()*0;stats={}
  size=batch['img'].shape[0];result=torch.cat((loss.reshape(-1),(size*self.lambda_cls*cls).reshape(1),(size*self.lambda_dfl*dfl).reshape(1)))
  self.last={'oldPositiveIndices':indices.detach().cpu().tolist(),'clsKL':float(cls.detach()),'dflKL':float(dfl.detach()),'teacherEval':not teacher.training,'oldPositiveImages':len(indices),'statistics':{k:v.cpu().tolist() for k,v in stats.items()}}
  self.last['batchImageIds']=list(image_ids);self.last['actualTeacherImageIds']=[image_ids[i] for i in self.last['oldPositiveIndices']];assert all(i in self.allowed_old_positive_ids for i in self.last['actualTeacherImageIds']);self.events.append(dict(self.last))
  return result,{**parts,'kd_cls':self.lambda_cls*cls.detach(),'kd_dfl':self.lambda_dfl*dfl.detach()}
