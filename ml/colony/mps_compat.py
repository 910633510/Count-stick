"""Keep Ultralytics' unmodified target-matching math on CPU for dense MPS batches.

MPS advanced boolean indexing can produce mismatched selection lengths in tal.py.
The assigner is no-grad by design, so its inputs/outputs can move without changing
network forward/backward gradients. This patch is process-local, not a library edit.
"""
import torch
from ultralytics.utils.tal import TaskAlignedAssigner
_ORIGINAL_FORWARD=TaskAlignedAssigner.forward

def install_cpu_assigner_for_mps():
    if getattr(TaskAlignedAssigner.forward,'_countscope_cpu_assignment',False):return
    @torch.no_grad()
    def forward(self,*args,**kwargs):
        device=args[0].device
        if device.type!='mps':return _ORIGINAL_FORWARD(self,*args,**kwargs)
        torch.mps.synchronize()
        cpu_args=tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in args)
        cpu_kwargs={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
        result=_ORIGINAL_FORWARD(self,*cpu_args,**cpu_kwargs)
        return tuple(x.to(device) if isinstance(x,torch.Tensor) else x for x in result)
    forward._countscope_cpu_assignment=True
    TaskAlignedAssigner.forward=forward

def check_equivalence():
    torch.manual_seed(123)
    scores=torch.rand(2,16,1)
    xy=torch.rand(2,16,2)*10
    boxes=torch.cat((xy,xy+2),dim=-1)
    anchors=torch.rand(16,2)*10
    labels=torch.zeros(2,3,1)
    gt=torch.tensor([[[1.,1.,6.,6.],[4.,4.,9.,9.],[0.,0.,3.,3.]]]*2)
    mask=torch.ones(2,3,1)
    inputs=(scores,boxes,anchors,labels,gt,mask)
    assigner=TaskAlignedAssigner(topk=3,num_classes=1)
    expected=_ORIGINAL_FORWARD(assigner,*inputs)
    install_cpu_assigner_for_mps()
    actual=assigner(*(x.to('mps') for x in inputs))
    assert len(actual)==len(expected)
    for x,y in zip(expected,actual):assert torch.equal(x,y.cpu())
    print('CPU assignment/MPS transfer equality PASS',flush=True)
if __name__=='__main__':check_equivalence()
