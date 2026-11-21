#!/usr/bin/env python3
"""Public FamNet few-shot counting replication/fine-tuning on anonymous FSC147 data.

Uses the inspected author's model/feature functions, with the original torchvision
0.5 ResNet50 weights. Exemplar boxes are provided by the dataset, NOT automatically
selected objects. This research model is not installed in CountScope.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/datasets/fsc147';AUTHOR=BASE/'author-code';RUN=ROOT/'output/training-fsc147-20261025'
os.environ['TORCH_HOME']=str(RUN/'torch-cache');os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import argparse,hashlib,importlib.util,json,time,random,math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
models=load_module('fsc_author_model',AUTHOR/'model.py')
utils=load_module('fsc_author_utils',AUTHOR/'utils.py')
LEGACY_URL='https://download.pytorch.org/models/resnet50-19c8e357.pth'
CACHE=RUN/'features';CACHE.mkdir(parents=True,exist_ok=True)
ANN=json.loads((BASE/'annotations/annotation_FSC147_384.json').read_text())
ORIGINAL_SPLITS=json.loads((BASE/'annotations/Train_Test_Val_FSC_147.json').read_text())
EXCLUDED={'train':['3471.jpg'],'val':['3477.jpg'],'test':[]}
SPLITS={k:[n for n in ORIGINAL_SPLITS[k] if n not in EXCLUDED[k]] for k in ['train','val','test']}
SEED=20261025

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class FrozenResnet50FPN(nn.Module):
    def __init__(self):
        super().__init__();base=resnet50(weights=None)
        state=torch.hub.load_state_dict_from_url(LEGACY_URL,map_location='cpu',check_hash=True,progress=False)
        base.load_state_dict(state);children=list(base.children())
        self.conv1=nn.Sequential(*children[:4]);self.conv2=children[4];self.conv3=children[5];self.conv4=children[6]
        self.requires_grad_(False)
    def forward(self,x):
        x=self.conv2(self.conv1(x));m3=self.conv3(x);m4=self.conv4(m3);return {'map3':m3,'map4':m4}

def sample(name):
    a=ANN[name];rects=[]
    for b in a['box_examples_coordinates']:
        x1,y1=b[0];x2,y2=b[2];rects.append([y1,x1,y2,x2])
    with Image.open(BASE/'extracted/images_384_VarV2'/name) as im:image=im.convert('RGB')
    return utils.Transform({'image':image,'lines_boxes':rects})

def freeze():
    p=RUN/'protocol.json'
    protocol={'datasetSource':'https://github.com/cvlab-stonybrook/LearningToCountEverything','registeredAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'seed':SEED,'method':'Published FamNet CountRegressor with frozen original ImageNet ResNet50FPN and dataset-provided exemplar boxes (3 to 6); no test-time adaptation.','backboneURL':LEGACY_URL,'regressorInitialSHA256':sha(AUTHOR/'FamNet_Save1.pth'),'sourceSHA256':{n:sha(AUTHOR/n) for n in ['model.py','utils.py','LICENSE']},'annotationSHA256':sha(BASE/'annotations/annotation_FSC147_384.json'),'splits':SPLITS,'excludedDueToDensityPointDisagreement':EXCLUDED,'training':{'epochs':1,'optimizer':'Adam','learningRate':1e-6,'loss':'Original mean-squared density loss; resized targets preserve total mass.','batchSize':1,'featureCache':'Float32; frozen extractor; no image augmentation','selection':'Minimum validation count MAE among initial public baseline and the fine-tuned epoch; test only after selection.'},'testCriteria':'All1190 official test images, unseen29categories, continuous countMAE/RMSE/bias and rounded exact count; reference exemplar boxes are supplied by benchmark.','automaticObjectSelectionImplemented':False,'shippingAppModified':False}
    if p.exists():
        old=json.loads(p.read_text());assert old['annotationSHA256']==protocol['annotationSHA256'] and old['splits']==SPLITS;return old
    p.write_text(json.dumps(protocol,indent=2)+'\n');return protocol

@torch.no_grad()
def features(backbone,name,device):
    p=CACHE/(Path(name).stem+'.npy')
    if p.exists():
        a=np.load(p,allow_pickle=False);assert np.isfinite(a).all();return torch.from_numpy(a).to(device)
    s=sample(name);feat=utils.extract_features(backbone,s['image'].unsqueeze(0).to(device),s['boxes'].unsqueeze(0),utils.MAPS,utils.Scales)
    a=feat.detach().cpu().numpy().astype('float32');assert np.isfinite(a).all();np.save(p,a,allow_pickle=False);return feat

def target(name,shape,device):
    d=np.load(BASE/'extracted/gt_density_map_adaptive_384_VarV2'/(Path(name).stem+'.npy'),allow_pickle=False).astype('float32')
    t=torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(device);mass=t.sum()
    if tuple(t.shape[-2:])!=tuple(shape):t=F.interpolate(t,size=shape,mode='bilinear',align_corners=False);t=t*(mass/t.sum().clamp_min(1e-8))
    return t

def regressor(device):
    r=models.CountRegressor(6,pool='mean');r.load_state_dict(torch.load(AUTHOR/'FamNet_Save1.pth',map_location='cpu',weights_only=True));return r.to(device)

def pilot():
    torch.set_num_threads(8);torch.manual_seed(SEED);freeze();backbone=FrozenResnet50FPN().eval().to('mps');r=regressor('mps').eval()
    names=SPLITS['train'][:3];results=[]
    for name in names:
        start=time.time()
        with torch.no_grad():f=features(backbone,name,'mps');pred=float(r(f).sum().cpu())
        results.append({'image':name,'truth':len(ANN[name]['points']),'prediction':pred,'seconds':time.time()-start,'featureShape':list(f.shape)});print(results[-1],flush=True)
    # Compare the entire extractor/regressor against CPU on the first validation image.
    name=SPLITS['val'][0];s=sample(name)
    with torch.no_grad():
        f=features(backbone,name,'mps');pred_mps=float(r(f).sum().cpu());backbone=backbone.cpu();r=r.cpu();cpu_features=utils.extract_features(backbone,s['image'].unsqueeze(0),s['boxes'].unsqueeze(0),utils.MAPS,utils.Scales);pred_cpu=float(r(cpu_features).sum())
    comparison={'image':name,'cpu':pred_cpu,'mps':pred_mps,'absoluteDifference':abs(pred_cpu-pred_mps)};assert abs(pred_cpu-pred_mps)<=max(.5,.01*abs(pred_cpu)),comparison
    (RUN/'pilot.json').write_text(json.dumps({'status':'PASS_FINITE_FORWARD_AND_CPU_MPS_COMPARISON','samples':results,'backendCheck':comparison},indent=2)+'\n');print('PILOT PASS',comparison,flush=True)

@torch.no_grad()
def validate(r,backbone,names,device,label):
    r.eval();rows=[];start=time.time()
    for idx,name in enumerate(names):
        out=r(features(backbone,name,device));pred=float(out.sum().cpu());assert math.isfinite(pred);truth=len(ANN[name]['points']);rows.append({'image':name,'truth':truth,'predicted':pred,'error':pred-truth})
        if (idx+1)%100==0:
            progress={'stage':label,'completedImages':idx+1,'totalImages':len(names),'runningMAE':sum(abs(x['error']) for x in rows)/len(rows),'elapsedSeconds':time.time()-start};(RUN/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(progress,flush=True)
    errors=np.array([x['error'] for x in rows]);summary={'images':len(rows),'MAE':float(abs(errors).mean()),'RMSE':float(np.sqrt((errors**2).mean())),'bias':float(errors.mean()),'roundedExactCountImages':sum(round(x['predicted'])==x['truth'] for x in rows)};return summary,rows

def train_and_test():
    torch.set_num_threads(8);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);protocol=freeze();assert (RUN/'pilot.json').exists()
    assert not (RUN/'test-report.json').exists(),'Do not rerun a completed test or tune on heldout results'
    device='mps';backbone=FrozenResnet50FPN().eval().to(device);r=regressor(device)
    baseline,base_rows=validate(r,backbone,SPLITS['val'],device,'baseline-val');(RUN/'baseline-validation.json').write_text(json.dumps({'summary':baseline,'images':base_rows},indent=2)+'\n');print('Baseline',baseline,flush=True)
    # One complete fine-tuning epoch over all valid official training images.
    r.train();optimizer=torch.optim.Adam(r.parameters(),lr=1e-6);names=SPLITS['train'][:];random.Random(SEED).shuffle(names);start=time.time();loss_sum=0.;count_error=0.
    for idx,name in enumerate(names):
        feat=features(backbone,name,device).detach().clone();optimizer.zero_grad(set_to_none=True);out=r(feat);gt=target(name,out.shape[-2:],device);loss=F.mse_loss(out,gt);assert torch.isfinite(loss);loss.backward();optimizer.step();loss_sum+=float(loss.detach().cpu());count_error+=abs(float(out.detach().sum().cpu())-len(ANN[name]['points']))
        if (idx+1)%100==0:
            progress={'stage':'fine-tune','completedImages':idx+1,'totalImages':len(names),'meanDensityMSE':loss_sum/(idx+1),'runningTrainCountMAE':count_error/(idx+1),'elapsedSeconds':time.time()-start};(RUN/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(progress,flush=True)
        if (idx+1)%500==0:torch.save({'model':{k:v.detach().cpu() for k,v in r.state_dict().items()},'optimizer':optimizer.state_dict(),'completedImages':idx+1,'order':names},RUN/'last-training-checkpoint.pth')
    final_weights=RUN/'FamNet-finetuned-one-epoch.pth';torch.save({k:v.detach().cpu() for k,v in r.state_dict().items()},final_weights)
    fine,fine_rows=validate(r,backbone,SPLITS['val'],device,'finetuned-val');(RUN/'finetuned-validation.json').write_text(json.dumps({'summary':fine,'images':fine_rows},indent=2)+'\n')
    chosen='finetuned' if fine['MAE']<baseline['MAE'] else 'original-public-baseline';source=final_weights if chosen=='finetuned' else AUTHOR/'FamNet_Save1.pth'
    selection={'frozenBeforeTestUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selected':chosen,'weights':str(source),'weightsSHA256':sha(source),'baselineValidation':baseline,'finetunedValidation':fine,'trainImages':len(names),'trainSteps':len(names),'fineTunedEpochs':1,'trainSeconds':time.time()-start,'protocolSHA256':sha(RUN/'protocol.json')};(RUN/'selection-before-test.json').write_text(json.dumps(selection,indent=2)+'\n');print('Frozen selection',selection,flush=True)
    r.load_state_dict(torch.load(source,map_location='cpu',weights_only=True));test,rows=validate(r,backbone,SPLITS['test'],device,'test');test.update({'state':'COMPLETE_OFFICIAL_UNSEEN_CATEGORY_TEST','method':'Few-shot FamNet using dataset-provided exemplar boxes; not fully automatic object selection.','selected':chosen,'modelSHA256':sha(source),'testUsedForRetraining':False});(RUN/'test-report.json').write_text(json.dumps({'summary':test,'images':rows},indent=2)+'\n');print('DONE',test,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','train']);a=p.parse_args();pilot() if a.phase=='pilot' else train_and_test()
