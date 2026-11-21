#!/usr/bin/env python3
"""Frozen class-balanced LOCA zero-shot screen; no exemplar boxes or text enter inference."""
from pathlib import Path
import os,sys,json,hashlib,time,resource,types,collections
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/loca-20261025';DATA=ROOT/'output/datasets/fsc147'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T

SEED_STRING='CountScope-LOCA-zero-shot-validation-screen-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def classes():return dict(line.split('\t',1) for line in (DATA/'annotations/ImageClasses_FSC147.txt').read_text().splitlines() if '\t' in line)

def freeze():
    clean=json.loads((ROOT/'output/training-fsc147-20261025/evaluation-split-before-test.json').read_text())['validation']
    label=classes();known={'7171.jpg':'Wrong exemplar boxes noted in CountGD NeurIPS2024 section4.1','7611.jpg':'Text should be yellow lego stud, not lego; CountGD NeurIPS2024 section4.1'}
    ann=json.loads((DATA/'annotations/annotation_FSC147_384.json').read_text());byclass=collections.defaultdict(list)
    for name in clean:
        if name not in known:byclass[label[name]].append(name)
    assert len(byclass)==29
    chosen=[]
    for cls,names in sorted(byclass.items()):
        order=sorted(names,key=lambda n:hashlib.sha256((SEED_STRING+'|'+n).encode()).hexdigest())
        assert len(order)>=2
        for name in order[:2]:chosen.append({'image':name,'class':cls,'truth':len(ann[name]['points']),'imageSHA256':sha(DATA/'extracted/images_384_VarV2'/name)})
    baseline_path=ROOT/'output/training-fsc147-20261025/baseline-validation-clean.json';base={r['image']:r for r in json.loads(baseline_path.read_text())['images']}
    assert all(r['image'] in base for r in chosen)
    for row in chosen:
        old=base[row['image']];assert old['truth']==row['truth'];row['famnetBaselinePrediction']=old['predicted']
    p=OUT/'protocol.json'
    original=json.loads((DATA/'annotations/Train_Test_Val_FSC_147.json').read_text())
    record={'frozenBeforeInferenceUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'purpose':'Feasibility screen on already-used validation, NOT a new independent test','selection':'Two per29clean validation classes; lowest SHA256(seed|filename), no count/error-based sampling.','seedString':SEED_STRING,'sampleCount':58,'images':chosen,'knownPaperCorrections':{n:{'issue':why,'officialSplits':[k for k in ['train','val','test'] if n in original[k]],'excludedIfInCleanVal':n in clean} for n,why in known.items()},'knownIssueSource':'https://proceedings.neurips.cc/paper_files/paper/2024/file/57c56985d9afe89bf78a8264c91071aa-Paper-Conference.pdf#page=6','input':'RGB image only. bboxes=None. No class text, GT exemplars, density maps or point coordinates passed to model.','groundTruthUse':'Only len(points) after/for fixed sample manifest and metric computation.','preprocessing':'Author validation pipeline: RGB tensor, resize to512x512, ImageNet mean/std. Explicit antialias=False matches torchvision0.12 tensor-resize default.','densityBuckets':{'sparse':'1-20','medium':'21-100','dense':'101-500','veryDense':'>500'},'baselineSourceSHA256':sha(baseline_path),'comparisonCaveat':'Cached FamNet is reference-conditioned; LOCA is not. This is a practical no-box comparison, not a same-input paper leaderboard.','deviceRule':'CPU pilot first2fixed samples; MPS only if both counts differ by at most max(0.05,0.1percent CPUcount); otherwise allCPU.','noTraining':True,'noOldTestRun':True,'source':json.loads((OUT/'source-receipt.json').read_text()),'checkpoint':json.loads((OUT/'checkpoint-receipt.json').read_text())}
    if p.exists():
        previous=json.loads(p.read_text());assert previous['images']==record['images'];return previous
    p.write_text(json.dumps(record,indent=2)+'\n');return record

def build():
    source=Path(json.loads((OUT/'source-receipt.json').read_text())['sourceDirectory']);sys.path.insert(0,str(source))
    from models import loca as lm
    from models import ope as om
    from utils.arg_parser import get_argparser
    original=lm.Backbone
    # Initialization-only adaptation: full author checkpoint is loaded strictly below.
    def backbone_no_download(name,pretrained,dilation,reduction,swav,requires_grad):
        return original(name,pretrained=False,dilation=dilation,reduction=reduction,swav=False,requires_grad=False)
    lm.Backbone=backbone_no_download
    def reject_roi_align(*args,**kwargs):raise AssertionError('Zero-shot graph attempted ROIAlign: no exemplars may be used')
    om.roi_align=reject_roi_align
    args=get_argparser().parse_args(['--backbone','resnet50','--swav_backbone','--reduction','8','--image_size','512','--num_enc_layers','3','--num_ope_iterative_steps','3','--emb_dim','256','--num_heads','8','--kernel_dim','3','--num_objects','3','--pre_norm','--zero_shot'])
    model=lm.build_model(args).eval().requires_grad_(False)
    checkpoint_path=OUT/'checkpoints/loca_zero_shot.pt';record=json.loads((OUT/'checkpoint-receipt.json').read_text());assert sha(checkpoint_path)==record['sha256']
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=True);state={k.removeprefix('module.'):v for k,v in ck['model'].items()}
    result=model.load_state_dict(state,strict=True);assert not result.missing_keys and not result.unexpected_keys
    assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
    return model,{'totalParameterElements':sum(p.numel() for p in model.parameters()),'totalStateElementsIncludingBuffers':sum(v.numel() for v in state.values()),'stateTensorCount':len(state),'strictCheckpointLoad':True,'allLoadedValuesEqualCheckpoint':True,'zeroShot':bool(model.zero_shot),'learnedPrototypeSlots':model.num_objects,'exemplarsProvided':0,'roiAlignGuardInstalled':True,'sourceArchitectureUnchanged':True,'adaptation':'Bypass CUDA/DDP runner; avoid redundant ImageNet/SwAV initialization downloads because complete checkpoint restores every tensor; useCPU/MPS device.'}

TRANSFORM=T.Compose([T.ToTensor(),T.Resize((512,512),antialias=False),T.Normalize([.485,.456,.406],[.229,.224,.225])])
def tensor(name):
    with Image.open(DATA/'extracted/images_384_VarV2'/name) as im:return TRANSFORM(im.convert('RGB')).unsqueeze(0)

@torch.no_grad()
def infer(model,x,device):
    start=time.perf_counter();out,aux=model(x.to(device),None);count=float(out.sum().cpu())
    assert np.isfinite(count) and bool(torch.isfinite(out).all())
    return count,time.perf_counter()-start

def metric(rows,key):
    errors=np.asarray([r[key]-r['truth'] for r in rows]);return {'images':len(rows),'MAE':float(np.abs(errors).mean()),'RMSE':float(np.sqrt(np.mean(errors**2))),'bias':float(errors.mean()),'roundedExactCountImages':sum(round(r[key])==r['truth'] for r in rows)}
def main():
    torch.set_num_threads(8);torch.manual_seed(20261025);protocol=freeze();assert not (OUT/'screen-result.json').exists(),'Existing result is immutable; create a new preregistered run'
    model,info=build();pilots=[];cpu_predictions={};mps_ok=torch.backends.mps.is_available();pilot_start=time.perf_counter()
    for row in protocol['images'][:2]:
        pred,seconds=infer(model,tensor(row['image']),'cpu');cpu_predictions[row['image']]=pred;pilots.append({'image':row['image'],'cpuCount':pred,'cpuSeconds':seconds})
    if mps_ok:
        try:
            model=model.to('mps')
            for pilot,row in zip(pilots,protocol['images'][:2]):
                pred,seconds=infer(model,tensor(row['image']),'mps');diff=abs(pred-pilot['cpuCount']);pilot.update(mpsCount=pred,mpsSeconds=seconds,absoluteDifference=diff,tolerance=max(.05,.001*abs(pilot['cpuCount'])))
                mps_ok &= diff<=pilot['tolerance']
        except Exception as e:mps_ok=False;info['mpsPilotError']=type(e).__name__+': '+str(e)
    device='mps' if mps_ok else 'cpu';model=model.to(device);(OUT/'backend-pilot.json').write_text(json.dumps({'chosenDevice':device,'pilots':pilots,'elapsedSeconds':time.perf_counter()-pilot_start,'model':info},indent=2)+'\n');print('Backend pilot',device,pilots,flush=True)
    results=[];start=time.perf_counter();max_mps_allocated=0;max_mps_driver=0
    for idx,row in enumerate(protocol['images']):
        assert sha(DATA/'extracted/images_384_VarV2'/row['image'])==row['imageSHA256']
        pred,seconds=infer(model,tensor(row['image']),device);new={**row,'locaZeroShotPrediction':pred,'inferenceSeconds':seconds};results.append(new)
        if device=='mps':max_mps_allocated=max(max_mps_allocated,torch.mps.current_allocated_memory());max_mps_driver=max(max_mps_driver,torch.mps.driver_allocated_memory())
        if (idx+1)%10==0:print('Screen',idx+1,'/58','MAE',metric(results,'locaZeroShotPrediction')['MAE'],flush=True)
    negative=[]
    for label,value in [('blank-white',255),('blank-gray',127)]:
        x=TRANSFORM(Image.new('RGB',(512,512),(value,value,value))).unsqueeze(0);pred,seconds=infer(model,x,device);negative.append({'syntheticControl':label,'truth':0,'prediction':pred,'seconds':seconds,'scope':'Synthetic solid image only, not real background/empty-plate acceptance.'})
    density={}
    for label,test in [('1-20',lambda n:n<=20),('21-100',lambda n:20<n<=100),('101-500',lambda n:100<n<=500),('>500',lambda n:n>500)]:
        subset=[r for r in results if test(r['truth'])]
        if subset:density[label]={'locaZeroShot':metric(subset,'locaZeroShotPrediction'),'famnetWithExemplars':metric(subset,'famnetBaselinePrediction')}
    result={'status':'COMPLETED_NO_EXEMPLAR_VALIDATION_SCREEN','completedAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocolSHA256':sha(OUT/'protocol.json'),'sourceCommit':protocol['source']['commit'],'weightsSHA256':protocol['checkpoint']['sha256'],'preprocessing':protocol['preprocessing'],'device':device,'model':info,'locaZeroShot':metric(results,'locaZeroShotPrediction'),'famnetWithExemplars':metric(results,'famnetBaselinePrediction'),'densityBuckets':density,'secondsFor58AndNegativeControls':time.perf_counter()-start,'meanImageInferenceSeconds':sum(r['inferenceSeconds'] for r in results)/len(results),'processPeakRSSBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'maxObservedMPSAllocatedBytes':max_mps_allocated,'maxObservedMPSDriverBytes':max_mps_driver,'memoryCaveat':'Process RSS includes checkpoint loading andCPU pilot; MPS values are observed after forwards, not guaranteed transient peaks or iPhone memory.','syntheticNegativeControls':negative,'images':results,'limitations':['Already-used class-balanced validation screen, not independent test or full benchmark.','FamNet comparison has visual exemplars; LOCA receives image only, so input conditions differ.','No automatic semantic selection guarantee in mixed-object images.','No training, no oldtest execution, no CoreML/iPhone conversion or validation, no App integration.']}
    (OUT/'screen-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','device','locaZeroShot','famnetWithExemplars','densityBuckets','meanImageInferenceSeconds','processPeakRSSBytes','maxObservedMPSDriverBytes','syntheticNegativeControls']},indent=2),flush=True)
if __name__=='__main__':main()
