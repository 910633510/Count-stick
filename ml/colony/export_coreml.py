#!/usr/bin/env python3
"""Export the evaluated candidate separately; compare real validation photos on both backends."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
import argparse,hashlib,json,shutil,time
import numpy as np
import torch
from ultralytics import YOLO
DATA=ROOT/'output/datasets/colony-figshare-22022540-v3/yolo-species-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--quantize',type=int,default=16);ap.add_argument('--check-existing',action='store_true');args=ap.parse_args();run=args.run.resolve()
    selection=json.loads((run/'evaluation/selection-before-test.json').read_text());report=json.loads((run/'evaluation/test-report.json').read_text());weights=run/'weights/best.pt';assert sha(weights)==selection['modelSHA256']
    out=ROOT/'output/models'/('CountScope-Colony-v1-fp'+str(args.quantize));out.mkdir(parents=True,exist_ok=True)
    target=out/'CountScopeColony1024.pt'
    if target.exists():assert sha(target)==sha(weights)
    else:shutil.copy2(weights,target)
    package=target.with_suffix('.mlpackage')
    assert args.check_existing or not package.exists(),'Preserve an existing export; select another precision or a new destination'
    torch.set_num_threads(8);model=YOLO(str(target));conf=selection['selectedConfidence']
    result=str(package) if args.check_existing else model.export(format='coreml',imgsz=1024,batch=1,quantize=args.quantize,nms=True,conf=conf,iou=0.7,max_det=1000,device='cpu')
    package=Path(result);assert package.exists()
    import coremltools as ct
    core=ct.models.MLModel(str(package));spec=core.get_spec()
    converted=YOLO(str(package),task='detect')
    manifest=json.loads((DATA/'manifest.json').read_text());files=manifest['splits']['val']
    ordered=sorted(files,key=lambda n:len((DATA/'labels/val'/(Path(n).stem+'.txt')).read_text().splitlines()))
    samples=[ordered[0],ordered[len(ordered)//2],ordered[-1]];pairs=[]
    for name in samples:
        p=DATA/'images/val'/name;kwargs=dict(imgsz=1024,rect=False,conf=conf,iou=0.7,max_det=1000,verbose=False)
        a=model.predict(str(p),device='cpu',**kwargs)[0];b=converted.predict(str(p),**kwargs)[0]
        pairs.append({'image':name,'torchCount':len(a.boxes),'coreMLCount':len(b.boxes),'countDifference':len(b.boxes)-len(a.boxes)})
        print('Runtime parity',pairs[-1],flush=True)
    tiled_pairs=[]
    if selection.get('selectedPipeline')=='automatic4tile':
        from evaluate_pipeline import predict_tiled
        a=predict_tiled(model,samples,'val',out/'torch-tile-parity-predictions.json')
        class SingleImageCoreML:
            def predict(self,sources,**kwargs):
                kwargs.pop('stream',None)
                for source in sources:
                    yield converted.predict(source,batch=1,stream=False,**kwargs)[0]
        b=predict_tiled(SingleImageCoreML(),samples,'val',out/'coreml-tile-parity-predictions.json')
        for name in samples:
            ca=sum(v>=conf for v in a[name]['confidence']);cb=sum(v>=conf for v in b[name]['confidence']);tiled_pairs.append({'image':name,'torchCount':ca,'coreMLCount':cb,'countDifference':cb-ca})
        print('Tiled count parity',tiled_pairs,flush=True)
    receipt={'status':'EXPORTED_WITH_RUNTIME_CHECK','createdAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sourceModelSHA256':sha(weights),'package':str(package),'quantize':args.quantize,'inputSize':1024,'preprocessing':'RGB YOLO fixed-square letterbox to1024, background114, scale1/255; port and verify this preprocessing before any app integration.','confidence':conf,'nmsIoU':0.7,'specificationVersion':spec.specificationVersion,'inputs':[i.name for i in spec.description.input],'outputs':[i.name for i in spec.description.output],'validationPhotoParity':pairs,'inferencePipeline':selection.get('selectedPipeline','whole'),'tiledValidationPhotoParity':tiled_pairs,'testMetricsBackend':'PyTorch CPU with the selected inference pipeline; CoreML only smoke-compared on three validation photos.','exactCountParity':all(p['countDifference']==0 for p in pairs+tiled_pairs),'testMetrics':report['summary'],'shippingAppModified':False,'limitations':'Mac CoreML smoke comparison only; not iPhone runtime/accuracy acceptance.'}
    receipt['packageFilesSHA256']={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()}
    (out/'export-report.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
