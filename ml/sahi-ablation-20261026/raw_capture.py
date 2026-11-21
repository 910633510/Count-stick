"""Resumable raw-capture stage. Real inference requires a root hash receipt.

Full postprocessor-grid/scoring orchestration is intentionally not complete;
this work was suspended when the user prioritized a separate pipe-end model.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,math,time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/sahi-ablation-20261026'
PARENT=ROOT/'output/research-20261025-hardcases/retention-savefix-v2'
ANCHOR=ROOT/'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/weights/best.pt'
ANCHOR_SHA='91d70c3067f2fdb77d2ba2541e6105bf3caf730da7bbc32e5f4d2c9310b0b337'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)

def prepare():
 assert sha(ANCHOR)==ANCHOR_SHA
 rows=[r for r in read(PARENT/'records.json') if r['split']=='val'];assert len(rows)==102
 save(OUT/'records.json',rows)
 for r in rows:assert sha(r['preparedImage'])==r['sha256']
 save(OUT/'data-binding.json',{'parentRecordsSHA256':sha(PARENT/'records.json'),'recordsSHA256':sha(OUT/'records.json'),'baselineCachePath':str(PARENT/'predictions/baseline.json'),'baselineCacheSHA256':sha(PARENT/'predictions/baseline.json'),'images':102,'modelSHA256':ANCHOR_SHA,'old63OrSealed36Included':False})

def capture(receipt_path):
 approval=read(receipt_path)
 assert approval['status']=='APPROVED_FOR_RAW_INFERENCE'
 assert approval['operatorSHA256']==sha(__file__) and approval['protocolSHA256']==sha(OUT/'protocol.json') and approval['dataBindingSHA256']==sha(OUT/'data-binding.json')
 assert sha(ANCHOR)==ANCHOR_SHA
 rows=read(OUT/'records.json');binding=read(OUT/'data-binding.json');assert binding['recordsSHA256']==sha(OUT/'records.json')
 torch.set_num_threads(8);model=YOLO(str(ANCHOR));kwargs=dict(imgsz=1024,rect=False,conf=.001,iou=.7,max_det=1000,device='cpu',verbose=False)
 for r in rows:
  dest=OUT/'raw'/f'{r["id"]}.json'
  if dest.exists():
   prior=read(dest);assert prior['imageSHA256']==r['sha256'] and prior['modelSHA256']==ANCHOR_SHA;continue
  assert sha(r['preparedImage'])==r['sha256']
  with Image.open(r['preparedImage']) as im:image=im.convert('RGB');w,h=image.size
  start=time.time();result=model.predict(r['preparedImage'],**kwargs)[0]
  whole={'xyxy':result.boxes.xyxy.cpu().numpy().tolist(),'confidence':result.boxes.conf.cpu().numpy().tolist(),'classes':result.boxes.cls.cpu().numpy().astype(int).tolist()}
  tw,th=math.ceil(w*.6),math.ceil(h*.6);tiles=[];parts=[]
  for row in range(2):
   for col in range(2):
    x=0 if col==0 else w-tw;y=0 if row==0 else h-th;tiles.append({'row':row,'col':col,'cropXYXY':[x,y,x+tw,y+th],'offset':[x,y]});parts.append(image.crop((x,y,x+tw,y+th)))
  for prediction,tile in zip(model.predict(parts,stream=True,**kwargs),tiles):
   local=prediction.boxes.xyxy.cpu().numpy().copy();global_boxes=local.copy();x,y=tile['offset'];global_boxes[:,[0,2]]+=x;global_boxes[:,[1,3]]+=y
   tile.update({'localXYXY':local.tolist(),'globalXYXY':global_boxes.tolist(),'confidence':prediction.boxes.conf.cpu().numpy().tolist(),'classes':prediction.boxes.cls.cpu().numpy().astype(int).tolist()})
   assert np.isfinite(global_boxes).all()
  save(dest,{'id':r['id'],'imageSHA256':r['sha256'],'size':[w,h],'modelSHA256':ANCHOR_SHA,'whole':whole,'tiles':tiles,'rawMeaning':'Per-pass model outputs afterconf.001/modelNMS.7/max1000, beforeanyownership orcross-viewpostprocess.','float32OffsetArithmetic':True,'protocolSHA256':sha(OUT/'protocol.json'),'operatorSHA256':sha(__file__),'seconds':time.time()-start})
  print('raw',r['id'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','capture']);p.add_argument('--receipt',type=Path);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 if a.phase=='prepare':prepare()
 else:assert a.receipt;capture(a.receipt)
