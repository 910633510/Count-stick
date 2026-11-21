#!/usr/bin/env python3
"""Offline FamNet reference-box counting; requires3-6 example rectangles, not auto target selection."""
from pathlib import Path
import argparse,importlib.util,json,hashlib
import torch
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--boxes',type=Path,required=True,help='JSON list of [x1,y1,x2,y2] rectangles in image pixels');p.add_argument('--bundle',type=Path,default=ROOT/'output/models/CountScope-FSC147-FamNet-v1');p.add_argument('--weights',choices=['baseline','finetuned'],default='baseline');p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(8)
 recipe=json.loads((a.bundle/'recipe.json').read_text());file=recipe['regressors'][a.weights]['file'];assert hashlib.sha256((a.bundle/file).read_bytes()).hexdigest()==recipe['regressors'][a.weights]['sha256']
 models=mod('famnet_export_model',a.bundle/'model.py');u=mod('famnet_export_utils',a.bundle/'utils.py')
 class Backbone(nn.Module):
  def __init__(self):
   super().__init__();r=resnet50(weights=None);r.load_state_dict(torch.load(a.bundle/recipe['backbone']['file'],map_location='cpu',weights_only=True));c=list(r.children());self.c1=nn.Sequential(*c[:4]);self.c2=c[4];self.c3=c[5];self.c4=c[6]
  def forward(self,x):
   x=self.c2(self.c1(x));x=self.c3(x);return {'map3':x,'map4':self.c4(x)}
 with Image.open(a.image) as im:image=im.convert('RGB')
 boxes=json.loads(a.boxes.read_text());assert 3<=len(boxes)<=6
 for x1,y1,x2,y2 in boxes:assert 0<=x1<x2<=image.width and 0<=y1<y2<=image.height
 sample=u.Transform({'image':image,'lines_boxes':[[y1,x1,y2,x2] for x1,y1,x2,y2 in boxes]});backbone=Backbone().eval().to(a.device);reg=models.CountRegressor(6,pool='mean').eval().to(a.device);reg.load_state_dict(torch.load(a.bundle/file,map_location='cpu',weights_only=True))
 with torch.no_grad():density=reg(u.extract_features(backbone,sample['image'].unsqueeze(0).to(a.device),sample['boxes'].unsqueeze(0),u.MAPS,u.Scales));count=float(density.sum().cpu())
 a.output.mkdir(parents=True,exist_ok=True);np.save(a.output/'density.npy',density.squeeze().cpu().numpy(),allow_pickle=False);report={'image':str(a.image),'count':count,'roundedCount':round(count),'weights':a.weights,'referenceBoxes':boxes,'automaticTargetSelection':False};(a.output/'prediction.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
