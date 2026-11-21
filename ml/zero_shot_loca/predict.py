#!/usr/bin/env python3
"""LOCA research inference from image only. No exemplar boxes or category text are accepted."""
from pathlib import Path
import argparse,json,time
from PIL import Image
import numpy as np
import torch
from screen import build,TRANSFORM

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 torch.set_num_threads(8);model,info=build();model=model.to(args.device)
 with Image.open(args.image) as im:x=TRANSFORM(im.convert('RGB')).unsqueeze(0)
 start=time.perf_counter()
 with torch.no_grad():density,_=model(x.to(args.device),None);value=float(density.sum().cpu())
 assert np.isfinite(value) and bool(torch.isfinite(density).all())
 args.output.mkdir(parents=True,exist_ok=True);np.save(args.output/'density.npy',density.squeeze().cpu().numpy(),allow_pickle=False)
 r={'model':'Official LOCA zero-shot checkpoint','inputImage':str(args.image),'count':value,'roundedCount':round(value),'visualExemplarsProvided':0,'categoryTextProvided':False,'device':args.device,'forwardAndTransferSeconds':time.perf_counter()-start,'densityShape':list(density.shape),'sourceArchitectureUnchanged':info['sourceArchitectureUnchanged'],'scope':'Research-only; no semantic target selection guarantee. Synthetic blank controls previously overcounted strongly. No CoreML/iPhone validation.'};(args.output/'prediction.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
