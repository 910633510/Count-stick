#!/usr/bin/env python3
"""Author-semantic-mask proposals for subsequent visual correction, not GT acceptance."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
from scipy.ndimage import label,find_objects
from PIL import Image,ImageDraw,ImageFont,ImageEnhance
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'output/research-20261025-hardcases/positives'
DATA=BASE/'raw/zenodo20178469/agar_plate_bacteria_colonies_semantic_segmentation_dataset'
OUT=BASE/'review/proposals';OUT.mkdir(parents=True,exist_ok=True)
CHOSEN=[105,23,48,84,19,18,45,10,7,68,65,69,6,67,46]
rows=json.loads((BASE/'review/mipro-image-index.json').read_text());meta={}
for p in DATA.glob('*.csv'):
 for r in csv.DictReader(p.open()):meta[r['rgb']]={**r,'sourceSplit':p.stem}
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
big=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',23)
ignore={(0,0,0),(250,50,83),(51,221,255)}
records=[]
for idx in CHOSEN:
 p=ROOT/rows[idx]['path'];r=meta[str(p.relative_to(DATA))];a=DATA/r['annotation']
 with Image.open(p) as im:im=im.convert('RGB');w,h=im.size
 with Image.open(a) as mm:m=np.array(mm.convert('RGB'))
 fg=np.ones(m.shape[:2],dtype=bool)
 for color in ignore:fg &= (m != color).any(2)
 components,n=label(fg,structure=np.ones((3,3),int));props=[]
 for number,slices in enumerate(find_objects(components),1):
  if slices is None:continue
  yy,xx=slices;props.append({'proposalId':number,'xyxy':[xx.start,yy.start,xx.stop,yy.stop],'maskArea':int((components[slices]==number).sum())})
 record={'id':'mipro-'+r['id'],'source':'zenodo20178469-hyperspectralRGB','index':idx,'sourceCSV':r,'groupId':'zenodo20178469:'+r['id'],'species':r['label'],'imagePath':str(p.relative_to(ROOT)),'sourceSemanticMaskPath':str(a.relative_to(ROOT)),'width':w,'height':h,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sourceMaskSHA256':hashlib.sha256(a.read_bytes()).hexdigest(),'authorMaskComponentProposals':props,'annotationQA':'PENDING_VISUAL_REVIEW','method':'Connected components of all author microbial/contaminant semantic pixels; these are proposals, NOT instance-count ground truth.'}
 records.append(record)
 # Full-size views are independent panels, with gain used for display only.
 for mode,gain in [('raw',1),('proposals',2)]:
  canvas=Image.new('RGB',(w,h+80),'#f4f5f7');d=ImageDraw.Draw(canvas)
  d.text((12,7),f'#{idx} {r["id"]} age={r["sample_age"]} date={r["date"]} | {mode}',font=big,fill='black')
  d.text((12,38),'Mask-component proposals are NOT accepted instance counts. Display gain='+str(gain)+'; original pixels unchanged.',font=font,fill='black')
  view=ImageEnhance.Brightness(im).enhance(gain)
  if mode=='proposals':
   dd=ImageDraw.Draw(view)
   for prop in props:
    x1,y1,x2,y2=prop['xyxy'];dd.rectangle((x1,y1,x2,y2),outline='#00e2ff',width=2);dd.rectangle((x1,y1,x1+35,y1+23),fill='#000000');dd.text((x1+1,y1),str(prop['proposalId']),fill='white',font=font)
  canvas.paste(view,(0,80));canvas.save(OUT/f'{idx:03d}-{r["id"]}-{mode}.jpg',quality=95)
(BASE/'review/author-assisted-proposals.json').write_text(json.dumps({'status':'PENDING_OUR_VISUAL_ANNOTATION','images':records},indent=2)+'\n')
print(json.dumps([{'index':r['index'],'group':r['groupId'],'proposals':len(r['authorMaskComponentProposals']),'boxes':r['authorMaskComponentProposals']} for r in records],indent=2))
