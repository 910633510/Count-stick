"""Read-only AGAR archive/pair/geometry audit plus diagnostic visualizations; no model."""
from pathlib import Path,PurePosixPath
import collections,hashlib,io,json,math,stat,zipfile
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'output/multiclass-data-20261029/colony';OUT=BASE/'independent';OUT.mkdir(exist_ok=True)
ARCHIVE=BASE/'archives/AGAR_demo.zip';EXPECTED_SHA='71113e092d241768186a5808fd0a19e2140bac904c57d64d97b4fbdc90208bb2';EXPECTED_BYTES=86605404
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert ARCHIVE.stat().st_size==EXPECTED_BYTES and sha(ARCHIVE)==EXPECTED_SHA
issues=[];records=[];entries=[];total=0
with zipfile.ZipFile(ARCHIVE) as z:
 names=set()
 for info in z.infolist():
  name=info.filename;path=PurePosixPath(name)
  assert name==info.orig_filename and '\\' not in name and not path.is_absolute() and '..' not in path.parts and ':' not in path.parts[0] and name not in names;names.add(name)
  assert not stat.S_ISLNK(info.external_attr>>16);assert not info.flag_bits&1
  assert info.file_size<64_000_000;total+=info.file_size;assert total<500_000_000
  if not info.is_dir():
   content=z.read(info);assert len(content)==info.file_size  # ZipFile verifies each member CRC.
   extracted=BASE/'raw/agar-demo'/name;assert extracted.is_file() and extracted.read_bytes()==content
   entries.append({'name':name,'bytes':info.file_size,'crc32':f'{info.CRC:08x}','sha256':hashlib.sha256(content).hexdigest()})
 images={PurePosixPath(x['name']).with_suffix(''):x for x in entries if x['name'].lower().endswith('.jpg')};labels={PurePosixPath(x['name']).with_suffix(''):x for x in entries if x['name'].lower().endswith('.json')};assert set(images)==set(labels) and len(images)==40 and len(entries)==80
 for stem in sorted(images,key=str):
  im_bytes=z.read(images[stem]['name']);annotation=json.loads(z.read(labels[stem]['name']))
  with Image.open(io.BytesIO(im_bytes)) as im:im.verify()
  with Image.open(io.BytesIO(im_bytes)) as im:w,h=im.size;mode=im.mode;orientation=im.getexif().get(274,1);assert mode=='RGB' and orientation in [1,None]
  count=annotation['colonies_number'];assert isinstance(count,int) and count>=-1;declared=set(annotation['classes']);boxes=[];classes=collections.Counter();ids=[]
  for box in annotation['labels']:
   assert all(isinstance(box[k],(int,float)) and math.isfinite(box[k]) for k in ['x','y','width','height']);x,y,bw,bh=[box[k] for k in ['x','y','width','height']];assert 0<=x<x+bw<=w and 0<=y<y+bh<=h;assert isinstance(box['class'],str) and box['class'];ids.append(box['id']);classes[box['class']]+=1;boxes.append([x,y,x+bw,y+bh,box['class']])
  if len(ids)!=len(set(ids)):issues.append({'sample':stem.name,'type':'duplicate_annotation_id'})
  if len({tuple(b) for b in boxes})!=len(boxes):issues.append({'sample':stem.name,'type':'exact_duplicate_box'})
  target=sum(n for c,n in classes.items() if c in declared);outside={c:n for c,n in classes.items() if c not in declared}
  if count>=0 and target!=count:issues.append({'sample':stem.name,'type':'author_target_count_mismatch','author':count,'declaredClassBoxes':target})
  if count==-1:assert not boxes
  records.append({'sample_id':int(stem.name),'condition':str(stem.parent),'imagePath':str((BASE/'raw/agar-demo'/images[stem]['name']).relative_to(ROOT)),'labelPath':str((BASE/'raw/agar-demo'/labels[stem]['name']).relative_to(ROOT)),'imageSHA256':images[stem]['sha256'],'labelSHA256':labels[stem]['sha256'],'width':w,'height':h,'authorCount':count,'declaredClasses':sorted(declared),'targetClassBoxCount':target,'allBoxCount':len(boxes),'otherClasses':outside,'allClassCounts':dict(classes),'status':'uncountable_not_negative' if count==-1 else 'countable','boxesForDiagnosticOnly':boxes})
assert not issues,issues
counts=collections.Counter(r['status'] for r in records);class_totals=collections.Counter()
for r in records:class_totals.update(r['allClassCounts'])
selected=[734,1399,4904,14581];byid={r['sample_id']:r for r in records};rendered=[]
try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',21)
except OSError:font=ImageFont.load_default()
for id in selected:
 r=byid[id];im=Image.open(ROOT/r['imagePath']).convert('RGB');scale=min(760/r['width'],820/r['height']);size=(round(r['width']*scale),round(r['height']*scale));raw=im.resize(size,Image.Resampling.LANCZOS);overlay=raw.copy();draw=ImageDraw.Draw(overlay)
 for x1,y1,x2,y2,kind in r['boxesForDiagnosticOnly']:
  color=(30,230,80) if kind in r['declaredClasses'] else (255,120,0);draw.rectangle((x1*scale,y1*scale,x2*scale,y2*scale),outline=color,width=2)
 panel=Image.new('RGB',(1560,950),'#15171b');panel.paste(raw,((780-size[0])//2,100));panel.paste(overlay,(780+(780-size[0])//2,100));d=ImageDraw.Draw(panel)
 d.text((18,12),f'AGAR {id} | original (left) / author boxes (right)',fill='white',font=font)
 detail=f'author target count: {r["authorCount"]} | target boxes: {r["targetClassBoxCount"]} | other: {r["otherClasses"]}'
 if r['authorCount']==-1:detail='AUTHOR UNCOUNTABLE (-1): no exhaustive instance labels; NEVER an empty negative'
 d.text((18,44),detail,fill='#ffca66',font=font);d.text((18,920),'Green: author-declared target species | Orange: other class (contamination). Diagnostic only; no relabel.',fill='white',font=font)
 path=OUT/f'agar-{id}-original-and-overlay.jpg';panel.save(path,quality=92);rendered.append({'sample_id':id,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'imageAndBoxesUnmodified':True,'displayScale':scale})
receipts=json.loads((BASE/'sources/download-receipts.json').read_text());sp=[]
for receipt in receipts:
 if receipt['id'].startswith('spsds-'):
  file=ROOT/receipt['localPath'];assert file.stat().st_size==receipt['bytes'] and sha(file)==receipt['sha256']==receipt['publishedSHA256']
  with Image.open(file) as im:im.verify()
  with Image.open(file) as im:sp.append({'id':receipt['id'],'sha256':sha(file),'bytes':file.stat().st_size,'size':list(im.size),'orientation':im.getexif().get(274,1),'authorAnnotationPresentLocally':False})
assert len(sp)==6
numeric={'status':'PASS_INDEPENDENT_CRC_PAIR_GEOMETRY_AND_TARGET_COUNT_SEMANTICS_PENDING_VISUAL','archivePath':str(ARCHIVE.relative_to(ROOT)),'archiveSHA256':EXPECTED_SHA,'archiveBytes':EXPECTED_BYTES,'allMemberCRCReadPass':True,'safePathsNoLinksEncryptedOrDuplicateEntries':True,'memberFiles':len(entries),'jpgCount':len(images),'jsonCount':len(labels),'uncompressedBytes':total,'pairSetsExactlyMatch':True,'extractedFilesByteEqualArchive':True,'statusCounts':dict(counts),'targetAuthorCountSum':sum(r['authorCount'] for r in records if r['authorCount']>=0),'totalBoxCount':sum(r['allBoxCount'] for r in records),'classTotals':dict(class_totals),'allBoxesFiniteInsideImage':True,'countMatchesDeclaredTargetClassesForAll39':True,'trueEmptyCount0Images':sum(r['authorCount']==0 for r in records),'uncountableNeverEmpty':True,'special734':{k:byid[734][k] for k in ['authorCount','declaredClasses','targetClassBoxCount','allBoxCount','otherClasses']},'records':[{k:v for k,v in r.items() if k!='boxesForDiagnosticOnly'} for r in records],'memberIndex':entries,'visualSamples':rendered,'SPSDS':{'status':'RAW_IMAGE_BYTES_DIMENSIONS_VERIFIED_NO_GT','images':sp,'count':6,'notEmptyNegativeLabels':True,'trainingReadiness':'Requires independent instance annotation and grouping review; no conversion to zero boxes.'},'newModelInference':False,'newTraining':False,'sourceAnnotationsModified':False,'oldFrozenLabelsModified':False,'externalCodeExecuted':False}
(OUT/'numeric-review.json').write_text(json.dumps(numeric,indent=2)+'\n');print('numericSHA',sha(OUT/'numeric-review.json'));print({k:numeric[k] for k in ['statusCounts','targetAuthorCountSum','totalBoxCount','trueEmptyCount0Images','special734']});print('visuals',rendered)
