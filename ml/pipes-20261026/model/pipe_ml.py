"""Shared pipe-only paths, data binding and numerical contracts."""
from pathlib import Path
import os,sys,json,hashlib,math,collections,re,subprocess
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/pipes-20261026/model'
SOURCE=Path(__file__).resolve().parent
WEIGHTS=OUT/'official-source/yolo11n.pt'
WEIGHTS_SHA='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics-config')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
sys.path.insert(0,str(ROOT/'ml/colony'))
import numpy as np
import torch
from PIL import Image

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');os.replace(temp,path)

def official_init_check():
 assert sha(WEIGHTS)==WEIGHTS_SHA
 audit=read(OUT/'official-source-audit.json');assert audit['status']=='PASS_OFFICIAL_SOURCE_FORMAT_ALLOWLIST_SAFE_LOAD_STRUCTURE' and audit['localSHA256']==WEIGHTS_SHA and audit['classes']==80

def code_hashes():
 files=['pipe_ml.py','train_pipe_ends.py','evaluate_pipe_ends.py','export_pipe_coreml.py','prepare_photos.swift','audit_official_init.py','bind_app_inputs.py']
 result={str((SOURCE/name).relative_to(ROOT)):sha(SOURCE/name) for name in files};result['ml/colony/mps_compat.py']=sha(ROOT/'ml/colony/mps_compat.py');return result

def app_source_hashes():
 return {'PhotoProcessing':sha(ROOT/'Count stick/StickDetector.swift'),'PipeLetterbox':sha(ROOT/'Count stick/PipeDetector.swift'),'ExportHelper':sha(ROOT/'Count stickTests/PipePreprocessingExportTests.swift')}

def root_training_approval(path):
 receipt=read(path);assert receipt['status']=='APPROVED_PIPE_BASELINE_TRAINING'
 protocol=read(OUT/'protocol.json');assert protocol['status']=='FROZEN_PIPE_BASELINE_PROTOCOL' and isinstance(protocol['samplesPerEpoch'],int) and protocol['samplesPerEpoch']>0
 assert protocol['appSourceCodeSHA256']==app_source_hashes(),'Approved App preprocessing/helper source changed'
 assert receipt['protocolSHA256']==sha(OUT/'protocol.json') and receipt['dataBindingSHA256']==sha(OUT/'data-binding.json') and receipt['codeSHA256']==code_hashes()
 assert receipt['officialWeightsSHA256']==WEIGHTS_SHA
 qa=Path(receipt['qaReportPath']);assert sha(qa)==receipt['qaReportSHA256'] and read(qa)['status'].startswith('PASS')
 binding=read(OUT/'data-binding.json');assert sha(binding['manifestPath'])==binding['manifestSHA256'] and sha(OUT/'records.json')==binding['recordsSHA256'] and sha(OUT/'dataset/data.yaml')==binding['dataYAMLSHA256']
 for r in read(OUT/'records.json'):assert sha(r['preparedImage'])==r['preparedImageSHA256'] and sha(r['preparedLabel'])==r['preparedLabelSHA256']
 return receipt

def prepare(manifest_path):
 official_init_check();manifest_path=Path(manifest_path).resolve();manifest=read(manifest_path)
 assert manifest['status']=='FROZEN_ANNOTATION_QA_PASSED' and manifest['testContentIncluded'] is False
 if (OUT/'data-binding.json').exists():
  b=read(OUT/'data-binding.json');assert b['manifestSHA256']==sha(manifest_path);return b
 rows=manifest['images'];assert rows and set(r['split'] for r in rows)=={'train','val'};ids=set();groups={};hash_splits={};items=[]
 for r in rows:
  assert re.fullmatch(r'[A-Za-z0-9_.-]+',r['id']) and r['id'] not in ids;ids.add(r['id'])
  assert r.get('dataNature')=='real' and r.get('isSynthetic') is False and r.get('hasAnnotationOverlay') is False,'Real baseline must exclude synthetic/markup/unknown provenance'
  assert r['annotationQA']=='PASS' and r['count']==len(r['xyxy']) and r['count']>=0
  if r['count']==0:assert r['negativeKind'] in ['empty_scene','non_pipe','side_only']
  else:assert r.get('subtype','round')=='round' and r.get('annotationBoundary')=='outer_rim'
  assert r['count']<=21504,'GT exceeds the declared1024single-class output-location capacity'
  group=str(r['groupId']);assert group not in groups or groups[group]==r['split'];groups[group]=r['split']
  assert r['sha256'] not in hash_splits or hash_splits[r['sha256']]==r['split'];hash_splits[r['sha256']]=r['split']
  path=(ROOT/r['imagePath']).resolve();assert path.is_relative_to(ROOT) and sha(path)==r['sha256']
  with Image.open(path) as image:image.verify()
  with Image.open(path) as image:assert image.size==(r['width'],r['height']) and image.getexif().get(274,1) in [None,1]
  for x1,y1,x2,y2 in r['xyxy']:assert 0<=x1<x2<=r['width'] and 0<=y1<y2<=r['height']
  items.append({'id':r['id'],'source':str(path),'destination':str(OUT/'dataset/images'/r['split']/(r['id']+'.png'))})
 assert all(any(r['split']==split and r['count']==0 for r in rows) and any(r['split']==split and r['count']>0 for r in rows) for split in ['train','val']),'Need positive and genuine negative train/val groups before release-quality training'
 save(OUT/'photo-preparation-input.json',items)
 executable=OUT/'tools/prepare_photos';executable.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['xcrun','swiftc',str(SOURCE/'prepare_photos.swift'),'-o',str(executable)],check=True)
 subprocess.run([str(executable),str(OUT/'photo-preparation-input.json'),str(OUT/'photo-preparation-output.json')],check=True)
 prepared={r['id']:r for r in read(OUT/'photo-preparation-output.json')};records=[]
 for r in rows:
  photo=prepared[r['id']];w,h=photo['width'],photo['height'];sx,sy=w/r['width'],h/r['height'];boxes=[[x1*sx,y1*sy,x2*sx,y2*sy] for x1,y1,x2,y2 in r['xyxy']]
  label=OUT/'dataset/labels'/r['split']/(r['id']+'.txt');label.parent.mkdir(parents=True,exist_ok=True);lines=[f'0 {(a+c)/2/w:.10f} {(b+d)/2/h:.10f} {(c-a)/w:.10f} {(d-b)/h:.10f}' for a,b,c,d in boxes];assert len(lines)==len(set(lines));label.write_text('\n'.join(lines)+('\n' if lines else ''))
  records.append({**r,'originalWidth':r['width'],'originalHeight':r['height'],'originalXYXY':r['xyxy'],'width':w,'height':h,'xyxy':boxes,'preparedImage':photo['destination'],'preparedImageSHA256':sha(photo['destination']),'preparedLabel':str(label),'preparedLabelSHA256':sha(label),'photoPreparation':'ImageIOmax2048','label':'pipe_end','species':'pipe_end','bucket':'positive' if r['count'] else 'negative'})
 save(OUT/'records.json',records);(OUT/'dataset/data.yaml').write_text('path: '+json.dumps(str(OUT/'dataset'))+'\ntrain: images/train\nval: images/val\nnames:\n  0: pipe_end\n')
 summary={s:{'images':sum(r['split']==s for r in records),'groups':len({r['groupId'] for r in records if r['split']==s}),'objects':sum(r['count'] for r in records if r['split']==s),'maxGT':max(r['count'] for r in records if r['split']==s),'negativeImages':sum(r['split']==s and r['count']==0 for r in records)} for s in ['train','val']}
 binding={'status':'PREPARED_PIPE_TRAIN_VAL_PENDING_ROOT_RECEIPT','manifestPath':str(manifest_path),'manifestSHA256':sha(manifest_path),'recordsSHA256':sha(OUT/'records.json'),'dataYAMLSHA256':sha(OUT/'dataset/data.yaml'),'summary':summary,'testContentIncluded':False,'officialWeightsSHA256':WEIGHTS_SHA,'photoPreparationSwiftSHA256':sha(SOURCE/'prepare_photos.swift'),'networkInputRendering':'App UIKit letterbox inputs still need separatebinding/parity before evaluation; no assumption OpenCV pixels equalCoreGraphics.'};save(OUT/'data-binding.json',binding);return binding

def pipeline_geometry(width,height,size=1024):
 gain=min(size/width,size/height);rw=round(width*gain);rh=round(height*gain);return {'gain':gain,'resizedWidth':rw,'resizedHeight':rh,'left':round((size-rw)/2-.1),'top':round((size-rh)/2-.1),'sourceWidth':width,'sourceHeight':height,'size':size}

def unletterbox(boxes,geometry):
 b=np.asarray(boxes,dtype=np.float32).reshape(-1,4).copy();g=geometry['gain'];b[:,[0,2]]=(b[:,[0,2]]-geometry['left'])/g;b[:,[1,3]]=(b[:,[1,3]]-geometry['top'])/g;b[:,[0,2]]=b[:,[0,2]].clip(0,geometry['sourceWidth']);b[:,[1,3]]=b[:,[1,3]].clip(0,geometry['sourceHeight']);valid=(b[:,2]>b[:,0])&(b[:,3]>b[:,1]);return b,valid
