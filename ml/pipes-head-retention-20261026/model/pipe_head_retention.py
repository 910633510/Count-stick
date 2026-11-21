"""Isolated head-only pipe retention data/protocol contracts. No heldout dataset reader."""
from pathlib import Path
import collections,datetime,hashlib,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-head-retention-20261026/model';SOURCE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'output/pipes-hardening-20261026/model'
PREVIOUS_RECORDS_SHA='cf6f0d541fd4c5830c12e02460672088eb7b6ddbb18c66207df07917dedb1f47';PREVIOUS_INPUTS_SHA='5572480c468a0958903172f1d1e573648077312b4931c9c3c6e82e650e3c3f5a';PREVIOUS_MANIFEST_SHA='680de25252f1d9cf91f335b0bdea6b42e18a9fbfe6d4db74ef0cbb41f65d29e8';PREVIOUS_SELECTION_SHA='f8659b52d87afc274f17892bb6339ca8eb7fcd3213e6865304cc64f24e208480'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics-config');os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
OLD=ROOT/'output/pipes-20261026/model';INIT=OLD/'checkpoints/epoch020.pt';INIT_SHA='48a77fd9198b23695f470b172397e918c9113238bfef2d421d8bbd431f69af12'
OLD_RECORDS_SHA='c01c9771fa7dd810c988df784d42fe6fd8d539b8253a7ca5518dcd0a78b13774';OLD_SELECTION_SHA='9eda69afbaa72634270f86f910e7f87719bbd55a5880d6dbb000e6d4a62209f8'
sys.path.insert(0,str(ROOT/'ml/pipes-20261026/model'));sys.path.insert(0,str(ROOT/'ml/colony'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def resolve(p):return Path(p).resolve() if Path(p).is_absolute() else (ROOT/p).resolve()
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def code_hashes():
 files=['pipe_head_retention.py','train_head_retention.py','evaluate_head_retention.py','bind_head_inputs.py','head_kd.py'];d={str((SOURCE/f).relative_to(ROOT)):sha(SOURCE/f) for f in files}
 for file,digest in read(PREVIOUS/'training-authorization.json')['codeSHA256'].items():assert sha(ROOT/file)==digest;d[file]=digest
 return d
APP_FILES={'PhotoProcessing':'Count stick/StickDetector.swift','PipeLetterbox':'Count stick/PipeDetector.swift','ExportHelper':'Count stickTests/PipePreprocessingExportTests.swift','NMS':'Count stick/PipeRawPostprocessor.swift'}
def app_hashes():return {k:sha(ROOT/v) for k,v in APP_FILES.items()}
def init_check():
 assert sha(INIT)==INIT_SHA and sha(OLD/'records.json')==OLD_RECORDS_SHA and sha(OLD/'selection.json')==OLD_SELECTION_SHA
 assert read(OLD/'selection.json')['selected']['modelSHA256']==INIT_SHA
 assert sha(PREVIOUS/'records.json')==PREVIOUS_RECORDS_SHA and sha(PREVIOUS/'evaluation-inputs.json')==PREVIOUS_INPUTS_SHA and sha(PREVIOUS/'selection.json')==PREVIOUS_SELECTION_SHA
 assert read(ROOT/'ml/pipes-20261026/parity/runtime-v3-independent-review.json')['status']=='PASS_PIPE_RUNTIME_V3_SOURCE_AMENDMENT'
def validate_data_nature(row,image_path):
 """A masked parent's unmodified crop is allowed only as a zero-GT training view."""
 from PIL import Image
 assert row['isSynthetic'] is False and row['hasAnnotationOverlay'] is False and row['annotationQA']=='PASS'
 if row['dataNature']=='real':
  assert row.get('sourceHadBlackMask',False) is False
  assert 'cropRectPixels' not in row and 'sourceCropXYXY' not in row,'Full natural-photo records cannot hide a crop'
  return
 assert row['dataNature']=='real_crop' and row['split']=='train' and row['count']==0 and row['xyxy']==[]
 assert row['sourceHadBlackMask'] is True and row['cropReview']=='PASS'
 assert str(row['groupId'])==str(row['parentGroupId']) and str(row['parentGroupId'])
 parent=resolve(row['sourceImagePath']);assert parent.is_relative_to(ROOT) and parent!=resolve(image_path) and sha(parent)==row['sourceImageSHA256']
 rect=row['cropRectPixels'];assert len(rect)==4 and all(type(x) is int for x in rect)
 if 'sourceCropXYXY' in row:assert row['sourceCropXYXY']==rect
 with Image.open(parent) as im:im.verify()
 with Image.open(parent) as im:
  assert im.getexif().get(274,1) in [None,1],'Crop coordinates must refer to upright original pixels'
  x1,y1,x2,y2=rect;assert 0<=x1<x2<=im.width and 0<=y1<y2<=im.height
  assert (x1,y1,x2,y2)!=(0,0,im.width,im.height),'A masked full image is not an unmodified internal crop'
  expected=im.convert('RGB').crop(tuple(rect));expected_size=expected.size;expected_pixels=expected.tobytes()
 with Image.open(image_path) as crop:
  assert crop.format=='PNG' and crop.mode=='RGB' and crop.size==expected_size==(row['width'],row['height'])
  assert crop.getexif().get(274,1) in [None,1] and crop.tobytes()==expected_pixels,'Derived crop changed source pixels'
def proposal():
 q=read(PREVIOUS/'protocol.json').copy();q.pop('registeredUTC',None);q.pop('manifestSHA256',None);q.pop('dataBindingSHA256',None);q['status']='PROPOSED_HEAD_RETENTION_PENDING_ROOT_FREEZE';q['training']=dict(q['training'],freeze=list(range(23)));q['dataReady']=False
 q['task']='head_only_round_pipe_retention';q['freezePolicy']='Freeze model.0..22 body/neck and fixed DFL integral. Train only model.23 Detect parameters. Force all BN running statistics, including head BN, to eval after each model.train call; head BN affine parameters remain trainable.'
 q['KD']={'teacherSHA256':INIT_SHA,'oldPositiveSourceRecordsSHA256':OLD_RECORDS_SHA,'oldPositiveScope':'Only original pipe training split=count>0 IDs, exactly156. Previous8 new positives and newMACO343 excluded.','classificationWeight':8.,'DFLWeight':1.,'temperature':1.,'classification':'Soft teacher Bernoulli KL; per-image teacher-p weighted foreground and (1-p) background normalized separately, averaged equally.','DFL':'Direct categoricalKL over16 bins per edge; edge mean, softteacher-p anchor weighting, per-image normalization.','batchScaling':'B times mean over old-positive subbatch for each KD term; zero when no old-positive images.','teacherOwnership':'Trainer only; plain loss callable holds weakref, never registers teacher on student; EMA created before criterion.','optimizer':'Teacher/body/fixedDFL excluded; teacher has no gradients.','serialization':'EMA/checkpoint has no teacher or customcriterion; standard DetectionModel and raw IO preserved.'}
 q['additionalAugmentation']='None. Preserve existing training transforms; no added downscale/blur.';q['previousRecordsSHA256']=PREVIOUS_RECORDS_SHA;q['previousInputsSHA256']=PREVIOUS_INPUTS_SHA;q['validationSourceManifestSHA256']=PREVIOUS_MANIFEST_SHA;q['originalFreshHoldoutPolicy']='Root verifies the same prior20 unseen identities are preserved; no full/test read by train operator.'
 return q

def prepare(manifest_path):
 from PIL import Image
 init_check();manifest_path=resolve(manifest_path);manifest=read(manifest_path);assert manifest['status']=='FROZEN_ANNOTATION_QA_PASSED' and manifest['testContentIncluded'] is False
 assert not (OUT/'data-binding.json').exists();parent_binding=read(PREVIOUS/'data-binding.json');assert parent_binding['recordsSHA256']==PREVIOUS_RECORDS_SHA and parent_binding['manifestSHA256']==PREVIOUS_MANIFEST_SHA;assert sha(parent_binding['manifestPath'])==PREVIOUS_MANIFEST_SHA
 old_tv=read(parent_binding['manifestPath'])['images'];new_tv=manifest['images'];assert len(old_tv)==49 and len(new_tv)==54 and len({r['id'] for r in new_tv})==54
 combined={r['id']:r for r in new_tv}
 for row in old_tv:assert combined[row['id']]==row,'The previous49 development rows must remain unchanged'
 previous_ids={r['id'] for r in old_tv};extra=[r for r in new_tv if r['id'] not in previous_ids];assert len(extra)==5 and all(r['split']=='train' for r in extra) and sum(r['count']>0 for r in extra)==1 and sum(r['count']==0 for r in extra)==4
 base=read(PREVIOUS/'records.json');assert len(base)==253 and sum(r['split']=='val' for r in base)==47
 ids={r['id'] for r in base};groups={str(r['groupId']):r['split'] for r in base};hash_splits={r['sha256']:r['split'] for r in base};items=[]
 for row in extra:
  assert row['id'] not in ids;ids.add(row['id']);assert row['count']==len(row['xyxy']) and 0<=row['count']<=20 and row['label']=='pipe_end' and row['dataNature']=='real'
  if row['count']:assert row['subtype']=='round' and row['annotationBoundary']=='outer_rim'
  else:assert row['negativeKind'] in ['non_pipe','empty_scene','side_only']
  assert groups.get(str(row['groupId']),'train')=='train' and hash_splits.get(row['sha256'],'train')=='train';groups[str(row['groupId'])]='train';hash_splits[row['sha256']]='train'
  path=resolve(row['imagePath']);assert path.is_relative_to(ROOT) and sha(path)==row['sha256']
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:assert im.size==(row['width'],row['height']) and im.getexif().get(274,1) in [None,1]
  validate_data_nature(row,path);assert all(0<=a<c<=row['width'] and 0<=b<d<=row['height'] for a,b,c,d in row['xyxy'])
  items.append({'id':row['id'],'source':str(path),'destination':str(OUT/'dataset/images/train'/(row['id']+'.png'))})
 OUT.mkdir(parents=True,exist_ok=True);save(OUT/'photo-preparation-input.json',items);exe=OUT/'tools/prepare_photos';exe.parent.mkdir(exist_ok=True)
 subprocess.run(['xcrun','swiftc',str(ROOT/'ml/pipes-20261026/model/prepare_photos.swift'),'-o',str(exe)],check=True);subprocess.run([str(exe),str(OUT/'photo-preparation-input.json'),str(OUT/'photo-preparation-output.json')],check=True);photos={r['id']:r for r in read(OUT/'photo-preparation-output.json')};records=[]
 for row in base:
  assert sha(row['preparedImage'])==row['preparedImageSHA256'] and sha(row['preparedLabel'])==row['preparedLabelSHA256'];image=OUT/'dataset/images'/row['split']/(row['id']+'.png');label=OUT/'dataset/labels'/row['split']/(row['id']+'.txt');image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(row['preparedImage'],image);shutil.copy2(row['preparedLabel'],label);records.append({**row,'preparedImage':str(image),'preparedLabel':str(label)})
 for row in extra:
  photo=photos[row['id']];w,h=photo['width'],photo['height'];sx,sy=w/row['width'],h/row['height'];boxes=[[a*sx,b*sy,c*sx,d*sy] for a,b,c,d in row['xyxy']];label=OUT/'dataset/labels/train'/(row['id']+'.txt');lines=[f'0 {(a+c)/2/w:.10f} {(b+d)/2/h:.10f} {(c-a)/w:.10f} {(d-b)/h:.10f}' for a,b,c,d in boxes];assert len(lines)==len(set(lines));label.write_text('\n'.join(lines)+('\n' if lines else ''));records.append({**row,'originalWidth':row['width'],'originalHeight':row['height'],'originalXYXY':row['xyxy'],'width':w,'height':h,'xyxy':boxes,'preparedImage':photo['destination'],'preparedImageSHA256':sha(photo['destination']),'preparedLabel':str(label),'preparedLabelSHA256':sha(label),'dataRole':'fresh','bucket':'new_positive' if row['count'] else 'negative'})
 assert len(records)==258 and sum(r['split']=='train' for r in records)==211 and sum(r['split']=='val' for r in records)==47
 save(OUT/'records.json',records);(OUT/'dataset/data.yaml').write_text('path: '+json.dumps(str(OUT/'dataset'))+'\ntrain: images/train\nval: images/val\nnames:\n  0: pipe_end\n')
 summary={s:{'images':sum(r['split']==s for r in records),'groups':len({r['groupId'] for r in records if r['split']==s}),'objects':sum(r['count'] for r in records if r['split']==s),'byRoleBucket':dict(collections.Counter(r['dataRole']+'/'+r['bucket'] for r in records if r['split']==s))} for s in ['train','val']}
 binding={'status':'PREPARED_HEAD_RETENTION_TRAIN_VAL_ONLY','manifestPath':str(manifest_path),'manifestSHA256':sha(manifest_path),'sealedFullManifestSHA256':manifest['sealedFullManifestSHA256'],'recordsSHA256':sha(OUT/'records.json'),'dataYAMLSHA256':sha(OUT/'dataset/data.yaml'),'previousRecordsSHA256':PREVIOUS_RECORDS_SHA,'previousInputsSHA256':PREVIOUS_INPUTS_SHA,'validationSourceManifestSHA256':PREVIOUS_MANIFEST_SHA,'initializationSHA256':INIT_SHA,'summary':summary,'old253CopiedExactly':True,'testContentIncluded':False};save(OUT/'data-binding.json',binding)
 protocol=proposal();protocol.update(status='FROZEN_HEAD_RETENTION_PROTOCOL_PENDING_ROOT_RECEIPT',registeredUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataReady=True,manifestSHA256=sha(manifest_path),dataBindingSHA256=sha(OUT/'data-binding.json'));save(OUT/'protocol.json',protocol)
 save(OUT/'root-training-receipt.template.json',{'status':'ROOT_APPROVAL_REQUIRED','protocolSHA256':sha(OUT/'protocol.json'),'dataBindingSHA256':sha(OUT/'data-binding.json'),'codeSHA256':code_hashes(),'initializationSHA256':INIT_SHA,'appSourceCodeSHA256':app_hashes(),'qaReportPath':None,'qaReportSHA256':None,'integrationSmokePath':str(OUT/'smoke/report.json'),'integrationSmokeSHA256':sha(OUT/'smoke/report.json'),'qaRequirements':['Original49TV rows unchanged; exactly5 added natural train views','Original156 teacher-old positive IDs only; earlier8new positives excluded','Same47validation and same20fresh holdout identities; groups/hash leakage excluded']});return binding

def verify(receipt_path):
 init_check();receipt=read(receipt_path);protocol=read(OUT/'protocol.json');binding=read(OUT/'data-binding.json');assert receipt['status']=='APPROVED_HEAD_RETENTION_TRAINING'
 assert receipt['protocolSHA256']==sha(OUT/'protocol.json') and receipt['dataBindingSHA256']==sha(OUT/'data-binding.json') and receipt['codeSHA256']==code_hashes() and receipt['initializationSHA256']==INIT_SHA
 assert protocol['status']=='FROZEN_HEAD_RETENTION_PROTOCOL_PENDING_ROOT_RECEIPT' and protocol['appSourceCodeSHA256']==receipt['appSourceCodeSHA256']==app_hashes();assert sha(binding['manifestPath'])==binding['manifestSHA256'] and sha(OUT/'records.json')==binding['recordsSHA256'] and sha(OUT/'dataset/data.yaml')==binding['dataYAMLSHA256']
 assert sha(resolve(receipt['qaReportPath']))==receipt['qaReportSHA256'] and read(resolve(receipt['qaReportPath']))['status'].startswith('PASS')
 smoke=resolve(receipt['integrationSmokePath']);assert sha(smoke)==receipt['integrationSmokeSHA256'];report=read(smoke);assert report['status']=='PASS_HEAD_RETENTION_TRAINER_INTEGRATION_NO_OPTIMIZER' and report['trainerSHA256']==sha(SOURCE/'train_head_retention.py') and report['criterionSHA256']==sha(SOURCE/'head_kd.py')
 for row in read(OUT/'records.json'):assert row['split'] in ['train','val'] and sha(row['preparedImage'])==row['preparedImageSHA256'] and sha(row['preparedLabel'])==row['preparedLabelSHA256']
 return receipt
