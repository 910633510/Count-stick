"""Isolated pipe hardening data/protocol contracts. No heldout dataset reader."""
from pathlib import Path
import collections,datetime,hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-hardening-20261026/model';SOURCE=Path(__file__).resolve().parent
OLD=ROOT/'output/pipes-20261026/model';INIT=OLD/'checkpoints/epoch020.pt';INIT_SHA='48a77fd9198b23695f470b172397e918c9113238bfef2d421d8bbd431f69af12'
OLD_RECORDS_SHA='c01c9771fa7dd810c988df784d42fe6fd8d539b8253a7ca5518dcd0a78b13774';OLD_SELECTION_SHA='9eda69afbaa72634270f86f910e7f87719bbd55a5880d6dbb000e6d4a62209f8'
sys.path.insert(0,str(ROOT/'ml/pipes-20261026/model'));sys.path.insert(0,str(ROOT/'ml/colony'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def resolve(p):return Path(p).resolve() if Path(p).is_absolute() else (ROOT/p).resolve()
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def code_hashes():
 files=['pipe_hardening.py','train_hardening.py','evaluate_hardening.py','bind_hardening_inputs.py'];d={str((SOURCE/f).relative_to(ROOT)):sha(SOURCE/f) for f in files}
 for file,digest in read(OLD/'training-authorization.json')['codeSHA256'].items():assert sha(ROOT/file)==digest;d[file]=digest
 return d
APP_FILES={'PhotoProcessing':'Count stick/StickDetector.swift','PipeLetterbox':'Count stick/PipeDetector.swift','ExportHelper':'Count stickTests/PipePreprocessingExportTests.swift','NMS':'Count stick/PipeRawPostprocessor.swift'}
def app_hashes():return {k:sha(ROOT/v) for k,v in APP_FILES.items()}
def init_check():
 assert sha(INIT)==INIT_SHA and sha(OLD/'records.json')==OLD_RECORDS_SHA and sha(OLD/'selection.json')==OLD_SELECTION_SHA
 assert read(OLD/'selection.json')['selected']['modelSHA256']==INIT_SHA
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
 t=read(OLD/'protocol.json')['training'].copy();t.update(epochs=12,lr0=.0001,warmup_bias_lr=.0001,freeze=4)
 return {'status':'PROPOSED_PIPE_HARDENING_PENDING_DATA_AND_ROOT_FREEZE','task':'round_hollow_pipe_end_hard_negative_low_count','initializationSHA256':INIT_SHA,'training':t,'checkpointEpochs':[4,8,12],'samplesPerEpoch':320,'bucketMasses':{'old_positive':.5,'new_positive':.25,'negative':.25},'sampling':'bucket mass; uniform source within bucket; uniform globally frozen physical group within source; uniform view within group; actual consumed IDs and postaugmentation boxes logged','confidence':.3,'nmsIoU':.7,'candidateCount':3,'freezePolicy':'Ultralytics freeze=4 freezes model.0 through model.3 parameters and BN running statistics; fixed DFL integral stays frozen. No teacher/KD and no negative pseudo-labels. Assert frozen live state unchanged each epoch and before save.','evaluationVariants':['gallery','camera'],'oldValidationPolicy':{'TP':'not lower','FP':'not higher','FN':'not higher','positiveCountWAPE':'not higher','sceneMacroWAPE':'not higher','eachSizeBinTP':'not lower','everyNegativeImage':'zero'},'freshValidationPolicy':read(OLD/'protocol.json')['qualification'],'candidateSelection':'Only eligible candidates; minimize worst-path fresh positiveCountMAE, then worst-path old positiveCountMAE, then earlier epoch. Fixed confidence .3 and NMS IoU .7; no threshold grid.','holdoutPolicy':'No previous37 test images/GT in training or selection. New physical-scene holdout frozen before optimization and owned by root; choose once from old40+freshval then single newholdout evaluation. Never use former test to select checkpoints/thresholds.','realCropPolicy':'Only root-reviewed, unmodified RGB PNG crops from known masked parents may supplement zero-target train images. Parent SHA/path, integer rectangle, exact decoded pixels, manual cropReview PASS and inherited canonical group are required. Never allow real_crop in val/test or count it as an independent physical scene.','sampleCoverageLimit':'Weighted image balance does not equal GT loss balance; record actual postaugmentation source/bucket box exposures. No guarantee of unseen-scene nonregression.','runtime':'Keep original App preprocessing and shared Swift raw NMS; future candidate requires new raw FP32 export/parity and fresh holdout qualification.','rawLocations':21504,'appSourceCodeSHA256':app_hashes(),'oldRecordsSHA256':OLD_RECORDS_SHA,'oldSelectionSHA256':OLD_SELECTION_SHA,'dataReady':False}
def prepare(manifest_path):
 from PIL import Image
 init_check();manifest_path=resolve(manifest_path);manifest=read(manifest_path);assert manifest['status']=='FROZEN_ANNOTATION_QA_PASSED' and manifest['testContentIncluded'] is False
 assert not (OUT/'data-binding.json').exists(),'Prepared experiment is immutable; use an explicit new version for repairs'
 oldrows=read(OLD/'records.json');assert len(oldrows)==204 and {r['split'] for r in oldrows}=={'train','val'};fresh=manifest['images'];assert fresh and {r['split'] for r in fresh}=={'train','val'}
 ids={r['id'] for r in oldrows};group_split={str(r['groupId']):r['split'] for r in oldrows};hash_split={r['sha256']:r['split'] for r in oldrows};items=[]
 for r in fresh:
  assert r['id'] not in ids;ids.add(r['id']);assert r['annotationQA']=='PASS' and r['dataNature'] in ['real','real_crop'] and r['isSynthetic'] is False and r['hasAnnotationOverlay'] is False
  assert r['count']==len(r['xyxy']) and 0<=r['count']<=20 and r['split'] in ['train','val'];assert r['label']=='pipe_end'
  if r['count']:assert r['subtype']=='round' and r['annotationBoundary']=='outer_rim'
  else:assert r['negativeKind'] in ['non_pipe','empty_scene','side_only']
  group=str(r['groupId']);assert group_split.get(group,r['split'])==r['split'];group_split[group]=r['split'];assert hash_split.get(r['sha256'],r['split'])==r['split'];hash_split[r['sha256']]=r['split']
  path=resolve(r['imagePath']);assert path.is_relative_to(ROOT) and sha(path)==r['sha256']
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:assert im.size==(r['width'],r['height']) and im.getexif().get(274,1) in [None,1]
  validate_data_nature(r,path)
  assert all(0<=a<c<=r['width'] and 0<=b<d<=r['height'] for a,b,c,d in r['xyxy'])
  items.append({'id':r['id'],'source':str(path),'destination':str(OUT/'dataset/images'/r['split']/(r['id']+'.png'))})
 assert all(any(r['split']==s and r['count']==0 for r in fresh) and any(r['split']==s and 1<=r['count']<=20 for r in fresh) for s in ['train','val'])
 OUT.mkdir(parents=True,exist_ok=True);save(OUT/'photo-preparation-input.json',items);exe=OUT/'tools/prepare_photos';exe.parent.mkdir(exist_ok=True)
 subprocess.run(['xcrun','swiftc',str(ROOT/'ml/pipes-20261026/model/prepare_photos.swift'),'-o',str(exe)],check=True);subprocess.run([str(exe),str(OUT/'photo-preparation-input.json'),str(OUT/'photo-preparation-output.json')],check=True);photos={x['id']:x for x in read(OUT/'photo-preparation-output.json')};records=[]
 for r in oldrows:
  assert sha(r['preparedImage'])==r['preparedImageSHA256'] and sha(r['preparedLabel'])==r['preparedLabelSHA256']
  image=OUT/'dataset/images'/r['split']/(r['id']+'.png');label=OUT/'dataset/labels'/r['split']/(r['id']+'.txt');image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r['preparedImage'],image);shutil.copy2(r['preparedLabel'],label)
  records.append({**r,'preparedImage':str(image),'preparedLabel':str(label),'dataRole':'old','bucket':'old_positive' if r['count'] else 'negative'})
 for r in fresh:
  photo=photos[r['id']];w,h=photo['width'],photo['height'];sx,sy=w/r['width'],h/r['height'];boxes=[[a*sx,b*sy,c*sx,d*sy] for a,b,c,d in r['xyxy']];label=OUT/'dataset/labels'/r['split']/(r['id']+'.txt');label.parent.mkdir(parents=True,exist_ok=True);lines=[f'0 {(a+c)/2/w:.10f} {(b+d)/2/h:.10f} {(c-a)/w:.10f} {(d-b)/h:.10f}' for a,b,c,d in boxes];assert len(lines)==len(set(lines));label.write_text('\n'.join(lines)+('\n' if lines else ''))
  records.append({**r,'originalWidth':r['width'],'originalHeight':r['height'],'originalXYXY':r['xyxy'],'width':w,'height':h,'xyxy':boxes,'preparedImage':photo['destination'],'preparedImageSHA256':sha(photo['destination']),'preparedLabel':str(label),'preparedLabelSHA256':sha(label),'dataRole':'fresh','bucket':'new_positive' if r['count'] else 'negative'})
 save(OUT/'records.json',records);(OUT/'dataset/data.yaml').write_text('path: '+json.dumps(str(OUT/'dataset'))+'\ntrain: images/train\nval: images/val\nnames:\n  0: pipe_end\n')
 summary={s:{'images':sum(r['split']==s for r in records),'groups':len({r['groupId'] for r in records if r['split']==s}),'objects':sum(r['count'] for r in records if r['split']==s),'byRoleBucket':dict(collections.Counter(r['dataRole']+'/'+r['bucket'] for r in records if r['split']==s)),'byDataNature':dict(collections.Counter(r['dataNature'] for r in records if r['split']==s)),'cropParentGroups':len({str(r['parentGroupId']) for r in records if r['split']==s and r['dataNature']=='real_crop'})} for s in ['train','val']}
 binding={'status':'PREPARED_HARDENING_TRAIN_VAL_ONLY','manifestPath':str(manifest_path),'manifestSHA256':sha(manifest_path),'sealedFullManifestSHA256':manifest['sealedFullManifestSHA256'],'recordsSHA256':sha(OUT/'records.json'),'dataYAMLSHA256':sha(OUT/'dataset/data.yaml'),'oldRecordsSHA256':OLD_RECORDS_SHA,'oldSelectionSHA256':OLD_SELECTION_SHA,'initializationSHA256':INIT_SHA,'summary':summary,'testContentIncluded':False};save(OUT/'data-binding.json',binding)
 protocol=proposal();protocol.update(status='FROZEN_PIPE_HARDENING_PROTOCOL_PENDING_ROOT_RECEIPT',registeredUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataReady=True,manifestSHA256=sha(manifest_path),dataBindingSHA256=sha(OUT/'data-binding.json'));save(OUT/'protocol.json',protocol)
 smoke=OUT/'smoke-v2/smoke-report.json';assert read(smoke)['status']=='PASS_ACTUAL_WEIGHTED_DATALOADER_AND_TRUE_CALLBACK_CHECKPOINT_LIFECYCLE'
 save(OUT/'root-training-receipt.template.json',{'status':'ROOT_APPROVAL_REQUIRED','protocolSHA256':sha(OUT/'protocol.json'),'dataBindingSHA256':sha(OUT/'data-binding.json'),'codeSHA256':code_hashes(),'initializationSHA256':INIT_SHA,'appSourceCodeSHA256':app_hashes(),'saveLifecycleSmokePath':str(smoke),'saveLifecycleSmokeSHA256':sha(smoke),'qaReportPath':None,'qaReportSHA256':None,'qaRequirements':['old37 excluded from train/val; grouped fresh test sealed before optimizer','real data/labels reviewed; source/group/pixel disjoint across train/val/test','fresh positive and nonpipe negative train+val present']});return binding

def verify(receipt_path):
 init_check();r=read(receipt_path);protocol=read(OUT/'protocol.json');b=read(OUT/'data-binding.json');assert r['status']=='APPROVED_PIPE_HARDENING_TRAINING'
 assert r['protocolSHA256']==sha(OUT/'protocol.json') and r['dataBindingSHA256']==sha(OUT/'data-binding.json') and r['codeSHA256']==code_hashes() and r['initializationSHA256']==INIT_SHA
 assert protocol['status']=='FROZEN_PIPE_HARDENING_PROTOCOL_PENDING_ROOT_RECEIPT' and protocol['appSourceCodeSHA256']==r['appSourceCodeSHA256']==app_hashes();assert sha(b['manifestPath'])==b['manifestSHA256'] and sha(OUT/'records.json')==b['recordsSHA256'] and sha(OUT/'dataset/data.yaml')==b['dataYAMLSHA256']
 assert sha(resolve(r['qaReportPath']))==r['qaReportSHA256'] and read(resolve(r['qaReportPath']))['status'].startswith('PASS')
 smoke_path=resolve(r['saveLifecycleSmokePath']);assert sha(smoke_path)==r['saveLifecycleSmokeSHA256'];smoke=read(smoke_path)
 assert smoke['status']=='PASS_ACTUAL_WEIGHTED_DATALOADER_AND_TRUE_CALLBACK_CHECKPOINT_LIFECYCLE' and smoke['operatorSHA256']==sha(SOURCE/'train_hardening.py') and smoke['optimizerSteps']==0 and smoke['samplerPrefixMatched'] and smoke['frozenLiveStateExact']
 for epoch,meta in smoke['snapshots'].items():assert sha(smoke_path.parent/'checkpoints'/f'epoch{int(epoch):03d}.pt')==meta['sha256']
 for row in read(OUT/'records.json'):assert row['split'] in ['train','val'] and sha(row['preparedImage'])==row['preparedImageSHA256'] and sha(row['preparedLabel'])==row['preparedLabelSHA256']
 return r
