"""Fourth pipe trial: audited natural data expansion only. No fallback manifest."""
from pathlib import Path
import collections,datetime,hashlib,json,math,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'output/pipes-data-expansion-20261029/model';SOURCE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'output/pipes-head-retention-20261026/model';OLD=ROOT/'output/pipes-20261026/model'
PREVIOUS_RECORDS_SHA='d4d4574d3cc8d7e4834fc04d92e0104ed4b6abcb4cb09fd69ee10983864d1592';PREVIOUS_INPUTS_SHA='365c908466a18d77a33e91c8eba8e7ce8d7d52a319750e761d2d7602f8cc3031';PREVIOUS_MANIFEST_SHA='b7185c895ed9c1d695d486a5b0dc3fbd9725510afbbb95c4b74b3eaea1e11393';PREVIOUS_SELECTION_SHA='d78abcdd21be0726b4ecfdf45ac20d7fb0cff08d496526cf971b06e2b72981a3'
VALIDATION_SOURCE_MANIFEST_SHA='680de25252f1d9cf91f335b0bdea6b42e18a9fbfe6d4db74ef0cbb41f65d29e8'
OLD_RECORDS_SHA='c01c9771fa7dd810c988df784d42fe6fd8d539b8253a7ca5518dcd0a78b13774';OLD_SELECTION_SHA='9eda69afbaa72634270f86f910e7f87719bbd55a5880d6dbb000e6d4a62209f8';INIT=OLD/'checkpoints/epoch020.pt';INIT_SHA='48a77fd9198b23695f470b172397e918c9113238bfef2d421d8bbd431f69af12'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics-config');os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1';sys.path.insert(0,str(ROOT/'ml/pipes-20261026/model'));sys.path.insert(0,str(ROOT/'ml/colony'))
APP_FILES={'PhotoProcessing':'Count stick/StickDetector.swift','PipeLetterbox':'Count stick/PipeDetector.swift','ExportHelper':'Count stickTests/PipePreprocessingExportTests.swift','NMS':'Count stick/PipeRawPostprocessor.swift'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def resolve(path):
 path=Path(path);path=(ROOT/path if not path.is_absolute() else path).resolve();assert path.is_relative_to(ROOT),'Path outside repository';return path
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def app_hashes():return {k:sha(ROOT/v) for k,v in APP_FILES.items()}
def code_hashes():
 files=['pipe_data_expansion.py','prepare_data.py','train_data_expansion.py','evaluate_data_expansion.py','bind_expansion_inputs.py','head_kd.py'];d={str((SOURCE/f).relative_to(ROOT)):sha(SOURCE/f) for f in files}
 for file,digest in read(PREVIOUS/'training-authorization.json')['codeSHA256'].items():assert sha(ROOT/file)==digest;d[file]=digest
 return d

def init_check():
 assert sha(INIT)==INIT_SHA and sha(OLD/'records.json')==OLD_RECORDS_SHA and sha(OLD/'selection.json')==OLD_SELECTION_SHA
 assert sha(PREVIOUS/'records.json')==PREVIOUS_RECORDS_SHA and sha(PREVIOUS/'evaluation-inputs.json')==PREVIOUS_INPUTS_SHA and sha(PREVIOUS/'selection.json')==PREVIOUS_SELECTION_SHA
 assert read(PREVIOUS/'selection.json')['selected'] is None

def proposal():
 previous=read(PREVIOUS/'protocol.json');keep=['task','initializationSHA256','training','checkpointEpochs','samplesPerEpoch','bucketMasses','sampling','confidence','nmsIoU','candidateCount','freezePolicy','evaluationVariants','oldValidationPolicy','freshValidationPolicy','candidateSelection','holdoutPolicy','sampleCoverageLimit','runtime','rawLocations','appSourceCodeSHA256','oldRecordsSHA256','oldSelectionSHA256','KD','additionalAugmentation'];q={k:previous[k] for k in keep};q.update(status='PENDING_DATA_EXPANSION_ROOT_FREEZE',task='natural_data_expansion_same_head_retention_recipe',dataReady=False,previousRecordsSHA256=PREVIOUS_RECORDS_SHA,previousInputsSHA256=PREVIOUS_INPUTS_SHA,validationSourceManifestSHA256=VALIDATION_SOURCE_MANIFEST_SHA)
 q['dataOnlyIntervention']='Only add root-audited complete natural training photos. No new degradation, architecture, KD coefficient, sampling rule, operating threshold or acceptance gate.'
 q['coveragePlanningTargets']={'positiveNewPhysicalGroups':24,'newPositiveObjects':150,'nativeMaxEdgeUnder512PositiveGroups':8,'positive1to5Groups':8,'positive6to20Groups':8,'obliqueOrLongBodyPositiveGroups':8,'naturalNegativeGroups':16,'cleanSmoothSolidBarGroups':8,'targetsAreHardAcceptanceGates':False,'note':'Coverage targets guide acquisition, not a universal dataset sufficiency rule. Root reviews actual quality and exposure before approval.'}
 q['resourceBound']={'epochs':12,'drawsPerEpoch':320,'totalDraws':3840,'validationInputs':94,'checkpoints':3,'predictedTrainingMinutes':'11–18 under similar contention','operatorBudgetTrainingMinutes':25,'automaticAdditionalRounds':0,'boundMeaning':'If a run is unexpectedly slow or fails, preserve diagnostics and consult root; do not silently extend or change batch.'}
 return q

def exposure_audit(records,added_ids):
 train=[r for r in records if r['split']=='train'];masses={'old_positive':.5,'new_positive':.25,'negative':.25};sources={b:{r['source'] for r in train if r['bucket']==b} for b in masses};groups={(b,s):{str(r['groupId']) for r in train if r['bucket']==b and r['source']==s} for b in masses for s in sources[b]};views=collections.Counter((r['bucket'],r['source'],str(r['groupId'])) for r in train);weights=[];draws=3840
 for r in train:
  b,s,g=r['bucket'],r['source'],str(r['groupId']);weights.append(masses[b]/len(sources[b])/len(groups[(b,s)])/views[(b,s,g)])
 assert abs(sum(weights)-1)<1e-9
 def rollup(indices):
  rr=[train[i] for i in indices];return {'views':len(rr),'physicalGroups':len({r['groupId'] for r in rr}),'originalGT':sum(r['count'] for r in rr),'expectedDraws':sum(weights[i]*draws for i in indices),'expectedOriginalGTExposure':sum(weights[i]*draws*train[i]['count'] for i in indices)}
 by_source={f'{b} | {s}':rollup([i for i,r in enumerate(train) if r['bucket']==b and r['source']==s]) for b in masses for s in sorted(sources[b])};per_group={str(g):rollup([i for i,r in enumerate(train) if str(r['groupId'])==g]) for g in sorted({str(r['groupId']) for r in train})};added=[r for r in train if r['id'] in added_ids];parent_groups={str(r['groupId']) for r in train if r['id'] not in added_ids}
 def group_count(predicate):return len({str(r['groupId']) for r in added if predicate(r)})
 return {'status':'ANALYTICAL_SAMPLING_EXPOSURE_NOT_LOSS_OR_ACTUAL_AUGMENTED_GT','byBucket':{b:rollup([i for i,r in enumerate(train) if r['bucket']==b]) for b in masses},'bySourceWithinBucket':by_source,'byPhysicalGroup':per_group,'addedData':rollup([i for i,r in enumerate(train) if r['id'] in added_ids]),'addedCoverage':{'positiveGroups':group_count(lambda r:r['count']>0),'positiveNovelGroupsBeyondParentTrain':len({str(r['groupId']) for r in added if r['count']>0}-parent_groups),'negativeNovelGroupsBeyondParentTrain':len({str(r['groupId']) for r in added if r['count']==0}-parent_groups),'addedViewsInPreviouslySeenTrainGroup':sum(str(r['groupId']) in parent_groups for r in added),'negativeGroups':group_count(lambda r:r['count']==0),'positiveGT':sum(r['count'] for r in added),'positive1to5Groups':group_count(lambda r:1<=r['count']<=5),'positive6to20Groups':group_count(lambda r:6<=r['count']<=20),'nativeMaxEdgeUnder512PositiveGroups':group_count(lambda r:r['count']>0 and max(r['originalWidth'],r['originalHeight'])<512),'cleanSmoothSolidBarGroupsDeclared':group_count(lambda r:r['count']==0 and r.get('confuserSubtype') in ['smooth_solid_bar','smooth_solid_rod_end','clean_smooth_solid_bar'])},'meaning':'Planned image/group/GT exposures only. They are not gradient magnitudes and not postaugmentation loss-box counts. Actual consumed images and GT remain logged by the unchanged trainer.'}

def prepare(freeze_receipt_path):
 # No default manifest and no data read before an explicit root-approved freeze receipt.
 receipt_path=resolve(freeze_receipt_path);receipt=read(receipt_path);assert receipt['status']=='APPROVED_DATA_EXPANSION_PREPARATION';init_check();manifest_path=resolve(receipt['manifestPath']);assert 'train-val' in manifest_path.name
 assert receipt['parentTrainValManifestSHA256']==PREVIOUS_MANIFEST_SHA;assert receipt['manifestSHA256'] not in {PREVIOUS_MANIFEST_SHA,VALIDATION_SOURCE_MANIFEST_SHA,'505d0a3de3424b75d53632e5349453d9e9f82faf2d831778f07f6be5a4072476'}
 assert sha(manifest_path)==receipt['manifestSHA256'];qa_path=resolve(receipt['qaReportPath']);assert sha(qa_path)==receipt['qaReportSHA256'] and read(qa_path)['status'].startswith('PASS')
 assert not (OUT/'data-binding.json').exists() and not (OUT/'dataset').exists(),'Do not overwrite previous preparation attempts'
 manifest=read(manifest_path);assert manifest['status']=='FROZEN_ANNOTATION_QA_PASSED' and manifest['testContentIncluded'] is False and manifest['sealedFullManifestSHA256']==receipt['sealedFullManifestSHA256']
 parent_binding=read(PREVIOUS/'data-binding.json');assert sha(parent_binding['manifestPath'])==PREVIOUS_MANIFEST_SHA
 parent_tv=read(parent_binding['manifestPath'])['images'];new_tv=manifest['images'];assert len(parent_tv)==54 and len(new_tv)>54 and len({r['id'] for r in new_tv})==len(new_tv);combined={r['id']:r for r in new_tv}
 for row in parent_tv:assert combined[row['id']]==row,'Previous54 train/val rows changed'
 prior_ids={r['id'] for r in parent_tv};extra=[r for r in new_tv if r['id'] not in prior_ids];assert extra and all(r['split']=='train' for r in extra),'All expansion views must be train-only'
 from PIL import Image
 base=read(PREVIOUS/'records.json');assert len(base)==258 and sum(r['split']=='val' for r in base)==47;ids={r['id'] for r in base};group_split={str(r['groupId']):r['split'] for r in base};all_shas={r['sha256'] for r in base};all_rgb={r['RGBPixelSHA256'] for r in base};items=[]
 for row in extra:
  assert row['id'] not in ids;ids.add(row['id']);assert row['sha256'] not in all_shas,'Exact duplicate is not new data';all_shas.add(row['sha256']);assert group_split.get(str(row['groupId']),'train')=='train';group_split[str(row['groupId'])]='train'
  assert row['annotationQA']=='PASS' and row['dataNature']=='real' and row['isSynthetic'] is False and row['hasAnnotationOverlay'] is False and not row.get('sourceHadBlackMask',False);assert not any(k in row for k in ['cropRectPixels','sourceCropXYXY'])
  assert row['label']=='pipe_end' and row['count']==len(row['xyxy']) and 0<=row['count']<=21504
  if row['count']:assert row['subtype']=='round' and row['annotationBoundary']=='outer_rim'
  else:assert row['negativeKind'] in ['non_pipe','empty_scene','side_only']
  path=resolve(row['imagePath']);assert sha(path)==row['sha256']
  with Image.open(path) as im:im.verify()
  with Image.open(path) as im:
   assert im.size==(row['width'],row['height']) and im.getexif().get(274,1) in [None,1];rgb_sha=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest();assert rgb_sha==row['RGBPixelSHA256'] and rgb_sha not in all_rgb,'Duplicate/changed decoded RGB is not new data';all_rgb.add(rgb_sha)
  assert all(all(math.isfinite(v) for v in [a,b,c,d]) and 0<=a<c<=row['width'] and 0<=b<d<=row['height'] for a,b,c,d in row['xyxy'])
  items.append({'id':row['id'],'source':str(path),'destination':str(OUT/'dataset/images/train'/(row['id']+'.png'))})
 OUT.mkdir(parents=True,exist_ok=True);save(OUT/'photo-preparation-input.json',items);exe=OUT/'tools/prepare_photos';exe.parent.mkdir(exist_ok=True);subprocess.run(['xcrun','swiftc',str(ROOT/'ml/pipes-20261026/model/prepare_photos.swift'),'-o',str(exe)],check=True);subprocess.run([str(exe),str(OUT/'photo-preparation-input.json'),str(OUT/'photo-preparation-output.json')],check=True);photos={r['id']:r for r in read(OUT/'photo-preparation-output.json')};records=[]
 for row in base:
  assert sha(row['preparedImage'])==row['preparedImageSHA256'] and sha(row['preparedLabel'])==row['preparedLabelSHA256'];image=OUT/'dataset/images'/row['split']/(row['id']+'.png');label=OUT/'dataset/labels'/row['split']/(row['id']+'.txt');image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(row['preparedImage'],image);shutil.copy2(row['preparedLabel'],label);records.append({**row,'preparedImage':str(image),'preparedLabel':str(label)})
 for row in extra:
  photo=photos[row['id']];w,h=photo['width'],photo['height'];sx,sy=w/row['width'],h/row['height'];boxes=[[a*sx,b*sy,c*sx,d*sy] for a,b,c,d in row['xyxy']];label=OUT/'dataset/labels/train'/(row['id']+'.txt');lines=[f'0 {(a+c)/2/w:.10f} {(b+d)/2/h:.10f} {(c-a)/w:.10f} {(d-b)/h:.10f}' for a,b,c,d in boxes];assert len(lines)==len(set(lines));label.write_text('\n'.join(lines)+('\n' if lines else ''));records.append({**row,'originalWidth':row['width'],'originalHeight':row['height'],'originalXYXY':row['xyxy'],'width':w,'height':h,'xyxy':boxes,'preparedImage':photo['destination'],'preparedImageSHA256':sha(photo['destination']),'preparedLabel':str(label),'preparedLabelSHA256':sha(label),'dataRole':'fresh','bucket':'new_positive' if row['count'] else 'negative'})
 assert len(records)==258+len(extra) and sum(r['split']=='val' for r in records)==47;save(OUT/'records.json',records);(OUT/'dataset/data.yaml').write_text('path: '+json.dumps(str(OUT/'dataset'))+'\ntrain: images/train\nval: images/val\nnames:\n  0: pipe_end\n');audit=exposure_audit(records,{r['id'] for r in extra});save(OUT/'planned-exposure.json',audit)
 binding={'status':'PREPARED_DATA_EXPANSION_TRAIN_ONLY_ADDITION','manifestPath':str(manifest_path),'manifestSHA256':sha(manifest_path),'sealedFullManifestSHA256':manifest['sealedFullManifestSHA256'],'freezeReceiptPath':str(receipt_path),'freezeReceiptSHA256':sha(receipt_path),'recordsSHA256':sha(OUT/'records.json'),'dataYAMLSHA256':sha(OUT/'dataset/data.yaml'),'previousRecordsSHA256':PREVIOUS_RECORDS_SHA,'previousInputsSHA256':PREVIOUS_INPUTS_SHA,'validationSourceManifestSHA256':VALIDATION_SOURCE_MANIFEST_SHA,'initializationSHA256':INIT_SHA,'addedTrainImages':len(extra),'summary':{'train':sum(r['split']=='train' for r in records),'val':47},'old258CopiedExactly':True,'testContentIncluded':False,'plannedExposureSHA256':sha(OUT/'planned-exposure.json')};save(OUT/'data-binding.json',binding)
 protocol=proposal();protocol.update(status='FROZEN_DATA_EXPANSION_PROTOCOL_PENDING_ROOT_RECEIPT',registeredUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataReady=True,manifestSHA256=sha(manifest_path),dataBindingSHA256=sha(OUT/'data-binding.json'));save(OUT/'protocol.json',protocol)
 save(OUT/'root-training-receipt.template.json',{'status':'ROOT_APPROVAL_REQUIRED','protocolSHA256':sha(OUT/'protocol.json'),'dataBindingSHA256':sha(OUT/'data-binding.json'),'codeSHA256':code_hashes(),'initializationSHA256':INIT_SHA,'appSourceCodeSHA256':app_hashes(),'qaReportPath':str(qa_path),'qaReportSHA256':sha(qa_path),'preflightPath':str(SOURCE/'implementation-preflight.json'),'preflightSHA256':sha(SOURCE/'implementation-preflight.json')});return binding

def verify(receipt_path):
 init_check();receipt=read(resolve(receipt_path));protocol=read(OUT/'protocol.json');binding=read(OUT/'data-binding.json');assert receipt['status']=='APPROVED_DATA_EXPANSION_TRAINING'
 assert receipt['protocolSHA256']==sha(OUT/'protocol.json') and receipt['dataBindingSHA256']==sha(OUT/'data-binding.json') and receipt['codeSHA256']==code_hashes() and receipt['initializationSHA256']==INIT_SHA
 assert protocol['status']=='FROZEN_DATA_EXPANSION_PROTOCOL_PENDING_ROOT_RECEIPT' and protocol['appSourceCodeSHA256']==receipt['appSourceCodeSHA256']==app_hashes();assert sha(binding['manifestPath'])==binding['manifestSHA256'] and sha(OUT/'records.json')==binding['recordsSHA256'] and sha(OUT/'dataset/data.yaml')==binding['dataYAMLSHA256']
 assert sha(binding['freezeReceiptPath'])==binding['freezeReceiptSHA256'] and read(binding['freezeReceiptPath'])['status']=='APPROVED_DATA_EXPANSION_PREPARATION'
 assert sha(resolve(receipt['qaReportPath']))==receipt['qaReportSHA256'] and read(resolve(receipt['qaReportPath']))['status'].startswith('PASS')
 preflight=resolve(receipt['preflightPath']);assert sha(preflight)==receipt['preflightSHA256'];pre=read(preflight);assert pre['status']=='PASS_SOURCE_REUSE_DATA_ADAPTER_PENDING_BINDINGS' and pre['codeSHA256']==code_hashes()
 for row in read(OUT/'records.json'):assert row['split'] in ['train','val'] and sha(row['preparedImage'])==row['preparedImageSHA256'] and sha(row['preparedLabel'])==row['preparedLabelSHA256']
 return receipt
