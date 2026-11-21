"""Append five reviewed training views; preserve all previous69 rows and holdout."""
from pathlib import Path
import copy,datetime,hashlib,json,re
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'ml/pipes-head-retention-20261026/data'
OUT=ROOT/'output/datasets/pipe-ends-head-retention-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def main():
 previous=ROOT/'output/datasets/pipe-ends-hardening-v1/manifest.json'
 previous_tv=previous.with_name('train-val-manifest.json')
 assert sha(previous)=='a4d64c40cf66779541d9edcdd0284c79712dcf4b465449129dfd2a40804eda50'
 assert sha(previous_tv)=='680de25252f1d9cf91f335b0bdea6b42e18a9fbfe6d4db74ef0cbb41f65d29e8'
 base=read(previous);old_rows=base['images'];assert len(old_rows)==69
 negative=ROOT/'output/pipes-head-retention-20261026/data/negatives/negative-candidates.json'
 assert sha(negative)=='ada7572a024d452bfa585bd06a99e7bed28015c7791f31fff48fd0f953ed5805'
 n=read(negative);assert n['status'].startswith('PASS') and len(n['images'])==4
 pos_review=ART/'positives/independent-maco-review.json'
 assert sha(pos_review)=='5c87866d42bb66ea0746d593ba973c31018d0bf8965eb6a2a9efe7e5966a07c2'
 p=next(x for x in read(pos_review)['images'] if x['id']=='new_maco_0343')
 assert p['status']=='PASS_TRAIN_ONLY_VISUAL_CANDIDATE' and p['authorCount']==p['independentCount']==12
 assert sha(ROOT/p['imagePath'])==p['imageSHA256'] and sha(ROOT/p['authorXMLPath'])==p['authorXMLSHA256']
 positive={'id':p['id'],'source':'zenodo17165968-MACO','groupId':'maco_round_global_00000343','split':'train',
           'imagePath':p['imagePath'],'sha256':p['imageSHA256'],'width':p['width'],'height':p['height'],
           'xyxy':p['authorBoxes'],'count':12,'label':'pipe_end','subtype':'round','annotationBoundary':'outer_rim',
           'annotationMethod':'AuthorHBB reused afterroot andindependent per-instance original/overlay review:10front mouths plus2partly occluded rear mouths. OriginalXML unchanged.',
           'annotationQA':'PASS','dataNature':'real','isSynthetic':False,'hasAnnotationOverlay':False,
           'sourceHadBlackMask':False,'negativeKind':None,'modelInferencePerformed':False,
           'authorLabelPath':p['authorXMLPath'],'authorLabelSHA256':p['authorXMLSHA256'],
           'sourcePage':'https://zenodo.org/records/17165968','bucket':'positive','finalTrainingAdmission':True,
           'targetPolicy':'One discernible front end of a metallic round pipe. Never count inner/rear aperture of the samepipe twice.'}
 extra=[positive]+copy.deepcopy(n['images'])
 reviews=[pos_review,ART/'negatives/dataset-qa.json',ART/'negatives/independent-visual-review.json']
 for r in reviews[1:]:assert read(r)['status'].startswith('PASS')
 bindings={str(p.relative_to(ROOT)):sha(p) for p in reviews+[negative,previous,previous_tv]}
 old241=read(ROOT/'output/datasets/pipe-ends-maco-v1/manifest.json')['images']
 used=old241+old_rows;used_ids={r['id'] for r in used};used_sha={r['sha256'] for r in used};used_rgb={r['RGBPixelSHA256'] for r in used};used_groups={r['groupId'] for r in used}
 for row in extra:
  assert re.fullmatch(r'[A-Za-z0-9_-]+',row['id']) and row['id'] not in used_ids
  assert row['split']=='train' and row['annotationQA']=='PASS' and row['dataNature']=='real'
  assert row['isSynthetic'] is False and row['hasAnnotationOverlay'] is False and row['sourceHadBlackMask'] is False
  assert row['modelInferencePerformed'] is False and row['count']==len(row['xyxy']) and 0<=row['count']<=20
  path=(ROOT/row['imagePath']).resolve();assert path.is_relative_to(ROOT) and sha(path)==row['sha256']
  with Image.open(path) as im:
   assert im.size==(row['width'],row['height']) and im.getexif().get(274,1) in [None,1]
   rgb=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
  if 'RGBPixelSHA256' in row:assert row['RGBPixelSHA256']==rgb
  row['RGBPixelSHA256']=rgb
  assert row['sha256'] not in used_sha and rgb not in used_rgb and row['groupId'] not in used_groups
  used_ids.add(row['id']);used_sha.add(row['sha256']);used_rgb.add(rgb);used_groups.add(row['groupId'])
  assert len(row['xyxy'])==len({tuple(b) for b in row['xyxy']})
  assert all(0<=a<c<=row['width'] and 0<=b<d<=row['height'] for a,b,c,d in row['xyxy'])
  row['finalTrainingAdmission']=True
 assert not OUT.exists()
 OUT.mkdir();labels=OUT/'added-labels';labels.mkdir()
 for row in extra:
  path=labels/(row['id']+'.txt');w,h=row['width'],row['height']
  text='\n'.join(f'0 {(a+c)/2/w:.10f} {(b+d)/2/h:.10f} {(c-a)/w:.10f} {(d-b)/h:.10f}' for a,b,c,d in row['xyxy'])
  path.write_text(text+('\n' if text else ''));row['labelPath']=str(path.relative_to(ROOT));row['labelSHA256']=sha(path)
 rows=copy.deepcopy(old_rows)+extra
 assert rows[:69]==old_rows
 assert [r for r in rows if r['split']=='test']==[r for r in old_rows if r['split']=='test']
 assert [r for r in rows if r['split']=='val']==[r for r in old_rows if r['split']=='val']
 summary={s:{'images':sum(r['split']==s for r in rows),'positiveImages':sum(r['split']==s and r['count']>0 for r in rows),'negativeImages':sum(r['split']==s and r['count']==0 for r in rows),'objects':sum(r['count'] for r in rows if r['split']==s),'groups':len({r['groupId'] for r in rows if r['split']==s})} for s in ['train','val','test']}
 manifest={'status':'FROZEN_ANNOTATION_QA_PASSED','createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'testContentIncluded':True,'images':rows,'summary':summary,'previousFullManifestSHA256':sha(previous),'previousTrainValManifestSHA256':sha(previous_tv),'original69RowsUnchanged':True,'fresh20RowsUnchanged':True,'fresh20RecognitionNeverRun':True,'new5TrainViewsRecognitionNeverRun':True,'new5QA':bindings,'freezerSHA256':sha(__file__),'historyNote':'Original69 row-level inference declarations describe their historical curation stage. Prior hardening evaluated the development rows; the20 fresh-test views remain untested. No attempt is made to relabel or resplit failed development examples.','excludedAdditionalCandidates':['MACO780 material/annotation completeness unresolved; notused','MACO528 partial opening scope unresolved; notused']}
 write(OUT/'manifest.json',manifest)
 tv={**manifest,'testContentIncluded':False,'images':[r for r in rows if r['split']!='test'],'summary':{s:summary[s] for s in ['train','val']},'sealedFullManifestSHA256':sha(OUT/'manifest.json')}
 assert len(tv['images'])==54 and tv['images'][:49]==read(previous_tv)['images']
 write(OUT/'train-val-manifest.json',tv)
 write(OUT/'identity-only.json',{'status':'FROZEN_IDENTITY_ONLY_NO_GT_OR_PIXELS','fullManifestSHA256':sha(OUT/'manifest.json'),'images':[{k:r[k] for k in ['id','split','groupId','sha256','RGBPixelSHA256']} for r in rows]})
 assert all(sha(ROOT/path)==digest for path,digest in bindings.items())
 qa={'status':'PASS_ROOT_HEAD_RETENTION_DATA_FREEZE','createdUTC':manifest['createdUTC'],'fullManifestPath':str((OUT/'manifest.json').relative_to(ROOT)),'fullManifestSHA256':sha(OUT/'manifest.json'),'trainValManifestPath':str((OUT/'train-val-manifest.json').relative_to(ROOT)),'trainValManifestSHA256':sha(OUT/'train-val-manifest.json'),'summary':summary,'original69RowsUnchanged':True,'original49DevelopmentRowsUnchanged':True,'same7FreshValidationRows':True,'same20FreshTestRows':True,'only5NaturalTrainViewsAdded':True,'noNewExactFilePixelGroupOverlap':True,'sourceReviewBindings':bindings,'newTrainingPerformed':False}
 write(ART/'root-freeze-qa.json',qa)
 print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
