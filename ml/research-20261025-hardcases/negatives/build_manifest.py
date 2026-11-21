#!/usr/bin/env python3
"""Bind visual zero-colony annotations, preserve quarantine, leave split to root."""
from pathlib import Path
import collections,hashlib,json,time
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/research-20261025-hardcases/negatives'
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 qa=read(HERE/'visual-review.json');assert len(qa['records'])==64
 m={r['filename']:r for r in read(OUT/'mendeley-review-index.json')}
 b={r['sourcePath']:r for r in read(OUT/'bruker-download-receipt.json')['records']}
 z={Path(r['imagePath']).name:r for r in read(OUT/'zenodo-empty-transfer.json')}
 items=[];excluded=[]
 for review in qa['records']:
  source=review['source'];name=review['filename']
  if source=='MendeleyNegative':
   original=m[name];image=ROOT/original['imagePath'];expected=original['normalizedSHA256'];origin=original['path'];source_sha=original['sha256'];group='mendeley_xrbtd74pfj_v3_negative_source_group';source_id='mendeley-xrbtd74pfj-v3';id_='negative_mendeley_'+Path(name).stem;provenance={'dataset':'https://data.mendeley.com/datasets/xrbtd74pfj/3','paper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC10024077/','downloadURL':original['sourceURL'],'authorCategory':'Negative/no growth','normalization':original['normalization'],'groupBasis':'Entire Negative source held together because public physical-plate IDs unavailable; visual similarities/pose variants cannot be assumed independent.','reviewPreview':original['reviewPreview']}
  elif source=='BRUKERAGAR':
   original=b[name];image=ROOT/original['path'];expected=original['sha256'];origin=original['path'];source_sha=expected;medium=name.split('/')[0];group='brukeragar_'+medium+'_plate0000001';source_id='brukeragar-public-5cbee6d';id_='negative_bruker_'+medium+'_'+('base' if Path(name).stem.endswith('0000001') else Path(name).stem.split('_')[-1]);provenance={'dataset':'https://github.com/michalcicatka/BRUKERAGAR-DATASET','commit':'5cbee6da8785b9bd6abf1b23a0a86e60afd7076d','paper':'https://doi.org/10.1016/j.heliyon.2024.e25714','downloadURL':original['sourceURL'],'authorCategory':'Uncultivated empty agar plate','normalization':'None; original verified LFS bytes.','groupBasis':'Identical medium/plate0000001 across all lighting/views.'}
  else:
   original=z[name];image=ROOT/original['imagePath'];expected=original['sha256'];origin=original['sourcePath'];source_sha=expected;group='zenodo18703636_session_20230710';source_id='zenodo18703636';id_='negative_zenodo_'+name.split('_iOS')[0];provenance={'dataset':'https://zenodo.org/records/18703636','originalDatasetAlias':'https://zenodo.org/records/18505210','archiveSHA256':'2c9762e8d89234b4dcd9c38a37b2d6478e4356ec4317c1f8d48b1ba7391bcef5','downloadReceipt':'output/research-20261025-hardcases/positives/sources/download-receipts.json','authorCategory':'COCO zero boxes','sourceOriginalSplit':original['sourceOriginalSplit'],'normalization':'None; byte-identical copy from verified original archive.','groupBasis':'All four empty images captured within46seconds; conservatively same acquisition-date group; root must union same-day positive capture groups before splitting.'}
  assert sha(image)==expected
  with Image.open(image) as im:im.load();assert im.mode=='RGB';width,height=im.size;assert im.getexif().get(274,1) in (1,None)
  r={'id':id_,'source':source_id,'species':'empty_plate','groupId':group,'bucket':'empty_negative','imagePath':str(image.relative_to(ROOT)),'width':width,'height':height,'sha256':expected,'sourceImagePath':origin,'sourceImageSHA256':source_sha,'annotationMethod':'Codex visual review of full real photograph, corroborated by author source annotation; zero denotes no visible colony, not certified biological sterility.','negativeKind':'real_empty_plate','visualReview':{'decision':review['decision'],'notes':review['notes'],'resolution':review['reviewedAt']},'provenance':provenance}
  if review['decision']=='PASS_ZERO':
   label=OUT/'labels'/(id_+'.txt');label.parent.mkdir(exist_ok=True);label.write_text('');r.update({'xyxy':[],'count':0,'annotationQA':'PASS','labelPath':str(label.relative_to(ROOT)),'labelSHA256':sha(label)});items.append(r)
  else:r.update({'xyxy':None,'count':None,'annotationQA':'QUARANTINED','trainingEligible':False});excluded.append(r)
 assert len(items)==40 and len(excluded)==24 and len({r['id'] for r in items+excluded})==64
 assert len({r['sha256'] for r in items+excluded})==64
 groups=collections.Counter(r['groupId'] for r in items)
 manifest={'status':'ANNOTATION_QA_PASSED_PENDING_ROOT_GLOBAL_SPLIT','createdUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'testContentIncluded':False,'prior36ClinicalHoldoutRead':False,'inferenceOrTrainingPerformed':False,'splitAssigned':False,'scope':'Real empty-plate candidates only. No synthetic blanks counted as data. Root must freeze global source/session groups and train/val split before training.','images':items,'reviewedRealImages':64,'acceptedRealImages':len(items),'quarantinedRealImages':len(excluded),'acceptedConservativeGroupCounts':dict(groups),'groupCountMeaning':'Four leakage-control groups, NOT40independentplates. Mendeley is one unresolved-source group; BRUKER groups are two physical plates; Zenodo is one acquisition-session group.','groupUnionRequired':'zenodo18703636_session_20230710 must union all positives captured same date/session.','visualReviewSHA256':sha(HERE/'visual-review.json'),'annotationPolicy':'ml/research-20261025-hardcases/annotation-policy.json','limitations':['Codex visual annotations are not certified microbiologist ground truth.','Mendeley uses a phone in controlled chamber, not freehand field acceptance.','40files span only4conservative groups and3sources; many are views of correlated plates.','Quarantine means insufficient visual certainty, not a medical diagnosis or confirmed author-label error.']}
 (HERE/'negative-candidates.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (HERE/'quarantine.json').write_text(json.dumps({'status':'EXCLUDED_NOT_ZERO_LABELED','images':excluded},indent=2)+'\n')
 (OUT/'negative-candidates.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'accepted':len(items),'quarantined':len(excluded),'groups':dict(groups),'manifestSHA256':sha(HERE/'negative-candidates.json')},indent=2))
if __name__=='__main__':main()
