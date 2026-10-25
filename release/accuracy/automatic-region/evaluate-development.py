#!/usr/bin/env python3
import datetime,hashlib,importlib.util,json,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evaluation',ROOT/'release/accuracy/web-set/evaluate.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
sets=[('v1',ROOT/'release/accuracy/web-set/manifest.json',ROOT/'output/web-annotation/evaluation-v1/predictions.json'),('v2',ROOT/'release/accuracy/web-set-v2/manifest.json',ROOT/'output/web-annotation-v2/evaluation/predictions.json')]
items=[];mappings={};cached={};provenance=[]
for name,manifest_path,pred_path in sets:
 _,active,_=ev.validate_manifest(manifest_path,ROOT)
 preds=json.loads(pred_path.read_text());byid={i['imageId']:i for i in preds['images']}
 for item in active:
  image=byid[item['id']];assert image['imageSha256']==item['sha256']
  variant=image['variants']['candidate'];assert variant['confidenceThreshold']==.25 and variant['iouThreshold']==.45
  det=[];mapping={}
  for p in variant['predictions']:
   identifier=str(uuid.uuid5(uuid.NAMESPACE_URL,item['id']+'/'+p['id'])).upper();mapping[identifier]=p
   x,y,w,h=p['bbox'];det.append({'id':identifier,'box':{'x':x/item['width'],'y':y/item['height'],'width':w/item['width'],'height':h/item['height']},'confidence':p['confidence']})
  items.append({**item,'dataset':name,'detections':det});mappings[item['id']]=mapping;cached[item['id']]=variant['predictions']
 provenance.append({'dataset':name,'manifestSha256':ev.digest(manifest_path),'cachedPredictionPath':str(pred_path.relative_to(ROOT)),'cachedPredictionSha256':ev.digest(pred_path),'originalDetectorSha256':preds.get('sourceDetectorSha256'),'detectorNotRerun':True})
protocol=HERE/'protocol-v1.json';plan=json.loads(protocol.read_text())
if 'declaredAtUTC' not in plan:
 plan['declaredAtUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();protocol.write_text(json.dumps(plan,indent=2)+'\n')
output=ROOT/'output/automatic-region/development-v1';output.mkdir(parents=True,exist_ok=True)
input_data={'protocolSha256':ev.digest(protocol),'geometrySourceSha256':ev.digest(ROOT/'Count stick/AutoCountRegion.swift'),'images':[{k:item[k] for k in ['id','path','sha256','width','height','detections']} for item in items]}
input_path=output/'input.json';input_path.write_text(json.dumps(input_data,indent=2)+'\n')
runner=output/'runner'
subprocess.run(['swiftc',str(ROOT/'Count stick/AutoCountRegion.swift'),str(HERE/'run-development.swift'),'-o',str(runner)],check=True)
result_path=output/'filter-results.json';subprocess.run([str(runner),str(input_path),str(result_path)],check=True)
results=json.loads(result_path.read_text());byid={i['imageId']:i for i in results['images']}
rows=[]
for item in items:
 out=byid[item['id']];row={'imageId':item['id'],'dataset':item['dataset'],'baseline':ev.score_image(item,cached[item['id']]),'ocrAreaCount':len(out['textAreas']),'variants':{}}
 for name,variant in out['variants'].items():
  kept=[mappings[item['id']][identifier] for identifier in variant['keptIDs']]
  row['variants'][name]={'metrics':ev.score_image(item,kept),'fallbackReason':variant['fallbackReason'],'regionCount':len(variant['regions']),'excludedByReason':{reason:sum(x['reason']==reason for x in variant['excluded']) for reason in ['text','isolated']},'excludedOriginalIds':[mappings[item['id']][x['id']]['id'] for x in variant['excluded']]}
 if 'ocrError' in out:row['ocrError']=out['ocrError']
 rows.append(row)
summary={}
for name in ['text_only','geometry_only','combined']:
 metrics=[r['variants'][name]['metrics'] for r in rows];countable=[m for m in metrics if m['exactCountEligible']]
 summary[name]={'TP':sum(m['TP'] for m in metrics),'FP':sum(m['FP'] for m in metrics),'FN':sum(m['FN'] for m in metrics),'countMAE':sum(m['absoluteCountError'] for m in countable)/len(countable),'exactCountImages':sum(m['exactCount'] for m in countable),'countEligibleImages':len(countable)}
baselines=[r['baseline'] for r in rows];countable=[m for m in baselines if m['exactCountEligible']]
summary['baseline']={'TP':sum(m['TP'] for m in baselines),'FP':sum(m['FP'] for m in baselines),'FN':sum(m['FN'] for m in baselines),'countMAE':sum(m['absoluteCountError'] for m in countable)/len(countable),'exactCountImages':sum(m['exactCount'] for m in countable),'countEligibleImages':len(countable)}
report={'status':'DEVELOPMENT_ONLY_NOT_BLIND_HOLDOUT','completedAtUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocolSha256':input_data['protocolSha256'],'geometrySourceSha256':input_data['geometrySourceSha256'],'provenance':provenance,'summary':summary,'images':rows,'limits':['Already-seen v1/v2 labels and cached detections were used for development diagnostics.','No new CoreML detector inference or weight/NMS change. Apple Vision OCR was run separately.','All assertions of accuracy must await a frozen new v3 evaluation.']}
(HERE/'development-v1-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,indent=2))
for r in rows:
 b=r['baseline'];print(r['imageId'],'baseline',(b['TP'],b['FP'],b['FN']), 'OCR',r['ocrAreaCount'])
 for name,v in r['variants'].items():
  m=v['metrics'];print(' ',name,(m['TP'],m['FP'],m['FN']),'count',m['predictedWholeImageCount'],v['excludedByReason'],'fallback',v['fallbackReason'])
