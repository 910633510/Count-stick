"""Bind old validated PNGs plus fresh actual-App validation PNGs; no model run."""
from pathlib import Path
import argparse
import pipe_hardening as p
from bind_app_inputs import inspect_manifest

def bind(paths):
 b=p.read(p.OUT/'data-binding.json');assert p.sha(p.OUT/'records.json')==b['recordsSHA256'];protocol=p.read(p.OUT/'protocol.json');assert protocol['appSourceCodeSHA256']==p.app_hashes()
 records={r['id']:r for r in p.read(p.OUT/'records.json') if r['split']=='val'};old={i:r for i,r in records.items() if r['dataRole']=='old'};fresh={i:r for i,r in records.items() if r['dataRole']=='fresh'}
 assert p.sha(p.OLD/'selection.json')==p.OLD_SELECTION_SHA
 assert p.sha(p.OLD/'evaluation-inputs.json')==p.read(p.OLD/'selection.json')['inputsManifestSHA256'],'Original baseline App inputs changed'
 original=p.read(p.OLD/'evaluation-inputs.json');lineage=p.read(p.ROOT/'ml/pipes-20261026/parity/runtime-v3-independent-review.json');assert lineage['status']=='PASS_PIPE_RUNTIME_V3_SOURCE_AMENDMENT';assert original['approvedAppSourceCodeSHA256']==lineage['oldAppSourceCodeSHA256'];assert {k:p.app_hashes()[k] for k in original['approvedAppSourceCodeSHA256']}==lineage['newAppSourceCodeSHA256']
 rows=[]
 for x in original['images']:
  assert x['id'] in old and x['split']=='val' and x['sourceImageSHA256']==old[x['id']]['sha256'];assert p.sha(x['networkImagePath'])==x['networkImageSHA256'] and p.sha(x['photoImagePath'])==x['photoImageSHA256'];rows.append({**x,'dataRole':'old'})
 assert len(rows)==80
 receipts=[];approved={k:v for k,v in protocol['appSourceCodeSHA256'].items() if k!='NMS'}
 for path in paths:
  manifest,rr=inspect_manifest(path);assert manifest['kind']=='frozen_train_val_photo' and manifest['sourceCodeSHA256']==approved;receipts.append({'path':str(Path(path).resolve()),'sha256':p.sha(path)})
  for row in rr:
   assert row['id'] in fresh and row['split']=='val' and row['sourceImageSHA256']==fresh[row['id']]['sha256'] and row['sourceManifestSHA256']==b['manifestSHA256'];rows.append({**row,'dataRole':'fresh'})
 assert len(rows)==2*len(records) and {(r['id'],r['variant']) for r in rows}=={(i,v) for i in records for v in ['gallery','camera']}
 result={'status':'VALIDATED_HARDENING_APP_INPUTS','testContentIncluded':False,'variants':['gallery','camera'],'recordsSHA256':p.sha(p.OUT/'records.json'),'approvedAppSourceCodeSHA256':p.app_hashes(),'images':rows,'oldInputManifestPath':str(p.OLD/'evaluation-inputs.json'),'oldInputManifestSHA256':p.sha(p.OLD/'evaluation-inputs.json'),'oldRuntimeReviewPath':str(p.ROOT/'ml/pipes-20261026/parity/runtime-v3-independent-review.json'),'oldRuntimeReviewSHA256':p.sha(p.ROOT/'ml/pipes-20261026/parity/runtime-v3-independent-review.json'),'freshAppExportManifests':receipts,'modelInferencePerformed':False};assert not (p.OUT/'evaluation-inputs.json').exists();p.save(p.OUT/'evaluation-inputs.json',result);return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('manifests',nargs='+',type=Path);args=parser.parse_args();print(bind(args.manifests)['status'])
