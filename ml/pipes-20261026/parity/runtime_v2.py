"""Explicit technical amendment chain; original frozen model code is untouched."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];MODEL=ROOT/'output/pipes-20261026/model';AMEND=Path(__file__).with_name('runtime-v2-amendment.json')
sys.path.insert(0,str(ROOT/'ml/pipes-20261026/model'))
import pipe_ml as p
from PIL import Image

def verify_authorization():
 a=p.read(MODEL/'training-authorization.json');m=p.read(AMEND);protocol=p.read(MODEL/'protocol.json');binding=p.read(MODEL/'data-binding.json')
 assert m['status']=='FROZEN_PIPE_RUNTIME_V2_AMENDMENT' and m['version']==2 and m['codeSHA256']==a['codeSHA256']==p.code_hashes()
 assert m['protocolSHA256']==a['protocolSHA256']==p.sha(MODEL/'protocol.json') and a['dataBindingSHA256']==p.sha(MODEL/'data-binding.json')
 assert p.sha(a['rootReceiptPath'])==a['rootReceiptSHA256']==m['trainingRootReceiptSHA256'] and p.read(a['rootReceiptPath'])['status']=='APPROVED_PIPE_BASELINE_TRAINING'
 assert m['selectionSHA256']==p.sha(MODEL/'selection.json') and m['modelSHA256']==p.read(MODEL/'selection.json')['selected']['modelSHA256']
 assert m['oldAppSourceCodeSHA256']==protocol['appSourceCodeSHA256'] and m['newAppSourceCodeSHA256']==p.app_source_hashes()
 before=ROOT/m['oldPipeDetectorSnapshotPath'];after=ROOT/'Count stick/PipeDetector.swift';old=before.read_text();new=after.read_text()
 assert p.sha(before)==m['oldPipeDetectorSHA256']==m['oldAppSourceCodeSHA256']['PipeLetterbox'] and p.sha(after)==m['newPipeDetectorSHA256']
 assert old.count('options.computeUnits = .all')==1 and old.count('label.confidence')==3
 assert old.replace('options.computeUnits = .all','options.computeUnits = .cpuAndNeuralEngine').replace('label.confidence','object.confidence')==new
 assert m['allowedChanges']==['computeUnits_all_to_cpuAndNeuralEngine','vision_score_label_to_object_confidence'] and m['pixelGeometryUnchanged'] is True
 block=lambda s:s[s.index('struct PipeLetterbox {'):s.index('enum PipeCountRegion {')]
 assert block(old)==block(new)
 for name in ['PhotoProcessing','ExportHelper']:assert m['oldAppSourceCodeSHA256'][name]==m['newAppSourceCodeSHA256'][name]
 assert p.sha(binding['manifestPath'])==binding['manifestSHA256'] and p.sha(MODEL/'records.json')==binding['recordsSHA256'] and p.sha(MODEL/'dataset/data.yaml')==binding['dataYAMLSHA256']
 for r in p.read(MODEL/'records.json'):assert p.sha(r['preparedImage'])==r['preparedImageSHA256'] and p.sha(r['preparedLabel'])==r['preparedLabelSHA256']
 p.official_init_check();return a

def app_inputs(path):
 verify_authorization();m=p.read(AMEND);manifest=p.read(path);assert manifest['status']=='VALIDATED_APP_PREPROCESSING_INPUTS' and manifest['testContentIncluded'] is False
 assert manifest['recordsSHA256']==p.sha(MODEL/'records.json') and manifest['approvedAppSourceCodeSHA256']==m['oldAppSourceCodeSHA256']
 assert manifest['variants']==['gallery','camera'] and len(manifest['images'])==80
 records={r['id']:r for r in p.read(MODEL/'records.json') if r['split']=='val'};assert len(records)==40
 assert {(r['variant'],r['id']) for r in manifest['images']}=={(v,id) for v in manifest['variants'] for id in records}
 for receipt in manifest['appExportManifests']:assert p.sha(receipt['path'])==receipt['sha256']
 for r in manifest['images']:
  assert r['sourceImageSHA256']==records[r['id']]['sha256'] and p.sha(r['networkImagePath'])==r['networkImageSHA256'] and p.sha(r['photoImagePath'])==r['photoImageSHA256']
  with Image.open(r['networkImagePath']) as im:assert im.size==(1024,1024)
  with Image.open(r['photoImagePath']) as im:assert im.size==(r['photoWidth'],r['photoHeight'])
  g=p.pipeline_geometry(r['photoWidth'],r['photoHeight']);assert all(abs(r['geometry'][k]-v)<1e-8 for k,v in g.items())
 return manifest,records
