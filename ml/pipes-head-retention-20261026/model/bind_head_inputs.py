"""Rebind the exact previous94 validated App inputs to unchanged47 validation records."""
import pipe_head_retention as p

def bind():
 p.init_check();assert not (p.OUT/'evaluation-inputs.json').exists();binding=p.read(p.OUT/'data-binding.json');assert p.sha(p.OUT/'records.json')==binding['recordsSHA256'];assert binding['validationSourceManifestSHA256']==p.PREVIOUS_MANIFEST_SHA
 parent_path=p.PREVIOUS/'evaluation-inputs.json';assert p.sha(parent_path)==p.PREVIOUS_INPUTS_SHA;previous=p.read(parent_path);assert previous['status']=='VALIDATED_HARDENING_APP_INPUTS' and previous['testContentIncluded'] is False and previous['approvedAppSourceCodeSHA256']==p.app_hashes()
 before={r['id']:r for r in p.read(p.PREVIOUS/'records.json') if r['split']=='val'};after={r['id']:r for r in p.read(p.OUT/'records.json') if r['split']=='val'};assert set(before)==set(after) and len(before)==47
 for id,row in after.items():
  assert {k:v for k,v in row.items() if k not in ['preparedImage','preparedLabel']}=={k:v for k,v in before[id].items() if k not in ['preparedImage','preparedLabel']}
  assert p.sha(row['preparedImage'])==row['preparedImageSHA256'] and p.sha(row['preparedLabel'])==row['preparedLabelSHA256']
 assert len(previous['images'])==94
 for row in previous['images']:assert row['id'] in after and p.sha(row['networkImagePath'])==row['networkImageSHA256'] and p.sha(row['photoImagePath'])==row['photoImageSHA256']
 output={**previous,'status':'VALIDATED_HEAD_RETENTION_APP_INPUTS','recordsSHA256':p.sha(p.OUT/'records.json'),'previousInputsPath':str(parent_path),'previousInputsSHA256':p.PREVIOUS_INPUTS_SHA,'validationDataUnchanged':True,'newAppExportPerformed':False};p.save(p.OUT/'evaluation-inputs.json',output)
 from evaluate_head_retention import inputs
 inputs();return output
if __name__=='__main__':print(bind()['status'])
