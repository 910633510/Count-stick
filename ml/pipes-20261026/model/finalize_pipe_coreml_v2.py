"""Explicit runtime-v2 finalization; preserves the original frozen eight files.

Preparation of this script is not approval. It can emit a usable configuration
only after root-bound source review, new runtime parity and fixed holdout PASS.
It imports only the stdlib-only test authorization gate, never its test runner.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/pipes-20261026/model'
SELF=Path(__file__).resolve()
GATE=SELF.with_name('evaluate_frozen_test.py')

def require(value,message):
 if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def resolve(path):
 path=Path(path);path=(ROOT/path if not path.is_absolute() else path).resolve();require(path.is_relative_to(ROOT),'Path escapes repository');return path
def write_once(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():require(read(path)==value,'Refuse to overwrite a different finalizedartifact');return
 with path.open('x') as file:json.dump(value,file,indent=2,allow_nan=False);file.write('\n');file.flush();os.fsync(file.fileno())

def verified_json(receipt,key):
 path=resolve(receipt[key+'Path']);require(sha(path)==receipt[key+'SHA256'],key+' file changed');return read(path),path

def training_data_chain(bindings,gate):
 """Original data/code checks, replacing only the already-reviewed App exception."""
 authorization=read(OUT/'training-authorization.json');protocol=read(OUT/'protocol.json');binding=read(OUT/'data-binding.json')
 require(authorization['codeSHA256']==bindings['codeSHA256']==gate.code_hashes(),'Frozen training code changed')
 require(authorization['protocolSHA256']==sha(OUT/'protocol.json')==bindings['protocolSHA256'],'Frozen training protocol changed')
 require(authorization['dataBindingSHA256']==sha(OUT/'data-binding.json'),'Training data binding changed')
 root_receipt=resolve(authorization['rootReceiptPath']);require(sha(root_receipt)==authorization['rootReceiptSHA256'],'Original root training receipt changed');original=read(root_receipt)
 require(original['status']=='APPROVED_PIPE_BASELINE_TRAINING' and original['codeSHA256']==bindings['codeSHA256'] and original['protocolSHA256']==bindings['protocolSHA256'] and original['dataBindingSHA256']==sha(OUT/'data-binding.json'),'Original training approval mismatch')
 require(protocol['status']=='FROZEN_PIPE_BASELINE_PROTOCOL','Training protocol not frozen')
 require(protocol['appSourceCodeSHA256']==bindings['trainingAppSourceCodeSHA256'],'Runtime chain lost original App source binding')
 qa=resolve(original['qaReportPath']);require(sha(qa)==original['qaReportSHA256'] and read(qa)['status'].startswith('PASS'),'Dataset QA changed')
 require(sha(resolve(binding['manifestPath']))==binding['manifestSHA256'],'Train-val source manifest changed')
 require(sha(OUT/'records.json')==binding['recordsSHA256'] and sha(OUT/'dataset/data.yaml')==binding['dataYAMLSHA256'],'Frozen records/YAML changed')
 for row in read(OUT/'records.json'):
  require(row['split'] in ['train','val'],'Unexpected nondevelopment record')
  require(sha(resolve(row['preparedImage']))==row['preparedImageSHA256'] and sha(resolve(row['preparedLabel']))==row['preparedLabelSHA256'],'Prepared image or label changed')
 official=OUT/'official-source/yolo11n.pt';require(sha(official)==original['officialWeightsSHA256']==authorization['officialWeightsSHA256'],'Official initialization changed')
 audit=read(OUT/'official-source-audit.json');require(audit['status']=='PASS_OFFICIAL_SOURCE_FORMAT_ALLOWLIST_SAFE_LOAD_STRUCTURE' and audit['localSHA256']==sha(official) and audit['classes']==80,'Initialization audit not valid')

def finalize(receipt_path):
 receipt_path=resolve(receipt_path);receipt=read(receipt_path)
 require(receipt['status']=='APPROVED_VALIDATED_PIPE_MODEL_AFTER_TEST_AND_PARITY' and receipt['runtimeVersion']==2,'No root runtime-v2 finalapproval')
 require(receipt['finalizerSHA256']==sha(SELF),'Finalizer differs from approved code')
 require(receipt['frozenTestEvaluatorSHA256']==sha(GATE),'Test authorization gate changed')
 test_receipt_path=resolve(receipt['rootTestReceiptPath']);require(sha(test_receipt_path)==receipt['rootTestReceiptSHA256'],'Dedicated root test receipt changed')
 spec=importlib.util.spec_from_file_location('pipe_frozen_gate_for_v2_finalize',GATE);gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 bindings=gate.gate(test_receipt_path)  # hashes/source approval only; does not read sealed test data.
 require(bindings is not None and bindings['runtimeVersion']==2,'No qualified runtime-v2 candidate')
 training_data_chain(bindings,gate)
 export=read(OUT/'export/export-status.json');require(receipt['exportStatusSHA256']==sha(OUT/'export/export-status.json'),'Export status changed')
 require(receipt['packageTreeSHA256']==export['packageTreeSHA256']==gate.tree_sha(export['packagePath']),'Export package changed')
 require(receipt['selectionSHA256']==export['selectionSHA256']==bindings['selectionSHA256'],'Selection changed')
 require(export['modelSHA256']==bindings['modelSHA256'] and export['exportID']==bindings['runtimeArtifacts']['exportID'],'Model/exportID changed')
 require({'confidence':export['confidenceThreshold'],'nmsIoU':export['iouThreshold']}==bindings['operatingPoint'],'Export operating point changed')
 review=read(resolve(bindings['runtimeAmendmentReviewPath']))
 require(receipt['runtimeAmendmentReviewSHA256']==bindings['runtimeAmendmentReviewSHA256'] and receipt['runtimeAmendmentSHA256']==bindings['runtimeAmendmentSHA256'],'Runtime amendment finalreceipt mismatch')
 parity,parity_path=verified_json(receipt,'parityReport');test,test_path=verified_json(receipt,'testReport')
 require(str(parity_path)==str(resolve(review['parityReportPath'])) and sha(parity_path)==review['parityReportSHA256'],'Finalparity is not independentlyreviewed v2parity')
 parity_protocol=resolve(review['parityProtocolPath']);require(sha(parity_protocol)==review['parityProtocolSHA256']==parity['parityProtocolSHA256'],'Newruntime parityprotocol changed')
 require(receipt['parityProtocolSHA256']==sha(parity_protocol),'Finalreceipt parityprotocol mismatch')
 for name,document,status in [('test',test,'PASS_PIPE_FROZEN_GROUP_HOLDOUT'),('parity',parity,'PASS_PIPE_TORCH_COREML_VISION_PARITY')]:
  require(document['status']==status,name+' qualification not PASS')
  for key in ['modelSHA256','selectionSHA256','protocolSHA256','operatingPoint','inputVariants']:
   require(document[key]==bindings[key],name+' binding mismatch: '+key)
  require(set(document['inputManifestBindings'])==set(bindings['inputVariants']),name+' missing App path')
  for item in document['inputManifestBindings'].values():require(sha(resolve(item['path']))==item['sha256'],name+' inputmanifest changed')
 require(parity['exportID']==export['exportID'] and parity['packageTreeSHA256']==export['packageTreeSHA256'],'Parity covers different package')
 require(test['sealedManifestSHA256']==bindings['sealedManifestSHA256'] and test['evaluatorSHA256']==bindings['evaluatorSHA256'],'Test covers different frozenholdout or evaluator')
 require(test['testImages']==37 and test['split']=='test' and test['selectionBasedOnTest'] is False and test['newTrainingPerformed'] is False,'Invalid heldout execution scope')
 require(all(test['variants'][v]['gates']['status']=='PASS' for v in bindings['inputVariants']),'A test App variant failed')
 config={'status':'VALIDATED_COREML_EXPORT','inputSize':1024,'classLabel':'pipe_end','confidenceThreshold':bindings['operatingPoint']['confidence'],'iouThreshold':bindings['operatingPoint']['nmsIoU'],'exportID':export['exportID'],'checkpointSHA256':bindings['modelSHA256'],'modelSHA256':bindings['modelSHA256'],'packageTreeSHA256':export['packageTreeSHA256'],'runtimeVersion':2,'computeUnits':'cpuAndNeuralEngine','confidenceSource':'VNRecognizedObjectObservation.confidence','runtimeAmendmentSHA256':bindings['runtimeAmendmentSHA256'],'runtimeReviewSHA256':bindings['runtimeAmendmentReviewSHA256'],'parityProtocolSHA256':sha(parity_protocol),'finalReceiptSHA256':sha(receipt_path)}
 write_once(OUT/'export/PipeEnds.json',config)
 write_once(OUT/'export/release-handoff-v2.json',{'status':'VALIDATED_RUNTIME_V2_MODEL_ARTIFACT_READY_FOR_APP_AGENT','configuration':config,'configSHA256':sha(OUT/'export/PipeEnds.json'),'packagePath':export['packagePath'],'runtimeSourceBinding':bindings,'testReportPath':str(test_path),'testReportSHA256':sha(test_path),'parityReportPath':str(parity_path),'parityReportSHA256':sha(parity_path),'originalAllFailurePreserved':True,'originalFrozenEightFilesModified':False,'appFilesEditedByFinalizer':False,'newInferencePerformedByFinalizer':False})
 return config

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();print(json.dumps(finalize(args.receipt),indent=2))
