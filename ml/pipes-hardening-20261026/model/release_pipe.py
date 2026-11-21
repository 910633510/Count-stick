"""Raw FP32 export and small explicit release bindings for the hardening round."""
from pathlib import Path
import argparse,hashlib,json,shutil,uuid
import pipe_hardening as p
PARITY_SOURCE=p.ROOT/'ml/pipes-20261026/parity/protocol.json'
PARITY_SHA='59861d73f76ad3bd21bd098d18d845d343d285352647b20880134e70780984b2'
NATIVE=p.ROOT/'ml/pipes-20261026/parity/native_parity_v3.swift'
NATIVE_SHA='b7dfde4920d731057378845bf2d30b3da58607688130c59fd64c80c45300afce'
NMS=p.ROOT/'Count stick/PipeRawPostprocessor.swift';NMS_SHA='4542c36ee0f171571d9651c5ccbeec7744bb71e8b4cf7fd429137afd302a3fbe'
HELPER=p.ROOT/'Count stickTests/PipeHardeningFrozenTestExportTests.swift'
def tree_sha(root):
 root=Path(root);h=hashlib.sha256()
 for file in sorted(x for x in root.rglob('*') if x.is_file()):h.update(str(file.relative_to(root)).encode()+b'\0'+file.read_bytes())
 return h.hexdigest()
def exclusive(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def code_hashes():
 assert p.sha(NATIVE)==NATIVE_SHA and p.sha(NMS)==NMS_SHA and p.sha(PARITY_SOURCE)==PARITY_SHA
 files=[p.SOURCE/'release_pipe.py',p.SOURCE/'parity_hardening.py',p.SOURCE/'evaluate_fresh_test.py',HELPER,NATIVE,NMS]
 return {str(f.relative_to(p.ROOT)):p.sha(f) for f in files}
def selected_binding():
 selection=p.read(p.OUT/'selection.json')
 if selection['selected'] is None:return None
 from evaluate_hardening import authorization
 authorization();selected=selection['selected'];assert selection['status']=='HARDENING_VAL_QUALIFIED_PENDING_FRESH_HOLDOUT' and selected['status']=='PASS' and selected in selection['candidates'] and len(selection['candidates'])==3
 assert selected['epoch'] in [4,8,12] and selected['confidence']==.3 and selected['nmsIoU']==.7
 assert all(selected['variants'][v][role]['gates']['status']=='PASS' for v in ['gallery','camera'] for role in ['old','fresh'])
 weights=p.OUT/'checkpoints'/f'epoch{selected["epoch"]:03d}.pt';assert p.sha(weights)==selected['modelSHA256']
 return {'selectionSHA256':p.sha(p.OUT/'selection.json'),'protocolSHA256':p.sha(p.OUT/'protocol.json'),'dataBindingSHA256':p.sha(p.OUT/'data-binding.json'),'modelSHA256':p.sha(weights),'epoch':selected['epoch'],'weightsPath':str(weights.relative_to(p.ROOT)),'operatingPoint':{'confidence':.3,'nmsIoU':.7},'inputVariants':['gallery','camera'],'codeSHA256':p.code_hashes(),'releaseCodeSHA256':code_hashes(),'appSourceCodeSHA256':p.app_hashes(),'runtimeVersion':3,'runtimeNMS':{'path':str(NMS.relative_to(p.ROOT)),'sha256':NMS_SHA}}
def export(receipt_path):
 expected=selected_binding()
 if expected is None:return {'status':'SKIP_NO_QUALIFIED_CANDIDATE','freshTestAccessed':False}
 receipt=p.read(receipt_path);assert receipt['status']=='APPROVED_HARDENING_RAW_PARITY_EXPORT'
 for key in ['selectionSHA256','protocolSHA256','dataBindingSHA256','modelSHA256','operatingPoint','releaseCodeSHA256']:assert receipt[key]==expected[key]
 folder=p.OUT/'export';assert not folder.exists();folder.mkdir();staging=folder/'PipeEnds.pt';shutil.copy2(p.resolve(expected['weightsPath']),staging)
 # Conversion only; no test data or runtime inference.
 from ultralytics import YOLO
 import coremltools as ct
 model=YOLO(str(staging));assert model.names=={0:'pipe_end'} and model.model.model[-1].nc==1
 package=Path(model.export(format='coreml',imgsz=1024,batch=1,dynamic=False,nms=False,quantize=32,device='cpu')).resolve();loaded=ct.models.MLModel(str(package),skip_model_load=True);spec=loaded.get_spec();assert len(spec.description.output)==1
 oldname=spec.description.output[0].name;ct.models.utils.rename_feature(spec,oldname,'raw_detections',rename_inputs=False,rename_outputs=True);export_id=str(uuid.uuid4());bound=ct.models.MLModel(spec,weights_dir=loaded.weights_dir,skip_model_load=True)
 bound.user_defined_metadata.update({'countscope.export_id':export_id,'countscope.checkpoint_sha256':expected['modelSHA256'],'countscope.confidence_threshold':'0.3','countscope.iou_threshold':'0.7','countscope.class_label':'pipe_end','countscope.input_size':'1024','countscope.runtime_version':'3','countscope.nms_mode':'external_swift_greedy','countscope.raw_output_name':'raw_detections'})
 temp=folder/'bound.mlpackage';bound.save(str(temp));package.rename(folder/'unbound-conversion.mlpackage');temp.rename(package);spec=ct.models.MLModel(str(package),skip_model_load=True).get_spec();output=spec.description.output[0];assert output.name=='raw_detections' and list(output.type.multiArrayType.shape)==[1,5,21504] and output.type.multiArrayType.dataType==65568
 image=next(x for x in spec.description.input if x.name=='image');assert image.type.imageType.width==image.type.imageType.height==1024
 status={**expected,'status':'EXPORTED_RAW_HARDENING_PENDING_PARITY_AND_FRESH_TEST','packagePath':str(package),'packageTreeSHA256':tree_sha(package),'exportID':export_id,'nms':False,'quantize':32,'inputSize':1024,'rawLocationCapacity':21504,'outputFeatureName':'raw_detections','outputShape':[1,5,21504],'outputDataType':65568,'confidenceThreshold':.3,'iouThreshold':.7,'rootExportReceiptSHA256':p.sha(receipt_path),'VALIDATED_COREML_EXPORTIssued':False};exclusive(folder/'export-status.json',status)
 old=p.read(PARITY_SOURCE);exclusive(p.OUT/'parity-protocol.json',{'status':'FROZEN_HARDENING_PARITY_BEFORE_PREDICTION','numericPolicy':old['numericPolicy'],'qualificationPolicy':old['qualificationPolicy'],'originalParityProtocolSHA256':PARITY_SHA,'nativeRunnerSHA256':NATIVE_SHA,'nmsSHA256':NMS_SHA,'exportStatusSHA256':p.sha(folder/'export-status.json'),'bindings':expected,'scope':'All old40+freshval gallery/camera inputs. Each role retains its frozen eligibility and cannot regress from selected Torch counts. No test.'})
 return status
def finalize(receipt_path):
 """Emit a config only for the exact independently reviewed root-approved release."""
 binding=selected_binding()
 assert binding is not None,'No development-qualified hardening candidate'
 receipt=p.read(receipt_path);assert receipt['status']=='APPROVED_HARDENING_VALIDATED_MODEL_AFTER_PARITY_AND_TEST'
 for key in ['selectionSHA256','protocolSHA256','dataBindingSHA256','modelSHA256','operatingPoint','releaseCodeSHA256']:assert receipt[key]==binding[key]
 from evaluate_fresh_test import gate
 test_receipt=p.resolve(receipt['rootFreshTestReceiptPath']);assert p.sha(test_receipt)==receipt['rootFreshTestReceiptSHA256'];approved=gate(test_receipt);assert approved is not None
 export=p.read(p.OUT/'export/export-status.json');assert receipt['exportStatusSHA256']==p.sha(p.OUT/'export/export-status.json') and receipt['packageTreeSHA256']==export['packageTreeSHA256']==tree_sha(export['packagePath'])
 assert export['modelSHA256']==binding['modelSHA256'] and export['selectionSHA256']==binding['selectionSHA256']
 reports={}
 for name,status in [('parityReport','PASS_HARDENING_TORCH_COREML_VISION_PARITY'),('freshTestReport','PASS_HARDENING_FRESH_GROUP_HOLDOUT')]:
  path=p.resolve(receipt[name+'Path']);assert p.sha(path)==receipt[name+'SHA256'];report=p.read(path);assert report['status']==status
  for key in ['selectionSHA256','protocolSHA256','dataBindingSHA256','modelSHA256','operatingPoint','releaseCodeSHA256','appSourceCodeSHA256']:assert report[key]==binding[key]
  assert report['inputVariants']==['gallery','camera'] and p.sha(report['inputManifestPath'])==report['inputManifestSHA256'];reports[name]=report
 parity=reports['parityReport'];test=reports['freshTestReport'];assert parity['exportID']==test['exportID']==export['exportID'] and parity['packageTreeSHA256']==test['packageTreeSHA256']==export['packageTreeSHA256']
 assert receipt['parityReportSHA256']==approved['parityReportSHA256'] and test['rootApprovalSHA256']==p.sha(test_receipt) and test['sealedManifestSHA256']==approved['sealedManifestSHA256'] and test['testImages']==approved['expectedTestImages']
 assert test['preTestProtocolSHA256']==p.sha(p.OUT/'test/pre-test-protocol.json') and p.read(p.OUT/'test/pre-test-protocol.json')['bindings']==approved
 assert test['selectionBasedOnTest'] is False and test['newTrainingPerformed'] is False and all(v['gates']['status']=='PASS' for v in test['variants'].values())
 config={'status':'VALIDATED_COREML_EXPORT','inputSize':1024,'classLabel':'pipe_end','confidenceThreshold':.3,'iouThreshold':.7,'exportID':export['exportID'],'checkpointSHA256':binding['modelSHA256'],'modelSHA256':binding['modelSHA256'],'packageTreeSHA256':export['packageTreeSHA256'],'runtimeVersion':3,'computeUnits':'cpuAndNeuralEngine','rawOutputFeatureName':'raw_detections','nmsSourceSHA256':NMS_SHA,'nmsImplementation':'shared_swift_greedy','finalReceiptSHA256':p.sha(receipt_path)}
 exclusive(p.OUT/'export/PipeEnds.json',config);exclusive(p.OUT/'export/release-handoff.json',{'status':'VALIDATED_HARDENING_MODEL_READY_FOR_APP_INTEGRATION','packagePath':export['packagePath'],'configuration':config,'configSHA256':p.sha(p.OUT/'export/PipeEnds.json'),'reports':{k:receipt[k] for k in ['parityReportPath','parityReportSHA256','freshTestReportPath','freshTestReportSHA256']},'physicalIPhoneAcceptanceClaimed':False});return config
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['export','finalize']);parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();print(json.dumps(export(args.receipt) if args.phase=='export' else finalize(args.receipt),indent=2))
