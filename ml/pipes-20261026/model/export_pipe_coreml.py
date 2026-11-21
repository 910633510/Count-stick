"""Export a qualified frozen pipe candidate; never equate conversion with release acceptance."""
import argparse,json,shutil,uuid
from pathlib import Path
import pipe_ml as p
from ultralytics import YOLO

def tree_sha(directory):
 import hashlib
 root=Path(directory);digest=hashlib.sha256()
 for file in sorted(x for x in root.rglob('*') if x.is_file()):digest.update(str(file.relative_to(root)).encode()+b'\0'+file.read_bytes())
 return digest.hexdigest()

def export(approval_path):
 from evaluate_pipe_ends import verify_authorization
 verify_authorization();approval=p.read(approval_path);selection=p.read(p.OUT/'selection.json')
 assert approval['status']=='APPROVED_PIPE_PARITY_EXPORT' and approval['selectionSHA256']==p.sha(p.OUT/'selection.json')
 assert selection['status']=='VAL_QUALIFIED_PENDING_FROZEN_SCENE_TEST' and selection['selected'] is not None
 chosen=selection['selected'];weights=p.OUT/'checkpoints'/f'epoch{chosen["epoch"]:03d}.pt';assert p.sha(weights)==chosen['modelSHA256']==approval['modelSHA256']
 folder=p.OUT/'export';assert not folder.exists(),'Do not overwrite a previous export';folder.mkdir();staging=folder/'PipeEnds.pt';shutil.copy2(weights,staging)
 model=YOLO(str(staging));assert model.names=={0:'pipe_end'} and model.model.model[-1].nc==1
 # Explicit32 avoids this installedexporter's nms=True implicitFP16 default.
 exported=model.export(format='coreml',imgsz=1024,batch=1,dynamic=False,nms=True,quantize=32,conf=chosen['confidence'],iou=chosen['nmsIoU'],max_det=21504,device='cpu')
 package=Path(exported).resolve();assert package.suffix=='.mlpackage' and package.name=='PipeEnds.mlpackage'
 import coremltools as ct
 loaded=ct.models.MLModel(str(package),skip_model_load=True);export_id=str(uuid.uuid4())
 loaded.user_defined_metadata.update({'countscope.export_id':export_id,'countscope.checkpoint_sha256':chosen['modelSHA256'],'countscope.confidence_threshold':str(chosen['confidence']),'countscope.iou_threshold':str(chosen['nmsIoU']),'countscope.class_label':'pipe_end','countscope.input_size':'1024'})
 bound=folder/'metadata-bound.mlpackage';loaded.save(str(bound));unbound=folder/'unbound-conversion.mlpackage';package.rename(unbound);bound.rename(package)
 loaded=ct.models.MLModel(str(package),skip_model_load=True);assert loaded.user_defined_metadata['countscope.export_id']==export_id and loaded.user_defined_metadata['countscope.checkpoint_sha256']==chosen['modelSHA256']
 spec=loaded.get_spec();inputs={x.name:x for x in spec.description.input};outputs={x.name:x for x in spec.description.output};assert 'image' in inputs and inputs['image'].type.WhichOneof('Type')=='imageType';image=inputs['image'].type.imageType;assert image.width==image.height==1024
 assert set(outputs)=={'confidence','coordinates'}
 for name in ['confidenceThreshold','iouThreshold']:assert name in inputs and inputs[name].type.WhichOneof('Type')=='doubleType'
 nms_models=[m.nonMaximumSuppression for m in spec.pipeline.models if m.WhichOneof('Type')=='nonMaximumSuppression'];assert len(nms_models)==1;nms=nms_models[0];labels=list(nms.stringClassLabels.vector);assert labels[0]=='pipe_end' and all(label==str(i) for i,label in enumerate(labels[1:],1))
 assert abs(nms.confidenceThreshold-chosen['confidence'])<1e-9 and abs(nms.iouThreshold-chosen['nmsIoU'])<1e-9
 contract={'status':'EXPORTED_PENDING_COREML_VISION_PARITY_AND_FROZEN_TEST','packagePath':str(package),'packageTreeSHA256':tree_sha(package),'modelSHA256':chosen['modelSHA256'],'selectionSHA256':p.sha(p.OUT/'selection.json'),'inputSize':1024,'imageInput':'image','colorInput':'RGB,scale1/255,bias0','classLabel':'pipe_end','semanticClasses':1,'confidenceChannelsIncludingOfficialZeroPadding':len(labels),'paddingLabels':labels[1:],'outputFeatures':['confidence','coordinates'],'coordinates':'normalizedcenterxywh inletterboxednetworkimage; directCoreML tensor usesYOLO top-left y convention; verifyVision bottom-left conversion empirically.','confidenceThreshold':chosen['confidence'],'iouThreshold':chosen['nmsIoU'],'quantize':32,'photoStage':'actualPhotoProcessing galleryImageIOmax2048/cameraUIImage normalizedmax2048','letterbox':'explicitUIKit gray114/center1024; geometry must beunmapped withrecordedgain/pad','arbitraryDetectionOrMarkerCap':False,'rawLocationCapacity':21504,'VALIDATED_COREML_EXPORTIssued':False,'requiredRemaining':['PyTorch/CoreML sameSwift-rendered1024RGB inputs parity','Vision observedlabel/bbox mapping includingpadding andorientation','Frozen physical-scene testqualification acrossactualAppphoto paths','root final acceptance receipt']}
 contract['exportID']=export_id;contract['checkpointSHA256']=chosen['modelSHA256'];contract['creatorDefinedMetadataKeys']=['countscope.export_id','countscope.checkpoint_sha256','countscope.confidence_threshold','countscope.iou_threshold','countscope.class_label','countscope.input_size']
 p.save(folder/'export-status.json',contract);p.save(folder/'io-contract.pending.json',contract);print(json.dumps(contract,indent=2))

def finalize(receipt_path):
 from evaluate_pipe_ends import verify_authorization
 verify_authorization()
 receipt=p.read(receipt_path);assert receipt['status']=='APPROVED_VALIDATED_PIPE_MODEL_AFTER_TEST_AND_PARITY'
 status=p.read(p.OUT/'export/export-status.json');assert receipt['exportStatusSHA256']==p.sha(p.OUT/'export/export-status.json') and receipt['packageTreeSHA256']==tree_sha(status['packagePath'])==status['packageTreeSHA256']
 expected_point={'confidence':status['confidenceThreshold'],'nmsIoU':status['iouThreshold']};variants=set(p.read(p.OUT/'protocol.json')['evaluationVariants'])
 for field in ['testReport','parityReport']:
  path=Path(receipt[field+'Path']);assert p.sha(path)==receipt[field+'SHA256'];report=p.read(path)
  expected_status='PASS_PIPE_FROZEN_GROUP_HOLDOUT' if field=='testReport' else 'PASS_PIPE_TORCH_COREML_VISION_PARITY'
  assert report['status']==expected_status and report['modelSHA256']==status['modelSHA256'] and report['selectionSHA256']==status['selectionSHA256'] and report['protocolSHA256']==p.sha(p.OUT/'protocol.json') and report['operatingPoint']==expected_point
  assert set(report['inputVariants'])==variants and len(report['inputVariants'])==len(variants)
  if field=='parityReport':
   assert report['exportID']==status['exportID'] and report['packageTreeSHA256']==status['packageTreeSHA256'] and set(report['inputManifestBindings'])==variants
   for binding in report['inputManifestBindings'].values():assert p.sha(binding['path'])==binding['sha256']
 assert receipt['selectionSHA256']==p.sha(p.OUT/'selection.json')==status['selectionSHA256']
 config={'status':'VALIDATED_COREML_EXPORT','inputSize':1024,'classLabel':'pipe_end','confidenceThreshold':status['confidenceThreshold'],'iouThreshold':status['iouThreshold'],'exportID':status['exportID'],'checkpointSHA256':status['modelSHA256'],'modelSHA256':status['modelSHA256'],'packageTreeSHA256':status['packageTreeSHA256'],'finalReceiptSHA256':p.sha(receipt_path)}
 p.save(p.OUT/'export/PipeEnds.json',config);p.save(p.OUT/'export/release-handoff.json',{'status':'VALIDATED_MODEL_ARTIFACT_READY_FOR_APP_AGENT','configuration':config,'packagePath':status['packagePath'],'configPath':str(p.OUT/'export/PipeEnds.json'),'appFilesEditedByModelOperator':False});print(config)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['export','finalize']);parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args()
 export(args.receipt) if args.phase=='export' else finalize(args.receipt)
