"""Audit the official YOLO11n asset before any pipeline is allowed to load it."""
from pathlib import Path
import hashlib,importlib,json,pickletools,zipfile
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/pipes-20261026/model'
PATH=OUT/'official-source/yolo11n.pt'
EXPECTED_SHA='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
EXPECTED_GLOBALS={
 '__builtin__ set','collections OrderedDict','torch FloatStorage','torch HalfStorage','torch LongStorage','torch Size','torch._utils _rebuild_parameter','torch._utils _rebuild_tensor_v2',
 'torch.nn.modules.activation SiLU','torch.nn.modules.batchnorm BatchNorm2d','torch.nn.modules.container ModuleList','torch.nn.modules.container Sequential','torch.nn.modules.conv Conv2d','torch.nn.modules.linear Identity','torch.nn.modules.pooling MaxPool2d','torch.nn.modules.upsampling Upsample',
 'ultralytics.nn.modules.block Attention','ultralytics.nn.modules.block Bottleneck','ultralytics.nn.modules.block C2PSA','ultralytics.nn.modules.block C3k','ultralytics.nn.modules.block C3k2','ultralytics.nn.modules.block DFL','ultralytics.nn.modules.block PSABlock','ultralytics.nn.modules.block SPPF','ultralytics.nn.modules.conv Concat','ultralytics.nn.modules.conv Conv','ultralytics.nn.modules.conv DWConv','ultralytics.nn.modules.head Detect','ultralytics.nn.tasks DetectionModel'}

def main():
 import torch
 from ultralytics.nn.tasks import DetectionModel
 assert hashlib.sha256(PATH.read_bytes()).hexdigest()==EXPECTED_SHA
 meta=json.loads((OUT/'official-source/asset-api.json').read_text());assert meta['id']==196201276 and meta['size']==PATH.stat().st_size and meta['browser_download_url']=='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
 with zipfile.ZipFile(PATH) as archive:
  names=archive.namelist();assert all(not n.startswith('/') and '..' not in Path(n).parts for n in names);assert sum(x.file_size for x in archive.infolist())<20_000_000
  p=next(n for n in names if n.endswith('/data.pkl'));ops=list(pickletools.genops(archive.read(p)));actual={arg for op,arg,pos in ops if op.name=='GLOBAL'}
  assert actual==EXPECTED_GLOBALS;assert not any(op.name in ['STACK_GLOBAL','EXT1','EXT2','EXT4','INST','OBJ'] for op,arg,pos in ops)
 allowed=[]
 for name in sorted(actual):
  module,symbol=name.split();module='builtins' if module=='__builtin__' else module;allowed.append(getattr(importlib.import_module(module),symbol))
 # Safe unpickler with only the already reviewed official model classes/functions.
 with torch.serialization.safe_globals(allowed):checkpoint=torch.load(PATH,map_location='cpu',weights_only=True)
 model=checkpoint.get('ema') or checkpoint.get('model');assert isinstance(model,DetectionModel)
 assert model.model[-1].nc==80 and len(model.names)==80 and list(model.stride.tolist())==[8.,16.,32.]
 assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
 assert not any(getattr(m,'_forward_hooks',{}) or getattr(m,'_forward_pre_hooks',{}) for m in model.modules())
 r=json.loads((OUT/'official-source-audit.json').read_text());r.update({'status':'PASS_OFFICIAL_SOURCE_FORMAT_ALLOWLIST_SAFE_LOAD_STRUCTURE','modelDeserialized':True,'deserialization':'torch.load weights_only=True with exact reviewed GLOBAL allowlist','modelClass':type(model).__name__,'classes':80,'stride':[8,16,32],'parameterCount':sum(p.numel() for p in model.parameters()),'stateTensors':len(model.state_dict()),'allFloatingStateFinite':True,'forwardHooksAbsent':True,'intendedPipeTask':'Fresh single-class pipe_end head trained from officialCOCO backbone/neck; no colony weights.','optimizerSteps':0,'modelInferencePerformed':False,'weightsModified':False})
 (OUT/'official-source-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

if __name__=='__main__':main()
