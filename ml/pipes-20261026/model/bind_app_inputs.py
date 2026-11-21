"""Audit actual UIKit PNGs and bind frozen validation inputs without model inference."""
import argparse,collections
from pathlib import Path
import pipe_ml as p
import numpy as np
import cv2
from PIL import Image

VARIANTS={'imageio-import':'gallery','camera-normalized':'camera'}

def inspect_manifest(path):
 path=Path(path).resolve();manifest=p.read(path);assert manifest['status']=='APP_PREPROCESSING_PIXELS_EXPORTED_NO_MODEL_INFERENCE';assert manifest['kind'] in ['synthetic_geometry_only','frozen_train_val_photo']
 rows=[]
 for r in manifest['records']:
  assert r['appPath'] in VARIANTS and r['outputOrientation']=='up' and r['networkSize']==1024 and r['paddingRGB']==[114,114,114]
  photo=(p.ROOT/r['preparedPNG']).resolve();network=(p.ROOT/r['networkPNG']).resolve();assert photo.is_relative_to(p.ROOT) and network.is_relative_to(p.ROOT);assert p.sha(photo)==r['preparedSHA256'] and p.sha(network)==r['networkSHA256']
  with Image.open(photo) as im:upright=np.asarray(im.convert('RGB')).copy();assert im.size==(r['preparedWidth'],r['preparedHeight'])
  with Image.open(network) as im:actual=np.asarray(im.convert('RGB')).copy();assert im.size==(1024,1024)
  geometry=p.pipeline_geometry(r['preparedWidth'],r['preparedHeight']);mapping={'resizedWidth':'resizedWidth','resizedHeight':'resizedHeight','left':'left','top':'top','gain':'gain'}
  assert all(abs(r[app]-geometry[key])<1e-8 for key,app in mapping.items());assert r['right']==1024-r['left']-r['resizedWidth'] and r['bottom']==1024-r['top']-r['resizedHeight']
  padding=np.ones((1024,1024),dtype=bool);padding[r['top']:r['top']+r['resizedHeight'],r['left']:r['left']+r['resizedWidth']]=False;assert np.all(actual[padding]==114)
  opencv=np.full((1024,1024,3),114,dtype=np.uint8);opencv[r['top']:r['top']+r['resizedHeight'],r['left']:r['left']+r['resizedWidth']]=cv2.resize(upright,(r['resizedWidth'],r['resizedHeight']),interpolation=cv2.INTER_LINEAR)
  delta=np.abs(actual.astype(np.int16)-opencv.astype(np.int16));comparison={'meanAbsoluteByteDifference':float(delta.mean()),'p99AbsoluteByteDifference':float(np.percentile(delta,99)),'maxAbsoluteByteDifference':int(delta.max()),'equalChannelFraction':float((delta==0).mean()),'pixelIdentical':bool(np.array_equal(actual,opencv))}
  rows.append({'id':r['id'],'variant':VARIANTS[r['appPath']],'kind':r['kind'],'split':r.get('split'),'sourceImageSHA256':r['sourceSHA256'],'sourceManifestSHA256':r.get('sourceManifestSHA256'),'networkImagePath':str(network),'networkImageSHA256':r['networkSHA256'],'photoImagePath':str(photo),'photoImageSHA256':r['preparedSHA256'],'photoWidth':r['preparedWidth'],'photoHeight':r['preparedHeight'],'geometry':geometry,'opencvComparison':comparison,'sourceEXIFOrientation':r['sourceEXIFOrientation']})
 return manifest,rows

def audit(paths):
 reports=[]
 for path in paths:
  manifest,rows=inspect_manifest(path);reports.append({'path':str(Path(path).resolve()),'sha256':p.sha(path),'kind':manifest['kind'],'sourceCodeSHA256':manifest['sourceCodeSHA256'],'images':rows})
 result={'status':'PASS_APP_PIXEL_HASH_GEOMETRY_GRAY114_WITH_MEASURED_RESIZE_DIFFERENCES','manifests':reports,'noPixelIdentityClaim':True,'modelInferencePerformed':False,'optimizerSteps':0};p.save(p.OUT/'app-preprocessing-audit.json',result);return result

def bind(paths):
 binding=p.read(p.OUT/'data-binding.json');assert p.sha(p.OUT/'records.json')==binding['recordsSHA256'];records={r['id']:r for r in p.read(p.OUT/'records.json')};val={r['id'] for r in records.values() if r['split']=='val'};rows=[];receipts=[]
 protocol=p.read(p.OUT/'protocol.json');approved=protocol['appSourceCodeSHA256'];assert approved==p.app_source_hashes()
 for path in paths:
  manifest,rr=inspect_manifest(path);assert manifest['kind']=='frozen_train_val_photo' and manifest['sourceCodeSHA256']==approved;receipts.append({'path':str(Path(path).resolve()),'sha256':p.sha(path),'appSourceCodeSHA256':manifest['sourceCodeSHA256']})
  for row in rr:
   assert row['id'] in records and row['split']==records[row['id']]['split'] and row['sourceImageSHA256']==records[row['id']]['sha256'] and row['sourceManifestSHA256']==binding['manifestSHA256']
   if row['split']=='val':rows.append(row)
 assert len(rows)==2*len(val) and {(r['variant'],r['id']) for r in rows}=={(variant,id) for variant in ['gallery','camera'] for id in val}
 output={'status':'VALIDATED_APP_PREPROCESSING_INPUTS','recordsSHA256':p.sha(p.OUT/'records.json'),'approvedAppSourceCodeSHA256':approved,'testContentIncluded':False,'variants':['gallery','camera'],'images':rows,'appExportManifests':receipts,'pixelClaim':'ActualAppPNG pixels willbe fedidentically toTorch/CoreML. MeasuredUIKit/OpenCV resizes may differ; no pixel-equivalence claim.','modelInferencePerformedByBinding':False}
 p.save(p.OUT/'evaluation-inputs.json',output);return output

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['audit','bind']);parser.add_argument('manifests',nargs='+',type=Path);args=parser.parse_args();result=audit(args.manifests) if args.phase=='audit' else bind(args.manifests);print(result['status'])
