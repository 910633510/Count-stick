#!/usr/bin/env python3
"""Anonymous, bounded range extraction of a pre-registered external test cohort."""
from pathlib import Path,PurePosixPath
import collections,concurrent.futures,hashlib,io,json,re,struct,time,urllib.request,zlib,xml.etree.ElementTree as ET
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'output/research-20261025';OUT=ROOT/'output/datasets/colony-external-zenodo18256864-v1'
SEED='CountScope-external-20261025-v1';PER_CLASS=4

def get_range(url,total,start,n):
 assert 0<=start<total and 0<n<=32*1024*1024 and start+n<=total
 for attempt in range(4):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+n-1}','User-Agent':'CountScope public research audit'})
   with urllib.request.urlopen(req,timeout=60) as res:
    assert res.status==206
    assert res.headers.get('Content-Range')==f'bytes {start}-{start+n-1}/{total}'
    data=res.read(n+1);assert len(data)==n
   return data
  except Exception:
   if attempt==3:raise
   time.sleep(2*(attempt+1))

def extract_one(meta,entry):
 url=meta['links']['self'];total=meta['size'];start=entry['headerOffset'];hdr=get_range(url,total,start,30)
 assert hdr[:4]==b'PK\x03\x04'
 fields=struct.unpack('<4s5H3I2H',hdr);flags=fields[2];method=fields[3];name_len,extra_len=fields[-2:]
 assert not (flags&1) and method in (0,8) and method==entry['compressType']
 compressed=get_range(url,total,start+30+name_len+extra_len,entry['compressedSize'])
 data=compressed if method==0 else zlib.decompress(compressed,-15)
 assert len(data)==entry['size'] and zlib.crc32(data)&0xffffffff==entry['CRC']
 return data

def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True);(OUT/'labels').mkdir(exist_ok=True)
 source=json.loads((WORK/'zenodo-18256864-metadata.json').read_text());image_records=json.loads((WORK/'images.zip-directory.json').read_text());label_records=json.loads((WORK/'label.zip-directory.json').read_text())
 images=[r for r in image_records if r['filename'].endswith('.jpg')];assert len(images)==950
 classes=collections.defaultdict(list)
 for r in images:classes[PurePosixPath(r['filename']).parent.name].append(r)
 assert len(classes)==19 and all(len(v)==50 for v in classes.values())
 chosen=[]
 for c,items in sorted(classes.items()):
  chosen.extend(sorted(items,key=lambda x:hashlib.sha256((SEED+'|'+x['filename']).encode()).hexdigest())[:PER_CLASS])
 labels={PurePosixPath(x['filename']).stem:x for x in label_records if x['filename'].endswith('.xml')};assert len(labels)==950
 protocol={'registeredBeforeInferenceUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sourceRecord':'https://zenodo.org/records/18256864','sourcePaper':'https://www.nature.com/articles/s41597-026-07095-5','selection':'Four original images per19species, ranked only bySHA256(seed|archive-filename), no model scores or box-count filtering. Augmented archives excluded.','seed':SEED,'selected':[{'species':PurePosixPath(x['filename']).parent.name,'filename':PurePosixPath(x['filename']).name,'imageArchiveEntry':x,'labelArchiveEntry':labels[PurePosixPath(x['filename']).stem]} for x in chosen],'purpose':'New-study external test only; no training/calibration or parameter selection on these images.','archiveIntegrity':'Per-entry CRC32 and extracted SHA256; full remote archive MD5 is not verified because only selected entries are downloaded.'}
 p=OUT/'selection-before-inference.json'
 if p.exists():
  old=json.loads(p.read_text());assert old['selected']==protocol['selected'];protocol=old
 else:p.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 image_meta=next(x for x in source['files'] if x['key']=='images.zip');label_meta=next(x for x in source['files'] if x['key']=='label.zip')
 def download(row):
  name=row['filename'];im=OUT/'images'/name;lab=OUT/'labels'/(Path(name).stem+'.xml')
  for path,meta,entry in [(im,image_meta,row['imageArchiveEntry']),(lab,label_meta,row['labelArchiveEntry'])]:
   if path.exists() and path.stat().st_size==entry['size'] and (zlib.crc32(path.read_bytes())&0xffffffff)==entry['CRC']:continue
   b=extract_one(meta,entry);tmp=path.with_suffix(path.suffix+'.part');tmp.write_bytes(b);tmp.replace(path)
  return name
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for i,f in enumerate(concurrent.futures.as_completed([pool.submit(download,r) for r in protocol['selected']])):
   print('Downloaded',i+1,'/',len(chosen),f.result(),flush=True)
 # Audit all annotations before any model inference. Invalid cases stop, never get replaced by easier images.
 old_manifest=json.loads((ROOT/'output/datasets/colony-figshare-22022540-v3/yolo-species-v1/manifest.json').read_text());old_hashes={r['sha256'] for r in old_manifest['images']}
 records=[]
 for row in protocol['selected']:
  name=row['filename'];p=OUT/'images'/name;l=OUT/'labels'/(Path(name).stem+'.xml');xml=ET.fromstring(l.read_bytes());width=int(xml.findtext('size/width'));height=int(xml.findtext('size/height'))
  with Image.open(p) as pil:pil.load();assert pil.size==(width,height);assert pil.getexif().get(274,1)==1
  boxes=[]
  for o in xml.findall('object'):
   b=[float(o.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']];assert 0<=b[0]<b[2]<=width and 0<=b[1]<b[3]<=height,(name,b,width,height)
   boxes.append(b)
  h=hashlib.sha256(p.read_bytes()).hexdigest();assert h not in old_hashes
  records.append({'filename':name,'species':row['species'],'width':width,'height':height,'sha256':h,'labelSHA256':hashlib.sha256(l.read_bytes()).hexdigest(),'count':len(boxes),'xyxy':boxes})
 assert len({r['sha256'] for r in records})==len(records)
 manifest={'status':'FROZEN_EXTERNAL_TEST_BEFORE_INFERENCE','sourceRecord':protocol['sourceRecord'],'sourcePaper':protocol['sourcePaper'],'images':records,'imageCount':len(records),'speciesCount':len(classes),'objects':sum(r['count'] for r in records),'selectionProtocolSHA256':hashlib.sha256((OUT/'selection-before-inference.json').read_bytes()).hexdigest(),'allEntryCRCsVerified':True,'exactOverlapWithOldDataset':False,'augmentationFilesIncluded':False,'authenticationUsed':False,'dataOnlyNoInference':True,'limitations':'Controlled imaging-system photos from a separate study; not iPhone field acceptance. Different images can still share strains/plates within this test study.'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print('EXTERNAL TEST READY',len(records),'images',manifest['objects'],'objects',flush=True)
if __name__=='__main__':main()
