#!/usr/bin/env python3
"""Bounded anonymous acquisition; never labels an image before visual review."""
from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request,time
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/research-20261025-hardcases/negatives'
SEED='CountScope-real-empty-dishes-20261025-v1'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 records=json.loads((OUT/'mendeley-negative-files.json').read_text())
 images=[r for r in records if r['filename'].lower().endswith('.jpg')]
 assert len(images)==500 and len({r['filename'] for r in images})==500
 chosen=sorted(images,key=lambda r:sha((SEED+'|'+r['filename']).encode()))[:48]
 protocol={'frozenUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sourceDataset':'https://data.mendeley.com/datasets/xrbtd74pfj/3','sourcePaper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC10024077/','sourceFolderMeaning':'Negative is author-defined no growth; 1CFU is in Uncertain, excluded.','selection':'Lowest48SHA256(seed|filename) among500Negativeoriginals. No model score, appearance, or outcome selected.','seed':SEED,'selected':chosen,'annotationPolicy':'Visual review required before zero-colony label; uncertain marks quarantine. No replacement of quarantined images in this batch.','groupingPolicy':'Same source file/variants stay together. No physical plate ID supplied in public image catalog; audit selected files for exact and high similarity duplicates before root split freeze.','trainOrValSplitAssigned':False,'noInference':True}
 path=OUT/'mendeley-selection-before-download.json'
 if path.exists():assert json.loads(path.read_text())['selected']==chosen
 else:path.write_text(json.dumps(protocol,indent=2)+'\n')
 def fetch(r):
  p=OUT/'originals/mendeley-negative'/r['filename'];p.parent.mkdir(parents=True,exist_ok=True);meta=r['content_details'];assert meta['size']<3000000
  if p.exists():b=p.read_bytes()
  else:
   req=urllib.request.Request(meta['download_url'],headers={'User-Agent':'Mozilla/5.0','Referer':'https://data.mendeley.com/datasets/xrbtd74pfj/3'})
   with urllib.request.urlopen(req,timeout=45) as res:b=res.read(3000001)
   assert len(b)==meta['size'] and sha(b)==meta['sha256_hash'];p.write_bytes(b)
  assert len(b)==meta['size'] and sha(b)==meta['sha256_hash'];return {'filename':r['filename'],'id':r['id'],'path':str(p.relative_to(ROOT)),'sha256':sha(b),'bytes':len(b),'sourceURL':meta['download_url']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  result=list(pool.map(fetch,chosen))
 (OUT/'mendeley-download-receipt.json').write_text(json.dumps({'downloaded':len(result),'totalBytes':sum(r['bytes'] for r in result),'authenticationUsed':False,'records':result},indent=2)+'\n')
 print('Downloaded',len(result),'images',sum(r['bytes'] for r in result),'bytes',flush=True)
if __name__=='__main__':main()
