#!/usr/bin/env python3
"""Download the published Figshare v3 files anonymously, checking every source MD5."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, time, urllib.request
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/datasets/colony-figshare-22022540-v3'
API = 'https://api.figshare.com/v2/articles/22022540/versions/3'

def md5(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def get_file(meta):
    name=meta['name'];assert Path(name).name==name
    dest=OUT/'raw'/name
    expected=meta.get('computed_md5') or meta['supplied_md5']
    if dest.exists() and dest.stat().st_size==meta['size'] and md5(dest)==expected:
        return {'name':name,'bytes':meta['size'],'md5':expected,'status':'VERIFIED_EXISTING'}
    for attempt in range(3):
        part=dest.with_name(dest.name+'.part')
        try:
            with urllib.request.urlopen(meta['download_url'],timeout=60) as r,part.open('wb') as f:
                while True:
                    b=r.read(1024*1024)
                    if not b:break
                    f.write(b)
            assert part.stat().st_size==meta['size'],name+' size mismatch'
            assert md5(part)==expected,name+' MD5 mismatch'
            os.replace(part,dest)
            return {'name':name,'bytes':meta['size'],'md5':expected,'status':'DOWNLOADED_VERIFIED'}
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))

def main():
    (OUT/'raw').mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(API,timeout=30) as r:data=json.load(r)
    (OUT/'source-metadata.json').write_text(json.dumps(data,indent=2)+'\n')
    files=data['files'];print('Downloading',len(files),'files;',sum(f['size'] for f in files),'bytes; no authentication',flush=True)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for f in concurrent.futures.as_completed([pool.submit(get_file,m) for m in files]):
            results.append(f.result())
            if len(results)%25==0 or len(results)==len(files):print('Verified',len(results),'/',len(files),flush=True)
    report={'source':API,'authenticationUsed':False,'files':sorted(results,key=lambda x:x['name']),'totalBytes':sum(f['bytes'] for f in results),'status':'COMPLETE_ALL_FILES_MD5_VERIFIED','downloadedAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (OUT/'download-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],flush=True)
if __name__=='__main__':main()
