#!/usr/bin/env python3
"""Anonymous bounded MACO pipe-only acquisition. Never loads model weights."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/pipes-20261026/data'
SOURCE=OUT/'sources/MACO-zenodo17165968.json'
CHOSEN={'Steel pipe.zip':'round-steel-pipe.zip','Square steel tube.zip':'square-steel-tube.zip'}


def hashes(path):
    sha,md5=hashlib.sha256(),hashlib.md5()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(2*1024*1024),b''):
            sha.update(block);md5.update(block)
    return sha,md5


def download(item):
    path=OUT/'archives'/CHOSEN[item['key']];path.parent.mkdir(parents=True,exist_ok=True)
    expected=item['size'];expected_md5=item['checksum'].removeprefix('md5:')
    if path.exists():
        sha,md5=hashes(path)
        assert path.stat().st_size==expected and md5.hexdigest()==expected_md5
        return {'sourceKey':item['key'],'path':str(path.relative_to(ROOT)),'status':'VERIFIED_EXISTING','bytes':expected,'sha256':sha.hexdigest(),'md5':md5.hexdigest()}
    part=path.with_suffix(path.suffix+'.part')
    started=time.time();attempts=[]
    for attempt in range(3):
        offset=part.stat().st_size if part.exists() else 0
        assert 0<=offset<=expected
        if offset==expected:break
        sha,md5=hashes(part) if offset else (hashlib.sha256(),hashlib.md5())
        headers={'User-Agent':'CountScope public steel-pipe dataset research'}
        if offset:headers['Range']=f'bytes={offset}-{expected-1}'
        try:
            req=urllib.request.Request(item['links']['self'],headers=headers)
            with urllib.request.urlopen(req,timeout=90) as response,part.open('ab' if offset else 'wb') as output:
                if offset:
                    assert response.status==206 and response.headers.get('Content-Range')==f'bytes {offset}-{expected-1}/{expected}'
                else:assert response.status==200
                received=offset;last=received
                while True:
                    block=response.read(2*1024*1024)
                    if not block:break
                    received+=len(block);assert received<=expected
                    output.write(block);sha.update(block);md5.update(block)
                    if received-last>=64*1024*1024:
                        print(json.dumps({'source':item['key'],'receivedMiB':round(received/2**20,1),'totalMiB':round(expected/2**20,1)}),flush=True);last=received
                assert received==expected
            break
        except Exception as error:
            attempts.append({'attempt':attempt+1,'startOffset':offset,'error':str(error)})
            print(json.dumps({'source':item['key'],'retry':attempt+1,'error':str(error)}),flush=True)
            if attempt==2:raise
            time.sleep(2)
    sha,md5=hashes(part)
    assert part.stat().st_size==expected and md5.hexdigest()==expected_md5,'Archive checksum mismatch'
    part.replace(path)
    return {'sourceKey':item['key'],'sourceRecord':'https://zenodo.org/records/17165968','url':item['links']['self'],'path':str(path.relative_to(ROOT)),'status':'DOWNLOADED_MD5_AND_SIZE_VERIFIED','bytes':expected,'sha256':sha.hexdigest(),'md5':md5.hexdigest(),'elapsedSeconds':round(time.time()-started,2),'retries':attempts,'authenticationUsed':False,'modelWeightsDownloaded':False}


def main():
    meta=json.loads(SOURCE.read_text());items=[f for f in meta['files'] if f['key'] in CHOSEN]
    assert len(items)==2 and sum(f['size'] for f in items)<3_000_000_000
    print(json.dumps({'plannedFiles':[f['key'] for f in items],'plannedUniqueDownloadBytes':sum(f['size'] for f in items)}),flush=True)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download,item) for item in items]):
            result=future.result();results.append(result)
            (OUT/'sources/download-receipts.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
