#!/usr/bin/env python3
"""Bounded ZIP directory inspection while complete archives download."""
from pathlib import Path
import collections,io,json,time,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/pipes-20261026/data'
class RangeFile(io.RawIOBase):
    def __init__(self,url,size):self.url,self.size,self.pos,self.requests=url,size,0,0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        if n>32*1024*1024:raise ValueError('remote read exceeds32MiB')
        start,end=self.pos,self.pos+n-1
        for attempt in range(3):
            req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','User-Agent':'CountScope public pipe ZIP audit'})
            try:
                with urllib.request.urlopen(req,timeout=60) as response:
                    assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{self.size}'
                    pieces=[];remaining=n+1
                    while remaining:
                        chunk=response.read(remaining)
                        if not chunk:break
                        pieces.append(chunk);remaining-=len(chunk)
                    data=b''.join(pieces)
                    if len(data)!=n:raise IOError(f'Short/long bounded range: expected{n},got{len(data)}')
                break
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
        self.pos+=n;self.requests+=1;return data

if __name__=='__main__':
    meta=json.loads((OUT/'sources/MACO-zenodo17165968.json').read_text())
    for key,tag in [('Steel pipe.zip','round'),('Square steel tube.zip','square')]:
        entry=next(f for f in meta['files'] if f['key']==key);remote=RangeFile(entry['links']['self'],entry['size'])
        with zipfile.ZipFile(remote) as archive:
            entries=[{'filename':i.filename,'size':i.file_size,'compressedSize':i.compress_size,'CRC':i.CRC,'headerOffset':i.header_offset,'compressType':i.compress_type,'flagBits':i.flag_bits} for i in archive.infolist()]
        (OUT/'sources'/(tag+'-archive-directory.json')).write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'type':tag,'members':len(entries),'uncompressedBytes':sum(e['size'] for e in entries),'extensions':dict(collections.Counter(Path(e['filename']).suffix for e in entries)),'examples':[e['filename'] for e in entries[:30]],'rangeRequests':remote.requests}),flush=True)
