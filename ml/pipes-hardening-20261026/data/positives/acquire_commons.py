from pathlib import Path
import urllib.request,urllib.parse,json,concurrent.futures,hashlib,time
from PIL import Image,ImageOps,ImageDraw
O=Path('output/pipes-hardening-20261026/data/positives');S=O/'sources';D=O/'commons-thumbnails';D.mkdir(parents=True,exist_ok=True)
headers={'User-Agent':'CountScopeDatasetReview/1.0 (manual research on public pipe photographs)'}
pages={}
for category in ['Steel pipes','Rusty pipes']:
 params={'action':'query','generator':'categorymembers','gcmtitle':'Category:'+category,'gcmtype':'file','gcmlimit':200,'prop':'imageinfo','iiprop':'url|extmetadata|size|timestamp','iiurlwidth':800,'format':'json'}
 url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)
 data=json.load(urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=35));(S/('commons-category-'+category.replace(' ','-')+'.json')).write_text(json.dumps(data,indent=2))
 pages.update(data.get('query',{}).get('pages',{}))
rows=[]
for page in sorted(pages.values(),key=lambda p:p['title']):
 title=page['title']
 if not title.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff')) or 'nude' in title.lower():continue
 info=page.get('imageinfo',[{}])[0]
 if not info.get('thumburl'):continue
 rows.append({'id':f'commons_{len(rows)+1:03d}','pageId':page['pageid'],'title':title,'originalURL':info['url'],'thumbnailURL':info['thumburl'],'pageURL':info['descriptionurl'],'sourceWidth':info['width'],'sourceHeight':info['height'],'sourceBytes':info.get('size'),'sourceMetadata':info.get('extmetadata',{})})
def download(r):
 dest=D/(r['id']+'.jpg');u=r['thumbnailURL']
 try:
  if not dest.exists():
   res=urllib.request.urlopen(urllib.request.Request(u,headers=headers),timeout=25);b=res.read(5_000_001);assert len(b)<=5_000_000;dest.write_bytes(b)
  with Image.open(dest) as im:im.verify()
  r['path']=str(dest);r['downloadBytes']=dest.stat().st_size;r['sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();return r
 except Exception as e:r['downloadError']=str(e);return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(download,rows))
(S/'commons-image-index.json').write_text(json.dumps(rows,indent=2))
ready=[r for r in rows if 'path' in r]
for start in range(0,len(ready),30):
 sheet=Image.new('RGB',(1500,1080),'#eee');dr=ImageDraw.Draw(sheet)
 for i,r in enumerate(ready[start:start+30]):
  with Image.open(r['path']) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((290,145))
  x=i%5*300;y=i//5*180;sheet.paste(im,(x,y));dr.text((x+3,y+148),r['id']+' '+r['title'][5:35],fill='black')
 sheet.save(O/'review'/f'commons-{start//30:02d}.jpg',quality=95)
print('Commons ready',len(ready),'downloadbytes',sum(r.get('downloadBytes',0) for r in rows),'failed',len(rows)-len(ready))
