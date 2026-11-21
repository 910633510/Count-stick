from pathlib import Path
import urllib.request,urllib.parse,json,concurrent.futures,hashlib
from PIL import Image,ImageOps,ImageDraw
O=Path('output/pipes-hardening-20261026/data/positives');S=O/'sources';headers={'User-Agent':'CountScopeDatasetReview/1.0'};old=json.load(open(S/'commons-image-index.json'));seen={r['pageId'] for r in old};pages={}
for category in ['Steel pipes','Rusty pipes']:
 prior=json.load(open(S/('commons-category-'+category.replace(' ','-')+'.json')));cont=prior.get('continue',{})
 if not cont:continue
 params={'action':'query','generator':'categorymembers','gcmtitle':'Category:'+category,'gcmtype':'file','gcmlimit':200,'prop':'imageinfo','iiprop':'url|extmetadata|size|timestamp','iiurlwidth':800,'format':'json',**cont}
 d=json.load(urllib.request.urlopen(urllib.request.Request('https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params),headers=headers),timeout=30));(S/('commons-category-'+category.replace(' ','-')+'-continued.json')).write_text(json.dumps(d,indent=2));pages.update(d.get('query',{}).get('pages',{}))
need=['Hoover Dam Low Flow','Open pipe in a Chico','Steel tubes to reinforce','Stacked pipes at LNG','Stacked up Pipes','Gas Pipes -','Irrigation pipes stacked']
for f in S.glob('commons-extra-*.json'):
 for k,v in json.load(open(f)).get('query',{}).get('pages',{}).items():
  if any(x in v['title'] for x in need):pages[k]=v
rows=[]
for p in sorted(pages.values(),key=lambda p:p['title']):
 if p['pageid'] in seen or 'imageinfo' not in p or 'nude' in p['title'].lower() or not p['title'].lower().endswith(('.jpg','.jpeg','.png','.tif')):continue
 i=p['imageinfo'][0]
 if not i.get('thumburl'):continue
 rows.append({'id':f'commons_{100+len(rows):03d}','pageId':p['pageid'],'title':p['title'],'originalURL':i['url'],'thumbnailURL':i['thumburl'],'pageURL':i['descriptionurl'],'sourceWidth':i['width'],'sourceHeight':i['height'],'sourceBytes':i.get('size'),'sourceMetadata':i.get('extmetadata',{})})
def get(r):
 try:
  p=O/'commons-thumbnails'/(r['id']+'.jpg');b=urllib.request.urlopen(urllib.request.Request(r['thumbnailURL'],headers=headers),timeout=25).read(5_000_001);assert len(b)<=5_000_000;p.write_bytes(b)
  with Image.open(p) as im:im.verify()
  r.update(path=str(p),downloadBytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 except Exception as e:r['downloadError']=str(e)
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(get,rows))
(S/'commons-additional-index.json').write_text(json.dumps(rows,indent=2));ready=[r for r in rows if 'path'in r]
for start in range(0,len(ready),30):
 canvas=Image.new('RGB',(1500,1080),'#eee');d=ImageDraw.Draw(canvas)
 for j,r in enumerate(ready[start:start+30]):
  with Image.open(r['path']) as im:im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((290,145))
  x=j%5*300;y=j//5*180;canvas.paste(im,(x,y));d.text((x+3,y+148),r['id']+' '+r['title'][5:35],fill='black')
 canvas.save(O/'review'/f'commons-extra-{start//30:02d}.jpg',quality=95)
print('MoreCommons',len(ready),'bytes',sum(r.get('downloadBytes',0) for r in rows))
