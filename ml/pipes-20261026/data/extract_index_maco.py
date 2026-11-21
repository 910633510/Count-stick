#!/usr/bin/env python3
"""Safe extraction and technical QA; no model or train/val/test assignment."""
from pathlib import Path,PurePosixPath,PureWindowsPath
import collections,hashlib,json,math,stat,zipfile,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/pipes-20261026/data'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(2*1024*1024),b''):h.update(block)
    return h.hexdigest()

def phash(image):
    array=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),float)
    coefficients=dctn(array,type=2,norm='ortho')[:8,:8];bits=(coefficients>np.median(coefficients.ravel()[1:])).ravel()
    return ''.join(f'{int("".join("1" if v else "0" for v in bits[i:i+4]),2):x}' for i in range(0,64,4))

def main():
    metadata=json.loads((OUT/'sources/MACO-zenodo17165968.json').read_text());receipts=json.loads((OUT/'sources/download-receipts.json').read_text());bykey={x['sourceKey']:x for x in receipts}
    inventory=[]
    for key,tag in [('Steel pipe.zip','round'),('Square steel tube.zip','square')]:
        receipt=bykey[key];path=ROOT/receipt['path'];assert digest(path)==receipt['sha256'];base=OUT/'raw'/tag;base.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist();assert sum(x.file_size for x in infos)<16_000_000_000
            assert len({x.filename for x in infos})==len(infos)
            for info in infos:
                rel=PurePosixPath(info.filename);assert not rel.is_absolute() and '..' not in rel.parts and not PureWindowsPath(info.filename).is_absolute()
                assert '\\' not in info.filename and stat.S_IFMT(info.external_attr>>16)!=stat.S_IFLNK and not(info.flag_bits&1)
                if info.is_dir():continue
                assert info.file_size<150_000_000
                destination=base.joinpath(*rel.parts);destination.parent.mkdir(parents=True,exist_ok=True)
                data=archive.read(info)  # zipfile verifies each member CRC.
                assert len(data)==info.file_size
                if destination.exists():assert destination.read_bytes()==data
                else:destination.write_bytes(data)
            inventory.append({'sourceKey':key,'archiveSHA256':receipt['sha256'],'memberCount':len(infos),'uncompressedBytes':sum(x.file_size for x in infos),'allMemberCRCsVerified':True,'safePathsNoEncryptedOrSymlinkMembers':True})
        print('Extracted',tag,flush=True)
    (OUT/'sources/extraction-receipt.json').write_text(json.dumps(inventory,indent=2)+'\n')
    records=[];pixel_groups=collections.defaultdict(list);byte_groups=collections.defaultdict(list);issues=[]
    for tag in ('round','square'):
        folder=OUT/'raw'/tag/('Steel pipe' if tag=='round' else 'Square steel tube')/'HBB'
        paths=sorted(folder.glob('*.jpg'));assert len(paths)==(1025 if tag=='round' else 577)
        for idx,image_path in enumerate(paths):
            label_path=image_path.with_suffix('.xml');assert label_path.exists();xml=ET.parse(label_path).getroot();objects=xml.findall('object')
            with Image.open(image_path) as image:
                image.load();w,h=image.size;mode=image.mode;orientation=image.getexif().get(274,1);rgb=image.convert('RGB');pixel_sha=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest();p=phash(rgb)
                exif_keys={str(k):str(v) for k,v in image.getexif().items() if k in (271,272,274,306,36867,36868)}
            rowissues=[]
            if (int(xml.findtext('size/width')),int(xml.findtext('size/height')))!=(w,h):rowissues.append('XML_IMAGE_DIMENSION_MISMATCH')
            if orientation not in (None,1):rowissues.append('EXIF_REQUIRES_EXPLICIT_GEOMETRY_NORMALIZATION')
            boxes=[];names=[]
            for obj in objects:
                box=[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')];names.append(obj.findtext('name'))
                if not all(math.isfinite(v) for v in box) or not(0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):rowissues.append('INVALID_BOX_BOUNDS')
                boxes.append(box)
            if len({tuple(b) for b in boxes})!=len(boxes):rowissues.append('EXACT_DUPLICATE_BOXES')
            valid_classes={'pipe'} if tag=='round' else {'square tube'}
            if set(names)-valid_classes:rowissues.append('UNEXPECTED_AUTHOR_CLASS')
            raw_sha=digest(image_path);identity=f'maco_{tag}_{image_path.stem}'
            record={'id':identity,'source':'zenodo17165968-MACO','sourceArchive':tag,'subtype':'round_steel_pipe' if tag=='round' else 'square_steel_tube','authorImageId':image_path.stem,'imagePath':str(image_path.relative_to(ROOT)),'labelPath':str(label_path.relative_to(ROOT)),'width':w,'height':h,'imageMode':mode,'sha256':raw_sha,'RGBPixelSHA256':pixel_sha,'labelSHA256':digest(label_path),'pHash64':p,'EXIF':exif_keys,'xyxy':boxes,'count':len(boxes),'authorClasses':sorted(set(names)),'annotationBoundary':'PENDING_VISUAL_AUDIT_OUTER_RIM_VS_OPENING','annotationQA':'QUARANTINE_TECHNICAL_ISSUE' if rowissues else 'PENDING_VISUAL_SCENE_AND_LABEL_QA','technicalIssues':sorted(set(rowissues)),'split':None,'groupId':None,'modelInferencePerformed':False}
            records.append(record);pixel_groups[pixel_sha].append(identity);byte_groups[raw_sha].append(identity)
            if rowissues:issues.append({'id':identity,'issues':sorted(set(rowissues))})
            if (idx+1)%100==0:print('Indexed',tag,idx+1,'/',len(paths),flush=True)
    report={'status':'TECHNICAL_INDEX_NOT_FROZEN_FOR_TRAINING','sourceRecord':'https://zenodo.org/records/17165968','images':records,'countsBySubtype':dict(collections.Counter(r['subtype'] for r in records)),'objectsBySubtype':{s:sum(r['count'] for r in records if r['subtype']==s) for s in {r['subtype'] for r in records}},'exactDuplicateByteGroups':[v for v in byte_groups.values() if len(v)>1],'exactDuplicatePixelGroups':[v for v in pixel_groups.values() if len(v)>1],'technicalIssues':issues,'originalFoldersAreAnnotationVariantsNotIndependentSplits':True,'modelInferencePerformed':False}
    (OUT/'technical-index.json').write_text(json.dumps(report,indent=2)+'\n')
    review=OUT/'review';review.mkdir(exist_ok=True);font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',15)
    for tag in ('round','square'):
        rows=[r for r in records if r['sourceArchive']==tag]
        for start in range(0,len(rows),50):
            canvas=Image.new('RGB',(10*170,5*160),'#f5f5f5');draw=ImageDraw.Draw(canvas)
            for pos,r in enumerate(rows[start:start+50]):
                with Image.open(ROOT/r['imagePath']) as image:image=image.convert('RGB');image.thumbnail((166,125))
                x=pos%10*170;y=pos//10*160;canvas.paste(image,(x+(170-image.width)//2,y));draw.text((x+3,y+126),f'{start+pos+1:04d}  GT {r["count"]}',font=font,fill='black');draw.text((x+3,y+143),f'{r["width"]}x{r["height"]}',font=font,fill='black')
            canvas.save(review/f'{tag}-contact-{start//50:02d}.jpg',quality=92)
    print(json.dumps({k:v for k,v in report.items() if k not in ('images',)}),flush=True)

if __name__=='__main__':main()
