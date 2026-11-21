"""Manually selected HBB coordinates from actually viewed photographs; no model output."""
from pathlib import Path
import json,hashlib,datetime
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'output/pipes-hardening-20261026/data/positives';ART=Path(__file__).parent
allcommons=json.load(open(OUT/'sources/commons-image-index.json'))+json.load(open(OUT/'sources/commons-additional-index.json'));meta={r['id']:r for r in allcommons}
boxes={
'commons_018':[[705,53,811,217]],
'commons_019':[[663,133,900,454]],
'commons_082':[[109,547,148,615],[132,558,174,630]],
'commons_091':[[582,331,651,410]],
'commons_121':[[344,331,432,483],[647,364,756,480]],
'commons_124':[[765,89,847,142]],
'commons_125':[[153,243,335,329],[432,803,613,950]],
'commons_129':[[590,313,804,548]],
'commons_133':[[232,326,291,393],[286,264,349,336],[327,215,379,271]],
'commons_145':[[80,67,888,745]],
'github_005':[[0,26,209,236],[0,172,147,343],[0,350,130,523],[169,174,350,352],[127,339,306,505],[330,138,540,345],[297,300,477,472],[473,272,662,457],[576,109,755,289],[645,266,796,427],[769,143,870,334],[801,306,870,477]]
}
groups={'commons_018':'commons_assortment_rusty_pipes_capture','commons_019':'commons_assortment_rusty_pipes_capture','commons_082':'commons_norderney_pipeworks_2025','commons_091':'commons_norderney_pipeworks_2025','commons_121':'commons_pipes042_ubt_scene','commons_124':'commons_hollandarod_mooring_pipes','commons_125':'commons_hollandarod_mooring_pipes','commons_129':'commons_furilden_rusty_pipe','commons_133':'commons_rusty_tubes_259086707','commons_145':'commons_torch_cut_pipe','github_005':'github_pipeCountingUNITN_pipe1_stock_scene'}
notes={
'commons_018':'One front-facing cut end; all other visible tubes are side walls. Same physical pile/capture family as019.',
'commons_019':'Closer view of same single open end/pile as018; not an independent scene.',
'commons_082':'Two overlapping open ends at foreground left. Distant attached pipeline has no discernible open mouth; do not count shafts or the rear lips of the same two pipes twice.',
'commons_091':'One visible front mouth, partly covered by attached flap. Flap itself is not a separate target; samebeach capture family as082.',
'commons_121':'Two separate rusty round pipe sections with visible front openings; rear flange rings and bolt holes not counted separately.',
'commons_124':'One visible open vertical pipe top. Small side drilled hole and clamp are not pipe ends; adjacent horizontal mouth occluded from this angle.',
'commons_125':'Two visible openings: upright post top and horizontal pipe end near clamp. Rear upright extends beyond frame, no visible end. Same physical mooring assembly as124.',
'commons_129':'One encrusted metal pipe cut end; box encloses the visible irregular outer rim, not the whole shaft.',
'commons_133':'Three cut metal tube mouths attached to industrial structure. Background beams and clamps are not mouths.',
'commons_145':'One torch-cut thick steel pipe face with slag/debris; inner opening and outer cut wall are one instance.',
'github_005':'Twelve visible steel pipe front faces, including cropped faces along left/right image edges; no extra instance for internal shading. Real photographic black-and-white source, not inference output.'}
rows=[];font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',20)
for id,xy in boxes.items():
 if id.startswith('commons'):
  r=meta[id];path=r['path'];provenance={'sourcePageURL':r['pageURL'],'sourceOriginalURL':r['originalURL'],'sourceDownloadURL':r['thumbnailURL'],'sourceResolution':'Commons served thumbnail actually used; do not claim originalfullresolution','sourceMetadata':r['sourceMetadata']}
 else:
  r=next(r for r in json.load(open(OUT/'sources/github-image-index.json')) if r['id']==id);path=r['path'];provenance={'sourcePageURL':'https://github.com/'+r['repo'],'sourceDownloadURL':r['url'],'sourceCommit':r['commit'],'sourceMember':r['sourcePath']}
 p=ROOT/path
 with Image.open(p) as image:w,h=image.size;display=image.convert('RGB')
 assert all(0<=a<c<=w and 0<=b<d<=h for a,b,c,d in xy)
 rows.append({'id':id,'imagePath':path,'width':w,'height':h,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'xyxy':xy,'count':len(xy),'groupId':groups[id],'source':'wikimedia_commons' if id.startswith('commons') else 'github_public_photo','subtype':'round','annotationBoundary':'outer_rim','annotationMethod':'Manual visual count and outer-rim HBB coordinates byassistant, no learned model proposals; numbered overlay inspection pending independentreview.','reviewNotes':notes[id],'annotationQA':'PENDING_DUAL_REVIEW','split':None,**provenance})
 d=ImageDraw.Draw(display)
 for i,(x1,y1,x2,y2) in enumerate(xy,1):d.rectangle((x1,y1,x2,y2),outline='#00ee44',width=2);d.text((x1+2,y1+2),str(i),font=font,fill='#ff2d55',stroke_width=1,stroke_fill='black')
 q=OUT/'review/general-annotations'/(id+'.jpg');q.parent.mkdir(parents=True,exist_ok=True);display.save(q,quality=97)
report={'status':'MANUAL_COORDINATES_WRITTEN_PENDING_OVERLAY_AND_INDEPENDENT_QA','createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'images':rows,'manualCount':sum(r['count'] for r in rows),'modelInferencePerformed':False,'sourceModified':False,'splitAssigned':False}
(ART/'general-manual-annotations.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('Manually annotated',len(rows),'images',report['manualCount'],'visible ends')
