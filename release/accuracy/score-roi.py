import json,math,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[2]
a=json.loads((root/'release/accuracy/annotations-review.json').read_text())
input_path=Path(sys.argv[2]) if len(sys.argv)>2 else root/'release/accuracy/results/crops-and-tiles.json'
x=json.loads(input_path.read_text())
labels=a['roi']['labels'];polygons=[r['polygon_pixels_full_image'] for r in a['roi']['ignore_regions']]
def point_in_polygon(p,poly):
 x,y=p;inside=False
 for i,(ax,ay) in enumerate(poly):
  bx,by=poly[(i+1)%len(poly)]
  if (ay>y)!=(by>y) and x<(bx-ax)*(y-ay)/(by-ay)+ax:inside=not inside
 return inside

def score(dets,mode,labels=labels,polygons=polygons):
 points=[(d['rectTopLeftPixels'][0]+d['rectTopLeftPixels'][2]/2,d['rectTopLeftPixels'][1]+d['rectTopLeftPixels'][3]/2) for d in dets]
 dp={0:(0,[])}
 for pi,p in enumerate(points):
  eligible=[]
  for gi,g in enumerate(labels):
   distance=math.dist(p,g['center_pixels_full_image'])
   if mode=='bbox_plus25':
    x1,y1,x2,y2=g['terminal_face_bbox_full_image'];ok=x1-25<=p[0]<=x2+25 and y1-25<=p[1]<=y2+25
   else:ok=distance<=mode
   if ok:eligible.append((gi,distance))
  updated=dict(dp)
  for mask,(cost,matches) in dp.items():
   for gi,distance in eligible:
    if mask&(1<<gi):continue
    newmask=mask|(1<<gi);newcost=cost+distance
    if newmask not in updated or newcost<updated[newmask][0]:updated[newmask]=(newcost,matches+[(pi,gi,distance)])
  dp=updated
 mask,(cost,matches)=min(dp.items(),key=lambda item:(-bin(item[0]).count("1"),item[1][0]))
 used={m[0] for m in matches};unmatched=[i for i in range(len(points)) if i not in used]
 ignored=[i for i in unmatched if any(point_in_polygon(points[i],p) for p in polygons)]
 false=[i for i in unmatched if i not in ignored]
 tp=len(matches);fp=len(false);fn=len(labels)-tp
 return {'TP':tp,'FP':fp,'FN':fn,'ignored':len(ignored),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/len(labels),'matched':[{'prediction':pi,'label':labels[gi]['id'],'distancePixels':round(dist,2)} for pi,gi,dist in matches],'falsePositiveIndices':false,'falsePositiveCenters':[points[i] for i in false],'missedLabels':[labels[i]['id'] for i in range(len(labels)) if not mask&(1<<i)]}
rows=[]
for method,key,detkey in [('wholeImage','wholeImageComparisons','detectionsInROI'),('independentCrop','roiCropComparisons','detections'),('overlappingTiles','tileComparisons','detectionsInROI')]:
 for r in x[key]:
  if r.get('fixture','bamboo-skewers-cc0.jpg')!='bamboo-skewers-cc0.jpg':continue
  row={k:v for k,v in r.items() if k not in ['detections','detectionsInROI']}
  row['method']=method
  row['metrics']={str(mode):score(r[detkey],mode) for mode in [25,50,80,110,'bbox_plus25']}
  rows.append(row)
holdout_rows=[]
if 'holdout_roi' in a:
 holdout=a['holdout_roi']; hlabels=holdout['labels']; hpolygons=[r['polygon_pixels_full_image'] for r in holdout['ignore_regions']]
 hx,hy,hw,hh=holdout['xywh']
 raw=json.loads((root/'release/accuracy/results/thresholds.json').read_text())['samples'][0]
 baseline=next(r for r in raw['baseline'] if r['mode']=='scaleFill')
 lower=next(r for r in raw['directPrediction'] if r['mode']=='scaleFill' and r['confidenceThreshold']==.25)
 for name,source in [('baseline_fill_conf025_iou070',baseline),('candidate_fill_conf025_iou045',lower)]:
  dets=[]
  for d in source['detections']:
   if 'rectBottomLeft' in d:
    x0,y0,w,h=d['rectBottomLeft'];rect=[x0*4000,(1-y0-h)*3000,w*4000,h*3000];confidence=d['confidence']
   else:
    cx,cy,w,h=d['centerXYWHOriginTopLeft'];rect=[(cx-w/2)*4000,(cy-h/2)*3000,w*4000,h*3000];confidence=d['score']
   cx=rect[0]+rect[2]/2;cy=rect[1]+rect[3]/2
   if hx<=cx<=hx+hw and hy<=cy<=hy+hh:dets.append({'rectTopLeftPixels':rect,'confidence':confidence})
  holdout_rows.append({'configuration':name,'detectionsInROI':dets,'metrics':{str(mode):score(dets,mode,hlabels,hpolygons) for mode in [25,50,80,110,'bbox_plus25']}})
(Path(sys.argv[3]) if len(sys.argv)>3 else root/'release/accuracy/results/roi-scores.json').write_text(json.dumps({'annotation':str(root/'release/accuracy/annotations-review.json'),'rows':rows,'holdoutRows':holdout_rows},indent=2))
for r in holdout_rows:
 print('HOLDOUT',r['configuration'],{k:(m['TP'],m['FP'],m['FN'],m['ignored']) for k,m in r['metrics'].items()})
for r in rows:
 if (r['method']=='wholeImage' and r['confidenceThreshold'] in [.25,.4,.5] and r['iouThreshold'] in [.45,.7]) or (r['method']=='overlappingTiles' and r['grid']==2 and r['globalIoU']==.45) or (r['method']=='independentCrop' and r['confidenceThreshold']==.25 and r['iouThreshold']==.45):
  print(r['method'],r['mode'],r['confidenceThreshold'],r.get('iouThreshold',r.get('globalIoU')),[(k,(r['metrics'][k]['TP'],r['metrics'][k]['FP'],r['metrics'][k]['FN'],r['metrics'][k]['ignored'])) for k in ['25','50','80','110','bbox_plus25']])
