"""Read cached validation boxes to attribute hard routing gaps; never infer on test."""
from pathlib import Path
import sys,json,collections,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'ml/colony'))
import numpy as np
from scipy.optimize import linear_sum_assignment
import evaluate as e

def matched(gt,b):
 i=e.iou_matrix(gt,b)
 if not i.size:return set()
 rr,cc=linear_sum_assignment(-((i>=.5)*1000+i));return {int(r) for r,c in zip(rr,cc) if i[r,c]>=.5}

def main():
 cache=ROOT/'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/evaluation';whole=json.loads((cache/'validation-predictions.json').read_text());tile=json.loads((cache/'validation-tiled-predictions.json').read_text());m=json.loads((e.DATA/'manifest.json').read_text());names=m['splits']['val'];result={}
 for cutoff in [16,24,32]:
  rows=[]
  for n in names:
   gt,(w,h)=e.gt(n,'val');bb={};sizes={};overlap={};selected={}
   for key,pred in [('whole',whole[n]),('tile',tile[n])]:
    b=np.asarray(pred['xyxy'],dtype=float).reshape(-1,4)[np.asarray(pred['confidence'])>=.3];bb[key]=b;sizes[key]=np.minimum(b[:,2]-b[:,0],b[:,3]-b[:,1])*1024/max(w,h);overlap[key]=e.iou_matrix(gt,b)
    keep=sizes[key]>=cutoff if key=='whole' else sizes[key]<cutoff;selected[key]=keep
   baseline_tp=matched(gt,bb['whole'])|matched(gt,bb['tile'])
   for i in sorted(baseline_tp):
    wh=overlap['whole'][i]>=.5;ti=overlap['tile'][i]>=.5
    kept=bool((wh&selected['whole']).any() or (ti&selected['tile']).any())
    if kept:continue
    reason='double_branch_size_disagreement' if wh.any() and ti.any() else 'whole_only_small_filtered' if wh.any() else 'tile_only_large_filtered'
    b=gt[i];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
    rows.append({'image':n,'species':n[:4],'gtIndex':i,'reason':reason,'gtMinSideAt1024':float(min(b[2]-b[0],b[3]-b[1])*1024/max(w,h)),'matchingWholePredictedSizes':sizes['whole'][wh].tolist(),'matchingTilePredictedSizes':sizes['tile'][ti].tolist(),'distanceToNearestQuadrantSeamAt1024':float(min(abs(cx-w/2),abs(cy-h/2))*1024/max(w,h))})
  result[str(cutoff)]={'confirmedBaselineTPWithoutAnyRetainedIoUCoverage':len(rows),'reasonCounts':dict(collections.Counter(r['reason'] for r in rows)),'speciesCounts':dict(collections.Counter(r['species'] for r in rows)),'cases':rows}
 report={'status':'COMPLETE_CACHED_VALIDATION_ROUTING_AUDIT','evaluatedAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Known validation only; no inference; no test accessed.','definition':'At confidence0.3, a GT was matched by at least one unchanged baseline, but no prediction overlapping it atIoU>=0.5 survives the hard scale gate before NMS. Thus it cannot become a fusionTP.','byThreshold':result,'quadrantOwnershipLimit':'The caches already applied per-tile quadrant ownership. Raw rejected tile boxes are unavailable, so this audit cannot causally attribute a missing tile result to that earlier ownership filter. Seam distance is descriptive only.'}
 (ROOT/'ml/research-20261025/fusion-boundary-audit.json').write_text(json.dumps(report,indent=2)+'\n');print({k:{n:v[n] for n in ['confirmedBaselineTPWithoutAnyRetainedIoUCoverage','reasonCounts','speciesCounts']} for k,v in result.items()})
if __name__=='__main__':main()
