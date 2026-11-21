#!/usr/bin/env python3
"""Scale-gated whole+tile prediction fusion; cached validation experiment only.

This is an engineering variant motivated by preserving whole-image context during
sliced inference. It is not a literal SAHI paper reproduction. Runtime selection
uses predicted geometry/confidence only, never a filename, species or ground truth.
"""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
import argparse,collections,hashlib,json,time
import numpy as np
import torch
from torchvision.ops import nms


def fuse_predictions(whole, tiled, image_size, min_side_threshold=16.0, nms_iou=.7,
                     confidence_floor=.001, max_detections=1000, fusion_mode='global_nms', containment_ratio=.9):
    """Return xyxy/confidence in original-image pixels from two prediction dicts.

    Whole predictions supply boxes at/above the full-image1024-normalized minimum
    side cutoff; tile predictions supply smaller boxes. Hard size disagreement
    between scales can leave gaps, so this function must pass subgroup evaluation.
    """
    width,height=image_size
    if width<=0 or height<=0:raise ValueError('Positive image dimensions required')
    if min_side_threshold<=0 or not 0<nms_iou<=1:raise ValueError('Invalid fusion parameters')
    boxes=[];scores=[];sources=[]
    for entry,source in [(whole,'whole'),(tiled,'tile')]:
        b=np.asarray(entry['xyxy'],dtype=np.float32).reshape(-1,4)
        s=np.asarray(entry['confidence'],dtype=np.float32)
        if len(b)!=len(s) or not np.isfinite(b).all() or not np.isfinite(s).all():raise ValueError('Invalid prediction arrays')
        side=np.minimum(b[:,2]-b[:,0],b[:,3]-b[:,1])*1024.0/max(width,height)
        keep=(s>=confidence_floor)&(side>0)
        keep &= (side>=min_side_threshold) if source=='whole' else (side<min_side_threshold)
        boxes.append(b[keep]);scores.append(s[keep]);sources.append(np.full(int(keep.sum()),0 if source=='whole' else 1,dtype=np.int8))
    b=np.concatenate(boxes);s=np.concatenate(scores)
    if not len(b):return {'xyxy':[],'confidence':[]}
    if fusion_mode=='global_nms':
        keep=nms(torch.from_numpy(b),torch.from_numpy(s),nms_iou)[:max_detections].numpy()
    elif fusion_mode=='cross_source_ios':
        source_ids=np.concatenate(sources);order=np.argsort(-s,kind='stable');chosen=[]
        area=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
        while len(order) and len(chosen)<max_detections:
            i=int(order[0]);chosen.append(i);rest=order[1:]
            if not len(rest):break
            cross=source_ids[rest]!=source_ids[i]
            tl=np.maximum(b[i,:2],b[rest,:2]);br=np.minimum(b[i,2:],b[rest,2:]);intersection=np.maximum(br-tl,0).prod(1)
            overlap=intersection/(area[i]+area[rest]-intersection+1e-9)
            suppress=cross&(overlap>=nms_iou)
            if source_ids[i]==0:
                center=(b[rest,:2]+b[rest,2:])/2
                inside=(center[:,0]>=b[i,0])&(center[:,0]<=b[i,2])&(center[:,1]>=b[i,1])&(center[:,1]<=b[i,3])
                ios=intersection/(np.minimum(area[i],area[rest])+1e-9)
                # Confidence order guarantees the full-image keeper is at least as confident.
                suppress|=cross&inside&(area[i]>=area[rest])&(ios>=containment_ratio)
            order=rest[~suppress]
        keep=np.asarray(chosen,dtype=np.int64)
    else:raise ValueError('Unknown fusion mode')
    return {'xyxy':b[keep].tolist(),'confidence':s[keep].tolist()}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def enrich(summary,rows):
    species={}
    for r in rows:
        s=species.setdefault(r['species'],{'images':0,'TP':0,'FP':0,'FN':0,'groundTruthObjects':0,'absoluteCountError':0,'signedCountError':0,'exactCountImages':0})
        s['images']+=1;s['TP']+=r['tp'];s['FP']+=r['fp'];s['FN']+=r['fn'];s['groundTruthObjects']+=r['truth'];s['absoluteCountError']+=abs(r['signedError']);s['signedCountError']+=r['signedError'];s['exactCountImages']+=r['signedError']==0
    for s in species.values():
        s['precision']=s['TP']/max(1,s['TP']+s['FP']);s['recall']=s['TP']/max(1,s['groundTruthObjects']);s['countMAE']=s['absoluteCountError']/s['images'];s['countBias']=s['signedCountError']/s['images']
    return {**summary,'F1':2*summary['TP']/max(1,2*summary['TP']+summary['FP']+summary['FN']),
            'macroSpeciesRecall':float(np.mean([s['recall'] for s in species.values()])),
            'macroSpeciesPrecision':float(np.mean([s['precision'] for s in species.values()])),
            'macroSpeciesMAE':float(np.mean([s['countMAE'] for s in species.values()])),
            'perSpecies':species}


def sanity_check():
    # Physical size gate uses full-image scale; overlapping retained boxes are deduplicated.
    whole={'xyxy':[[0,0,40,40],[100,100,110,110]],'confidence':[.9,.8]}
    tile={'xyxy':[[101,101,111,111],[200,200,240,240]],'confidence':[.85,.95]}
    r=fuse_predictions(whole,tile,(1024,1024),16,.7)
    assert len(r['xyxy'])==2 and sorted(r['confidence'])==sorted([float(np.float32(.9)),float(np.float32(.85))])
    empty=fuse_predictions({'xyxy':[],'confidence':[]},{'xyxy':[],'confidence':[]},(1024,1024))
    assert empty=={'xyxy':[],'confidence':[]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'output/research-20261025/fusion');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise RuntimeError('Existing frozen experiment; do not overwrite it')
    import evaluate as e
    from evaluate_pipeline import ap_from_predictions
    torch.set_num_threads(8);sanity_check()
    cache=ROOT/'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/evaluation'
    files={'whole':cache/'validation-predictions.json','tile':cache/'validation-tiled-predictions.json'}
    preds={k:json.loads(v.read_text()) for k,v in files.items()}
    manifest=json.loads((e.DATA/'manifest.json').read_text());names=manifest['splits']['val'];images={x['file']:x for x in manifest['images']}
    assert len(names)==56 and all(set(p)==set(names) for p in preds.values())
    confidence=.3;references={}
    for key in ['whole','tile']:
        s,rows=e.metrics(preds[key],names,'val',confidence,113.0);s=enrich(s,rows);s.update(ap_from_predictions(preds[key],names,'val'));references[key]=s
    w,t=references['whole'],references['tile']
    thresholds={
      'countMAEMax':min(w['countMAE'],t['countMAE'])*.95,
      'macroSpeciesMAEMax':min(w['macroSpeciesMAE'],t['macroSpeciesMAE'])*.95,
      'TPMin':max(w['TP'],t['TP']), 'FPMax':min(w['FP'],t['FP']), 'FNMax':min(w['FN'],t['FN']),
      'macroSpeciesRecallMin':max(w['macroSpeciesRecall'],t['macroSpeciesRecall']),
      'AP50Min':max(w['AP50'],t['AP50'])-.01,'AP50_95Min':max(w['AP50_95'],t['AP50_95'])-.01,
      'species':{s:{'recallMin':max(w['perSpecies'][s]['recall'],t['perSpecies'][s]['recall'])-.02,
                    'precisionMin':max(w['perSpecies'][s]['precision'],t['perSpecies'][s]['precision'])-.02,
                    'countMAEMax':min(w['perSpecies'][s]['countMAE'],t['perSpecies'][s]['countMAE'])+max(1,.05*min(w['perSpecies'][s]['countMAE'],t['perSpecies'][s]['countMAE']))} for s in w['perSpecies']},
      'sizeRecallMin':{k:max(w['sizeRecall'][k]['recall'],t['sizeRecall'][k]['recall'])-.02 for k in w['sizeRecall']}}
    protocol={'frozenBeforeCandidateScoringUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'EXPOSED_VALIDATION_DEVELOPMENT_ONLY','modelSHA256':'91d70c3067f2fdb77d2ba2541e6105bf3caf730da7bbc32e5f4d2c9310b0b337','manifestSHA256':digest(e.DATA/'manifest.json'),'cacheSHA256':{k:digest(v) for k,v in files.items()},'sourceSHA256':digest(Path(__file__)),'operatingConfidence':confidence,'predictionFloorForAP':.001,'maxDetections':1000,'grid':{'minimumSideAt1024':[16,24,32],'NMSIoU':[.5,.7]},'gatePolicy':'Strict pre-registered engineering Pareto/non-inferiority screen: improve count and macroMAE5%; keep totalTP/FN at least the better reference and totalFP no worse than the lower-FP reference; macro recall non-decreasing; per-species P/R and size recall may lose no more than2percentage points versus their stronger reference; speciesMAE tolerance1object or5%; AP drop<=1point. No gate relaxation after results.','thresholds':thresholds,'references':references,'oldTestInferenceAllowed':False,'newExternalHoldoutInferenceAllowed':False,'deploymentApproval':False,'literatureRelation':'Motivated by standard full+slice prediction retention; hard size gating is an engineering variant, not strict SAHI replication.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    artifact=ROOT/'ml/research-20261025/fusion-protocol.json';artifact.write_text(json.dumps(protocol,indent=2)+'\n')
    candidates=[];start=time.time()
    for side in [16,24,32]:
      for iou in [.5,.7]:
        key=f'side{side}-nms{iou:.1f}';fused={n:fuse_predictions(preds['whole'][n],preds['tile'][n],(images[n]['width'],images[n]['height']),side,iou) for n in names}
        s,rows=e.metrics(fused,names,'val',confidence,113.0);s=enrich(s,rows);s.update(ap_from_predictions(fused,names,'val'))
        checks={'countMAE':s['countMAE']<=thresholds['countMAEMax'],'macroSpeciesMAE':s['macroSpeciesMAE']<=thresholds['macroSpeciesMAEMax'],'TP':s['TP']>=thresholds['TPMin'],'FP':s['FP']<=thresholds['FPMax'],'FN':s['FN']<=thresholds['FNMax'],'macroSpeciesRecall':s['macroSpeciesRecall']>=thresholds['macroSpeciesRecallMin'],'AP50':s['AP50']>=thresholds['AP50Min'],'AP50_95':s['AP50_95']>=thresholds['AP50_95Min']}
        for sp,limits in thresholds['species'].items():
            actual=s['perSpecies'][sp];checks['recall.'+sp]=actual['recall']>=limits['recallMin'];checks['precision.'+sp]=actual['precision']>=limits['precisionMin'];checks['countMAE.'+sp]=actual['countMAE']<=limits['countMAEMax']
        for size,minimum in thresholds['sizeRecallMin'].items():checks['recall.'+size]=s['sizeRecall'][size]['recall']>=minimum
        record={'id':key,'config':{'minSideThreshold':side,'nmsIoU':iou,'confidence':confidence},'status':'PASS_DEVELOPMENT_SCREEN' if all(checks.values()) else 'FAIL_DEVELOPMENT_SCREEN','metrics':s,'checks':checks,'failedChecks':[k for k,v in checks.items() if not v],'images':rows}
        (out/(key+'-predictions.json')).write_text(json.dumps(fused)+'\n');(out/(key+'-report.json')).write_text(json.dumps(record,indent=2)+'\n');candidates.append(record)
        print(key,record['status'],'MAE',round(s['countMAE'],3),'macroMAE',round(s['macroSpeciesMAE'],3),'TP',s['TP'],'FP',s['FP'],'failed',record['failedChecks'],flush=True)
    passed=[c for c in candidates if c['status']=='PASS_DEVELOPMENT_SCREEN'];chosen=min(passed,key=lambda c:(c['metrics']['macroSpeciesMAE'],c['metrics']['countMAE'])) if passed else None
    result={'status':'DEVELOPMENT_CANDIDATE_FOUND_PENDING_NEW_HOLDOUT' if chosen else 'NO_CANDIDATE_PASSED_FROZEN_GATES','completedAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocolSHA256':digest(out/'protocol.json'),'selectedConfig':chosen['config'] if chosen else None,'candidateCount':len(candidates),'noGatesRelaxed':True,'newModelInferencePerformed':False,'testImagesReadOrScored':False,'appModelDataChanged':False,'postprocessingAndScoringSeconds':time.time()-start,'references':references,'candidates':[{k:v for k,v in c.items() if k!='images'} for c in candidates],'limitations':['Known validation data, not new blind evaluation.','No iPhone/CoreML end-to-end latency or parity measurement.','Hard size partition can create gaps when the two models estimate opposite sides of the cutoff.','Not model-promotion or App Store approval.']}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'ml/research-20261025/fusion-results.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'ml/research-20261025/fusion-selected-config.json').write_text(json.dumps({'status':result['status'],'config':result['selectedConfig'],'protocolSHA256':result['protocolSHA256'],'deploymentApproved':False},indent=2)+'\n');print(result['status'],flush=True)
if __name__=='__main__':main()
