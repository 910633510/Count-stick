"""One extra registered cached-validation experiment; no new inference/test access."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'ml/colony'))
import scale_fusion as sf
import evaluate as e
from evaluate_pipeline import ap_from_predictions
import torch

def main():
    torch.set_num_threads(8);out=ROOT/'output/research-20261025/fusion';old=json.loads((out/'protocol.json').read_text());limits=old['thresholds'];m=json.loads((e.DATA/'manifest.json').read_text());names=m['splits']['val'];images={r['file']:r for r in m['images']}
    cache=ROOT/'output/training-20261025/runs/colony-yolo11n-v2-cpuassign/evaluation';whole=json.loads((cache/'validation-predictions.json').read_text());tile=json.loads((cache/'validation-tiled-predictions.json').read_text())
    protocol={'frozenBeforeExtraCandidateScoringUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'EXPOSED_VALIDATION_ONLY','previousProtocolSHA256':sf.digest(out/'protocol.json'),'previousGridStatus':'NO_CANDIDATE_PASSED_FROZEN_GATES','reason':'GlobalNMS0.5 also suppresses boxes within one source, although that source already usedNMS0.7. This candidate preserves same-source predictions and only deduplicates cross-source boxes.','config':{'minSideThreshold':16,'nmsIoU':.5,'confidence':.3,'fusion_mode':'cross_source_ios','containment_ratio':.9},'rule':'Confidence-ordered greedy suppression only across sources: IoU>=0.5; additionally a higher/equal-confidence full-image box may suppress a smaller-area tile box when it contains the tile center and intersection/min-area>=0.9. No same-source re-NMS.','gatesUnchanged':True,'thresholds':limits,'sourceSHA256':sf.digest(ROOT/'ml/colony/scale_fusion.py'),'newModelInference':False,'oldTestOrExternalHoldoutUsed':False}
    pp=out/'ios-protocol.json';assert not pp.exists();pp.write_text(json.dumps(protocol,indent=2)+'\n');(ROOT/'ml/research-20261025/fusion-ios-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Verify that nested weaker tile fragments disappear and overlapping same-source boxes survive.
    w={'xyxy':[[0,0,40,40],[1,1,41,41]],'confidence':[.9,.8]};t={'xyxy':[[5,5,15,15]],'confidence':[.7]}
    sanity=sf.fuse_predictions(w,t,(1024,1024),16,.5,fusion_mode='cross_source_ios');assert len(sanity['xyxy'])==2
    fused={n:sf.fuse_predictions(whole[n],tile[n],(images[n]['width'],images[n]['height']),16,.5,fusion_mode='cross_source_ios',containment_ratio=.9) for n in names}
    s,rows=e.metrics(fused,names,'val',.3,113.);s=sf.enrich(s,rows);s.update(ap_from_predictions(fused,names,'val'))
    checks={'countMAE':s['countMAE']<=limits['countMAEMax'],'macroSpeciesMAE':s['macroSpeciesMAE']<=limits['macroSpeciesMAEMax'],'TP':s['TP']>=limits['TPMin'],'FP':s['FP']<=limits['FPMax'],'FN':s['FN']<=limits['FNMax'],'macroSpeciesRecall':s['macroSpeciesRecall']>=limits['macroSpeciesRecallMin'],'AP50':s['AP50']>=limits['AP50Min'],'AP50_95':s['AP50_95']>=limits['AP50_95Min']}
    for sp,t in limits['species'].items():
        a=s['perSpecies'][sp];checks['recall.'+sp]=a['recall']>=t['recallMin'];checks['precision.'+sp]=a['precision']>=t['precisionMin'];checks['countMAE.'+sp]=a['countMAE']<=t['countMAEMax']
    for k,v in limits['sizeRecallMin'].items():checks['recall.'+k]=s['sizeRecall'][k]['recall']>=v
    record={'id':'side16-cross-source-ios0.9-iou0.5','status':'PASS_DEVELOPMENT_SCREEN' if all(checks.values()) else 'FAIL_DEVELOPMENT_SCREEN','protocolSHA256':sf.digest(pp),'metrics':s,'checks':checks,'failedChecks':[k for k,v in checks.items() if not v],'images':rows,'noGatesRelaxed':True,'newInferencePerformed':False,'testUsed':False}
    (out/'ios-predictions.json').write_text(json.dumps(fused)+'\n');(out/'ios-report.json').write_text(json.dumps(record,indent=2)+'\n');(ROOT/'ml/research-20261025/fusion-ios-results.json').write_text(json.dumps({k:v for k,v in record.items() if k!='images'},indent=2)+'\n')
    final={'status':'DEVELOPMENT_CANDIDATE_FOUND_PENDING_NEW_HOLDOUT' if all(checks.values()) else 'NO_CANDIDATE_PASSED_SEVEN_FROZEN_SCREENS','selectedConfig':protocol['config'] if all(checks.values()) else None,'deploymentApproved':False,'noGatesRelaxed':True,'candidateCount':7,'bestAggregateCandidateIsNotApproved':'side16-nms0.5','bestAggregateMAE':11.535714285714286,'explanation':'Averages improved, but subgroup protection failures prevent promotion.'};(ROOT/'ml/research-20261025/fusion-final-selection.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({'status':record['status'],'MAE':s['countMAE'],'TP':s['TP'],'FP':s['FP'],'failedChecks':record['failedChecks']},indent=2),flush=True)
if __name__=='__main__':main()
