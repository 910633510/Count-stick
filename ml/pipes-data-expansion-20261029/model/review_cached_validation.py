"""Independent read-only review of the three completed data-expansion validation caches.

No model construction, torch.load, forward call, image decoding, test manifest
reader, threshold search, or training operation is present. Reuses the prior
independent count/AP equations, not the candidate evaluator's scoring functions.
"""
from pathlib import Path
import argparse,collections,datetime,hashlib,importlib.util,json,re

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/pipes-data-expansion-20261029/model'
PREVIOUS=ROOT/'output/pipes-head-retention-20261026/model'
OLD=ROOT/'output/pipes-20261026/model'
HELPER=ROOT/'ml/pipes-20261026/model/review_frozen_test.py'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(p):return Path(p) if Path(p).is_absolute() else ROOT/p


def old_gates(candidate,baseline):
    checks={'TP':candidate['TP']>=baseline['TP']}
    for key in ['FP','FN','positiveCountWAPE','sceneMacroWAPE']:
        checks[key]=candidate[key]<=baseline[key]+(1e-12 if isinstance(baseline[key],float) else 0)
    assert candidate['perScene'].keys()==baseline['perScene'].keys()
    for name in baseline['sizeRecall']:
        assert candidate['sizeRecall'][name]['GT']==baseline['sizeRecall'][name]['GT']
        checks['sizeTP/'+name]=candidate['sizeRecall'][name]['TP']>=baseline['sizeRecall'][name]['TP']
    checks['allNegativeImagesZero']=candidate['negativeFalsePositiveImages']==0
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}


def require_completed(selection_sha256):
    # No numerical imports or scoring before a parent-supplied immutable selection
    # and all282 raw/sidecar files are available. Never score a partial run.
    if not re.fullmatch('[0-9a-f]{64}',selection_sha256):
        raise ValueError('A parent-provided completed selection SHA256 is required')
    selection_path=OUT/'selection.json'
    if not selection_path.is_file() or sha(selection_path)!=selection_sha256:
        raise ValueError('Completed selection is absent or differs from the approved snapshot')
    selection=read(selection_path)
    if selection.get('status') not in ['DATA_EXPANSION_VAL_QUALIFIED_PENDING_FRESH_HOLDOUT','FAIL_DATA_EXPANSION_VALIDATION']:
        raise ValueError('Selection is not complete')
    assert len(selection['candidates'])==3 and {r['epoch'] for r in selection['candidates']}=={4,8,12}
    completion=read(OUT/'training-complete.json')
    assert completion['status']=='COMPLETE_DATA_EXPANSION_12_EPOCHS_WITH_REGISTERED_CHECKPOINTS'
    assert set(completion['checkpoints'])=={'4','8','12'}
    inputs=read(OUT/'evaluation-inputs.json')
    assert inputs['testContentIncluded'] is False and len(inputs['images'])==94
    assert all(r['split']=='val' for r in inputs['images'])
    for epoch in [4,8,12]:
        folder=OUT/'predictions'/f'epoch{epoch:03d}'
        assert len(list(folder.glob('*/*.npz')))==94
        assert (OUT/'scores'/f'epoch{epoch:03d}.json').is_file()
        for row in inputs['images']:
            cache=folder/row['variant']/(row['id']+'.npz')
            assert cache.is_file() and cache.with_suffix('.json').is_file()
    return selection_path,selection


def main(selection_sha256):
    selection_path,selection=require_completed(selection_sha256)
    import numpy as np
    spec=importlib.util.spec_from_file_location('prior_independent_equations_only',HELPER)
    ind=importlib.util.module_from_spec(spec);spec.loader.exec_module(ind)
    ind.torch.set_num_threads(4)
    protocol=read(OUT/'protocol.json');authorization=read(OUT/'training-authorization.json')
    binding=read(OUT/'data-binding.json');inputs=read(OUT/'evaluation-inputs.json')
    completion=read(OUT/'training-complete.json')
    assert protocol['checkpointEpochs']==[4,8,12] and protocol['candidateCount']==3
    assert protocol['confidence']==.3 and protocol['nmsIoU']==.7 and protocol['rawLocations']==21504
    assert selection['protocolSHA256']==authorization['protocolSHA256']==sha(OUT/'protocol.json')
    assert authorization['dataBindingSHA256']==sha(OUT/'data-binding.json')
    assert sha(resolve(binding['manifestPath']))==binding['manifestSHA256']
    assert sha(resolve(binding['freezeReceiptPath']))==binding['freezeReceiptSHA256']
    assert selection['inputsManifestSHA256']==sha(OUT/'evaluation-inputs.json')
    assert authorization['rootReceiptSHA256']==sha(resolve(authorization['rootReceiptPath']))
    assert selection['codeSHA256']==authorization['codeSHA256']
    for path,digest in authorization['codeSHA256'].items():assert sha(ROOT/path)==digest
    assert completion['status']=='COMPLETE_DATA_EXPANSION_12_EPOCHS_WITH_REGISTERED_CHECKPOINTS'
    assert completion['rootReceiptSHA256']==authorization['rootReceiptSHA256'] and set(completion['checkpoints'])=={'4','8','12'}
    assert inputs['status']=='VALIDATED_DATA_EXPANSION_APP_INPUTS' and inputs['testContentIncluded'] is False
    assert inputs['recordsSHA256']==binding['recordsSHA256']==sha(OUT/'records.json')
    record_list=read(OUT/'records.json');assert all(r['split'] in ['train','val'] for r in record_list)
    records={r['id']:r for r in record_list if r['split']=='val'}
    assert len(records)==47 and collections.Counter(r['dataRole'] for r in records.values())=={'old':40,'fresh':7}
    assert len(record_list)==291
    previous_inputs=read(PREVIOUS/'evaluation-inputs.json')
    assert sha(PREVIOUS/'evaluation-inputs.json')==inputs['previousInputsSHA256']==protocol['previousInputsSHA256']
    assert inputs['images']==previous_inputs['images'] and inputs['validationDataUnchanged'] is True
    assert sha(PREVIOUS/'records.json')==protocol['previousRecordsSHA256']
    previous_val={r['id']:r for r in read(PREVIOUS/'records.json') if r['split']=='val'}
    assert set(previous_val)==set(records)
    for identity,row in records.items():
        for key in ['originalWidth','originalHeight','originalXYXY','count','sha256','groupId','dataRole']:
            assert row[key]==previous_val[identity][key],(identity,key,'Validation changed')
    assert len(inputs['images'])==94 and inputs['variants']==['gallery','camera']
    assert {(x['id'],x['variant']) for x in inputs['images']}=={(i,v) for i in records for v in ['gallery','camera']}
    for x in inputs['images']:
        r=records[x['id']];assert x['split']=='val' and x['dataRole']==r['dataRole'] and x['sourceImageSHA256']==r['sha256']
        assert sha(resolve(x['networkImagePath']))==x['networkImageSHA256'] and sha(resolve(x['photoImagePath']))==x['photoImageSHA256']
    baseline_selection=read(OLD/'selection.json')
    assert sha(OLD/'selection.json')==selection['oldBaselineSelectionSHA256']==protocol['oldSelectionSHA256']
    baseline=baseline_selection['selected'];assert baseline['confidence']==.3 and baseline['nmsIoU']==.7
    previous_audit=ROOT/'ml/pipes-20261026/model/independent-validation-review.json'
    assert read(previous_audit)['validationAudit']['selectionSHA256']==sha(OLD/'selection.json')
    assert read(previous_audit)['validationAudit']['status'].startswith('PASS')
    # Feed original-image GT to independent equations, retaining split=val.
    metric_records={i:{**r,'width':r['originalWidth'],'height':r['originalHeight'],'xyxy':r['originalXYXY']} for i,r in records.items()}
    candidates=[];cache_hashes={};models={};input_order=inputs['images']
    for epoch in [4,8,12]:
        weights=OUT/'checkpoints'/f'epoch{epoch:03d}.pt';meta=read(weights.with_suffix('.json'))
        # This audit is explicitly cache-only: bind checkpoint registration to
        # completion/selection, without opening or deserializing weight bytes.
        assert meta==completion['checkpoints'][str(epoch)]
        assert meta['sha256']==next(r['modelSHA256'] for r in selection['candidates'] if r['epoch']==epoch)
        assert meta['protocolSHA256']==selection['protocolSHA256'] and meta['dataBindingSHA256']==sha(OUT/'data-binding.json')
        models[str(epoch)]=meta['sha256']
        folder=OUT/'predictions'/f'epoch{epoch:03d}';cb=read(folder/'binding.json')
        assert cb['modelSHA256']==meta['sha256'] and cb['inputsManifestSHA256']==selection['inputsManifestSHA256']
        assert cb['recordsSHA256']==binding['recordsSHA256'] and cb['codeSHA256']==selection['codeSHA256']
        assert cb['preNMS'] is True and cb['confidenceFloor']==.001
        cache={}
        for x in input_order:
            path=folder/x['variant']/(x['id']+'.npz');side=read(path.with_suffix('.json'));digest=sha(path)
            assert digest==side['npzSHA256'] and side['id']==x['id'] and side['variant']==x['variant']
            assert side['modelSHA256']==meta['sha256'] and side['networkImageSHA256']==x['networkImageSHA256']
            assert side['rawLocations']==21504 and side['NMSApplied'] is False and side['outputCapApplied'] is False
            with np.load(path,allow_pickle=False) as data:
                boxes=data['xyxy'].copy();confidence=data['confidence'].copy()
            assert boxes.dtype==confidence.dtype==np.float32 and boxes.shape==(len(confidence),4)
            assert len(confidence)==side['retainedAfterFloor']<=21504
            assert np.isfinite(boxes).all() and np.isfinite(confidence).all() and np.all((confidence>=.001)&(confidence<=1))
            cache[(x['variant'],x['id'])]=(boxes,confidence);cache_hashes[str(path.relative_to(ROOT))]=digest
        result={'epoch':epoch,'confidence':.3,'nmsIoU':.7,'modelSHA256':meta['sha256'],'variants':{}}
        for variant in ['gallery','camera']:
            result['variants'][variant]={}
            for role in ['old','fresh']:
                part=[x for x in input_order if x['variant']==variant and x['dataRole']==role]
                rows=[];ap_matches=[];ap_scores=[];gt_total=0
                for x in part:
                    boxes,scores=cache[(variant,x['id'])]
                    operating=ind.pp(boxes,scores,x['geometry'],.3,.7)
                    row,truth=ind.metric_row(metric_records[x['id']],x,operating);rows.append(row)
                    ap_prediction=ind.pp(boxes,scores,x['geometry'],.001,.7)
                    ap_matches.append(ind.ap_match(truth,ap_prediction['xyxy']));ap_scores.extend(ap_prediction['confidence']);gt_total+=len(truth)
                summary=ind.summary(rows);summary.update(ind.ap_from(ap_matches,ap_scores,gt_total))
                score={'summary':summary,'images':rows,'qualification':ind.gates(summary)}
                score['gates']=old_gates(summary,baseline['variants'][variant]['summary']) if role=='old' else ind.gates(summary)
                result['variants'][variant][role]=score
        result['status']='PASS' if all(result['variants'][v][role]['gates']['status']=='PASS' for v in ['gallery','camera'] for role in ['old','fresh']) else 'FAIL'
        ind.eq(result,read(OUT/'scores'/f'epoch{epoch:03d}.json'))
        candidates.append(result);print('RECOMPUTED',epoch,result['status'],flush=True)
    assert len(cache_hashes)==282
    ind.eq(candidates,selection['candidates'])
    rank=lambda r:(max(r['variants'][v]['fresh']['summary']['positiveCountMAE'] for v in ['gallery','camera']),max(r['variants'][v]['old']['summary']['positiveCountMAE'] for v in ['gallery','camera']),r['epoch'])
    passing=[r for r in candidates if r['status']=='PASS'];chosen=min(passing,key=rank) if passing else None
    ind.eq(chosen,selection['selected']);ind.eq(min(candidates,key=rank),selection['bestResearchCandidateNotAutomaticallyUsable'])
    assert selection['status']==('DATA_EXPANSION_VAL_QUALIFIED_PENDING_FRESH_HOLDOUT' if passing else 'FAIL_DATA_EXPANSION_VALIDATION')
    assert selection['old37TestAccessed'] is False and selection['freshTestAccessed'] is False and selection['VALIDATED_COREML_EXPORTIssued'] is False
    compact=[]
    for r in candidates:
        compact.append({'epoch':r['epoch'],'status':r['status'],'variants':{v:{role:{'summary':r['variants'][v][role]['summary'],'gates':r['variants'][v][role]['gates']} for role in ['old','fresh']} for v in ['gallery','camera']}})
    final=candidates[-1]
    details={v:{x['id']:{k:x[k] for k in ['GT','predicted','TP','FP','FN','absoluteError']} for x in final['variants'][v]['fresh']['images']} for v in ['gallery','camera']}
    report={'status':'PASS_INDEPENDENT_AUDIT_OF_ALL3_DATA_EXPANSION_VALIDATION_FAILURES' if not passing else 'PASS_INDEPENDENT_DATA_EXPANSION_VALIDATION_RECOMPUTATION',
            'createdUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selectionPath':str(selection_path.relative_to(ROOT)),
            'selectionSHA256':sha(selection_path),'auditScriptSHA256':sha(__file__),'independentEquationsSHA256':sha(HELPER),
            'protocolSHA256':selection['protocolSHA256'],'codeSHA256':selection['codeSHA256'],'recordsSHA256':binding['recordsSHA256'],
            'inputsManifestSHA256':selection['inputsManifestSHA256'],'oldBaselineSelectionSHA256':sha(OLD/'selection.json'),
            'priorBaselineIndependentAuditSHA256':sha(previous_audit),'modelSHA256ByEpoch':models,
            'validationPhotos':{'old':40,'fresh':7},'inputVariants':['gallery','camera'],'rawCacheFilesRecomputed':282,'rawCacheSHA256':cache_hashes,
            'operatingPoint':{'confidence':.3,'nmsIoU':.7},'candidateCount':3,'qualifiedCount':len(passing),'selected':chosen,
            'previousInputsSHA256':sha(PREVIOUS/'evaluation-inputs.json'),'previousRecordsSHA256':sha(PREVIOUS/'records.json'),
            'all94InputRowsAnd47ValidationGTRecordsUnchangedFromPreviousRound':True,'totalTrainValRecordCount':291,
            'freshDetailsByEpoch':{str(r['epoch']):{v:{x['id']:{k:x[k] for k in ['GT','predicted','TP','FP','FN','absoluteError']} for x in r['variants'][v]['fresh']['images']} for v in ['gallery','camera']} for r in candidates},
            'researchBestEpochNotQualified':min(candidates,key=rank)['epoch'],'allCacheProvenanceDtypeShapeFiniteChecksPassed':True,
            'allPerImageCountsMatchingSizesSummariesAPGatesAndRankingMatch':True,'oldBaselineSummary':{v:baseline['variants'][v]['summary'] for v in ['gallery','camera']},
            'candidates':compact,'epoch12FreshDetails':details,'modelInferencePerformed':False,'modelWeightsDeserialized':False,'modelWeightBytesOpened':False,
            'validationImageBytesHashed':True,'imagePixelsDecoded':False,'freshTestRead':False,'old37TestRead':False,
            'metricsOrLabelsModified':False,'selectionChangedByReviewer':False,
            'limitations':['NMS uses the same frozen CPU torchvision kernel, but unpadding/count matching/size metrics/scene metrics/AP and gates are reproduced by a separate prior-reviewed implementation.',
                           'Independent recomputation verifies arithmetic/provenance, not a new annotation exercise or a field-accuracy claim.','Checkpoint identity is bound through registration/completion/selection/cache metadata; weight files were intentionally not opened.','The old baseline metrics were already independently audited; this run recomputes all282 data-expansion caches and compares against that hash-bound baseline.']}
    assert sha(selection_path)==selection_sha256,'Selection changed during independent review'
    assert sha(resolve(binding['manifestPath']))==binding['manifestSHA256']
    assert sha(resolve(binding['freezeReceiptPath']))==binding['freezeReceiptSHA256']
    report['freezeReceiptSHA256']=binding['freezeReceiptSHA256']
    report['trainValManifestSHA256']=binding['manifestSHA256']
    report['parentSuppliedSelectionSHA256']=selection_sha256
    report['all282CachesPresentBeforeAnyMetricComputation']=True
    destination=Path(__file__).with_name('independent-cache-review.json');assert not destination.exists()
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(destination,sha(destination),report['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-sha256',required=True,help='Completed selection hash supplied in parent readiness notification')
    main(parser.parse_args().selection_sha256)
