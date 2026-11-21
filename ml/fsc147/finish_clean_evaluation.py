#!/usr/bin/env python3
"""Finish the full fine-tuning epoch, then evaluate pre-registered overlap-free holdouts."""
from experiment import *
import signal,subprocess,shutil

def summary(rows):
    errors=np.array([r['predicted']-r['truth'] for r in rows],dtype=float)
    return {'images':len(rows),'MAE':float(np.abs(errors).mean()),'RMSE':float(np.sqrt((errors**2).mean())),'bias':float(errors.mean()),'roundedExactCountImages':sum(round(r['predicted'])==r['truth'] for r in rows),'within10PercentImages':sum(abs(r['predicted']-r['truth'])<=.1*r['truth'] for r in rows)}

def register():
    raw=json.loads((BASE/'image-hash-audit.json').read_text());extra={'val':[],'test':[]}
    for group in raw['crossSplitExactDuplicateGroups']:
        for row in group:
            if row['split'] in extra:extra[row['split']].append(row['name'])
    clean={k:[n for n in SPLITS[k] if n not in extra[k]] for k in ['val','test']}
    p=RUN/'evaluation-split-before-test.json'
    r={'registeredBeforeTestUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'trainUnchanged':SPLITS['train'],'validation':clean['val'],'test':clean['test'],'densityExceptions':EXCLUDED,'additionalExactCrossSplitDuplicateExclusions':extra,'hashAuditSHA256':sha(BASE/'image-hash-audit.json'),'testImages':len(clean['test']),'validationImages':len(clean['val']),'selectionRule':'Lower cleaned-validation MAE of original public baseline and completed one-epoch fine-tuned model, frozen before first test inference.','note':'Within-split repeated pixels may represent different exemplar/target queries and are retained. This is an official-class benchmark with exact cross-split image duplicates removed, not an independent real-world acquisition set.'}
    if p.exists():
        old=json.loads(p.read_text());assert old['validation']==r['validation'] and old['test']==r['test'];return old
    assert not (RUN/'test-report.json').exists(),'Register the clean cohort before any test result'
    p.write_text(json.dumps(r,indent=2)+'\n');print('Frozen clean split',len(r['validation']),len(r['test']),extra,flush=True);return r

def wait_for_epoch(pid):
    path=RUN/'FamNet-finetuned-one-epoch.pth';start=time.time()
    while time.time()-start<1800:
        if path.exists():
            time.sleep(1)
            try:torch.load(path,map_location='cpu',weights_only=True)
            except Exception:time.sleep(1);continue
            assert not (RUN/'test-report.json').exists(),'Original job reached test before clean-cohort handoff'
            command=subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip()
            assert 'ml/fsc147/experiment.py train' in command,command
            os.kill(pid,signal.SIGINT)
            for _ in range(30):
                if subprocess.run(['ps','-p',str(pid)],capture_output=True).returncode!=0:break
                time.sleep(1)
            else:raise RuntimeError('Training process did not stop after saved-epoch handoff')
            # A full epoch has completed: this file is written only after the loop over all training examples.
            assert sha(path)!=sha(AUTHOR/'FamNet_Save1.pth')
            (RUN/'training-epoch-complete.json').write_text(json.dumps({'status':'COMPLETE_ONE_FULL_FINE_TUNING_EPOCH','trainingImages':len(SPLITS['train']),'optimizerSteps':len(SPLITS['train']),'weights':str(path),'sha256':sha(path),'originalWeightsSHA256':sha(AUTHOR/'FamNet_Save1.pth'),'handoff':'Stopped optional original evaluation after the complete weights file was saved, to apply pre-registered exact-duplicate exclusions before test.'},indent=2)+'\n')
            print('Completed epoch saved; original evaluation stopped before test',flush=True);return
        if subprocess.run(['ps','-p',str(pid)],capture_output=True).returncode!=0:raise RuntimeError('Original training ended before complete checkpoint')
        time.sleep(.5)
    raise TimeoutError('Epoch handoff timed out; no test or new training run started')

def evaluate_clean(split):
    torch.set_num_threads(8);device='mps';backbone=FrozenResnet50FPN().eval().to(device);r=regressor(device)
    original=json.loads((RUN/'baseline-validation.json').read_text());rows=[row for row in original['images'] if row['image'] in split['validation']];assert len(rows)==len(split['validation']);base=summary(rows);(RUN/'baseline-validation-clean.json').write_text(json.dumps({'summary':base,'images':rows},indent=2)+'\n')
    path=RUN/'FamNet-finetuned-one-epoch.pth';r.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));fine,rows=validate(r,backbone,split['validation'],device,'finetuned-clean-val');fine=summary(rows);(RUN/'finetuned-validation-clean.json').write_text(json.dumps({'summary':fine,'images':rows},indent=2)+'\n')
    chosen='finetuned' if fine['MAE']<base['MAE'] else 'original-public-baseline';chosen_path=path if chosen=='finetuned' else AUTHOR/'FamNet_Save1.pth'
    select={'frozenBeforeTestUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selected':chosen,'weights':str(chosen_path),'weightsSHA256':sha(chosen_path),'baselineValidation':base,'fineTunedValidation':fine,'trainingImages':len(SPLITS['train']),'fineTunedEpochs':1,'evaluationSplitSHA256':sha(RUN/'evaluation-split-before-test.json')};(RUN/'selection-clean-before-test.json').write_text(json.dumps(select,indent=2)+'\n');print('Selected before test',select,flush=True)
    r.load_state_dict(torch.load(chosen_path,map_location='cpu',weights_only=True));test,rows=validate(r,backbone,split['test'],device,'clean-test');test=summary(rows);test.update({'state':'COMPLETED_UNSEEN_CATEGORY_TEST_WITH_EXACT_OVERLAP_REMOVED','method':'FamNet few-shot counting with dataset-provided3-6exemplar boxes; no automatic object selection and no test-time adaptation.','selected':chosen,'weightsSHA256':sha(chosen_path),'testUsedForRetraining':False,'testClasses':29})
    (RUN/'test-report-clean.json').write_text(json.dumps({'summary':test,'images':rows},indent=2)+'\n')
    (RUN/'complete.json').write_text(json.dumps({'state':'COMPLETE_TRAIN_AND_CLEAN_TEST','trainingImages':len(SPLITS['train']),'fineTunedEpochs':1,'test':test,'selection':select},indent=2)+'\n');print('DONE',test,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--evaluate-only',action='store_true');a=p.parse_args();s=register()
    if not a.evaluate_only:wait_for_epoch(a.pid)
    evaluate_clean(s)
