#!/usr/bin/env python3
"""Train a single-class colony detector; never write into the shipping app."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'output/training-20261025/ultralytics-config')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
import argparse, hashlib, json, time
import torch
from ultralytics import YOLO,settings
DATA=ROOT/'output/datasets/colony-figshare-22022540-v3/yolo-species-v1'
RUNS=ROOT/'output/training-20261025/runs'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--batch',type=int,default=4);ap.add_argument('--imgsz',type=int,default=1024);ap.add_argument('--name',default='colony-yolo11n-v1');ap.add_argument('--resume',type=Path);ap.add_argument('--initial-weights',type=Path);ap.add_argument('--lr',type=float,default=0.001);ap.add_argument('--degrees',type=float,default=90.0);ap.add_argument('--patience',type=int,default=10);ap.add_argument('--warmup-bias-lr',type=float);args=ap.parse_args()
    from mps_compat import install_cpu_assigner_for_mps
    install_cpu_assigner_for_mps()
    torch.set_num_threads(8)
    settings.update({'sync':False,'wandb':False,'mlflow':False,'comet':False,'clearml':False,'neptune':False,'tensorboard':False})
    manifest=DATA/'manifest.json';audit=json.loads((DATA/'audit-report.json').read_text());assert hashlib.sha256(manifest.read_bytes()).hexdigest()==audit['manifestSHA256'];assert torch.backends.mps.is_available()
    config=dict(data=str(DATA/'data.yaml'),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,device='mps',workers=2,seed=20261025,deterministic=True,amp=False,optimizer='AdamW',nbs=16,lr0=args.lr,lrf=0.01,weight_decay=0.0005,warmup_epochs=2,patience=args.patience,mosaic=0.0,mixup=0.0,copy_paste=0.0,degrees=args.degrees,translate=0.08,scale=0.25,fliplr=0.5,flipud=0.5,hsv_h=0.01,hsv_s=0.2,hsv_v=0.2,cache=False,max_det=1000,plots=True,save=True,save_period=5,project=str(RUNS),name=args.name,exist_ok=False,verbose=False)
    if args.warmup_bias_lr is not None:config['warmup_bias_lr']=args.warmup_bias_lr
    experiment=ROOT/'output/training-20261025'/('protocol-'+args.name+'.json')
    protocol={'createdAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'datasetManifestSHA256':audit['manifestSHA256'],'pretrainedWeights':str(args.initial_weights.resolve()) if args.initial_weights else str(ROOT/'output/training-20261025/yolo11n.pt'),'trainConfig':config,'assignmentBackend':'Original Ultralytics no-grad TaskAlignedAssigner on CPU; model forward/backward on MPS to avoid dense advanced-indexing failure.', 'selection':'Highest validation mAP50-95 checkpoint; no test inference until model and confidence threshold are fixed.','confidenceSelection':'Validation count-MAE minimum among [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50], ties favor higher threshold.','testProtocol':'Single whole-image evaluation at trained resolution; IoU0.50 detection matching, AP50/AP50-95, count MAE/RMSE/bias/exact-count rate. No test-driven retraining in this run.','trainingPurpose':'Initial research candidate for colony counting, not species diagnosis or a production-ready replacement.'}
    protocol['pretrainedSHA256']=hashlib.sha256(Path(protocol['pretrainedWeights']).read_bytes()).hexdigest()
    if experiment.exists():assert json.loads(experiment.read_text())['trainConfig']==config,'Do not alter a frozen training protocol'
    else:experiment.write_text(json.dumps(protocol,indent=2)+'\n')
    model=YOLO(str(args.resume) if args.resume else protocol['pretrainedWeights'])
    def serial(value):
        if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
        if hasattr(value,'detach'):return value.detach().cpu().tolist()
        return value
    def progress(trainer):
        info={'epoch':trainer.epoch+1,'epochs':trainer.epochs,'loss':serial(trainer.tloss),'metrics':{k:float(v) for k,v in trainer.metrics.items()},'bestFitness':float(trainer.best_fitness) if trainer.best_fitness is not None else None,'updatedAtUTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        (Path(trainer.save_dir)/'progress.json').write_text(json.dumps(info,indent=2)+'\n')
    model.add_callback('on_fit_epoch_end',progress)
    start=time.time();result=model.train(resume=True) if args.resume else model.train(**config)
    elapsed=time.time()-start
    end={'status':'TRAINING_COMPLETED','secondsThisProcess':elapsed,'resumedFrom':str(args.resume) if args.resume else None,'saveDirectory':str(model.trainer.save_dir),'bestCheckpoint':str(model.trainer.best),'metrics':{k:float(v) for k,v in result.results_dict.items()},'datasetManifestSHA256':audit['manifestSHA256']}
    (Path(model.trainer.save_dir)/'training-complete.json').write_text(json.dumps(end,indent=2)+'\n');print(json.dumps(end,indent=2),flush=True)
if __name__=='__main__':main()
