#!/usr/bin/env python3
"""One predeclared calibration-only confidence sweep, then one frozen winner on holdout.

No labels, production code, weights, scaling, or IoU threshold are changed by this script.
Run `calibrate` once and save its selection before running `holdout` once. V1 is untouched.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dataset_evaluation', HERE / 'evaluate.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
THRESHOLDS = (0.25, 0.35, 0.45, 0.55)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path, value):
    with path.open('x') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write('\n')


def checkpoint(repository):
    model = repository / 'Count stick/best.mlpackage'
    return {'sourceDetectorSha256': evaluation.digest(repository / 'Count stick/StickDetector.swift'),
            'modelPath': str(model),
            'modelFileHashes': {str(p.relative_to(model)): evaluation.digest(p) for p in sorted(model.rglob('*')) if p.is_file()}}


def predictions_for(items, variants, phase, manifest_sha, model_checkpoint, output):
    validated = {'validationPassed': True, 'labelsFrozen': True, 'splitLocked': True, 'manifestSha256': manifest_sha,
                 **model_checkpoint, 'phase': phase, 'variants': variants,
                 'images': [{k: image[k] for k in ('id', 'path', 'sha256', 'width', 'height', 'split')} for image in items]}
    input_path = output / (phase + '-input.json')
    prediction_path = output / (phase + '-predictions.json')
    write_new(input_path, validated)
    subprocess.run(['swift', str(HERE / 'evaluate-confidence-model.swift'), str(input_path), str(prediction_path)], check=True)
    predictions = json.loads(prediction_path.read_text())
    if predictions['manifestSha256'] != manifest_sha or predictions['phase'] != phase:
        raise ValueError('Prediction phase or dataset identity changed')
    if predictions['modelFileHashes'] != model_checkpoint['modelFileHashes'] or predictions['sourceDetectorSha256'] != model_checkpoint['sourceDetectorSha256']:
        raise ValueError('Prediction checkpoint identity changed')
    by_id = {p['imageId']: p for p in predictions['images']}
    if set(by_id) != {item['id'] for item in items} or len(by_id) != len(items):
        raise ValueError('Prediction IDs do not match declared split')
    scored = {variant['name']: [] for variant in variants}
    for item in items:
        p = by_id[item['id']]
        if p['imageSha256'] != item['sha256']:
            raise ValueError('Image changed: ' + item['id'])
        for variant in variants:
            predicted = p['variants'][variant['name']]
            if predicted['confidenceThreshold'] != variant['confidenceThreshold'] or predicted['iouThreshold'] != 0.45 or predicted['cropScale'] != 'scaleFill':
                raise ValueError('Unexpected prediction parameters')
            metrics = evaluation.score_image(item, predicted['predictions'])
            tp, fp, fn = metrics['TP'], metrics['FP'], metrics['FN']
            metrics['F1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
            scored[variant['name']].append({'imageId': item['id'], 'sourceGroup': item['sourceGroup'], 'metrics': metrics})
    return scored, prediction_path


def aggregate(rows):
    count_rows = [row['metrics'] for row in rows if row['metrics']['exactCountEligible']]
    tp = sum(row['metrics']['TP'] for row in rows)
    fp = sum(row['metrics']['FP'] for row in rows)
    fn = sum(row['metrics']['FN'] for row in rows)
    return {'images': len(rows), 'macroPerImageF1': sum(row['metrics']['F1'] for row in rows) / len(rows),
            'TP': tp, 'FP': fp, 'FN': fn,
            'microPrecision': tp / (tp + fp) if tp + fp else None,
            'microRecall': tp / (tp + fn) if tp + fn else None,
            'countEligibleImages': len(count_rows),
            'countMAE': sum(row['absoluteCountError'] for row in count_rows) / len(count_rows) if count_rows else None,
            'exactCountImages': sum(row['exactCount'] for row in count_rows),
            'exactCountRate': sum(row['exactCount'] for row in count_rows) / len(count_rows) if count_rows else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('calibrate', 'holdout'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--repository', type=Path, default=HERE.parents[2])
    parser.add_argument('--output-dir', type=Path, default=Path('output/web-annotation/confidence-sweep-v2'))
    args = parser.parse_args()
    repository, manifest_path, output = args.repository.resolve(), args.manifest.resolve(), args.output_dir.resolve()
    _, active, _ = evaluation.validate_manifest(manifest_path, repository)
    manifest_sha = evaluation.digest(manifest_path)
    model_checkpoint = checkpoint(repository)
    calibration = [item for item in active if item['split'] == 'calibration']
    holdout = [item for item in active if item['split'] == 'holdout']
    if {i['id'] for i in calibration} != {'root-flat-black', 'acquisition-04'} or len(holdout) != 3:
        raise ValueError('This predeclared experiment requires the two specified calibration images and the three frozen holdout images')
    output.mkdir(parents=True, exist_ok=True)
    plan_path, selection_path = output / 'plan.json', output / 'selection.json'
    if args.phase == 'calibrate':
        plan = {'schemaVersion': 1, 'declaredAtUTC': now(), 'manifestSha256': manifest_sha,
                'calibrationImageIds': [item['id'] for item in calibration], 'holdoutImageIds': [item['id'] for item in holdout],
                'confidenceThresholds': list(THRESHOLDS), 'iouThreshold': 0.45, 'cropScale': 'scaleFill',
                'selectionRule': 'Maximize equally weighted per-image F1 on calibration only; ties within1e-12 use lower countMAE on exact-count-eligible calibration images only; remaining tie chooses lower confidence.',
                'baselineHoldoutPreviouslyObserved': True, 'newThresholdHoldoutUsedForSelection': False,
                'holdoutRule': 'Persist selection.json before new winner holdout inference. Run only that winner once on all three holdout images. Do not retune or edit labels after seeing its holdout results.',
                'labelsAndSplitsUnchanged': True, 'productionCodeUnchanged': True, **model_checkpoint,
                'scriptSha256': evaluation.digest(Path(__file__)), 'modelRunnerSha256': evaluation.digest(HERE / 'evaluate-confidence-model.swift')}
        write_new(plan_path, plan)  # Refuses accidental repeated sweeps in the same experiment directory.
        variants = [{'name': 'confidence_' + str(int(c * 100)).zfill(3), 'confidenceThreshold': c, 'iouThreshold': 0.45} for c in THRESHOLDS]
        scored, prediction_path = predictions_for(calibration, variants, 'calibration', manifest_sha, model_checkpoint, output)
        summaries = [{**variant, 'aggregate': aggregate(scored[variant['name']]), 'images': scored[variant['name']]} for variant in variants]
        best_f1 = max(s['aggregate']['macroPerImageF1'] for s in summaries)
        tied = [s for s in summaries if abs(s['aggregate']['macroPerImageF1'] - best_f1) <= 1e-12]
        winner = min(tied, key=lambda s: (s['aggregate']['countMAE'] if s['aggregate']['countMAE'] is not None else float('inf'), s['confidenceThreshold']))
        report = {'schemaVersion': 1, 'phase': 'calibration', 'completedAtUTC': now(), 'planSha256': evaluation.digest(plan_path),
                  'predictionSha256': evaluation.digest(prediction_path), 'variants': summaries}
        report_path = output / 'calibration-report.json'; write_new(report_path, report)
        selection = {'schemaVersion': 1, 'selectedAtUTC': now(), 'manifestSha256': manifest_sha,
                     'planSha256': evaluation.digest(plan_path), 'calibrationReportSha256': evaluation.digest(report_path),
                     'selectedConfidenceThreshold': winner['confidenceThreshold'], 'iouThreshold': 0.45, 'cropScale': 'scaleFill',
                     'selectionBasis': 'CALIBRATION_ONLY', 'aggregate': winner['aggregate'],
                     'holdoutNotRunInThisPhase': True, 'productionCodeChanged': False, **model_checkpoint}
        write_new(selection_path, selection)
        print(json.dumps({'selectionPath': str(selection_path), 'selectedConfidenceThreshold': winner['confidenceThreshold'], 'calibrationSummaries': [{k:s[k] for k in ('confidenceThreshold','aggregate')} for s in summaries]}, indent=2))
    else:
        plan = json.loads(plan_path.read_text()); selection = json.loads(selection_path.read_text())
        if selection['manifestSha256'] != manifest_sha or selection['planSha256'] != evaluation.digest(plan_path):
            raise ValueError('Frozen selection/dataset identity mismatch')
        if selection['calibrationReportSha256'] != evaluation.digest(output / 'calibration-report.json'):
            raise ValueError('Calibration report changed after selection')
        if any(selection[k] != model_checkpoint[k] for k in model_checkpoint):
            raise ValueError('Checkpoint changed after selection')
        if plan['scriptSha256'] != evaluation.digest(Path(__file__)) or plan['modelRunnerSha256'] != evaluation.digest(HERE / 'evaluate-confidence-model.swift'):
            raise ValueError('Experiment scripts changed after declaration')
        confidence = selection['selectedConfidenceThreshold']
        if confidence not in THRESHOLDS: raise ValueError('Winner outside declared search')
        write_new(output / 'holdout-started.json', {'startedAtUTC': now(), 'selectionSha256': evaluation.digest(selection_path), 'confidenceThreshold': confidence})
        variant = {'name': 'frozen_winner', 'confidenceThreshold': confidence, 'iouThreshold': 0.45}
        scored, prediction_path = predictions_for(holdout, [variant], 'holdout', manifest_sha, model_checkpoint, output)
        report = {'schemaVersion': 1, 'phase': 'holdout_once', 'completedAtUTC': now(), 'manifestSha256': manifest_sha,
                  'selectionSha256': evaluation.digest(selection_path), 'predictionSha256': evaluation.digest(prediction_path),
                  'confidenceThreshold': confidence, 'iouThreshold': 0.45, 'cropScale': 'scaleFill',
                  'aggregate': aggregate(scored['frozen_winner']), 'images': scored['frozen_winner'],
                  'productionCodeChanged': False, 'noRetuningAfterHoldout': True,
                  'limitations': ['Only five web photos, including one restaurant photograph, were available.', 'Visible endpoint counts are not hidden physical-stick or retail-package quantities.', 'Mac CPU-only evaluation is not physical-device validation.', 'Completion is not a high-accuracy or release-readiness claim.']}
        report_path = output / 'holdout-report.json'; write_new(report_path, report)
        print(json.dumps({'holdoutReport': str(report_path), 'aggregate': report['aggregate'], 'counts': [{'imageId': r['imageId'], 'predicted': r['metrics']['predictedWholeImageCount'], 'visibleGroundTruth': r['metrics']['confirmedVisibleEndpointCount'], 'signedError': r['metrics']['signedCountError']} for r in report['images']]}, indent=2))
    if evaluation.digest(manifest_path) != manifest_sha or checkpoint(repository) != model_checkpoint:
        raise ValueError('Frozen dataset or checkpoint changed during evaluation; do not use outputs')
    for image in active:
        if evaluation.digest(image['path']) != image['sha256'] or evaluation.digest(image['annotationPath']) != image['annotationSha256']:
            raise ValueError('Frozen input changed during evaluation; do not use outputs: ' + image['id'])

if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print('Experiment blocked: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
