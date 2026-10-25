#!/usr/bin/env python3
"""Validate frozen blind labels, run fixed CoreML variants, and score endpoint/count errors.

Canonical schema is documented in evaluate-schema.json. No inference runs during --validate-only.
A source group cannot cross calibration/holdout. Byte-identical or explicitly duplicated images
cannot both be eligible. All labels and their matching radii must be frozen before inference.
"""
import argparse
import collections
import datetime
import hashlib
import heapq
import json
import math
from pathlib import Path
import subprocess
import sys

VARIANTS = ('baseline', 'candidate')
COORDINATES = 'upright image pixels, top-left'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

def resolve_input(value, manifest_path, repository):
    if not isinstance(value, str) or not value:
        raise ValueError('A nonempty input path is required')
    raw = Path(value)
    candidates = [raw.resolve()] if raw.is_absolute() else [(manifest_path.parent / raw).resolve(), (repository / raw).resolve()]
    paths = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if len(paths) != 1:
        raise ValueError('Missing or ambiguous input path: ' + value)
    path = paths[0]
    if repository not in path.parents:
        raise ValueError('Dataset input must stay inside the repository: ' + value)
    return path

def validate_manifest(path, repository):
    path = path.resolve(); repository = repository.resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schemaVersion') != 1:
        raise ValueError('schemaVersion must equal 1')
    if manifest.get('labelsFrozen') is not True or manifest.get('splitLocked') is not True or not manifest.get('frozenAt'):
        raise ValueError('Refusing inference: labelsFrozen, frozenAt, and splitLocked are required')
    images = manifest.get('images')
    if not isinstance(images, list) or not images:
        raise ValueError('Manifest images must be a nonempty list')
    seen_ids, seen_sha, group_splits = set(), {}, {}
    active, excluded = [], []
    for image in images:
        image_id = image.get('id')
        if not isinstance(image_id, str) or not image_id or image_id in seen_ids:
            raise ValueError('Missing or duplicate image id: ' + str(image_id))
        seen_ids.add(image_id)
        group, split = image.get('sourceGroup'), image.get('split')
        if not isinstance(group, str) or not group or split not in ('calibration', 'holdout'):
            raise ValueError(image_id + ': sourceGroup and calibration/holdout split required')
        if group in group_splits and group_splits[group] != split:
            raise ValueError('Source-group leakage across splits: ' + group)
        group_splits[group] = split
        if image.get('eligible') is not True:
            excluded.append({'imageId': image_id, 'status': image.get('status'), 'reason': image.get('exclusionReason', 'not eligible')})
            continue
        if image.get('duplicateOf'):
            raise ValueError(image_id + ': an explicit duplicate must be excluded, not scored twice')
        photo_path = resolve_input(image.get('path'), path, repository)
        photo_sha = digest(photo_path)
        if image.get('imageSha256', image.get('sha256')) != photo_sha:
            raise ValueError(image_id + ': missing or incorrect imageSha256')
        if photo_sha in seen_sha:
            raise ValueError('Duplicate eligible image content: ' + seen_sha[photo_sha] + ' and ' + image_id)
        seen_sha[photo_sha] = image_id
        annotation_path = resolve_input(image.get('annotationPath'), path, repository)
        annotation_sha = digest(annotation_path)
        if image.get('annotationSha256') != annotation_sha:
            raise ValueError(image_id + ': annotationSha256 missing or stale; labels must be frozen before prediction')
        annotation = json.loads(annotation_path.read_text())
        if annotation.get('schemaVersion') != 1 or annotation.get('imageId') != image_id or annotation.get('imageSha256') != photo_sha:
            raise ValueError(image_id + ': annotation identity mismatch')
        if annotation.get('coordinateSystem') != COORDINATES:
            raise ValueError(image_id + ': annotations must use upright image pixels, top-left')
        review = annotation.get('review', {})
        if review.get('status') != 'reviewed_frozen' or not review.get('annotator') or not review.get('reviewer'):
            raise ValueError(image_id + ': blind annotation and review must be frozen before inference')
        width, height = annotation.get('width'), annotation.get('height')
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(image_id + ': positive integer upright dimensions required')
        scope = annotation.get('scope')
        if scope not in ('full_image', 'roi'):
            raise ValueError(image_id + ': scope must be full_image or roi')
        region = annotation.get('evaluationRegion', {}).get('xywh', [0, 0, width, height] if scope == 'full_image' else None)
        if not isinstance(region, list) or len(region) != 4 or not all(number(v) for v in region):
            raise ValueError(image_id + ': evaluationRegion.xywh required')
        x, y, w, h = region
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError(image_id + ': invalid evaluation region')
        if scope == 'full_image' and region != [0, 0, width, height]:
            raise ValueError(image_id + ': full_image must cover the whole upright image')
        points = annotation.get('points')
        if not isinstance(points, list):
            raise ValueError(image_id + ': points array required (empty is valid for a verified negative)')
        point_ids = set()
        default_radius = annotation.get('matching', {}).get('radiusPx')
        for point in points:
            point_id = point.get('id')
            px, py, radius = point.get('x'), point.get('y'), point.get('matchRadius', default_radius)
            if not point_id or point_id in point_ids or not all(number(v) for v in (px, py, radius)) or radius <= 0:
                raise ValueError(image_id + ': every point needs unique id, finite coordinates, positive matchRadius')
            if not (x <= px <= x + w and y <= py <= y + h):
                raise ValueError(image_id + ': point lies outside evaluation region: ' + str(point_id))
            point_ids.add(point_id)
            point['matchRadius'] = radius
            if 'bbox' in point and (len(point['bbox']) != 4 or not all(number(v) for v in point['bbox']) or min(point['bbox'][2:]) <= 0):
                raise ValueError(image_id + ': bbox must be [x,y,width,height]')
        if annotation.get('visibleEndpointCount') != len(points):
            raise ValueError(image_id + ': visibleEndpointCount must match manually labeled points, not package quantity')
        ignore = annotation.get('ignoreRegions', [])
        if not isinstance(ignore, list):
            raise ValueError(image_id + ': ignoreRegions must be a list')
        for area in ignore:
            polygon = area.get('polygon', [])
            if len(polygon) < 3 or not area.get('reason') or any(len(p) != 2 or not all(number(v) for v in p) for p in polygon):
                raise ValueError(image_id + ': ignored uncertainty needs polygon and reason')
        complete = annotation.get('completeCountVerified') is True
        if complete and (scope != 'full_image' or ignore):
            raise ValueError(image_id + ': exact count requires full_image, no uncertain ignore regions, and manual verification')
        active.append({'id': image_id, 'path': str(photo_path), 'sha256': photo_sha, 'annotationPath': str(annotation_path),
                       'annotationSha256': annotation_sha, 'sourceGroup': group, 'split': split, 'width': width, 'height': height,
                       'region': region, 'annotation': annotation})
    if not active:
        raise ValueError('No eligible frozen images; inference will not run')
    return manifest, active, excluded

def inside_polygon(point, polygon):
    x, y = point
    inside = False
    for index, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(index + 1) % len(polygon)]
        # Boundary points count as ignored too; do not create spurious negatives on uncertainty edges.
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-8 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside

def minimum_cost_maximum_matching(predictions, labels):
    """Successive shortest paths: maximum one-to-one cardinality, then minimum pixel distance."""
    p_count, l_count = len(predictions), len(labels)
    source, label_start, sink = 0, 1 + p_count, 1 + p_count + l_count
    graph = [[] for _ in range(sink + 1)]
    links = []
    def edge(a, b, capacity, cost):
        forward = [b, len(graph[b]), capacity, cost]
        reverse = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward); graph[b].append(reverse)
        return forward
    for pi, prediction in enumerate(predictions):
        edge(source, 1 + pi, 1, 0.0)
        for gi, label in enumerate(labels):
            distance = math.hypot(prediction['x'] - label['x'], prediction['y'] - label['y'])
            if distance <= label['matchRadius']:
                link = edge(1 + pi, label_start + gi, 1, distance)
                links.append((pi, gi, distance, link))
    for gi in range(l_count):
        edge(label_start + gi, sink, 1, 0.0)
    potential = [0.0] * len(graph)
    while True:
        distance = [math.inf] * len(graph); distance[source] = 0.0
        previous = [None] * len(graph)
        queue = [(0.0, source)]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > distance[node] + 1e-10: continue
            for ei, item in enumerate(graph[node]):
                target, _, capacity, weight = item
                if not capacity: continue
                candidate = cost + weight + potential[node] - potential[target]
                if candidate < distance[target] - 1e-10:
                    distance[target] = candidate; previous[target] = (node, ei)
                    heapq.heappush(queue, (candidate, target))
        if previous[sink] is None: break
        for node, value in enumerate(distance):
            if math.isfinite(value): potential[node] += value
        node = sink
        while node != source:
            parent, ei = previous[node]; item = graph[parent][ei]
            item[2] -= 1; graph[node][item[1]][2] += 1; node = parent
    return [{'predictionIndex': pi, 'labelIndex': gi, 'distancePx': distance}
            for pi, gi, distance, link in links if link[2] == 0]

def score_image(item, predictions):
    annotation = item['annotation']; labels = annotation['points']
    rx, ry, rw, rh = item['region']
    in_region = [p for p in predictions if rx <= p['x'] <= rx + rw and ry <= p['y'] <= ry + rh]
    matches = minimum_cost_maximum_matching(in_region, labels)
    used_predictions = {m['predictionIndex'] for m in matches}; used_labels = {m['labelIndex'] for m in matches}
    ignored, false_positives, duplicates = [], [], []
    for pi, prediction in enumerate(in_region):
        if pi in used_predictions: continue
        if any(inside_polygon((prediction['x'], prediction['y']), area['polygon']) for area in annotation.get('ignoreRegions', [])):
            ignored.append(pi); continue
        false_positives.append(pi)
        if any(math.hypot(prediction['x'] - labels[gi]['x'], prediction['y'] - labels[gi]['y']) <= labels[gi]['matchRadius'] for gi in used_labels):
            duplicates.append(pi)
    tp, fp, fn = len(matches), len(false_positives), len(labels) - len(matches)
    complete = annotation.get('completeCountVerified') is True and annotation.get('scope') == 'full_image' and not annotation.get('ignoreRegions')
    count_error = len(predictions) - len(labels) if complete else None
    return {'TP': tp, 'FP': fp, 'FN': fn, 'ignoredPredictions': len(ignored), 'duplicateFalsePositives': len(duplicates),
            'otherFalsePositives': fp - len(duplicates), 'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / len(labels) if labels else None, 'predictedWholeImageCount': len(predictions),
            'predictedRegionCount': len(in_region), 'confirmedVisibleEndpointCount': len(labels),
            'exactCountEligible': complete, 'signedCountError': count_error,
            'absoluteCountError': abs(count_error) if complete else None, 'exactCount': count_error == 0 if complete else None,
            'matches': [{**m, 'labelId': labels[m['labelIndex']]['id'], 'predictionId': in_region[m['predictionIndex']].get('id')} for m in matches],
            'falsePositivePredictionIds': [in_region[i].get('id') for i in false_positives],
            'duplicatePredictionIds': [in_region[i].get('id') for i in duplicates],
            'missedLabelIds': [label['id'] for gi, label in enumerate(labels) if gi not in used_labels]}

def summarize(rows):
    result = {}
    for variant in VARIANTS:
        metrics = [row['variants'][variant] for row in rows]
        totals = {key: sum(m[key] for m in metrics) for key in ('TP', 'FP', 'FN', 'ignoredPredictions', 'duplicateFalsePositives')}
        tp, fp, fn = (totals[k] for k in ('TP', 'FP', 'FN'))
        count_metrics = [m for m in metrics if m['exactCountEligible']]
        result[variant] = {**totals, 'images': len(rows), 'sourceGroups': len({r['sourceGroup'] for r in rows}),
                           'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None,
                           'countEligibleImages': len(count_metrics),
                           'countMAE': sum(m['absoluteCountError'] for m in count_metrics) / len(count_metrics) if count_metrics else None,
                           'exactCountRate': sum(m['exactCount'] for m in count_metrics) / len(count_metrics) if count_metrics else None}
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--predictions', type=Path, help='Score already-recorded predictions with the identical frozen manifest SHA')
    args = parser.parse_args(argv)
    repository = args.repository.resolve(); manifest_path = args.manifest.resolve()
    manifest, active, excluded = validate_manifest(manifest_path, repository)
    manifest_sha = digest(manifest_path)
    validation = {'validationPassed': True, 'eligibleImages': len(active), 'excludedImages': len(excluded),
                  'sourceGroups': len({x['sourceGroup'] for x in active}), 'manifestSha256': manifest_sha,
                  'splitCounts': dict(collections.Counter(x['split'] for x in active))}
    if args.validate_only:
        print(json.dumps(validation, indent=2)); return 0
    output_dir = (args.output_dir or manifest_path.parent / 'evaluation').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = repository / 'Count stick/best.mlpackage'
    helper = Path(__file__).with_name('evaluate-model.swift')
    model_hashes = {str(p.relative_to(model_path)): digest(p) for p in sorted(model_path.rglob('*')) if p.is_file()}
    detector_sha = digest(repository / 'Count stick/StickDetector.swift')
    validated = {'validationPassed': True, 'labelsFrozen': True, 'splitLocked': True, 'manifestSha256': manifest_sha,
                 'modelPath': str(model_path), 'modelFileHashes': model_hashes, 'sourceDetectorSha256': detector_sha, 'images': [{k: item[k] for k in ('id', 'path', 'sha256', 'width', 'height')} for item in active]}
    validated_path = output_dir / 'validated-input.json'; validated_path.write_text(json.dumps(validated, indent=2) + '\n')
    prediction_path = args.predictions.resolve() if args.predictions else output_dir / 'predictions.json'
    if not args.predictions:
        subprocess.run(['swift', str(helper), str(validated_path), str(prediction_path)], check=True)
    if digest(manifest_path) != manifest_sha:
        raise ValueError('Manifest changed while evaluation was running')
    for item in active:
        if digest(item['annotationPath']) != item['annotationSha256'] or digest(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed while evaluation was running: ' + item['id'])
    predictions = json.loads(prediction_path.read_text())
    if predictions.get('modelFileHashes') != model_hashes or predictions.get('sourceDetectorSha256') != detector_sha:
        raise ValueError('Prediction model/source hashes do not match the evaluated checkpoint')
    if {str(p.relative_to(model_path)): digest(p) for p in sorted(model_path.rglob('*')) if p.is_file()} != model_hashes or digest(repository / 'Count stick/StickDetector.swift') != detector_sha:
        raise ValueError('Model or detector source changed while evaluating')
    if predictions.get('manifestSha256') != manifest_sha:
        raise ValueError('Predictions were not run against this exact frozen manifest')
    by_id = {image['imageId']: image for image in predictions['images']}
    if len(by_id) != len(predictions['images']) or set(by_id) != {item['id'] for item in active}:
        raise ValueError('Prediction image IDs are duplicated or do not match the frozen eligible set')
    rows = []
    for item in active:
        prediction = by_id[item['id']]
        if prediction.get('imageSha256') != item['sha256']:
            raise ValueError('Prediction image digest mismatch: ' + item['id'])
        variants = prediction['variants']
        for name in VARIANTS:
            if variants[name]['confidenceThreshold'] != 0.25 or variants[name]['iouThreshold'] != (0.70 if name == 'baseline' else 0.45) or variants[name].get('cropScale') != 'scaleFill':
                raise ValueError('Unexpected variant parameters: ' + name)
            for p in variants[name]['predictions']:
                if not all(number(p.get(key)) for key in ('x', 'y', 'confidence')):
                    raise ValueError('Non-finite prediction on ' + item['id'])
        rows.append({'imageId': item['id'], 'sourceGroup': item['sourceGroup'], 'split': item['split'],
                     'imageSha256': item['sha256'], 'annotationSha256': item['annotationSha256'],
                     'variants': {name: score_image(item, variants[name]['predictions']) for name in VARIANTS}})
    report = {'schemaVersion': 1, 'evaluatedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'validation': validation, 'executionPlatform': predictions.get('executionPlatform'),
              'sourceDetectorSha256': detector_sha,
              'modelDefinitionSha256': digest(model_path / 'Data/com.apple.CoreML/model.mlmodel'),
              'modelFileHashes': model_hashes,
              'predictionFileSha256': digest(prediction_path), 'excludedImages': excluded,
              'matchingProtocol': 'Maximum-cardinality, minimum-total-pixel-distance one-to-one matches inside each frozen per-point radius. Unmatched predictions in ignored polygons excluded; extra matches to already-matched heads are duplicate FP.',
              'countProtocol': 'Exact-count rate/MAE only for full_image + completeCountVerified + no ignored uncertainty. These are visible-endpoint counts, never inferred package or physical hidden-stick quantities.',
              'overall': summarize(rows), 'bySplit': {split: summarize([row for row in rows if row['split'] == split]) for split in ('calibration', 'holdout')},
              'bySourceGroup': {group: summarize([row for row in rows if row['sourceGroup'] == group]) for group in sorted({row['sourceGroup'] for row in rows})},
              'images': rows,
              'limitations': ['Web imagery may differ from restaurant camera captures.', 'Mac CPU-only inference is not physical iPhone validation.', 'No release-quality acceptance is implied by completion of this evaluation.']}
    report_path = output_dir / 'report.json'; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(report_path), 'bySplit': report['bySplit']}, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print('Evaluation blocked: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
