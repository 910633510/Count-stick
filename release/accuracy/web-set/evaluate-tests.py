#!/usr/bin/env python3
"""Small synthetic checks of evaluation bookkeeping. No image inference or accuracy claims."""
import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).with_name('evaluate.py'))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

class EvaluationChecks(unittest.TestCase):
    def test_matching_prefers_maximum_cardinality_over_greedy_nearest(self):
        labels = [{'id': 'A', 'x': 0, 'y': 0, 'matchRadius': 3}, {'id': 'B', 'x': 3, 'y': 0, 'matchRadius': 2}]
        predictions = [{'x': 1, 'y': 0}, {'x': -1, 'y': 0}]
        matched = evaluation.minimum_cost_maximum_matching(predictions, labels)
        self.assertEqual({(m['predictionIndex'], m['labelIndex']) for m in matched}, {(0, 1), (1, 0)})

    def test_matching_agrees_with_bruteforce_on_small_ambiguous_cases(self):
        generator = random.Random(17)
        for _ in range(30):
            labels = [{'id': str(i), 'x': generator.uniform(0,10), 'y': generator.uniform(0,10), 'matchRadius': generator.uniform(2,7)} for i in range(4)]
            predictions = [{'x': generator.uniform(0,10), 'y': generator.uniform(0,10)} for _ in range(5)]
            optimum = [-1, float('inf')]
            def enumerate_assignments(index, used, count, cost):
                if index == len(predictions):
                    if count > optimum[0] or (count == optimum[0] and cost < optimum[1]): optimum[:] = [count, cost]
                    return
                enumerate_assignments(index+1, used, count, cost)
                for gi, label in enumerate(labels):
                    distance = evaluation.math.hypot(predictions[index]['x']-label['x'], predictions[index]['y']-label['y'])
                    if gi not in used and distance <= label['matchRadius']:
                        enumerate_assignments(index+1, used | {gi}, count+1, cost+distance)
            enumerate_assignments(0, set(), 0, 0)
            matched = evaluation.minimum_cost_maximum_matching(predictions, labels)
            self.assertEqual(len(matched), optimum[0])
            self.assertAlmostEqual(sum(m['distancePx'] for m in matched), optimum[1], places=7)

    def test_duplicate_is_false_positive_and_count_error(self):
        item = self.item(complete=True)
        result = evaluation.score_image(item, [{'id': 'a', 'x': 10, 'y': 10}, {'id': 'b', 'x': 11, 'y': 10}])
        self.assertEqual((result['TP'], result['FP'], result['FN'], result['duplicateFalsePositives']), (1, 1, 0, 1))
        self.assertEqual(result['absoluteCountError'], 1)
        self.assertFalse(result['exactCount'])

    def test_uncertain_polygon_is_excluded_from_false_positives_and_exact_counts(self):
        item = self.item(complete=False)
        item['annotation']['ignoreRegions'] = [{'polygon': [[20,20],[30,20],[30,30],[20,30]], 'reason': 'ambiguous end'}]
        result = evaluation.score_image(item, [{'id': 'a', 'x': 10, 'y': 10}, {'id': 'b', 'x': 20, 'y': 25}])
        self.assertEqual((result['TP'], result['FP'], result['ignoredPredictions']), (1, 0, 1))
        self.assertIsNone(result['absoluteCountError'])
        self.assertFalse(result['exactCountEligible'])

    def test_roi_never_gets_whole_image_exact_count(self):
        item = self.item(complete=True)
        item['annotation']['scope'] = 'roi'
        result = evaluation.score_image(item, [{'id': 'a', 'x': 10, 'y': 10}])
        self.assertFalse(result['exactCountEligible'])

    def test_label_hash_changes_and_unfrozen_status_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest_path, manifest, annotation_path = self.dataset(Path(directory))
            evaluation.validate_manifest(manifest_path, root)
            original = json.loads(annotation_path.read_text())
            original['points'][0]['x'] = 11
            annotation_path.write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'annotationSha256'):
                evaluation.validate_manifest(manifest_path, root)
            manifest['labelsFrozen'] = False
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Refusing inference'):
                evaluation.validate_manifest(manifest_path, root)

    def test_source_group_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, manifest, _ = self.dataset(Path(directory))
            second = copy.deepcopy(manifest['images'][0]); second.update(id='other', split='holdout', eligible=False)
            manifest['images'].append(second); path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Source-group leakage'):
                evaluation.validate_manifest(path, root)

    def test_duplicate_eligible_photo_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, manifest, _ = self.dataset(Path(directory))
            second = copy.deepcopy(manifest['images'][0]); second.update(id='other')
            manifest['images'].append(second); path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Duplicate eligible image'):
                evaluation.validate_manifest(path, root)

    @staticmethod
    def item(complete):
        return {'region': [0,0,100,100], 'annotation': {'scope': 'full_image', 'points': [{'id':'one','x':10,'y':10,'matchRadius':3}], 'ignoreRegions': [], 'completeCountVerified':complete}}

    def dataset(self, root):
        photo = root/'sample.jpg'; photo.write_bytes(b'validator-unit-test-placeholder-not-an-inference-image')
        image_sha = evaluation.digest(photo)
        annotation = {**self.item(True)['annotation'], 'schemaVersion':1, 'imageId':'sample', 'imageSha256':image_sha, 'width':100, 'height':100,
                      'coordinateSystem':evaluation.COORDINATES, 'visibleEndpointCount':1, 'review': {'status':'reviewed_frozen','annotator':'A','reviewer':'B'}}
        annotation_path = root/'annotation.json'; annotation_path.write_text(json.dumps(annotation))
        manifest = {'schemaVersion':1,'labelsFrozen':True,'frozenAt':'2026-10-23T12:00:00Z','splitLocked':True,
                    'images':[{'id':'sample','path':'sample.jpg','imageSha256':image_sha,'sourceGroup':'group-A','split':'calibration',
                               'annotationPath':'annotation.json','annotationSha256':evaluation.digest(annotation_path),'eligible':True}]}
        path=root/'manifest.json';path.write_text(json.dumps(manifest))
        return root,path,manifest,annotation_path

if __name__ == '__main__': unittest.main()
