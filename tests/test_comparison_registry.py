import json
from pathlib import Path
import re
import unittest

from comparison_suite import Method, validate_identity, final_text, numeric
from comparison_training import TrainConfig

ROOT=Path(__file__).resolve().parents[1]

class RegistryTests(unittest.TestCase):
    def test_empty_final_marker_is_not_a_crash_or_answer(self):
        self.assertEqual(final_text('Final answer:'),'')
        self.assertIsNone(numeric('#### '))

    def test_all_method_and_training_presets_validate(self):
        methods=list((ROOT/'experiments/comparison/methods').glob('*.json'))
        training=list((ROOT/'experiments/comparison/training').glob('*.json'))
        self.assertEqual(len(methods),9);self.assertEqual(len(training),5)
        for p in methods:Method(**json.loads(p.read_text()))
        for p in training:TrainConfig(**json.loads(p.read_text()))

    def test_primary_evidence_is_immutable_and_status_explicit(self):
        rows=json.loads((ROOT/'experiments/comparison/papers.json').read_text())['papers']
        self.assertEqual(len(rows),22);self.assertEqual(len({r['id'] for r in rows}),22)
        for r in rows:
            self.assertTrue(r['implementation_status'])
            if r['evidence_status']=='upstream-repository':
                self.assertTrue(re.fullmatch('[a-f0-9]{40}',r['inspected_readme_blob']))
                self.assertIn(r['inspected_readme_blob'],r['evidence_url'])
        thinkless=next(r for r in rows if r['id']=='thinkless')
        self.assertIn('upstream-output-import',thinkless['implementation_status'])
        self.assertEqual(thinkless['inspected_code']['commit'],'55817cbeedaf4fd862844cc3b471dbaf3aa43227')

    def test_dataset_support_does_not_fake_game24(self):
        data=json.loads((ROOT/'experiments/comparison/datasets.json').read_text())['datasets']
        self.assertEqual(len(data),14)
        game=next(r for r in data if r['id']=='game24')
        self.assertIn('upstream',game['scorer'])
        self.assertIn('Not an implemented native scorer',game['purpose'])

    def test_identity_examples_are_not_runnable_fake_measurements(self):
        with self.assertRaises(ValueError):validate_identity(json.loads((ROOT/'experiments/comparison/identity.example.json').read_text()))
        with self.assertRaises(ValueError):validate_identity({})

    def test_study_is_plan_not_results(self):
        data=json.loads((ROOT/'experiments/comparison/study.json').read_text())
        self.assertEqual(data['training_seeds'],[7,17,27])
        self.assertIn('not-registered-result',data['status'])
        self.assertIn('targets_not_forecasts',data)

if __name__=='__main__':unittest.main()
