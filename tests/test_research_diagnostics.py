"""Audit the supplementary scientific artifacts against their source cases."""
import json
from pathlib import Path
import unittest

import numpy as np

from heatcast.evaluation import metrics
from heatcast.issued_data import event_rule
from heatcast.models import LSTM, ConvLSTM


class ResearchDiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(Path('results/neural_sensitivity/metrics.json').exists(), 'Run the seed-sensitivity study first')
    def test_all_seeds_preserve_cases_and_reported_scores(self):
        report=json.loads(Path('results/neural_sensitivity/metrics.json').read_text())
        self.assertEqual(len(report['experiments']),15)
        for name in {r['experiment'] for r in report['experiments']}:
            group=[r for r in report['experiments'] if r['experiment']==name]
            self.assertEqual({r['seed'] for r in group},{7,21,42})
            reference=None
            for record in group:
                z=np.load(f"results/neural_sensitivity/{name}_seed{record['seed']}_predictions.npz")
                if reference is None:reference=z['truth']
                np.testing.assert_array_equal(z['truth'],reference)
                actual=metrics(z['truth'],z['probability'],float(z['cutoff']))
                self.assertAlmostEqual(actual['f1'],record['test']['f1'],places=12)
                self.assertAlmostEqual(actual['brier'],record['test']['brier'],places=12)
                for original_key in ['f1','brier','precision','recall']:
                    self.assertTrue(np.isfinite(record['test'][original_key]))

    def test_neural_parameter_counts_match_architecture(self):
        actual=[sum(p.numel() for p in model.parameters()) for model in [LSTM(3),LSTM(7),ConvLSTM(3),ConvLSTM(2)]]
        self.assertEqual(actual,[4769,5281,6541,6109])

    @unittest.skipUnless(Path('results/error_analysis/metrics.json').exists(), 'Run the error audit first')
    def test_error_traces_reconstruct_event_labels(self):
        report=json.loads(Path('results/error_analysis/metrics.json').read_text())
        self.assertEqual(len(report['cases']),6)
        for case in report['cases']:
            task=case['experiment'].split('_lead')[0]
            observed=np.array([r['observed_c'] for r in case['trajectory']])
            forecast=np.array([r['forecast_c'] for r in case['trajectory']])
            threshold=np.array([r['threshold_c'] for r in case['trajectory']])
            self.assertEqual(bool(event_rule(observed,threshold,task)),case['truth'])
            self.assertEqual(bool(event_rule(forecast,threshold,task)),case['raw_event'])
            self.assertEqual(case['probability']>=case['cutoff'],case['predicted'])

    @unittest.skipUnless(Path('results/error_analysis/metrics.json').exists(), 'Run the error audit first')
    def test_alert_change_decomposition_balances(self):
        report=json.loads(Path('results/error_analysis/metrics.json').read_text())
        for entry in report['experiments'].values():
            change=entry['adjusted_false_alerts']-entry['raw_false_alerts']
            self.assertEqual(change,entry['added_false_alerts']-entry['removed_false_alerts'])


if __name__=='__main__':unittest.main()
