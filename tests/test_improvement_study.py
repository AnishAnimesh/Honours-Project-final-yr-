import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from heatcast.improvement_study import (CANDIDATES, choose_macro_cutoff,
    crossfit_candidate, fit_candidate, margin_features, predict_candidate,
    purged_folds, select_development, forecast, check_artifacts)


class ImprovementStudyTests(unittest.TestCase):
    def setUp(self):
        self.dates = np.arange('2026-05-01', '2026-06-23', dtype='datetime64[D]')

    def test_purge_keeps_whole_days_out_of_training(self):
        seen = []
        for train, valid in purged_folds(self.dates):
            self.assertGreaterEqual(len(train), 20)
            delta = np.abs((self.dates[train, None] - self.dates[valid]).astype(int))
            self.assertTrue((delta > 7).all())
            seen.extend(valid)
        np.testing.assert_array_equal(np.sort(seen), np.arange(len(self.dates)))

    def test_duplicate_or_insufficient_dates_rejected(self):
        with self.assertRaises(ValueError):
            purged_folds(self.dates[:10])
        with self.assertRaises(ValueError):
            purged_folds(np.r_[self.dates[:1], self.dates])

    def test_cutoff_optimizes_equal_weight_sites_and_conservative_ties(self):
        truth = np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=bool)
        probability = np.array([[.9, .3], [.8, .3], [.7, .6], [.2, .1]])
        cutoff = choose_macro_cutoff(truth, probability)
        self.assertEqual(cutoff, .6)
        # With no positive support, all cutoffs tie: choose no alerts.
        self.assertEqual(choose_macro_cutoff(np.zeros_like(truth), probability), 1.)
        self.assertEqual(choose_macro_cutoff(np.zeros_like(truth), np.ones_like(probability)), 1.)

    def test_crossfit_empirical_bank_cannot_use_validation_truth(self):
        shape = (len(self.dates), 2, 3)
        forecast = np.zeros(shape)
        threshold = np.ones(shape)
        errors = np.zeros(shape)
        valid = np.arange(10)
        train = np.arange(20, len(self.dates))
        # Only held-out residuals would turn the event on.
        errors[valid] = 10
        spec = dict(name='test', kind='empirical', scale=1.)
        bundle = dict(spec=spec, task='ongoing', error_bank=errors[train])
        np.testing.assert_equal(predict_candidate(bundle, forecast[valid], threshold[valid]), 0)

    def test_logistic_scaler_fits_training_only_and_inference_is_run_only(self):
        rng = np.random.default_rng(3)
        run = rng.normal(size=(30, 2, 4))
        threshold = np.zeros_like(run)
        truth = run[..., 1] > 0
        spec = next(c for c in CANDIDATES if c['kind'] == 'logistic')
        bundle = fit_candidate(spec, run, threshold, truth, 'onset')
        features = margin_features(run, threshold)
        np.testing.assert_allclose(bundle['estimator'][0].mean_, features.reshape(-1, features.shape[-1]).mean(0))
        prediction = predict_candidate(bundle, run, threshold)
        self.assertEqual(prediction.shape, truth.shape)
        self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())
        self.assertNotIn('truth', bundle)

    def test_single_class_logistic_fit_is_explicit_constant(self):
        run = np.zeros((25, 2, 4))
        spec = next(c for c in CANDIDATES if c['kind'] == 'logistic')
        bundle = fit_candidate(spec, run, run, np.zeros((25, 2)), 'onset')
        self.assertEqual(bundle['constant'], 0.)
        np.testing.assert_equal(predict_candidate(bundle, run, run), 0.)

    def test_development_selection_has_no_test_input(self):
        truth = np.array([[1, 0], [0, 1]], dtype=bool)
        probabilities = [np.full(truth.shape, .5) for _ in CANDIDATES]
        probabilities[0] = truth.astype(float)
        entries, selected = select_development(truth, probabilities)
        self.assertEqual(selected, 0)
        self.assertEqual(entries[selected]['cutoff'], .5)
        self.assertNotIn('test', entries[selected])

    def test_inference_archive_needs_no_verification_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = np.arange('2026-07-01', '2026-07-13', dtype='datetime64[D]')[None, :]
            np.savez(root / 'daily_runs.npz', sites=['test_site'],
                     initializations=np.array(['2026-07-01'], dtype='datetime64[D]'),
                     valid_dates=dates, forecast_tmax=np.full((1, 12, 1), 35.))
            bundle = dict(spec=CANDIDATES[0], task='ongoing', sites=['test_site'],
                          reference_table=np.full((365, 1), 30.), cutoff=.5,
                          original_development_available_at=np.datetime64('2026-06-30', 'ns'),
                          designed_at_utc='2026-11-03T00:00:00Z')
            joblib.dump(bundle, root / 'ongoing_lead1_selected.joblib')
            frame = forecast('2026-07-02', 1, 'ongoing', root, root)
            self.assertEqual(frame['probability'].iloc[0], 1.)
            self.assertTrue(frame['predicted_event'].iloc[0])
            self.assertIn('Exploratory', frame['study_status'].iloc[0])
            with np.load(root / 'daily_runs.npz') as archive:
                changed = {key: archive[key] for key in archive.files}
            changed.update(verification_dates=np.array(['2026-07-03'], dtype='datetime64[D]'),
                           verification_tmax=np.full((1, 1), -9999.))
            np.savez(root / 'daily_runs.npz', **changed)
            modified = forecast('2026-07-02', 1, 'ongoing', root, root)
            np.testing.assert_array_equal(frame['probability'], modified['probability'])

    @unittest.skipUnless(Path('results/improvement_study/metrics.json').exists(), 'Study artifacts not generated')
    def test_all_saved_predictions_reproduce_without_verification(self):
        self.assertEqual(check_artifacts(), 3570)


if __name__ == '__main__':
    unittest.main()
