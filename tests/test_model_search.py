import tempfile
import json
import hashlib
import copy
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from heatcast.model_search import (engineered_features, temporal_folds,
                                  threshold_policy, score, predict_bundle,
                                  candidate_specs, difference_intervals)
from heatcast.model_search_forecast import forecast
from heatcast import model_search
from heatcast.model_search_resume import transfer


class ModelSearchTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2018-01-01', periods=240).to_numpy()
        self.raw = np.random.default_rng(2).normal(size=(240, 2, 6)).astype('float32')
        self.threshold = np.zeros((240, 2))
        self.lat, self.lon = np.array([40, 50]), np.array([2, 5])

    def test_all_new_features_ignore_weather_after_issue(self):
        for lead in [1, 3, 7]:
            idx, x, _ = engineered_features(self.raw, self.dates, self.threshold,
                                             self.lat, self.lon, lead)
            row = 20
            issue = idx[row]-lead
            changed = self.raw.copy()
            changed[issue+1:] = 999
            _, other, _ = engineered_features(changed, self.dates, self.threshold,
                                               self.lat, self.lon, lead)
            np.testing.assert_array_equal(x[row], other[row])

    def test_streak_and_precipitation_match_trailing_history(self):
        raw = np.zeros_like(self.raw)
        raw[:, :, 5] = 2
        raw[140:145, :, 0] = 1
        idx, x, names = engineered_features(raw, self.dates, self.threshold,
                                             self.lat, self.lon, 1)
        row = np.flatnonzero(idx == 145)[0]
        np.testing.assert_array_equal(x[row, :, names.index('hot_streak_capped30')], [5, 5])
        np.testing.assert_array_equal(x[row, :, names.index('precipitation_total_30')], [60, 60])

    def test_folds_never_train_on_later_labels_or_split_sites(self):
        dates = pd.date_range('1981-01-01', '2025-12-31').to_numpy()
        idx = np.arange(30, len(dates)-2)
        folds = temporal_folds(dates, idx)
        for tr, va in folds:
            self.assertLess(dates[idx[tr]+2].max(), dates[idx[va]].min())
            self.assertGreaterEqual(pd.Timestamp(dates[idx[va]].min()).year, 2011)
            self.assertEqual(len(np.intersect1d(tr, va)), 0)

    def test_accuracy_policy_cannot_trade_away_required_validation_recall(self):
        y = np.array([[0], [0], [0], [1], [1]], dtype=bool)
        p = np.array([[.1], [.2], [.3], [.15], [.25]])
        cutoff = threshold_policy(y, p, 'accuracy', min_recall=.85)
        self.assertGreaterEqual(score(y, p, cutoff)['recall'], .85)

    def test_bundle_serialization_and_equal_weight_predictions(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(20, 2, 3))
        y = (x[:, :, 0] > 0).ravel()
        models = [ExtraTreesClassifier(n_estimators=5, random_state=s).fit(x.reshape(-1, 3), y)
                  for s in [1, 2]]
        bundle = dict(members=[dict(model=m, variant='engineered') for m in models],
                      calibrator=None, thresholds={'f1': .4, 'accuracy': .6})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bundle.joblib'
            joblib.dump(bundle, path)
            p, pred = predict_bundle(joblib.load(path), {'engineered': x})
        expected = np.mean([m.predict_proba(x.reshape(-1, 3))[:, 1].reshape(20, 2)
                            for m in models], axis=0)
        np.testing.assert_allclose(p, expected)
        np.testing.assert_array_equal(pred, p >= .4)

    def test_search_is_seeded_and_contains_all_six_families(self):
        self.assertEqual(candidate_specs(), candidate_specs())
        self.assertEqual({s['family'] for s in candidate_specs()},
                         {'hist', 'xgboost', 'catboost', 'lightgbm', 'extra_trees', 'random_forest'})

    def test_identical_forecasts_have_zero_paired_difference(self):
        dates = np.r_[pd.date_range('2020-06-01', periods=10).to_numpy(),
                      pd.date_range('2021-06-01', periods=10).to_numpy()]
        p = np.tile([[.2, .8]], (20, 1))
        result = difference_intervals(p > .5, p, .5, p, .5, dates, repeats=10)
        self.assertEqual(result['accuracy_difference_95'], [0, 0])
        self.assertEqual(result['f1_difference_95'], [0, 0])

    def test_csv_forecast_uses_only_history_and_saved_schema(self):
        idx, x, names = engineered_features(self.raw, self.dates, self.threshold,
                                             self.lat, self.lon, 1)
        model = ExtraTreesClassifier(n_estimators=5, random_state=1).fit(
            x.reshape(-1, x.shape[-1]), (x[:, :, 0] > 0).ravel())
        target_row = 20
        target = str(self.dates[idx[target_row]])[:10]
        bundle = dict(members=[dict(model=model, variant='engineered')],
                      calibrator=None, thresholds={'f1': .4, 'accuracy': .6},
                      feature_names={'engineered': names}, climate=np.zeros((365, 2)),
                      sites=np.array(['A', 'B']), lat=self.lat, lon=self.lon)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            joblib.dump(bundle, root/'ongoing_lead1_test_bundle.joblib')
            (root/'metrics.json').write_text(json.dumps({'experiments': {
                'ongoing_lead1': {'selection': {'f1_champion': 'test'}}}}))
            data = root/'data.npz'
            np.savez(data, x=self.raw, dates=self.dates, sites=bundle['sites'],
                     lat=self.lat, lon=self.lon)
            actual = forecast(data, root, target)
            expected = model.predict_proba(x[target_row])[:, 1]
            np.testing.assert_allclose(actual['probability'], expected)
            changed = self.raw.copy()
            changed[idx[target_row]:] = 999
            np.savez(data, x=changed, dates=self.dates, sites=bundle['sites'],
                     lat=self.lat, lon=self.lon)
            np.testing.assert_allclose(forecast(data, root, target)['probability'], expected)

    def test_cache_transfer_reuses_only_identical_candidate_and_rejects_feature_change(self):
        source_path = Path(model_search.__file__)
        snapshot = source_path.read_text()
        specs = candidate_specs()[:2]
        changed_spec = copy.deepcopy(specs[1])
        changed_spec['params']['max_iter'] = 999
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'old'
            (old/'cache').mkdir(parents=True)
            data = root/'data.npz'
            data.write_bytes(b'fixed data fixture')
            hashes = {str(source_path): hashlib.sha256(snapshot.encode()).hexdigest(),
                      'data': hashlib.sha256(data.read_bytes()).hexdigest()}
            manifest = dict(hashes=hashes, candidates=[specs[0], changed_spec], versions={},
                            leads=[1], tasks=['ongoing'], fingerprint='fixture')
            (old/'model_search_source.py').write_text(snapshot)
            (old/'manifest.json').write_text(json.dumps(manifest))
            for spec in specs:
                np.savez(old/'cache'/f'ongoing_lead1_{spec["name"]}_oof.npz', probability=[.2])
            transfer(old, root/'new', data)
            self.assertEqual(len(list((root/'new'/'cache').glob('*.npz'))), 1)
            changed_source = snapshot.replace("'hot_streak_capped30'", "'different_feature'")
            (old/'model_search_source.py').write_text(changed_source)
            manifest['hashes'][str(source_path)] = hashlib.sha256(changed_source.encode()).hexdigest()
            (old/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError, 'engineered_features'):
                transfer(old, root/'rejected', data)


if __name__ == '__main__':
    unittest.main()
