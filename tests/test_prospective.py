from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import json

import numpy as np

from heatcast.prospective import forecast_rows, immutable_json, verify


class ProspectiveTests(unittest.TestCase):
    def test_actual_lead_and_out_of_season_abstention(self):
        days = np.arange('2026-11-02', '2026-11-14', dtype='datetime64[D]')
        receipt = datetime(2026, 11, 3, 18, 30, tzinfo=timezone.utc)
        rows, hashes = forecast_rows(days, np.full((12, 2), 20), ['A', 'B'], receipt, 'unused')
        self.assertEqual(len(rows), 12)
        self.assertFalse(hashes)
        for row in rows:
            self.assertGreaterEqual(row['actual_lead_hours'], row['minimum_lead_days'] * 24)
            self.assertIsNone(row['probability'])
            self.assertIsNone(row['predicted_event'])
        self.assertEqual(rows[0]['target_date'], '2026-11-05')

    def test_missing_horizon_fails(self):
        days = np.arange('2026-11-02', '2026-11-05', dtype='datetime64[D]')
        with self.assertRaisesRegex(ValueError, 'trajectory'):
            forecast_rows(days, np.full((3, 1), 20), ['A'], datetime(2026, 11, 3, tzinfo=timezone.utc), 'unused')

    def test_capture_files_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'forecast.json'
            immutable_json(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                immutable_json(path, {'a': 2})
            self.assertIn('1', path.read_text())

    def test_verify_refuses_future_outcome_without_network(self):
        from heatcast.prospective import sha
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / 'response.json').write_bytes(b'{}')
            immutable_json(directory / 'forecast.json', dict(response_sha256=sha(b'{}'), warnings=[dict(trajectory_dates=['2099-05-01'])]))
            session = Mock()
            with self.assertRaisesRegex(ValueError, 'not yet available'):
                verify(directory, session)
            session.get.assert_not_called()

    def test_summer_inference_uses_bundle_only(self):
        receipt = datetime(2027, 6, 1, 12, tzinfo=timezone.utc)
        days = np.arange('2027-05-31', '2027-06-12', dtype='datetime64[D]')
        bundle = dict(sites=['A'], freeze=np.datetime64('2026-06-30', 'ns'),
                      reference_table=np.full((365, 1), 25), cutoff=.5)
        bundle['error_bank'] = np.zeros((25, 1, 3))
        def load(path):
            result = dict(bundle)
            result['error_bank'] = np.zeros((25, 1, 4 if 'onset' in str(path) else 3))
            return result
        with patch('heatcast.prospective.joblib.load', side_effect=load), patch.object(Path, 'read_bytes', return_value=b'bundle'):
            rows, hashes = forecast_rows(days, np.full((12, 1), 30), ['A'], receipt, 'bundles')
        self.assertEqual(len(hashes), 6)
        self.assertTrue(all(row['probability'] == (1 if row['task'] == 'ongoing' else 0) for row in rows))
        self.assertTrue(all(row['local_hot_threshold_c'] == [25] * len(row['trajectory_dates']) for row in rows))

    def test_verification_uses_captured_threshold_and_preserves_forecast(self):
        from heatcast.prospective import sha
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / 'response.json').write_bytes(b'{}')
            record = dict(response_sha256=sha(b'{}'), sites=['A'], request=dict(
                latitude='48.75', longitude='2.25', timezone='GMT', elevation='nan', cell_selection='nearest'),
                warnings=[dict(site='A', task='onset', minimum_lead_days=1,
                    target_date='2026-05-02', trajectory_dates=['2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04'],
                    forecast_tmax_c=[20, 30, 30, 30], local_hot_threshold_c=[25] * 4,
                    probability=.8, predicted_event=True)])
            immutable_json(directory / 'forecast.json', record)
            original = (directory / 'forecast.json').read_bytes()
            payload = dict(latitude=48.75, longitude=2.25, utc_offset_seconds=0,
                           daily_units={'temperature_2m_max': '°C'}, daily=dict(
                               time=record['warnings'][0]['trajectory_dates'], temperature_2m_max=[20, 30, 30, 30]))
            response = Mock(content=json.dumps(payload).encode())
            response.json.return_value = payload
            session = Mock()
            session.get.return_value = response
            with patch('heatcast.prospective.utc_now', return_value=datetime(2026, 6, 1, tzinfo=timezone.utc)), \
                 patch('heatcast.prospective.np.load', side_effect=AssertionError('No verification-time model or reference access')):
                result = verify(directory, session)
            self.assertTrue(result['results'][0]['truth'])
            self.assertEqual(result['results'][0]['trajectory_mae_c'], 0)
            self.assertEqual(original, (directory / 'forecast.json').read_bytes())
            self.assertEqual(len(list(directory.glob('verification-*.json'))), 2)


if __name__ == '__main__':
    unittest.main()
