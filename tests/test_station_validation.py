import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from heatcast.station_validation import read_station, station_reference, station_cases


class StationValidationTests(unittest.TestCase):
    def test_incomplete_reference_is_rejected(self):
        frame = pd.DataFrame(dict(date=pd.date_range('1981-01-01', periods=365), quality_ok=True, tmax_c=20))
        with self.assertRaisesRegex(ValueError, '1982'):
            station_reference(frame)

    def test_reference_excludes_verification_years(self):
        dates = pd.date_range('1981-01-01', '2010-12-31')
        frame = pd.DataFrame(dict(date=dates, quality_ok=True, tmax_c=20))
        future = pd.DataFrame(dict(date=pd.date_range('2026-01-01', periods=365), quality_ok=True, tmax_c=60))
        np.testing.assert_equal(station_reference(pd.concat([frame, future])), np.full(365, 20))

    def test_duplicate_and_quality_flags(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'station.csv'
            path.write_text('site,station_id,date,tmax_c,quality_ok\nA,X,2026-05-01,30,maybe\n')
            with self.assertRaisesRegex(ValueError, 'quality_ok'):
                read_station(path)
            path.write_text('site,station_id,date,tmax_c,quality_ok\nA,X,2026-05-01,30,true\nA,X,2026-05-01,30,true\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_station(path)

    def test_missing_or_rejected_event_days_excluded(self):
        z = dict(sites=np.array(['A']), target_dates=np.array(['2026-05-02'], dtype='datetime64[D]'),
                 forecast_trajectory=np.array([[[20, 30, 30, 30]]]), joint_error_adjusted=np.array([[.8]]), cutoff=np.array(.5))
        frame = pd.DataFrame(dict(site='A', station_id='X', date=pd.date_range('2026-05-01', periods=4),
                                  tmax_c=[20, 30, 30, 30], quality_ok=[True, True, False, True]))
        cases, excluded = station_cases(z, frame, 'onset', {'A': np.full(365, 25)})
        self.assertFalse(cases)
        self.assertEqual(len(excluded), 1)
        frame['quality_ok'] = True
        cases, excluded = station_cases(z, frame, 'onset', {'A': np.full(365, 25)})
        self.assertTrue(cases[0]['station_event'])
        self.assertEqual(cases[0]['target_temperature_error_c'], 0)


if __name__ == '__main__':
    unittest.main()
