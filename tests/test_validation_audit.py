import unittest
import numpy as np
from heatcast.validation_audit import reliability, event_episodes, dependence_intervals, aligned_arrays


class ValidationAuditTests(unittest.TestCase):
    def test_reliability_keeps_one_and_empty_bins(self):
        r = reliability([0, 1], [0, 1], [0.5, 0.5])
        self.assertEqual(r['brier_skill_score'], 1)
        self.assertEqual(sum(row['count'] for row in r['bins']), 2)
        self.assertEqual(r['bins'][-1]['count'], 1)
        self.assertIsNone(r['bins'][1]['observed_frequency'])

    def test_episodes_never_bridge_gaps_or_sites(self):
        dates = np.array(['2021-09-29', '2021-09-30', '2022-05-01', '2022-05-02'])
        y = np.ones((4, 2), dtype=bool)
        p = np.array([[0, 1], [1, 0], [0, 0], [0, 1]], dtype=bool)
        rows = event_episodes(y, p, dates, ['A', 'B'])
        self.assertEqual(len(rows), 4)
        self.assertEqual(sum(row['detected'] for row in rows), 3)
        self.assertEqual(rows[0]['days_to_first_detected_ongoing_day'], 1)

    def test_bootstrap_is_paired_and_deterministic(self):
        dates = np.arange('2021-05-01', '2021-05-31', dtype='datetime64[D]')
        y = np.arange(60).reshape(30, 2) % 3 == 0
        p = y * 0.8 + 0.1
        r = dependence_intervals(y, p, .5, p, .5, dates, repeats=20)
        for row in r:
            self.assertEqual(row['f1_difference_95_interval'], [0, 0])
        self.assertEqual(r, dependence_intervals(y, p, .5, p, .5, dates, repeats=20))

    def test_short_segments_are_reported_not_bridged(self):
        dates = np.array(['2021-09-30', '2022-05-01'])
        y = np.array([[0], [1]])
        r = dependence_intervals(y, y, .5, y, .5, dates, repeats=10)
        self.assertTrue(all(not row['available'] for row in r))

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            reliability([0, 1], [0, np.nan])
        with self.assertRaises(ValueError):
            aligned_arrays([[1], [0]], [[.5], [.6]], ['2021-01-02', '2021-01-01'])


if __name__ == '__main__':
    unittest.main()
