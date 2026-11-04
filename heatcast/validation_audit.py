"""Read-only reliability, event and dependence audits of saved held-out forecasts.

No fitting, cutoff selection or model replacement occurs in this module.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def aligned_arrays(truth, probability, dates):
    y, p = np.asarray(truth), np.asarray(probability, dtype=float)
    days = np.asarray(dates).astype('datetime64[D]')
    if y.ndim != 2 or y.shape != p.shape or y.shape[0] != len(days) or not y.size:
        raise ValueError('Expected aligned day-by-site arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Truth must be binary and probabilities finite in [0, 1]')
    if np.isnat(days).any() or (np.diff(days).astype(int) <= 0).any():
        raise ValueError('Dates must be unique and increasing')
    return y.astype(bool), p, days


def reliability(truth, probability, baseline=None, bins=10):
    """Fixed equal-width bins; empty bins are explicit, not zero observed risk."""
    y, p = np.asarray(truth).ravel(), np.asarray(probability, dtype=float).ravel()
    if y.size == 0 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected nonempty aligned binary truth and probabilities')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or bins < 1:
        raise ValueError('Invalid probabilities or bin count')
    membership = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        keep = membership == i
        rows.append(dict(lower=i / bins, upper=(i + 1) / bins, count=int(keep.sum()),
                         mean_probability=float(p[keep].mean()) if keep.any() else None,
                         observed_frequency=float(y[keep].mean()) if keep.any() else None))
    brier = float(np.mean((p - y) ** 2))
    result = dict(n=int(y.size), positives=int(y.sum()), brier=brier, bins=rows,
                  expected_calibration_error=float(sum(
                      row['count'] / y.size * abs(row['mean_probability'] - row['observed_frequency'])
                      for row in rows if row['count'])),
                  note='Descriptive fixed-bin reliability; small bin support limits interpretation.')
    if baseline is not None:
        base = np.asarray(baseline, dtype=float).ravel()
        if base.shape != y.shape or not np.isfinite(base).all() or ((base < 0) | (base > 1)).any():
            raise ValueError('Invalid matched baseline')
        reference = float(np.mean((base - y) ** 2))
        result.update(baseline_brier=reference, brier_skill_score=1 - brier / reference if reference > 0 else None)
    return result


def event_episodes(truth, prediction, dates, sites):
    """Collapse ongoing positive days into episodes; never join seasonal gaps.

    This measures detection of third-and-later hot days, not warning before onset.
    Onset labels should instead be scored as individual exact-date events.
    """
    y, p, days = aligned_arrays(truth, np.asarray(prediction, dtype=float), dates)
    if not np.isin(p, [0, 1]).all() or y.shape[1] != len(sites):
        raise ValueError('Expected binary predictions and aligned sites')
    pred = p.astype(bool)
    rows = []
    for s, site in enumerate(sites):
        positives = np.flatnonzero(y[:, s])
        if not len(positives):
            continue
        cuts = np.flatnonzero(np.diff(days[positives]).astype(int) != 1) + 1
        for episode in np.split(positives, cuts):
            detected = episode[pred[episode, s]]
            rows.append(dict(site=str(site), first_ongoing_day=str(days[episode[0]]),
                             last_ongoing_day=str(days[episode[-1]]), ongoing_days=len(episode),
                             hit_days=int(pred[episode, s].sum()), detected=bool(len(detected)),
                             days_to_first_detected_ongoing_day=(
                                 int((days[detected[0]] - days[episode[0]]).astype(int)) if len(detected) else None)))
    return rows


def dependence_intervals(truth, probability, cutoff, baseline, baseline_cutoff, dates,
                         block_lengths=(3, 7, 14), repeats=500, seed=42):
    """Paired full-length moving blocks, stratified by contiguous season.

    Blocks do not cross date gaps; each season retains its original sample size.
    Whole location fields stay together. Intervals condition on fitted models.
    """
    y, p, days = aligned_arrays(truth, probability, dates)
    if not np.isfinite([cutoff, baseline_cutoff]).all() or not 0 <= cutoff <= 1 or not 0 <= baseline_cutoff <= 1:
        raise ValueError('Cutoffs must be finite in [0, 1]')
    base = np.asarray(baseline)
    if base.shape != y.shape or not np.isfinite(base).all() or ((base < 0) | (base > 1)).any():
        raise ValueError('Expected an aligned probability baseline')
    if repeats < 2:
        raise ValueError('Need at least two bootstrap repeats')
    def counts(pred):
        return np.stack([(pred & y).sum(1), (pred & ~y).sum(1), (~pred & y).sum(1)], axis=1)
    def f1(c):
        tp, fp, fn = c
        return float(2 * tp / max(2 * tp + fp + fn, 1))
    a, b = counts(p >= cutoff), counts(base >= baseline_cutoff)
    seasons = np.split(np.arange(len(days)), np.flatnonzero(np.diff(days).astype(int) != 1) + 1)
    result = []
    for length in block_lengths:
        if length < 1:
            raise ValueError('Block lengths must be positive')
        if any(len(season) < length for season in seasons):
            result.append(dict(block_days=length, available=False, reason='A contiguous segment is shorter than the block'))
            continue
        rng = np.random.default_rng(seed)
        scores, differences = [], []
        for _ in range(repeats):
            pieces = []
            for season in seasons:
                starts = rng.integers(0, len(season) - length + 1,
                                      size=int(np.ceil(len(season) / length)))
                positions = np.concatenate([np.arange(start, start + length) for start in starts])[:len(season)]
                pieces.append(season[positions])
            sample = np.concatenate(pieces)
            score = f1(a[sample].sum(0))
            scores.append(score)
            differences.append(score - f1(b[sample].sum(0)))
        result.append(dict(block_days=length, available=True, repeats=repeats,
                           f1_95_interval=np.quantile(scores, [.025, .975]).tolist(),
                           f1_difference_95_interval=np.quantile(differences, [.025, .975]).tolist()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    parser.add_argument('--output', default='results/validation_audit')
    parser.add_argument('--repeats', type=int, default=500)
    args = parser.parse_args()
    root, out = Path(args.results), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(scope='Descriptive audit of already inspected held-out results; no model selection.',
                  bootstrap='Paired full-length moving blocks within each contiguous season; whole spatial fields.',
                  experiments={})
    event_rows, bin_rows = [], []
    for study in ['research', 'issued_forecasts']:
        for task in ['ongoing', 'onset']:
            for lead in [1, 3, 7]:
                name = f'{task}_lead{lead}'
                path = root / study / f'{name}_predictions.npz'
                z = np.load(path)
                research = study == 'research'
                y = z['truth']
                dates = z['dates' if research else 'target_dates']
                cutoff = float(z['threshold' if research else 'cutoff'])
                p = z['probability' if research else 'joint_error_adjusted']
                base = z['persistence' if research else 'raw_gfs']
                climate = z['climatology' if research else 'historical_climatology']
                y, p, dates = aligned_arrays(y, p, dates)
                key = f'{study}/{name}'
                entry = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             reliability=reliability(y, p, climate),
                             baseline='temperature persistence' if research else 'raw GFS',
                             uncertainty=dependence_intervals(y, p, cutoff, base, .5, dates, repeats=args.repeats),
                             sites={})
                for s, site in enumerate(z['sites']):
                    local = reliability(y[:, s], p[:, s], climate[:, s])
                    entry['sites'][str(site)] = {k: v for k, v in local.items() if k != 'bins'}
                    for row in local['bins']:
                        bin_rows.append(dict(experiment=key, site=str(site), **row))
                if task == 'ongoing':
                    episodes = event_episodes(y, p >= cutoff, dates, z['sites'])
                    event_rows.extend(dict(experiment=key, **row) for row in episodes)
                    hits = sum(row['detected'] for row in episodes)
                    entry['episodes'] = dict(n=len(episodes), detected=hits, missed=len(episodes) - hits,
                                             detection_fraction=hits / len(episodes) if episodes else None,
                                             note='Episodes comprise third-and-later hot days; this is not advance-onset recall.')
                else:
                    entry['exact_onsets'] = dict(n=int(y.sum()), detected=int(((p >= cutoff) & y).sum()),
                                                false_alerts=int(((p >= cutoff) & ~y).sum()))
                report['experiments'][key] = entry
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame(event_rows).to_csv(out / 'ongoing_episodes.csv', index=False)
    pd.DataFrame(bin_rows).to_csv(out / 'reliability_bins.csv', index=False)
    lines = ['# Additional validation audit', '', report['scope'], '',
             'Brier skill compares with each experiment’s saved historical climatology probabilities. Negative skill means worse squared-probability error than that baseline.', '',
             '| Experiment | Brier skill | Event support | Detection |', '| --- | --- | --- | --- |']
    for key, entry in report['experiments'].items():
        support = entry.get('episodes', entry.get('exact_onsets'))
        detected = support['detected']
        lines.append(f"| {key} | {entry['reliability']['brier_skill_score']:.3f} | {support['n']} | {detected} |")
    lines.extend(['', 'Ongoing support counts episodes of labelled ongoing days. Onset support counts exact first days; these are different outcomes.', '',
                  'Intervals for 3-, 7- and 14-day blocks and site/bin support are in metrics.json and reliability_bins.csv. They do not measure retraining uncertainty or establish station accuracy.', '',
                  'The source benchmark cutoffs and artifacts are unchanged. These diagnostics are not a new untouched test.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Audited {len(report["experiments"])} experiments; outputs: {out}')


if __name__ == '__main__':
    main()
