"""Development-selected forecast refinements; explicitly exploratory reused-test study."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .download_issued import LEADS, digest, target_dates
from .issued_data import (adjusted_risk, eligible_targets, event_rule, offsets,
                          trajectory, validate_timeline, verification)
from .issued_evaluation import FREEZE, evaluate
from .research_data import day_index
from .validation_audit import dependence_intervals, reliability

PROTOCOL = Path('IMPROVEMENT_STUDY_PROTOCOL.md')
STATUS = 'Exploratory amendment after inspecting original test; not prospective or an untouched test.'
CANDIDATES = [dict(name='raw_gfs', kind='raw')]
CANDIDATES += [dict(name=f'error_scale_{scale:g}', kind='empirical', scale=scale)
               for scale in (0.25, 0.5, 0.75, 1.0)]
CANDIDATES += [dict(name=f'logistic_C{c:g}_{weight or "unweighted"}', kind='logistic',
                   c=c, class_weight=weight) for c in (0.1, 1.0) for weight in (None, 'balanced')]


def purged_folds(dates, folds=5, purge_days=7):
    """Disjoint contiguous validation blocks; training excludes calendar neighbors."""
    dates = np.asarray(dates).astype('datetime64[D]')
    if len(dates) < folds or np.isnat(dates).any() or (np.diff(dates).astype(int) <= 0).any():
        raise ValueError('Need unique ordered development dates')
    result = []
    for valid in np.array_split(np.arange(len(dates)), folds):
        distance = np.abs((dates[:, None] - dates[valid][None, :]).astype(int))
        train = np.flatnonzero((distance > purge_days).all(axis=1))
        if len(train) < 20:
            raise ValueError('Fewer than 20 purged development training dates')
        result.append((train, valid))
    return result


def margin_features(forecast, threshold):
    """Run-only features, (date, location, feature), no truth or residual inputs."""
    margin = np.asarray(forecast, dtype=float) - np.asarray(threshold, dtype=float)
    if margin.ndim != 3 or not np.isfinite(margin).all():
        raise ValueError('Expected finite aligned date/location/event-day arrays')
    return np.concatenate([margin, margin.min(-1, keepdims=True), margin.max(-1, keepdims=True),
                           margin.mean(-1, keepdims=True), margin.std(-1, keepdims=True),
                           np.diff(margin, axis=-1)], axis=-1)


def choose_macro_cutoff(truth, probability):
    """Exact development macro-site F1 optimum; conservative tie resolution."""
    y, p = np.asarray(truth, dtype=bool), np.asarray(probability, dtype=float)
    if y.shape != p.shape or y.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('Expected aligned binary targets and probabilities')
    thresholds = np.unique(np.r_[p.ravel(), 1.])
    best = (-1., 0.)
    for cutoff in thresholds:
        pred = p >= cutoff
        tp = (pred & y).sum(axis=0)
        denominator = pred.sum(axis=0) + y.sum(axis=0)
        score = float(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float),
                                where=denominator > 0).mean())
        if score >= best[0] - 1e-12:
            best = (score, float(cutoff))
    return best[1]


def fit_candidate(spec, forecast, threshold, truth, task):
    if spec['kind'] == 'raw':
        return dict(spec=spec, task=task)
    if spec['kind'] == 'empirical':
        raise ValueError('Empirical candidate requires explicit development error bank')
    features = margin_features(forecast, threshold)
    labels = np.asarray(truth).ravel()
    if len(np.unique(labels)) < 2:
        return dict(spec=spec, task=task, constant=float(labels.mean()))
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=spec['c'], class_weight=spec['class_weight'], max_iter=1000, random_state=42))
    model.fit(features.reshape(-1, features.shape[-1]), labels)
    return dict(spec=spec, task=task, estimator=model)


def predict_candidate(bundle, forecast, threshold):
    spec, task = bundle['spec'], bundle['task']
    if spec['kind'] == 'raw':
        return event_rule(forecast, threshold, task).astype(float)
    if spec['kind'] == 'empirical':
        return adjusted_risk(forecast, threshold, bundle['error_bank'] * spec['scale'], task)
    if 'constant' in bundle:
        return np.full(forecast.shape[:2], bundle['constant'])
    features = margin_features(forecast, threshold)
    return bundle['estimator'].predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(features.shape[:2])


def crossfit_candidate(spec, forecast, threshold, truth, error_bank, task, folds):
    probability = np.full(np.asarray(truth).shape, np.nan, dtype=float)
    for train, valid in folds:
        if spec['kind'] == 'empirical':
            bundle = dict(spec=spec, task=task, error_bank=error_bank[train])
        else:
            bundle = fit_candidate(spec, forecast[train], threshold[train], truth[train], task)
        probability[valid] = predict_candidate(bundle, forecast[valid], threshold[valid])
    if not np.isfinite(probability).all():
        raise ValueError('Cross-fitting failed to produce every development prediction')
    return probability


def select_development(truth, probabilities):
    """Accepts development data only; never receives test labels or test scores."""
    entries = []
    for spec, probability in zip(CANDIDATES, probabilities):
        cutoff = .5 if spec['kind'] == 'raw' else choose_macro_cutoff(truth, probability)
        scores = evaluate(truth, probability, cutoff)
        entries.append(dict(candidate=spec, cutoff=cutoff, development=scores))
    best = max(range(len(entries)), key=lambda i: (
        entries[i]['development']['macro_site_f1'], -entries[i]['development']['brier'], -i))
    return entries, best


def thresholds_for(table, part, task, n_sites):
    return table[day_index(part['valid_dates'].ravel())].reshape(
        len(part['target']), len(offsets(task)), n_sites).transpose(0, 2, 1)


def forecast(issue, lead=1, task='ongoing', data='data/issued_forecasts', results='results/improvement_study'):
    """Reconstruct using a saved development-only bundle; verification arrays unused."""
    bundle = joblib.load(Path(results) / f'{task}_lead{lead}_selected.joblib')
    issue = np.datetime64(issue, 'ns')
    target = issue + np.timedelta64(lead, 'D')
    if issue <= bundle['original_development_available_at']:
        raise ValueError('Issue precedes original development availability boundary')
    if int(str(target)[5:7]) not in (5, 6, 7, 8, 9):
        raise ValueError('Model is evaluated for May–September only')
    with np.load(Path(data) / 'daily_runs.npz') as archive:
        if archive['sites'].tolist() != bundle['sites']:
            raise ValueError('Site order differs from development')
        run_only = {key: archive[key] for key in ('initializations', 'valid_dates', 'forecast_tmax')}
    part = trajectory(run_only, np.array([target]), lead, task)
    threshold = thresholds_for(bundle['reference_table'], part, task, len(bundle['sites']))
    probability = predict_candidate(bundle, part['forecast'], threshold)[0]
    return pd.DataFrame(dict(site=bundle['sites'], task=task, lead_days=lead,
                             initialization=str(part['initialization'][0]), issue_time=str(issue),
                             target_date=str(target)[:10], method=bundle['spec']['name'],
                             probability=probability, cutoff=bundle['cutoff'],
                             predicted_event=probability >= bundle['cutoff'],
                             study_status=STATUS, designed_at_utc=bundle['designed_at_utc']))


def run(data='data/issued_forecasts', output='results/improvement_study'):
    root, out = Path(data), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    if digest(root / 'daily_runs.npz') != manifest['prepared_sha256']:
        raise ValueError('Prepared forecast archive checksum differs')
    if digest('ISSUED_FORECAST_PROTOCOL.md') != manifest['protocol_sha256']:
        raise ValueError('Original timing protocol differs from archive provenance')
    archive = np.load(root / 'daily_runs.npz')
    climate = np.load('results/research/climatology.npz')
    if not np.array_equal(archive['sites'], climate['sites']):
        raise ValueError('Climatology site order differs')
    sites, table = archive['sites'], climate['table']
    dates, coverage = {}, {}
    for split, nominal in target_dates().items():
        dates[split], excluded = eligible_targets(archive, nominal)
        coverage[split] = dict(retained=len(dates[split]), excluded=excluded)
    folds = purged_folds(dates['development'])
    designed_at = pd.Timestamp.now(tz='UTC').isoformat()
    original_report = json.loads(Path('results/issued_forecasts/metrics.json').read_text())
    report = dict(status=STATUS, designed_at_utc=designed_at,
                  protocol_sha256=digest(PROTOCOL), archive_sha256=manifest['prepared_sha256'],
                  original_metrics_sha256=digest('results/issued_forecasts/metrics.json'),
                  climatology_sha256=digest('results/research/climatology.npz'),
                  original_development_available_at=str(FREEZE),
                  development='2026-05-01 to 2026-06-22', test='2026-07-08 to 2026-09-30',
                  selection='Five contiguous development folds, seven-calendar-day purge; macro-site F1, then Brier.',
                  coverage=coverage, folds=[dict(training_dates=dates['development'][tr].astype(str).tolist(),
                                              validation_dates=dates['development'][va].astype(str).tolist())
                                          for tr, va in folds], experiments={})
    comparisons, region_rows = [], []
    for lead in LEADS:
        for task in ('ongoing', 'onset'):
            name = f'{task}_lead{lead}'
            dev = trajectory(archive, dates['development'], lead, task)
            test = trajectory(archive, dates['test'], lead, task)
            validate_timeline(dev, test, FREEZE)
            dev_threshold = thresholds_for(table, dev, task, len(sites))
            dev_actual = verification(archive, dev['valid_dates'])
            dev_truth = event_rule(dev_actual, dev_threshold, task)
            errors = dev_actual - dev['forecast']
            probabilities = [crossfit_candidate(spec, dev['forecast'], dev_threshold, dev_truth,
                                               errors, task, folds) for spec in CANDIDATES]
            entries, selected = select_development(dev_truth, probabilities)
            # Persist the development-only choice before consulting test verification.
            (out / f'{name}_selection.json').write_text(json.dumps(dict(
                status=STATUS, designed_at_utc=designed_at, selected=entries[selected],
                candidates=entries, protocol_sha256=report['protocol_sha256']), indent=2))
            spec = CANDIDATES[selected]
            selected_bundle = (dict(spec=spec, task=task, error_bank=errors) if spec['kind'] == 'empirical'
                               else fit_candidate(spec, dev['forecast'], dev_threshold, dev_truth, task))
            selected_bundle.update(reference_table=table, sites=sites.tolist(), lead=lead,
                                   cutoff=entries[selected]['cutoff'], designed_at_utc=designed_at,
                                   original_development_available_at=FREEZE,
                                   protocol_sha256=report['protocol_sha256'],
                                   archive_sha256=report['archive_sha256'], status=STATUS)
            joblib.dump(selected_bundle, out / f'{name}_selected.joblib')
            test_threshold = thresholds_for(table, test, task, len(sites))
            test_truth = event_rule(verification(archive, test['valid_dates']), test_threshold, task)
            test_predictions = []
            for i, candidate in enumerate(CANDIDATES):
                bundle = (dict(spec=candidate, task=task, error_bank=errors)
                          if candidate['kind'] == 'empirical' else fit_candidate(
                              candidate, dev['forecast'], dev_threshold, dev_truth, task))
                probability = predict_candidate(bundle, test['forecast'], test_threshold)
                test_predictions.append(probability)
                entries[i]['test'] = evaluate(test_truth, probability, entries[i]['cutoff'])
                comparisons.append(dict(experiment=name, method=candidate['name'], selected=i == selected,
                                        development_macro_f1=entries[i]['development']['macro_site_f1'],
                                        **entries[i]['test']))
            chosen = test_predictions[selected]
            baseline = np.broadcast_to(dev_truth.mean(axis=0), test_truth.shape)
            regional = []
            for s, site in enumerate(sites):
                row = dict(experiment=name, site=str(site), method=spec['name'],
                           positive_days=int(test_truth[:, s].sum()),
                           development_positive_days=int(dev_truth[:, s].sum()),
                           **evaluate(test_truth[:, s:s+1], chosen[:, s:s+1], entries[selected]['cutoff']))
                region_rows.append(row)
                regional.append(dict(site=str(site), **reliability(test_truth[:, s], chosen[:, s], baseline[:, s])))
            uncertainty = dependence_intervals(test_truth, chosen, entries[selected]['cutoff'],
                                               test_predictions[0], .5, dates['test'])
            original_adjusted = original_report['experiments'][name]['methods']['joint_error_adjusted']
            report['experiments'][name] = dict(selected_method=spec['name'], cutoff=entries[selected]['cutoff'],
                development_positive_days=int(dev_truth.sum()), test_positive_days=int(test_truth.sum()),
                candidates=entries, reliability=reliability(test_truth, chosen, baseline),
                regional_reliability=regional, selected_vs_raw_uncertainty=uncertainty,
                original_fixed_adjusted=dict(cutoff=original_adjusted['cutoff'], test=original_adjusted['test'],
                    note='Unmodified original study; its crossfit exclusion/cutoff policy differs from this amendment.'))
            np.savez_compressed(out / f'{name}_predictions.npz', sites=sites, target_dates=dates['test'],
                truth=test_truth, probability=chosen, cutoff=entries[selected]['cutoff'],
                raw_gfs=test_predictions[0], development_probability=probabilities[selected],
                development_truth=dev_truth, development_dates=dates['development'],
                forecast_trajectory=test['forecast'], threshold_trajectory=test_threshold,
                **{candidate['name']: probability for candidate, probability in zip(CANDIDATES[1:], test_predictions[1:])})
            print(f"{name}: selected {spec['name']}; test F1={entries[selected]['test']['f1']:.4f}; raw={entries[0]['test']['f1']:.4f}", flush=True)
    pd.DataFrame(comparisons).to_csv(out / 'comparison.csv', index=False)
    pd.DataFrame(region_rows).to_csv(out / 'regional_metrics.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / 'PROTOCOL_AT_EVALUATION.md').write_text(PROTOCOL.read_text())
    write_report(report, out)
    check_artifacts(data, output)
    return report


def check_artifacts(data='data/issued_forecasts', output='results/improvement_study'):
    """Check all 3,570 selected outputs against run-only inference and original labels."""
    root = Path(output)
    report = json.loads((root / 'metrics.json').read_text())
    if digest(PROTOCOL) != report['protocol_sha256']:
        raise ValueError('Amendment protocol differs from evaluated version')
    if digest(Path(data) / 'daily_runs.npz') != report['archive_sha256']:
        raise ValueError('Forecast archive differs from evaluated source')
    if digest('results/issued_forecasts/metrics.json') != report['original_metrics_sha256']:
        raise ValueError('Original benchmark metrics changed')
    with np.load(Path(data) / 'daily_runs.npz') as archive:
        runs = {key: archive[key] for key in ('initializations', 'valid_dates', 'forecast_tmax')}
    checked = 0
    for name, entry in report['experiments'].items():
        saved = np.load(root / f'{name}_predictions.npz')
        original = np.load(Path('results/issued_forecasts') / f'{name}_predictions.npz')
        for key in ('sites', 'target_dates', 'truth'):
            if not np.array_equal(saved[key], original[key]):
                raise ValueError(f'{name}: original comparison cases differ: {key}')
        if not np.array_equal(saved['raw_gfs'], original['raw_gfs']):
            raise ValueError(f'{name}: original raw baseline differs')
        bundle = joblib.load(root / f'{name}_selected.joblib')
        if bundle['spec']['name'] != entry['selected_method'] or bundle['cutoff'] != entry['cutoff']:
            raise ValueError(f'{name}: selected inference bundle differs')
        part = trajectory(runs, saved['target_dates'], bundle['lead'], bundle['task'])
        threshold = thresholds_for(bundle['reference_table'], part, bundle['task'], len(bundle['sites']))
        probability = predict_candidate(bundle, part['forecast'], threshold)
        if not np.allclose(probability, saved['probability'], rtol=0, atol=1e-12):
            raise ValueError(f'{name}: saved inference does not reproduce probabilities')
        original_fixed = report['experiments'][name]['original_fixed_adjusted']
        fixed_scores = evaluate(saved['truth'], original['joint_error_adjusted'], original_fixed['cutoff'])
        if fixed_scores != original_fixed['test']:
            raise ValueError(f'{name}: original adjusted metrics do not reproduce')
        checked += probability.size
    print(f'Checked {checked} selected probabilities against forecast-only inference and original labels.', flush=True)
    return checked


def write_report(report, out):
    lines = ['# Exploratory forecast-refinement study', '', report['status'], '',
             'The new methods were designed after reviewing original test weaknesses. '
             'Selection uses development only, but this reused test is not untouched evidence. '
             'Original published benchmarks remain unchanged.', '',
             '| Task / lead | Development-selected method | Selected test F1 | Original adjusted F1 | Raw test F1 | Selected macro-site F1 | Test positives |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for name, entry in report['experiments'].items():
        chosen = next(c for c in entry['candidates'] if c['candidate']['name'] == entry['selected_method'])
        raw = entry['candidates'][0]['test']
        lines.append(f"| {name} | {entry['selected_method']} | {chosen['test']['f1']:.4f} | {entry['original_fixed_adjusted']['test']['f1']:.4f} | {raw['f1']:.4f} | {chosen['test']['macro_site_f1']:.4f} | {entry['test_positive_days']} |")
    lines += ['', '## Interpretation', '',
              '- Selection and probability cutoffs use five development blocks, with seven-day purging and whole locations kept together.',
              '- Development scores are used for tuning and are not unbiased estimates. Every candidate test result is retained in `comparison.csv`.',
              '- Classification and probability quality are separate: see Brier, reliability bins, bin support, and per-location metrics in `metrics.json`.',
              '- No candidate is promoted because it wins on test. A selected method can still worsen the original result.',
              '- Original adjusted scores are copied unchanged for context. The new error_scale_1 candidate uses different purged folds and macro-site cutoff tuning, so it is not the original adjustment.',
              '- The empirical error scales are sensitivity/shrinkage experiments, not a physical ensemble. Logistic probabilities are not guaranteed calibrated.',
              '- Per-site development event frequency is the frozen probability baseline for Brier skill; this differs from the original historical-climatology baseline.',
              '- Paired 3/7/14-day block intervals compare the selected method with raw GFS; they condition on this fitted model and one season.',
              '- Seven grid locations, ERA5 verification, assumed historic receipt timing, and seasonal transfer remain limitations. New independent seasons and station observations are required.',
              '', '## Reproduction and inference', '', '```bash',
              'python3 -m heatcast.improvement_study',
              'python3 -m heatcast.improvement_study --check',
              'python3 -m unittest discover -s tests -p test_improvement_study.py -v',
              'python3 -m heatcast.improvement_study --issue 2026-07-25 --lead 7 --task onset --forecast-output /tmp/improved-warning.csv',
              '```', '', 'Saved bundles contain development-only fitted quantities and historical thresholds. '
              'The inference path supplies forecast arrays only; test verification arrays never enter prediction. '
              'Archived inference is a reconstruction, not proof the new model issued a warning in 2026.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/issued_forecasts')
    parser.add_argument('--output', default='results/improvement_study')
    parser.add_argument('--issue')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--lead', type=int, choices=LEADS, default=1)
    parser.add_argument('--task', choices=['ongoing', 'onset'], default='ongoing')
    parser.add_argument('--forecast-output', default='improved_warning.csv')
    args = parser.parse_args()
    if args.check:
        check_artifacts(args.data, args.output)
    elif args.issue:
        frame = forecast(args.issue, args.lead, args.task, args.data, args.output)
        frame.to_csv(args.forecast_output, index=False)
        print(frame.to_string(index=False))
    else:
        run(args.data, args.output)


if __name__ == '__main__':
    main()
