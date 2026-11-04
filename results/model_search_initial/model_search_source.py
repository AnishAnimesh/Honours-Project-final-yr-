"""Exploratory, chronological tree-model and causal-feature comparison."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from .evaluation import metrics
from .research_data import (day_index, event_targets, fit_climatology, make_features,
                            macro_f1, validate_daily)


def engineered_features(raw, dates, threshold, lat, lon, lead):
    """All weather values end at issue; every rolling window is trailing."""
    idx, base, names = make_features(raw, dates, threshold, lat, lon, lead)
    keep = idx - lead >= 29
    idx, base = idx[keep], base[keep]
    issue = idx - lead
    margin = raw[:, :, 0] - threshold
    columns = [base]
    names = list(names)

    def add(name, values):
        columns.append(np.asarray(values, dtype='float32')[:, :, None])
        names.append(name)

    for lag in range(14):
        add(f'exact_margin_lag{lag}', margin[issue-lag])
    streak = np.zeros((len(idx), len(lat)), dtype='float32')
    continuing = np.ones_like(streak, dtype=bool)
    for lag in range(30):
        continuing &= margin[issue-lag] > 0
        streak += continuing
    add('hot_streak_capped30', streak)
    for length in [3, 7, 14]:
        history = np.stack([margin[issue-k] for k in range(length)], axis=0)
        nights = np.stack([raw[issue-k, :, 1] for k in range(length)], axis=0)
        add(f'margin_max_{length}', history.max(0))
        add(f'margin_min_{length}', history.min(0))
        add(f'margin_std_{length}', history.std(0))
        add(f'margin_slope_{length}', (history[0]-history[-1])/(length-1))
        add(f'night_mean_{length}', nights.mean(0))
        add(f'night_change_{length}', nights[0]-nights[-1])
    add('diurnal_range', raw[issue, :, 0]-raw[issue, :, 1])
    for length in [7, 14, 30]:
        add(f'precipitation_total_{length}',
            np.stack([raw[issue-k, :, 5] for k in range(length)]).sum(0))
        add(f'soil_change_{length}', raw[issue, :, 3]-raw[issue-length+1, :, 3])
    for site in range(len(lat)):
        add(f'site_indicator_{site}', np.broadcast_to(
            (np.arange(len(lat)) == site)[None, :], (len(idx), len(lat))))
    x = np.concatenate(columns, axis=-1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite engineered features')
    return idx, x, names


def temporal_folds(dates, idx):
    years = pd.DatetimeIndex(dates).year.to_numpy()
    return [(np.flatnonzero(years[idx+2] < start),
             np.flatnonzero((years[idx] >= start) & (years[idx+2] <= end)))
            for start, end in [(2011, 2012), (2013, 2014), (2015, 2016)]]


def threshold_policy(y, p, policy='f1', min_recall=.85):
    """Pooled threshold; F1 gives equal weight to sites; accuracy constrains recall."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p)
    thresholds = np.unique(np.r_[0., np.quantile(p, np.linspace(0, 1, 201)),
                                  np.nextafter(1., 2.)])
    pred = p[None, :, :] >= thresholds[:, None, None]
    tp = (pred & y).sum(axis=1)
    fp = (pred & ~y).sum(axis=1)
    fn = (~pred & y).sum(axis=1)
    mf = (2*tp/np.maximum(2*tp+fp+fn, 1)).mean(axis=1)
    recall = tp.sum(1)/max(int(y.sum()), 1)
    accuracy = (pred == y).mean(axis=(1, 2))
    if policy == 'f1':
        order = np.lexsort((thresholds, accuracy, mf))
    elif policy == 'accuracy':
        score = np.where(recall >= min_recall, accuracy, -np.inf)
        order = np.lexsort((thresholds, mf, score))
    else:
        raise ValueError('Unknown threshold policy')
    chosen = int(order[-1])
    return float(thresholds[chosen])


def score(y, p, cutoff):
    result = metrics(y, p, cutoff)
    tn, fp = result['confusion_matrix'][0]
    fn, tp = result['confusion_matrix'][1]
    specificity = tn/max(tn+fp, 1)
    result.update(macro_site_f1=macro_f1(y, p, cutoff), specificity=specificity,
                  balanced_accuracy=(specificity+tp/max(tp+fn, 1))/2,
                  false_alerts=fp, misses=fn)
    return result


def candidate_specs(trials=3, seed=42):
    rng = np.random.default_rng(seed)
    specs = []
    # Fixed-parameter feature ablation and original benchmark candidate set.
    for variant in ['temperature_only', 'no_soil', 'all_features']:
        for leaves in [7, 15]:
            specs.append(dict(name=f'original_{variant}_{leaves}', family='hist',
                              variant=variant, params=dict(max_iter=160,
                              max_leaf_nodes=leaves, learning_rate=.07,
                              l2_regularization=10)))
    specs.append(dict(name='engineered_matched_15', family='hist', variant='engineered',
                      params=dict(max_iter=160, max_leaf_nodes=15,
                                  learning_rate=.07, l2_regularization=10)))
    for family in ['hist', 'xgboost', 'catboost', 'lightgbm', 'extra_trees', 'random_forest']:
        for trial in range(trials):
            def pick(values):
                return rng.choice(values).item()
            if family == 'hist':
                params = dict(max_iter=pick([160, 300, 600]),
                              max_leaf_nodes=pick([7, 15, 31]),
                              learning_rate=pick([.03, .07, .1]),
                              min_samples_leaf=pick([20, 50, 100]),
                              l2_regularization=pick([1., 10., 30.]))
            elif family == 'xgboost':
                params = dict(n_estimators=pick([160, 300, 600]), max_depth=pick([3, 5, 7]),
                              learning_rate=pick([.03, .07, .1]),
                              min_child_weight=pick([5, 20, 50]),
                              subsample=pick([.7, 1.]), colsample_bytree=pick([.7, 1.]),
                              reg_lambda=pick([1., 10., 30.]))
            elif family == 'catboost':
                params = dict(iterations=pick([160, 300, 600]), depth=pick([4, 6, 8]),
                              learning_rate=pick([.03, .07, .1]),
                              l2_leaf_reg=pick([3., 10., 30.]))
            elif family == 'lightgbm':
                params = dict(n_estimators=pick([160, 300, 600]), num_leaves=pick([7, 15, 31]),
                              learning_rate=pick([.03, .07, .1]),
                              min_child_samples=pick([20, 50, 100]),
                              reg_lambda=pick([1., 10., 30.]), colsample_bytree=pick([.7, 1.]))
            else:
                params = dict(n_estimators=200, max_depth=pick([8, 14, 20]),
                              min_samples_leaf=pick([5, 20, 50]),
                              max_features=pick([.5, 1.]), class_weight=None)
            specs.append(dict(name=f'{family}_{trial}', family=family,
                              variant='engineered', params=params))
    return specs


def build_model(spec):
    params = dict(spec['params'])
    family = spec['family']
    if family == 'hist':
        return HistGradientBoostingClassifier(**params, early_stopping=False, random_state=42)
    if family == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**params, n_jobs=4, random_state=42, tree_method='hist',
                             eval_metric='logloss')
    if family == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, thread_count=4, random_seed=42,
                                  verbose=False, allow_writing_files=False)
    if family == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, n_jobs=4, random_state=42, verbosity=-1)
    cls = ExtraTreesClassifier if family == 'extra_trees' else RandomForestClassifier
    return cls(**params, n_jobs=4, random_state=42)


def logit(p):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def calibrated(calibrator, p):
    return calibrator.predict_proba(logit(p))[:, 1].reshape(p.shape)


def predict_bundle(bundle, features, policy='f1'):
    """Inference from feature arrays only; verification targets are never needed."""
    parts = []
    shape = None
    for member in bundle['members']:
        x = features[member['variant']]
        shape = x.shape[:2]
        parts.append(member['model'].predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(shape))
    p = np.mean(parts, axis=0)
    if bundle['calibrator'] is not None:
        p = calibrated(bundle['calibrator'], p)
    return p, p >= bundle['thresholds'][policy]


def difference_intervals(y, p, cutoff, baseline_p, baseline_cutoff, dates, repeats=300):
    """Full seven-day blocks, whole sites, resampled within each contiguous season."""
    cuts = np.r_[0, np.flatnonzero(np.diff(dates).astype('timedelta64[D]') !=
                                 np.timedelta64(1, 'D'))+1, len(dates)]
    rng = np.random.default_rng(42)
    pred, baseline = p >= cutoff, baseline_p >= baseline_cutoff
    deltas = []
    for _ in range(repeats):
        samples = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            length = min(7, hi-lo)
            starts = rng.integers(lo, hi-length+1, size=int(np.ceil((hi-lo)/length)))
            samples.extend((starts[:, None]+np.arange(length)).ravel()[:hi-lo])
        s = np.asarray(samples)
        def f1(a):
            tp = int((a & y[s]).sum())
            return 2*tp/max(int(a.sum()+y[s].sum()), 1)
        deltas.append([(pred[s] == y[s]).mean()-(baseline[s] == y[s]).mean(),
                       f1(pred[s])-f1(baseline[s])])
    return dict(method=f'{repeats} paired seasonal 7-day blocks; whole sites',
                accuracy_difference_95=np.quantile(np.array(deltas)[:, 0], [.025, .975]).tolist(),
                f1_difference_95=np.quantile(np.array(deltas)[:, 1], [.025, .975]).tolist())


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    specs = candidate_specs(args.trials)
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'lightgbm']}
    source_paths = [Path(__file__), Path('heatcast/research_data.py'),
                    Path('heatcast/evaluation.py'), Path('MODEL_SEARCH_PROTOCOL.md')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    hashes['data'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    identity = dict(hashes=hashes, candidates=specs, versions=versions,
                    leads=args.leads, tasks=args.tasks)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity['fingerprint'] = fingerprint
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Output belongs to another source/config/data version; use a new --output')
    manifest.write_text(json.dumps(identity, indent=2))
    (out/'PROTOCOL_AT_FIT.md').write_text(Path('MODEL_SEARCH_PROTOCOL.md').read_text())
    z = np.load(args.data)
    raw, dates, sites = z['x'], z['dates'], z['sites']
    validate_daily(raw, dates, sites)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    climate = fit_climatology(raw[:, :, 0], dates, (years >= 1981) & (years <= 2010))
    thresholds = climate[day_index(dates)]
    labels = event_targets(raw[:, :, 0] > thresholds, dates)
    rows, regional, experiments = [], [], {}
    for lead in args.leads:
        idx, enhanced, feature_names = engineered_features(raw, dates, thresholds, z['lat'], z['lon'], lead)
        features = {'engineered': enhanced}
        feature_names_by_variant = {'engineered': feature_names}
        for variant in ['temperature_only', 'no_soil', 'all_features']:
            old_idx, old_x, old_names = make_features(raw, dates, thresholds, z['lat'], z['lon'], lead, variant)
            features[variant] = old_x[np.searchsorted(old_idx, idx)]
            feature_names_by_variant[variant] = old_names
        train = np.flatnonzero(years[idx+2] <= 2016)
        val = np.flatnonzero((years[idx] >= 2017) & (years[idx+2] <= 2020))
        test = np.flatnonzero((years[idx] >= 2021) & (years[idx+2] <= 2025))
        folds = temporal_folds(dates, idx)
        oof_idx = np.concatenate([v for _, v in folds])
        for task in args.tasks:
            name = f'{task}_lead{lead}'
            y = labels[task][idx]
            oof = {}
            rankings = []
            for spec in specs:
                key = spec['name']
                path = cache/f'{name}_{key}_oof.npz'
                if path.exists():
                    p = np.load(path)['probability']
                else:
                    parts = []
                    x = features[spec['variant']]
                    for fold, (tr, va) in enumerate(folds):
                        model = build_model(spec)
                        model.fit(x[tr].reshape(-1, x.shape[-1]), y[tr].ravel())
                        parts.append(model.predict_proba(x[va].reshape(-1, x.shape[-1]))[:, 1].reshape(len(va), -1))
                    p = np.concatenate(parts)
                    np.savez_compressed(path, probability=p)
                oof[key] = p
                cutoff = threshold_policy(y[oof_idx], p)
                selection = score(y[oof_idx], p, cutoff)
                rankings.append(dict(**spec, selection=selection))
                print(f'{name} {key}: OOF macro F1={selection["macro_site_f1"]:.3f}', flush=True)
            family_winners = []
            for family in ['hist', 'xgboost', 'catboost', 'lightgbm', 'extra_trees', 'random_forest']:
                winner = max([r for r in rankings if r['family'] == family],
                             key=lambda r: (r['selection']['macro_site_f1'], -r['selection']['brier']))
                family_winners.append(winner)
            family_winners.sort(key=lambda r: r['selection']['macro_site_f1'], reverse=True)
            members = {s['name']: [s['name']] for s in specs}
            for count in [3, 6]:
                key = f'ensemble_top{count}'
                members[key] = [r['name'] for r in family_winners[:count]]
                oof[key] = np.mean([oof[k] for k in members[key]], axis=0)
                cutoff = threshold_policy(y[oof_idx], oof[key])
                rankings.append(dict(name=key, family='ensemble', variant='mixed', params={},
                                     selection=score(y[oof_idx], oof[key], cutoff)))
            f1_champion = max(rankings, key=lambda r: (r['selection']['macro_site_f1'],
                                                      -r['selection']['brier']))['name']
            accuracy_scores = {}
            for key, p in oof.items():
                cutoff = threshold_policy(y[oof_idx], p, 'accuracy')
                accuracy_scores[key] = score(y[oof_idx], p, cutoff)
            accuracy_champion = max(accuracy_scores, key=lambda k:
                                    (accuracy_scores[k]['accuracy'], accuracy_scores[k]['macro_site_f1']))
            selection_record = dict(f1_champion=f1_champion, accuracy_champion=accuracy_champion,
                                    members=members, candidates=rankings,
                                    accuracy_selection_scores=accuracy_scores)
            (out/f'{name}_SELECTION.json').write_text(json.dumps(selection_record, indent=2))
            # Champion identities are frozen before any access to held-out labels.
            needed = set(members[f1_champion]+members[accuracy_champion])
            fitted, pv_raw, pt_raw = {}, {}, {}
            for spec in specs:
                key = spec['name']
                x = features[spec['variant']]
                model = build_model(spec)
                model.fit(x[train].reshape(-1, x.shape[-1]), y[train].ravel())
                pv_raw[key] = model.predict_proba(x[val].reshape(-1, x.shape[-1]))[:, 1].reshape(len(val), -1)
                pt_raw[key] = model.predict_proba(x[test].reshape(-1, x.shape[-1]))[:, 1].reshape(len(test), -1)
                if key in needed:
                    fitted[key] = dict(model=model, variant=spec['variant'], name=key)
            for key in ['ensemble_top3', 'ensemble_top6']:
                pv_raw[key] = np.mean([pv_raw[k] for k in members[key]], axis=0)
                pt_raw[key] = np.mean([pt_raw[k] for k in members[key]], axis=0)
            results = {}
            probabilities = {}
            for key in oof:
                calibrator = LogisticRegression(C=1., max_iter=500)
                calibrator.fit(logit(oof[key]), y[oof_idx].ravel())
                cv = calibrated(calibrator, pv_raw[key])
                raw_brier = float(np.mean((pv_raw[key]-y[val])**2))
                calibrated_brier = float(np.mean((cv-y[val])**2))
                use_calibration = calibrated_brier < raw_brier
                pv = cv if use_calibration else pv_raw[key]
                pt = calibrated(calibrator, pt_raw[key]) if use_calibration else pt_raw[key]
                policies = {}
                cutoffs = {}
                for policy in ['f1', 'accuracy']:
                    cutoff = threshold_policy(y[val], pv, policy)
                    cutoffs[policy] = cutoff
                    vm, tm = score(y[val], pv, cutoff), score(y[test], pt, cutoff)
                    policies[policy] = dict(validation=vm, test=tm)
                    rows.append(dict(experiment=name, candidate=key, policy=policy,
                                     f1_champion=key == f1_champion,
                                     accuracy_champion=key == accuracy_champion,
                                     calibration='sigmoid' if use_calibration else 'raw', **tm))
                    if key in [f1_champion, accuracy_champion]:
                        for s, site in enumerate(sites):
                            regional.append(dict(experiment=name, candidate=key, policy=policy,
                                                 site=str(site), **score(y[test, s:s+1], pt[:, s:s+1], cutoff)))
                results[key] = dict(calibration='sigmoid' if use_calibration else 'raw',
                                    validation_raw_brier=raw_brier,
                                    validation_calibrated_brier=calibrated_brier, policies=policies)
                probabilities[key] = pt
                if key in [f1_champion, accuracy_champion]:
                    bundle = dict(members=[fitted[k] for k in members[key]],
                                  calibrator=calibrator if use_calibration else None,
                                  thresholds=cutoffs, feature_names=feature_names_by_variant,
                                  climate=climate, sites=sites, lat=z['lat'], lon=z['lon'],
                                  task=task, lead=lead, fingerprint=fingerprint,
                                  status='Exploratory retrospective ERA5-history amendment')
                    joblib.dump(bundle, out/f'{name}_{key}_bundle.joblib', compress=3)
                    check_p, _ = predict_bundle(bundle, {v: x[test] for v, x in features.items()})
                    np.testing.assert_allclose(check_p, pt, atol=1e-12)
            original_path = Path('results/research')/f'{name}_predictions.npz'
            original = np.load(original_path)
            if not np.array_equal(original['dates'], dates[idx[test]]):
                raise ValueError('Original benchmark dates do not match')
            np.testing.assert_array_equal(original['truth'], y[test])
            original_score = score(y[test], original['probability'], float(original['threshold']))
            uncertainty = {}
            for policy, key in [('f1', f1_champion), ('accuracy', accuracy_champion)]:
                cutoff = results[key]['policies'][policy]['test']['threshold']
                uncertainty[policy] = difference_intervals(y[test], probabilities[key], cutoff,
                    original['probability'], float(original['threshold']), dates[idx[test]])
            experiments[name] = dict(selection=selection_record, results=results,
                                     original=original_score, uncertainty=uncertainty,
                                     all_negative=score(y[test], np.zeros_like(y[test], dtype=float), .5),
                                     n_training_dates=len(train), n_validation_dates=len(val),
                                     n_test_dates=len(test), n_features={k: len(v) for k, v in feature_names_by_variant.items()})
            np.savez_compressed(out/f'{name}_predictions.npz', dates=dates[idx[test]],
                                truth=y[test], validation_dates=dates[idx[val]],
                                validation_truth=y[val], oof_dates=dates[idx[oof_idx]],
                                oof_truth=y[oof_idx], candidates=np.array(list(probabilities)),
                                probability=np.stack(list(probabilities.values())),
                                raw_probability=np.stack([pt_raw[k] for k in probabilities]),
                                validation_probability=np.stack([
                                    calibrated(LogisticRegression(C=1., max_iter=500).fit(logit(oof[k]), y[oof_idx].ravel()), pv_raw[k])
                                    if results[k]['calibration'] == 'sigmoid' else pv_raw[k]
                                    for k in probabilities]),
                                oof_probability=np.stack([oof[k] for k in probabilities]), sites=sites)
            pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
            pd.DataFrame(regional).to_csv(out/'regional_metrics.csv', index=False)
            report = dict(status='Exploratory; reused test, no prospective or operational claim',
                          fingerprint=fingerprint, experiments=experiments)
            (out/'metrics.json').write_text(json.dumps(report, indent=2))
            write_report(report, out)
            print(f'FINISHED {name}: F1 champion {f1_champion}; accuracy champion {accuracy_champion}', flush=True)


def write_report(report, out):
    lines = ['# Exploratory model and feature search', '',
             'Selection uses chronological development folds. Test years were already inspected; '
             'these are exploratory results requiring independent confirmation. '
             'All values below are held-out-period measurements, not promises of future skill.', '',
             '| Task | Policy | Development-selected model | Accuracy | F1 | Recall | Original accuracy | Original F1 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for name, exp in report['experiments'].items():
        for policy in ['f1', 'accuracy']:
            key = exp['selection'][f'{policy}_champion']
            m = exp['results'][key]['policies'][policy]['test']
            old = exp['original']
            lines.append(f'| {name} | {policy} | {key} | {m["accuracy"]:.4f} | {m["f1"]:.4f} | '
                         f'{m["recall"]:.4f} | {old["accuracy"]:.4f} | {old["f1"]:.4f} |')
    lines += ['', 'The accuracy policy requires at least 85% pooled recall on development/validation; '
              'that is not a guarantee of 85% test recall. F1 cutoffs maximize equal-site F1. '
              'Raw versus sigmoid calibration is chosen by validation Brier, not test outcomes.', '',
              'The six original-feature boosting settings and fixed-parameter engineered comparator '
              'allow feature ablation; all other family trials use engineered features. '
              'Tree ensembles use date-aligned out-of-fold predictions, with equal weights. '
              'OOF/validation results are selection scores.', '',
              'See `comparison.csv` for every candidate under both policies, `regional_metrics.csv` '
              'for champion location scores, and `metrics.json` for confusion matrices, calibration '
              'decisions and paired seasonal block intervals versus the original model. '
              'Champion identities were frozen in `*_SELECTION.json` before test scoring. '
              'Inference bundles reproduce saved champion probabilities without test labels.', '',
              '## Reproduce', '', '```bash', 'python3 -m pip install -r requirements-model-search.txt',
              'python3 -m heatcast.model_search',
              'python3 -m unittest discover -s tests -p test_model_search.py -v', '```', '',
              'No original benchmark artifact was replaced. This uses retrospective ERA5 history; '
              'future station verification and real-time input availability are not established.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/europe_sites/daily.npz')
    parser.add_argument('--output', default='results/model_search')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--leads', type=int, nargs='+', choices=[1, 3, 7], default=[1, 3, 7])
    parser.add_argument('--tasks', nargs='+', choices=['ongoing', 'onset'], default=['ongoing', 'onset'])
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    run(args)


if __name__ == '__main__':
    main()
