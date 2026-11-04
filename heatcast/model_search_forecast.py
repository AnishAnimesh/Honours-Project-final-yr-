"""Export a development-selected exploratory tree-model forecast."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from .model_search import engineered_features, predict_bundle
from .research_data import day_index, make_features


def forecast(daily, results, target_date, task='ongoing', lead=1, policy='f1'):
    root = Path(results)
    entry = json.loads((root/'metrics.json').read_text())['experiments'][f'{task}_lead{lead}']
    champion = entry['selection'][f'{policy}_champion']
    bundle = joblib.load(root/f'{task}_lead{lead}_{champion}_bundle.joblib')
    target = np.datetime64(target_date, 'ns')
    issue = target-np.timedelta64(lead, 'D')
    if pd.Timestamp(target).month not in [5, 6, 7, 8, 9]:
        raise ValueError('The benchmark covers May–September targets only')
    z = np.load(daily)
    for key in ['sites', 'lat', 'lon']:
        if not np.array_equal(z[key], bundle[key]):
            raise ValueError(f'{key} differs from training order/coordinates')
    # Discard all future observations before constructing any model input.
    keep = z['dates'] <= issue
    dates, raw = z['dates'][keep], z['x'][keep]
    if len(dates) < 30 or dates[-1] != issue:
        raise ValueError('Need observations through the issue day and 30 days of history')
    if not np.all(np.diff(dates[-30:]).astype('timedelta64[D]') == np.timedelta64(1, 'D')):
        raise ValueError('Need 30 consecutive daily observations')
    # Only the latest 30 observed days are necessary. Future values are placeholders.
    dates, raw = dates[-30:], raw[-30:]
    future = np.arange(issue+np.timedelta64(1, 'D'), target+np.timedelta64(3, 'D'),
                       np.timedelta64(1, 'D'))
    dates = np.concatenate([dates, future])
    raw = np.concatenate([raw, np.zeros((len(future), *raw.shape[1:]), dtype='float32')])
    threshold = bundle['climate'][day_index(dates)]
    features = {}
    for variant in {member['variant'] for member in bundle['members']}:
        if variant == 'engineered':
            idx, x, names = engineered_features(raw, dates, threshold, z['lat'], z['lon'], lead)
        else:
            idx, x, names = make_features(raw, dates, threshold, z['lat'], z['lon'], lead, variant)
        if names != bundle['feature_names'][variant]:
            raise ValueError('Feature schema differs from trained bundle')
        position = np.flatnonzero(dates[idx] == target)
        if len(position) != 1:
            raise ValueError('Target features could not be constructed')
        features[variant] = x[position]
    probability, decision = predict_bundle(bundle, features, policy)
    return pd.DataFrame(dict(site=z['sites'], latitude=z['lat'], longitude=z['lon'],
        issue_date=str(issue)[:10], target_date=str(target)[:10], lead_days=lead,
        task=task, model=champion, policy=policy, probability=probability[0],
        cutoff=bundle['thresholds'][policy], predicted_event=decision[0],
        status='Exploratory retrospective ERA5-history forecast'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily', default='data/europe_sites/daily.npz')
    parser.add_argument('--results', default='results/model_search')
    parser.add_argument('--date', required=True)
    parser.add_argument('--task', choices=['ongoing', 'onset'], default='ongoing')
    parser.add_argument('--lead', type=int, choices=[1, 3, 7], default=1)
    parser.add_argument('--policy', choices=['f1', 'accuracy'], default='f1')
    parser.add_argument('--output', default='model_search_forecast.csv')
    args = parser.parse_args()
    frame = forecast(args.daily, args.results, args.date, args.task, args.lead, args.policy)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
