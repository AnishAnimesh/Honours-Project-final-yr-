"""Verify saved search metrics and causal CSV inference against stored predictions."""
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from heatcast.model_search import score, threshold_policy
from heatcast.model_search_forecast import forecast


def check(root=Path('results/model_search'), allow_partial=False):
    report = json.loads((root/'metrics.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    for path, expected in manifest['hashes'].items():
        actual_path = Path('data/europe_sites/daily.npz') if path == 'data' else Path(path)
        assert hashlib.sha256(actual_path.read_bytes()).hexdigest() == expected, path
    source = np.load('data/europe_sites/daily.npz')
    for name, entry in report['experiments'].items():
        z = np.load(root/f'{name}_predictions.npz')
        names = z['candidates'].tolist()
        selection_scores, accuracy_scores = {}, {}
        for i, candidate in enumerate(names):
            cutoff = threshold_policy(z['oof_truth'], z['oof_probability'][i])
            selection_scores[candidate] = score(z['oof_truth'], z['oof_probability'][i], cutoff)
            accuracy_cutoff = threshold_policy(z['oof_truth'], z['oof_probability'][i], 'accuracy')
            accuracy_scores[candidate] = score(z['oof_truth'], z['oof_probability'][i], accuracy_cutoff)
            for policy in ['f1', 'accuracy']:
                saved = entry['results'][candidate]['policies'][policy]
                cutoff = threshold_policy(z['validation_truth'], z['validation_probability'][i], policy)
                np.testing.assert_allclose(cutoff, saved['test']['threshold'], atol=1e-12)
                measured = score(z['truth'], z['probability'][i], cutoff)
                for metric in ['accuracy', 'f1', 'recall', 'precision', 'brier', 'macro_site_f1', 'balanced_accuracy']:
                    np.testing.assert_allclose(measured[metric], saved['test'][metric], atol=1e-12)
        f1_champion = max(names, key=lambda k: (selection_scores[k]['macro_site_f1'],
                                               -selection_scores[k]['brier']))
        accuracy_champion = max(names, key=lambda k: (accuracy_scores[k]['accuracy'],
                                                     accuracy_scores[k]['macro_site_f1']))
        assert f1_champion == entry['selection']['f1_champion']
        assert accuracy_champion == entry['selection']['accuracy_champion']
        for ensemble in ['ensemble_top3', 'ensemble_top6']:
            members = entry['selection']['members'][ensemble]
            expected = np.mean([z['oof_probability'][names.index(k)] for k in members], axis=0)
            np.testing.assert_allclose(expected, z['oof_probability'][names.index(ensemble)])
        task, lead_text = name.split('_lead')
        lead = int(lead_text)
        date = str(z['dates'][len(z['dates'])//2])[:10]
        row = np.flatnonzero(z['dates'] == np.datetime64(date))[0]
        for policy in ['f1', 'accuracy']:
            candidate = entry['selection'][f'{policy}_champion']
            expected = z['probability'][names.index(candidate), row]
            actual = forecast('data/europe_sites/daily.npz', root, date, task, lead, policy)
            np.testing.assert_allclose(actual['probability'], expected, atol=1e-12)
        # All observations after the issue date may change without affecting inference.
        issue = np.datetime64(date)-np.timedelta64(lead, 'D')
        changed = source['x'].copy()
        changed[source['dates'] > issue] = 999
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'changed.npz'
            np.savez(path, x=changed, dates=source['dates'], sites=source['sites'],
                     lat=source['lat'], lon=source['lon'])
            after = forecast(path, root, date, task, lead, 'f1')
        champion = entry['selection']['f1_champion']
        np.testing.assert_allclose(after['probability'], z['probability'][names.index(champion), row], atol=1e-12)
        print(f'{name}: selection, ensembles, all candidate metrics, thresholds, champion inference and future isolation verified')
    frame = pd.read_csv(root/'comparison.csv')
    assert len(frame) == sum(2*len(e['results']) for e in report['experiments'].values())
    if not allow_partial:
        assert len(report['experiments']) == 6, 'The default full experiment must cover all six task/lead combinations'


if __name__ == '__main__':
    check()
