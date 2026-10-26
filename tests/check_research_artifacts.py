"""Audit saved long-history experiment without retraining."""
import hashlib
import json
from pathlib import Path
import sys
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from heatcast.research_data import make_features,day_index,event_targets
from heatcast.evaluation import metrics
from heatcast.research_forecast import forecast


def main():
    root=Path('results/research');data=Path('data/europe_sites/daily.npz')
    r=json.loads((root/'metrics.json').read_text());manifest=json.loads(Path('data/europe_sites/manifest.json').read_text())
    digest=hashlib.sha256(data.read_bytes()).hexdigest()
    assert digest==r['data_sha256']==manifest['daily_sha256']
    assert hashlib.sha256(Path('RESEARCH_PROTOCOL.md').read_bytes()).hexdigest()==r['protocol_sha256']
    if 'protocol_sha256_at_fit' in r:
        assert hashlib.sha256((root/'PROTOCOL_AT_FIT.md').read_bytes()).hexdigest()==r['protocol_sha256_at_fit']
    for site in manifest['sites']:
        for entry in site['files']:
            path=data.parent/entry['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    z=np.load(data);table=np.load(root/'climatology.npz')['table'];threshold=table[day_index(z['dates'])]
    targets=event_targets(z['x'][:,:,0]>threshold,z['dates'])
    for name,entry in r['experiments'].items():
        task,lead=name.split('_lead');lead=int(lead);saved=np.load(root/f'{name}_predictions.npz')
        idx,x,_=make_features(z['x'],z['dates'],threshold,z['lat'],z['lon'],lead,entry['selected']['variant'])
        positions=np.flatnonzero(np.isin(z['dates'][idx],saved['dates']))
        np.testing.assert_equal(z['dates'][idx[positions]],saved['dates'])
        np.testing.assert_equal(targets[task][idx[positions]],saved['truth'])
        np.testing.assert_equal((saved['dates']-saved['issue_dates']).astype('timedelta64[D]'),np.full(len(saved['dates']),lead,dtype='timedelta64[D]'))
        model=joblib.load(root/f'{name}.joblib');sample=np.array([0,len(positions)//2,len(positions)-1])
        prob=model.predict_proba(x[positions[sample]].reshape(-1,x.shape[-1]))[:,1].reshape(3,-1)
        np.testing.assert_allclose(prob,saved['probability'][sample],atol=1e-7)
        row=len(saved['dates'])//2
        reconstructed=forecast(data,root,str(saved['dates'][row])[:10],task,lead,'primary')
        np.testing.assert_allclose(reconstructed.probability.to_numpy(),saved['probability'][row],atol=1e-7)
        cutoff=entry['selected']['threshold'];assert cutoff==float(saved['threshold'])
        result=metrics(saved['truth'],saved['probability'],cutoff)
        for key in ['f1','precision','recall','roc_auc','brier','average_precision']:
            assert np.isclose(result[key],entry['test'][key]),(name,key)
        if 'validation_champion' in entry:
            scores={'gradient_boosting':entry['selected']['validation']['macro_site_f1']}
            scores.update({n:v['validation']['macro_site_f1'] for n,v in entry['matched_comparisons'].items()})
            assert entry['validation_champion']==max(scores,key=scores.get)
    assert len(r['spatial_transfer'])==2*len(r['sites']) and len(r['backtests'])==2
    print('Research artifact audit passed: raw response hashes, protocol, labels, lead dates, all six saved models, cutoffs, metrics, issue-time-only inference and validation champion selection.')

if __name__=='__main__':main()
