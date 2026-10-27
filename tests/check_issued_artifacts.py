"""Audit archive provenance, chronological availability and warning reproduction."""
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from heatcast.download_issued import digest,target_dates
from heatcast.issued_data import trajectory,verification,event_rule,adjusted_risk,offsets,latest_usable_era5,eligible_targets
from heatcast.issued_evaluation import FREEZE
from heatcast.issued_forecast import forecast
from heatcast.evaluation import metrics
from heatcast.research_data import day_index


def main():
    root=Path('data/issued_forecasts');out=Path('results/issued_forecasts')
    manifest=json.loads((root/'manifest.json').read_text());report=json.loads((out/'metrics.json').read_text());archive=np.load(root/'daily_runs.npz')
    assert digest(root/'daily_runs.npz')==manifest['prepared_sha256']==report['data_sha256']
    assert digest('ISSUED_FORECAST_PROTOCOL.md')==manifest['protocol_sha256']==report['protocol_sha256']
    for record in manifest['runs']+[manifest['verification']]:assert digest(root/record['file'])==record['sha256']
    assert digest('data/europe_sites/daily.npz')==report['history_sha256']
    assert digest('results/research/climatology.npz')==report['climatology_sha256']
    common=None
    table=np.load('results/research/climatology.npz')['table']
    for name,entry in report['experiments'].items():
        task,lead=name.split('_lead');lead=int(lead);z=np.load(out/f'{name}_predictions.npz');bundle=joblib.load(out/f'{name}_adjustment.joblib')
        if common is None:common=z['target_dates']
        else:np.testing.assert_equal(common,z['target_dates'])
        np.testing.assert_equal(z['target_dates']-z['issue_times'],np.full(len(common),lead,dtype='timedelta64[D]'))
        assert np.all(z['initializations']<z['assumed_available']) and np.all(z['assumed_available']<z['issue_times'])
        assert np.all(z['issue_times']>FREEZE)
        assert np.max(z['development_dates']+np.timedelta64(int(offsets(task).max())+6,'D'))<=FREEZE
        assert bundle['protocol_sha256']==report['protocol_sha256']
        np.testing.assert_equal(bundle['bank_dates'],z['development_dates']);np.testing.assert_equal(bundle['error_bank'],z['development_error_bank'])
        dev_part=trajectory(archive,z['development_dates'],lead,task)
        np.testing.assert_allclose(verification(archive,dev_part['valid_dates'])-dev_part['forecast'],z['development_error_bank'],atol=1e-6)
        test_part=trajectory(archive,common,lead,task)
        th=table[day_index(test_part['valid_dates'].ravel())].reshape(len(common),len(offsets(task)),len(archive['sites'])).transpose(0,2,1)
        np.testing.assert_equal(event_rule(verification(archive,test_part['valid_dates']),th,task),z['truth'])
        np.testing.assert_allclose(adjusted_risk(test_part['forecast'],th,bundle['error_bank'],task),z['joint_error_adjusted'])
        for method,values in entry['methods'].items():
            result=metrics(z['truth'],z[method],values['cutoff'])
            for key in ['f1','precision','recall','brier','average_precision']:
                assert np.isclose(result[key],values['test'][key]),(name,method,key)
        index=len(common)//2
        replay=forecast(str(z['issue_times'][index])[:10],lead,task)
        np.testing.assert_allclose(replay.adjusted_probability,z['joint_error_adjusted'][index])
        # Future/recent ERA5 observations are inaccessible to warning inference.
        with tempfile.TemporaryDirectory() as temporary:
            temporary=Path(temporary);copy={k:archive[k] for k in archive.files};copy['verification_tmax']=np.full_like(copy['verification_tmax'],9999)
            np.savez_compressed(temporary/'daily_runs.npz',**copy)
            altered=forecast(str(z['issue_times'][index])[:10],lead,task,data=temporary)
            np.testing.assert_equal(replay.adjusted_probability.to_numpy(),altered.adjusted_probability.to_numpy())
    for split,nominal in target_dates().items():
        eligible,excluded=eligible_targets(archive,nominal)
        assert len(eligible)==report['coverage'][split]['retained_target_days'] and len(excluded)==len(report['coverage'][split]['excluded'])
    audit=pd.read_csv(out/'warning_audit.csv')
    issue=pd.to_datetime(audit.issue_time);init=pd.to_datetime(audit.initialization);available=pd.to_datetime(audit.assumed_available)
    target=pd.to_datetime(audit.target_date)
    assert (init<available).all() and (available<issue).all() and (issue>pd.Timestamp(FREEZE)).all()
    np.testing.assert_equal((target-issue).dt.total_seconds().to_numpy()/3600,audit.warning_lead_hours.to_numpy())
    np.testing.assert_equal((target-init).dt.total_seconds().to_numpy()/3600,audit.initialization_to_target_hours.to_numpy())
    for lead in [1,3,7]:
        issues=archive['initializations']+np.timedelta64(1,'D')
        usable=latest_usable_era5(issues);assert np.all(usable+np.timedelta64(6,'D')<=issues)
    print('Issued-forecast audit passed: source hashes, frozen exact runs, common coverage, development-only error banks, ERA5 publication delay, 24/72/168-hour warning leads, saved scores and forecast reproduction without verification inputs.')

if __name__=='__main__':main()
