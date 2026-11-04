"""Descriptive error traces and distribution diagnostics; never tune predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .issued_data import verification


def main():
    out = Path('results/error_analysis'); out.mkdir(parents=True, exist_ok=True)
    archive = np.load('data/issued_forecasts/daily_runs.npz')
    report = dict(status='Post hoc explanation of saved forecasts; no refitting, cutoff tuning or causal attribution.', experiments={}, cases=[])
    rows = []
    for task, lead, status in [('onset', 1, 'miss'), ('onset', 1, 'false_alert'),
                               ('ongoing', 7, 'introduced_false_alert'), ('ongoing', 7, 'rescued_hit'),
                               ('onset', 7, 'miss'), ('onset', 7, 'false_alert')]:
        name = f'{task}_lead{lead}'
        z = np.load(f'results/issued_forecasts/{name}_predictions.npz')
        y = z['truth'].astype(bool); pred = z['joint_error_adjusted'] >= float(z['cutoff']); raw = z['raw_gfs'].astype(bool)
        choices = dict(miss=y & ~pred, false_alert=~y & pred,
                       introduced_false_alert=~y & pred & ~raw, rescued_hit=y & pred & ~raw)[status]
        positions = np.argwhere(choices)
        if not len(positions): continue
        i, s = map(int, positions[0])
        observed = verification(archive, z['valid_dates'][i:i+1])[0, s]
        case = dict(experiment=name, status=status, site=str(z['sites'][s]), target_date=str(z['target_dates'][i])[:10],
                    issue_time=str(z['issue_times'][i]), initialization=str(z['initializations'][i]),
                    probability=float(z['joint_error_adjusted'][i,s]), cutoff=float(z['cutoff']),
                    truth=bool(y[i,s]), raw_event=bool(raw[i,s]), predicted=bool(pred[i,s]), trajectory=[])
        for d, forecast, actual, threshold in zip(z['valid_dates'][i], z['forecast_trajectory'][i,s], observed, z['threshold_trajectory'][i,s]):
            row = dict(experiment=name, status=status, site=case['site'], target_date=case['target_date'],
                       valid_date=str(d)[:10], forecast_c=float(forecast), observed_c=float(actual),
                       threshold_c=float(threshold), forecast_margin_c=float(forecast-threshold), observed_margin_c=float(actual-threshold))
            rows.append(row); case['trajectory'].append(row)
        report['cases'].append(case)

    drift = []
    for task in ['ongoing','onset']:
        for lead in [1,3,7]:
            name=f'{task}_lead{lead}'; z=np.load(f'results/issued_forecasts/{name}_predictions.npz')
            errors=verification(archive,z['valid_dates'])-z['forecast_trajectory']
            y=z['truth'].astype(bool); pred=z['joint_error_adjusted']>=float(z['cutoff']); raw=z['raw_gfs'].astype(bool)
            for s,site in enumerate(z['sites']):
                drift.append(dict(experiment=name,site=str(site),development_error_mean_c=float(z['development_error_bank'][:,s].mean()),
                                  test_error_mean_c=float(errors[:,s].mean()),
                                  development_error_sd_c=float(z['development_error_bank'][:,s].std()),test_error_sd_c=float(errors[:,s].std()),
                                  added_false_alerts=int((pred[:,s]&~y[:,s]&~raw[:,s]).sum()),
                                  removed_false_alerts=int((~pred[:,s]&~y[:,s]&raw[:,s]).sum()),
                                  rescued_positive_days=int((pred[:,s]&y[:,s]&~raw[:,s]).sum()),
                                  newly_missed_positive_days=int((~pred[:,s]&y[:,s]&raw[:,s]).sum())))
            report['experiments'][name]=dict(cutoff=float(z['cutoff']), raw_alerts=int(raw.sum()),adjusted_alerts=int(pred.sum()),
                raw_false_alerts=int((raw&~y).sum()),adjusted_false_alerts=int((pred&~y).sum()),
                added_false_alerts=int((pred&~y&~raw).sum()),removed_false_alerts=int((~pred&~y&raw).sum()),
                rescued_positive_days=int((pred&y&~raw).sum()),newly_missed_positive_days=int((~pred&y&raw).sum()),
                development_error_mean_c=float(z['development_error_bank'].mean()),test_error_mean_c=float(errors.mean()))
    r3=np.load('results/research/onset_lead3_predictions.npz');r7=np.load('results/research/onset_lead7_predictions.npz')
    report['onset_cutoff_diagnostic']={}
    for lead,z in [(3,r3),(7,r7)]:
        scores={}
        for label,t in [('own_validation_cutoff',float(z['threshold'])),('three_day_validation_cutoff',float(r3['threshold']))]:
            p=z['probability']>=t;y=z['truth'].astype(bool)
            scores[label]=dict(cutoff=t,alerts=int(p.sum()),hits=int((p&y).sum()),false_alerts=int((p&~y).sum()),recall=float((p&y).sum()/y.sum()))
        report['onset_cutoff_diagnostic'][str(lead)]=scores
    grid=json.loads(Path('results/metrics.json').read_text())
    report['spatial_prevalence']={k:dict(target_days=v['n_days'],prevalence=v['prevalence']) for k,v in grid['splits'].items()}
    report['spatial_neural_diagnostics']={}
    for name in ['lstm_Paris','convlstm','convlstm_no_soil']:
        e=grid['models'][name];hist=e['history'];best=min(hist,key=lambda h:h['validation_loss'])
        report['spatial_neural_diagnostics'][name]=dict(cutoff=e['test']['threshold'],best_validation_epoch=best['epoch'],
            epochs_completed=len(hist),test_precision=e['test']['precision'],test_recall=e['test']['recall'],
            test_confusion_matrix=e['test']['confusion_matrix'],test_brier=e['test']['brier'])
    pd.DataFrame(rows).to_csv(out/'case_trajectories.csv',index=False)
    pd.DataFrame(drift).to_csv(out/'regional_error_shift.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
