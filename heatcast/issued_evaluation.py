"""Evaluate archived forecasts as warnings with explicit availability assumptions."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .download_issued import target_dates,LEADS,digest
from .issued_data import trajectory,verification,event_rule,adjusted_risk,latest_usable_era5,offsets,validate_timeline,eligible_targets
from .research_data import day_index,event_targets,macro_f1
from .evaluation import metrics,choose_threshold,block_bootstrap

FREEZE=np.datetime64('2026-06-30T00:00:00','ns')


def evaluate(y,p,cutoff):
    result=metrics(y,p,cutoff)
    result['macro_site_f1']=macro_f1(y,p,cutoff)
    matrix=np.array(result['confusion_matrix']);tn,fp,fn,tp=matrix.ravel()
    result.update(event_hits=int(tp),missed_event_days=int(fn),false_alerts=int(fp),false_alerts_per_100_location_days=float(fp/y.size*100))
    return result


def climatology(history,reference_table,targets,task):
    dates=history['dates'];day=day_index(dates)
    hot=history['x'][:,:,0]>reference_table[day]
    labels=event_targets(hot,dates)[task]
    train=pd.DatetimeIndex(dates).year<=2016
    result=[]
    for target_day in day_index(targets):
        delta=np.abs(day-target_day);keep=train & (np.minimum(delta,365-delta)<=15)
        result.append(labels[keep].mean(0))
    return np.stack(result)


def write_warning_rows(rows,name,method,part,truth,probability,cutoff,sites,lead,threshold):
    for i,target in enumerate(part['target']):
        for s,site in enumerate(sites):
            rows.append(dict(experiment=name,method=method,site=str(site),lead_days=lead,
                initialization=str(part['initialization'][i]),assumed_available=str(part['assumed_available'][i]),
                issue_time=str(part['issue'][i]),target_date=str(target)[:10],warning_lead_hours=24*lead,
                initialization_to_target_hours=24*(lead+1),calibration_frozen_at=str(FREEZE),
                first_verification_day=str(part['valid_dates'][i,0])[:10],last_verification_day=str(part['valid_dates'][i,-1])[:10],
                truth=bool(truth[i,s]),score=float(probability[i,s]),cutoff=float(cutoff),predicted_event=bool(probability[i,s]>=cutoff),
                status='hit' if truth[i,s] and probability[i,s]>=cutoff else 'miss' if truth[i,s] else 'false_alert' if probability[i,s]>=cutoff else 'correct_negative'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/issued_forecasts');p.add_argument('--output',default='results/issued_forecasts');a=p.parse_args()
    root=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    archive=np.load(root/'daily_runs.npz');manifest=json.loads((root/'manifest.json').read_text())
    if digest(root/'daily_runs.npz')!=manifest['prepared_sha256']:raise ValueError('Prepared archive checksum mismatch')
    if digest('ISSUED_FORECAST_PROTOCOL.md')!=manifest['protocol_sha256']:raise ValueError('Protocol changed since archive acquisition; record an amendment before evaluation')
    history=np.load('data/europe_sites/daily.npz');climate=np.load('results/research/climatology.npz')
    if not np.array_equal(archive['sites'],history['sites']) or not np.array_equal(archive['sites'],climate['sites']):raise ValueError('Site order differs')
    table=climate['table'];sites=archive['sites'];nominal_dates=target_dates();dates={};coverage={}
    for split,td in nominal_dates.items():
        dates[split],excluded=eligible_targets(archive,td)
        coverage[split]=dict(nominal_target_days=len(td),retained_target_days=len(dates[split]),excluded=excluded)
        if len(dates[split])<30:raise ValueError('Fewer than 30 complete matched targets; report archive coverage before attempting evaluation')
    report=dict(protocol_sha256=digest('ISSUED_FORECAST_PROTOCOL.md'),data_sha256=manifest['prepared_sha256'],
        history_sha256=digest('data/europe_sites/daily.npz'),climatology_sha256=digest('results/research/climatology.npz'),
        data=dict(model='gfs_global',sites=sites.tolist(),run_count=len(archive['initializations']),forecast_start=str(archive['initializations'][0]),forecast_end=str(archive['initializations'][-1]),
                  verification='ERA5 hourly-derived daily maximum, actual historical reference grid coordinates',
                  publication_policy='Run initialization +12h assumed available; warning issued +24h. Individual historical receipt times unavailable.'),
        coverage=coverage,development='2026-05-01 through 2026-06-22; verification available by 2026-06-30',freeze=str(FREEZE),
        test='2026-07-08 through 2026-09-30; common target/site cases across leads',
        reference='Fixed 1981–2010 local 90th percentile, five-calendar-day window',experiments={},limitations=[
            'Retrospectively retrieved explicit 2026 GFS runs, one season, seven locations; not a prospective deployment.',
            'Historical API receipt timestamps are unavailable; the 24-hour initialization-to-decision buffer assumes no longer publication outage.',
            'Model runs are 24 hours old at warning issuance, so this conservative policy does not measure freshest-run skill.',
            'Verification uses ERA5, not independent station observations; GFS and ERA5 grid coordinates can differ.',
            'Development covers early summer and test covers later summer; joint-error stationarity is an assumption.',
            'Empirical error scenarios are not an NWP ensemble and do not guarantee calibrated probabilities.',
            'Existing reanalysis-history ML scores use a different input/availability design and cannot be compared as if operational.'])
    warning_rows=[];regional_rows=[];temperature_rows=[];comparison=[]
    for lead in LEADS:
        for task in ['ongoing','onset']:
            name=f'{task}_lead{lead}';parts={};truth={};threshold={};error={}
            for split,td in dates.items():
                part=trajectory(archive,td,lead,task);parts[split]=part
                actual=verification(archive,part['valid_dates'])
                th=table[day_index(part['valid_dates'].ravel())].reshape(len(td),len(offsets(task)),len(sites)).transpose(0,2,1)
                threshold[split]=th;truth[split]=event_rule(actual,th,task);error[split]=actual-part['forecast']
            validate_timeline(parts['development'],parts['test'],FREEZE)
            dev_prob=adjusted_risk(parts['development']['forecast'],threshold['development'],error['development'],task,
                dates['development'],dates['development'],exclude_days=7)
            cutoff=choose_threshold(truth['development'],dev_prob)
            test_prob=adjusted_risk(parts['test']['forecast'],threshold['test'],error['development'],task)
            predictions={};dev_predictions={};cutoffs={}
            for split in ['development','test']:
                part=parts[split];td=dates[split]
                raw=event_rule(part['forecast'],threshold[split],task).astype(float)
                usable=latest_usable_era5(part['issue'])
                verify_dates=archive['verification_dates'].astype('datetime64[ns]')
                pos=np.searchsorted(verify_dates,usable)
                if not np.array_equal(verify_dates[pos],usable):raise ValueError('Missing delayed baseline observation')
                last_temp=archive['verification_tmax'][pos]
                persistence=event_rule(np.broadcast_to(last_temp[:,:,None],part['forecast'].shape),threshold[split],task).astype(float)
                values=dict(raw_gfs=raw,joint_error_adjusted=dev_prob if split=='development' else test_prob,
                            delayed_era5_persistence=persistence,historical_climatology=climatology(history,table,td,task))
                if split=='development':dev_predictions=values
                else:predictions=values
            for method in predictions:
                cutoffs[method]=cutoff if method=='joint_error_adjusted' else choose_threshold(truth['development'],dev_predictions[method]) if method=='historical_climatology' else .5
            entry=dict(development_crossfit_exclusion_days=7,development_positive_days=int(truth['development'].sum()),test_positive_days=int(truth['test'].sum()),
                       n_test_target_days=len(dates['test']),n_test_location_days=truth['test'].size,methods={})
            for method,prob in predictions.items():
                t=cutoffs[method];scores=evaluate(truth['test'],prob,t)
                entry['methods'][method]=dict(cutoff=t,development=evaluate(truth['development'],dev_predictions[method],t),test=scores)
                comparison.append(dict(experiment=name,method=method,**scores))
                write_warning_rows(warning_rows,name,method,parts['test'],truth['test'],prob,t,sites,lead,threshold['test'])
                for s,site in enumerate(sites):
                    regional_rows.append(dict(experiment=name,method=method,site=str(site),positive_days=int(truth['test'][:,s].sum()),**metrics(truth['test'][:,s],prob[:,s],t)))
            uncertainty=block_bootstrap(truth['test'],test_prob,cutoff,predictions['raw_gfs'],dates['test'],seed=42)
            entry['adjusted_vs_raw_uncertainty']=dict(method=uncertainty['method'],adjusted_f1_95_interval=uncertainty['f1_95_interval'],f1_difference_vs_raw_95_interval=uncertainty['f1_difference_from_persistence_95_interval'])
            # Record temperature errors at the target day (rather than treating event F1 as temperature skill).
            target_offset=2 if task=='ongoing' else 1
            delta=error['test'][:,:,target_offset]
            for s,site in enumerate(sites):temperature_rows.append(dict(experiment=name,site=str(site),raw_gfs_mae=float(np.abs(delta[:,s]).mean()),raw_gfs_rmse=float(np.sqrt((delta[:,s]**2).mean())),mean_era5_minus_gfs=float(delta[:,s].mean())))
            entry['raw_target_temperature_mae']=float(np.abs(delta).mean());entry['raw_target_temperature_rmse']=float(np.sqrt((delta**2).mean()))
            report['experiments'][name]=entry
            np.savez_compressed(out/f'{name}_predictions.npz',target_dates=dates['test'],issue_times=parts['test']['issue'],initializations=parts['test']['initialization'],
                assumed_available=parts['test']['assumed_available'],valid_dates=parts['test']['valid_dates'],truth=truth['test'],forecast_trajectory=parts['test']['forecast'],
                threshold_trajectory=threshold['test'],development_error_bank=error['development'],development_dates=dates['development'],development_probability=dev_prob,sites=sites,
                cutoff=cutoff,**predictions)
            # Lightweight inference bundle uses only fixed reference + development residuals, never test labels.
            joblib.dump(dict(task=task,lead=lead,sites=sites.tolist(),reference_table=table,error_bank=error['development'],bank_dates=dates['development'],
                             cutoff=cutoff,freeze=FREEZE,protocol_sha256=report['protocol_sha256']),out/f'{name}_adjustment.joblib')
            print(f"{name}: raw F1={entry['methods']['raw_gfs']['test']['f1']:.3f}; adjusted F1={entry['methods']['joint_error_adjusted']['test']['f1']:.3f}; positive location-days={entry['test_positive_days']}",flush=True)
    pd.DataFrame(comparison).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(warning_rows).to_csv(out/'warning_audit.csv',index=False)
    pd.DataFrame(regional_rows).to_csv(out/'regional_metrics.csv',index=False)
    pd.DataFrame(temperature_rows).to_csv(out/'temperature_metrics.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    (out/'PROTOCOL_AT_EVALUATION.md').write_text(Path('ISSUED_FORECAST_PROTOCOL.md').read_text())
    from .issued_report import write_report
    write_report(report,out)

if __name__=='__main__':main()
