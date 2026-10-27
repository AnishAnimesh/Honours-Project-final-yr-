"""Reconstruct a warning from a frozen archived run without verification inputs."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .issued_data import trajectory,adjusted_risk,event_rule
from .research_data import day_index


def forecast(issue_date,lead=1,task='ongoing',data='data/issued_forecasts',results='results/issued_forecasts'):
    root=Path(results);archive=np.load(Path(data)/'daily_runs.npz');name=f'{task}_lead{lead}'
    bundle=joblib.load(root/f'{name}_adjustment.joblib')
    issue=np.datetime64(issue_date,'ns');target=issue+np.timedelta64(lead,'D')
    if issue<=bundle['freeze']:raise ValueError('Adjustment was not frozen before this warning time')
    if list(archive['sites'])!=bundle['sites']:raise ValueError('Archive site order differs')
    # Only forecast arrays are supplied: future verification data cannot enter inference.
    run_only={k:archive[k] for k in ['initializations','valid_dates','forecast_tmax']}
    part=trajectory(run_only,np.array([target]),lead,task)
    days=part['valid_dates'];threshold=bundle['reference_table'][day_index(days.ravel())].reshape(1,days.shape[1],len(bundle['sites'])).transpose(0,2,1)
    prob=adjusted_risk(part['forecast'],threshold,bundle['error_bank'],task)[0]
    raw=event_rule(part['forecast'],threshold,task)[0]
    frame=pd.DataFrame(dict(site=bundle['sites'],task=task,lead_days=lead,warning_lead_hours=24*lead,
        initialization=str(part['initialization'][0]),assumed_available=str(part['assumed_available'][0]),
        issue_time=str(issue),target_date=str(target)[:10],raw_gfs_event=raw,
        adjusted_probability=prob,cutoff=bundle['cutoff'],adjusted_event=prob>=bundle['cutoff']))
    return frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--issue',required=True);p.add_argument('--lead',type=int,choices=[1,3,7],default=1);p.add_argument('--task',choices=['ongoing','onset'],default='ongoing');p.add_argument('--data',default='data/issued_forecasts');p.add_argument('--results',default='results/issued_forecasts');p.add_argument('--output',default='issued_warning.csv');a=p.parse_args()
    frame=forecast(a.issue,a.lead,a.task,a.data,a.results);frame.to_csv(a.output,index=False);print(frame.to_string(index=False))

if __name__=='__main__':main()
