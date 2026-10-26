"""Issue an observation-only site forecast for a specified summer target day."""
import argparse
import json
import os
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .models import LSTM
from .research_data import day_index,make_features


def forecast(daily,results,target_date,task='ongoing',lead=1,method='primary'):
    root=Path(results);z=np.load(daily);report=json.loads((root/'metrics.json').read_text())
    name=f'{task}_lead{lead}';entry=report['experiments'][name];target=np.datetime64(target_date,'ns');issue=target-np.timedelta64(lead,'D')
    dates=z['dates'];keep=dates<=issue
    if not keep.any() or dates[keep][-1]!=issue:raise ValueError('The end-of-issue-day observation is unavailable')
    if target.astype('datetime64[M]').astype(int)%12+1 not in [5,6,7,8,9]:raise ValueError('The benchmark forecasts May–September targets only')
    dates=dates[keep];raw=z['x'][keep]
    if not np.all(np.diff(dates[-14:]).astype('timedelta64[D]')==np.timedelta64(1,'D')):raise ValueError('Need 14 consecutive daily observations')
    saved=np.load(root/'climatology.npz')
    if not np.array_equal(z['sites'],saved['sites']) or not np.array_equal(z['lat'],saved['lat']) or not np.array_equal(z['lon'],saved['lon']):raise ValueError('Site order and coordinates differ from training')
    # Placeholder future rows permit target feature indexing. None can enter the target's predictors.
    future=np.arange(issue+np.timedelta64(1,'D'),target+np.timedelta64(3,'D'),np.timedelta64(1,'D'))
    dates=np.concatenate([dates,future]);raw=np.concatenate([raw,np.zeros((len(future),*raw.shape[1:]),dtype='float32')])
    threshold=saved['table'][day_index(dates)]
    model_name=entry.get('validation_champion','gradient_boosting') if method=='champion' else 'gradient_boosting'
    variant=entry['selected']['variant'] if model_name=='gradient_boosting' else 'all_features'
    idx,x,_=make_features(raw,dates,threshold,z['lat'],z['lon'],lead,variant)
    position=np.flatnonzero(dates[idx]==target)
    if len(position)!=1:raise ValueError('Target could not be constructed')
    if model_name in ['gradient_boosting','logistic']:
        path=root/(f'{name}.joblib' if model_name=='gradient_boosting' else f'{name}_logistic.joblib')
        prob=joblib.load(path).predict_proba(x[position[0]])[:,1]
        cutoff=entry['selected']['threshold'] if model_name=='gradient_boosting' else entry['matched_comparisons'][model_name]['validation']['threshold']
    else:
        checkpoint=torch.load(root/f'{name}_lstm.pt',map_location='cpu',weights_only=True)
        values=np.concatenate([raw,(raw[:,:,0]-threshold)[:,:,None]],-1)
        values=(values-np.array(checkpoint['mean']))/np.array(checkpoint['std'])
        t=int(np.flatnonzero(dates==issue)[0]);seq=values[t-13:t+1].transpose(1,0,2).astype('float32')
        model=LSTM(7);model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.no_grad():prob=model(torch.from_numpy(seq.copy())).sigmoid().numpy()
        cutoff=checkpoint['threshold']
    return pd.DataFrame(dict(site=z['sites'],latitude=z['lat'],longitude=z['lon'],issue_date=str(issue)[:10],target_date=str(target)[:10],lead_days=lead,task=task,model=model_name,probability=prob,cutoff=cutoff,predicted_event=prob>=cutoff))


def main():
    p=argparse.ArgumentParser();p.add_argument('--daily',default='data/europe_sites/daily.npz');p.add_argument('--results',default='results/research');p.add_argument('--date',required=True);p.add_argument('--task',choices=['ongoing','onset'],default='ongoing');p.add_argument('--lead',type=int,choices=[1,3,7],default=1);p.add_argument('--method',choices=['primary','champion'],default='primary');p.add_argument('--output',default='site_forecast.csv');a=p.parse_args()
    frame=forecast(a.daily,a.results,a.date,a.task,a.lead,a.method);frame.to_csv(a.output,index=False);print(frame.to_string(index=False))

if __name__=='__main__':main()
