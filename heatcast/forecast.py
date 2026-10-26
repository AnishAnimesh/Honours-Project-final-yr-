"""Forecast the day after the last complete day in a prepared daily NPZ."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .data import thresholds, labels, calendar_day
from .models import ConvLSTM
from .run import tabular


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--daily',required=True,help='Daily NPZ containing x, dates, lat and lon, including training reference years')
    p.add_argument('--results',default='results')
    p.add_argument('--output',default='next_day_forecast.csv')
    a=p.parse_args(); root=Path(a.results)
    z=np.load(a.daily); raw=z['x']; dates=z['dates']
    if len(dates)<7 or not np.all(np.diff(dates[-7:]).astype('timedelta64[D]')==np.timedelta64(1,'D')):
        raise SystemExit('Need seven complete consecutive daily observations.')
    preprocessing=np.load(root/'preprocessing.npz')
    mask=preprocessing['mask']; mean=preprocessing['mean']; std=preprocessing['std']
    if raw.shape[2:]!=mask.shape:
        raise SystemExit('Input grid shape differs from training grid.')
    saved=np.load(root/'test_predictions.npz')
    if not np.array_equal(z['lat'],saved['lat']) or not np.array_equal(z['lon'],saved['lon']):
        raise SystemExit('Input grid coordinates differ from training grid.')
    ref=pd.to_datetime(dates).year<=2022
    # Fixed historical reference must match the training cache exactly.
    original=np.load(root/'preprocessing.npz')['dates']
    if not np.array_equal(dates[ref],original[pd.to_datetime(original).year<=2022]):
        raise SystemExit('Daily file must contain the original complete training reference dates.')
    threshold=thresholds(raw[:,0],dates,ref)
    x=np.nan_to_num((raw-mean)/np.maximum(std,1e-6)).astype('float32')*mask[None,None]
    report=json.loads((root/'metrics.json').read_text()); name=report['selected_grid_model']
    cutoff=report['models'][name]['validation']['threshold']
    next_date=dates[-1]+np.timedelta64(1,'D')
    if name=='gradient_boosting':
        extended_dates=np.append(dates,next_date)
        feat=tabular(x,raw[:,0]-threshold,raw[:,0]>threshold,extended_dates,np.array([len(dates)]),mask)
        probability=joblib.load(root/'gradient_boosting.joblib').predict_proba(feat)[:,1]
    elif name.startswith('convlstm'):
        checkpoint=torch.load(root/f'{name}.pt',map_location='cpu',weights_only=True)
        channels=checkpoint['channels']; model=ConvLSTM(len(channels))
        model.load_state_dict(checkpoint['state_dict']); model.eval()
        with torch.no_grad():
            probability=model(torch.from_numpy(x[-7:,channels][None].copy())).sigmoid().numpy()[0][mask]
    elif name=='persistence_grid':
        probability=labels(raw[:,0],threshold,dates)[-1][mask].astype(float)
    else:
        day=calendar_day(dates); target_day=calendar_day([next_date])[0]
        probability=labels(raw[:,0],threshold,dates)[ref & (abs(day-target_day)<=15)].mean(0)[mask]
    lat,lon=np.meshgrid(z['lat'],z['lon'],indexing='ij')
    np.savetxt(a.output,np.column_stack([lat[mask],lon[mask],probability,probability>=cutoff]),delimiter=',',header='latitude,longitude,probability,predicted_heatwave',comments='')
    print(f'{name}: forecast {str(next_date)[:10]} at threshold {cutoff:.5f} -> {a.output}')

if __name__=='__main__': main()
