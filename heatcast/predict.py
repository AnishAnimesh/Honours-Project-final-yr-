"""Reproduce a saved test forecast as a latitude/longitude CSV."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='results')
    p.add_argument('--date',required=True)
    p.add_argument('--output',default='forecast.csv')
    a=p.parse_args()
    root=Path(a.results)
    report=json.loads((root/'metrics.json').read_text())
    z=np.load(root/'test_predictions.npz')
    dates=z['dates'].astype('datetime64[D]')
    positions=np.flatnonzero(dates==np.datetime64(a.date,'D'))
    if not len(positions):
        raise SystemExit('Date is not among the saved contiguous held-out forecast dates.')
    name=report['selected_grid_model']; t=report['models'][name]['test']['threshold']
    latitude,longitude=np.meshgrid(z['lat'],z['lon'],indexing='ij')
    mask=z['mask']; probability=z[name][positions[0]]
    values=np.column_stack([latitude[mask],longitude[mask],probability,probability>=t])
    np.savetxt(a.output,values,delimiter=',',header='latitude,longitude,probability,predicted_heatwave',comments='')
    print(f'{name}: exported {len(probability)} land cells for {a.date} to {a.output}')

if __name__=='__main__': main()
