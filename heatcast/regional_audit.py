"""Diagnose geographical failure in the original full-grid experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluation import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results');a=p.parse_args()
    root=Path(a.results);out=root/'regional_audit';out.mkdir(exist_ok=True)
    report=json.loads((root/'metrics.json').read_text());z=np.load(root/'test_predictions.npz')
    y=z['truth'];name=report['selected_grid_model'];prob=z[name];t=report['models'][name]['validation']['threshold']
    pred=prob>=t;tp=(pred&y).sum(0);fp=(pred&~y).sum(0);fn=(~pred&y).sum(0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0)
    recall=np.divide(tp,tp+fn,out=np.zeros_like(tp,dtype=float),where=(tp+fn)>0)
    base=z['persistence_grid']>=.5;btp=(base&y).sum(0);bfp=(base&~y).sum(0);bfn=(~base&y).sum(0)
    bf1=np.divide(2*btp,2*btp+bfp+bfn,out=np.zeros_like(btp,dtype=float),where=(2*btp+bfp+bfn)>0)
    lat,lon=np.meshgrid(z['lat'],z['lon'],indexing='ij');lat=lat[z['mask']];lon=lon[z['mask']]
    frame=pd.DataFrame(dict(latitude=lat,longitude=lon,positive_days=y.sum(0),f1=f1,precision=precision,recall=recall,persistence_f1=bf1,f1_difference=f1-bf1))
    frame['support']=np.where(frame.positive_days>=10,'at_least_10_positive_days','sparse_or_no_events')
    frame.to_csv(out/'cell_metrics.csv',index=False)
    rows=[]
    for north in [False,True]:
        for east in [False,True]:
            keep=((lat>=46.5)==north)&((lon>=1.5)==east)
            region=('north' if north else 'south')+('_east' if east else '_west')
            cell=metrics(y[:,keep],prob[:,keep],t);baseline=metrics(y[:,keep],base[:,keep].astype(float),.5)
            rows.append(dict(region=region,n_cells=int(keep.sum()),**cell,persistence_f1=baseline['f1'],f1_difference=cell['f1']-baseline['f1']))
    pd.DataFrame(rows).to_csv(out/'subregions.csv',index=False)
    years=pd.DatetimeIndex(z['dates']).year
    pd.DataFrame([dict(year=int(year),**metrics(y[years==year],prob[years==year],t)) for year in np.unique(years)]).to_csv(out/'years.csv',index=False)
    from .plots import matplotlib
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    values=[f1,f1-bf1,y.sum(0)];titles=['Cell F1 (fewer than 10 positives hidden)','F1 difference from persistence','Number of positive test days']
    for ax,values,title,cmap,lo,hi in zip(axes,values,titles,['viridis','RdBu','magma'],[0,-.5,0],[1,.5,int(y.sum(0).max())]):
        grid=np.full(z['mask'].shape,np.nan);grid[z['mask']]=values
        if title.startswith('Cell F1'):
            supported=np.zeros_like(z['mask']);supported[z['mask']]=y.sum(0)>=10;grid[~supported]=np.nan
        im=ax.imshow(grid,cmap=cmap,vmin=lo,vmax=hi,extent=[z['lon'].min(),z['lon'].max(),z['lat'].min(),z['lat'].max()]);ax.set(title=title,xlabel='Longitude',ylabel='Latitude');fig.colorbar(im,ax=ax,shrink=.75)
    fig.tight_layout();fig.savefig(out/'spatial_skill.png',dpi=160);plt.close(fig)
    supported=frame[frame.positive_days>=10]
    worst=supported.nsmallest(10,'f1')
    lines=['# Original-grid regional failure audit','',f'Model: {name}; validation-selected cutoff {t:.4f}. No retraining or test-based cutoff changes.','',
        f'{len(frame)} evaluated cells; {len(supported)} have at least 10 positive test days. Low-support cells are identified rather than silently removed from overall scores.','',
        '## Geographic quadrants','',pd.DataFrame(rows)[['region','n_cells','f1','recall','persistence_f1','f1_difference']].to_csv(index=False),'',
        '## Lowest F1 among cells with at least 10 positives','',worst.to_csv(index=False),'',
        'The quadrant boundaries (46.5°N and 1.5°E) are simple geographic bins, not climate or administrative regions. Test failures are descriptive findings and must not be used to tune the model on this same test set.','',
        '![Spatial skill](spatial_skill.png)']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(pd.DataFrame(rows)[['region','f1','recall','f1_difference']].to_string(index=False))

if __name__=='__main__':main()
