"""Build the presentation dataset from the project's real model artifacts."""
from pathlib import Path
import json
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]/'dist'


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,)):return None if not np.isfinite(value) else float(value)
    if isinstance(value,float):return value if np.isfinite(value) else None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def save(name,data):
    (OUT/'data'/name).write_text(json.dumps(clean(data),separators=(',',':'),allow_nan=False))


def date_strings(dates):return np.asarray(dates).astype('datetime64[D]').astype(str).tolist()


def main():
    (OUT/'data').mkdir(parents=True,exist_ok=True);(OUT/'reports').mkdir(exist_ok=True)
    history=np.load(ROOT/'data/europe_sites/daily.npz')
    issued_archive=np.load(ROOT/'data/issued_forecasts/daily_runs.npz')
    climate=np.load(ROOT/'results/research/climatology.npz')
    issued_report=json.loads((ROOT/'results/issued_forecasts/metrics.json').read_text())
    research_report=json.loads((ROOT/'results/research/metrics.json').read_text())
    countries=dict(Paris='France',London='United Kingdom',Madrid='Spain',Rome='Italy',Athens='Greece',Berlin='Germany',Warsaw='Poland')
    sites=[dict(id=str(site).lower(),name=str(site),country=countries[str(site)],lat=float(history['lat'][s]),lon=float(history['lon'][s]),
                gridLat=float(issued_archive['latitude'][s]),gridLon=float(issued_archive['longitude'][s])) for s,site in enumerate(history['sites'])]
    save('manifest.json',dict(generatedAt=datetime.now(timezone.utc).isoformat(),sites=sites,defaultDate='2026-08-12',
        sources=dict(issued=dict(label='Archived GFS warnings',start='2026-07-08',end='2026-09-30',dates=85,locations=7,mode='Historical replay'),
                     research=dict(label='Climate model study',start='2021-05-01',end='2025-09-30',dates=765,locations=7,mode='Historical experiment')),
        dataHashes=dict(issued=issued_report['data_sha256'],research=research_report['data_sha256']),
        live=dict(endpoint='https://api.open-meteo.com/v1/forecast',model='gfs_global',validatedMonths=[5,6,7,8,9],
                  note='Live provider feed; not the frozen-run historical experiment. Error adjustment was developed on early summer 2026.')))
    issued=dict(experiments={},report=issued_report,regions=pd.read_csv(ROOT/'results/issued_forecasts/regional_metrics.csv').to_dict('records'),
                observations=dict(dates=date_strings(issued_archive['verification_dates']),tmax=issued_archive['verification_tmax']),
                runs=dict(initializations=date_strings(issued_archive['initializations']),dates=[date_strings(row) for row in issued_archive['valid_dates']],tmax=issued_archive['forecast_tmax']),
                thresholdTable=climate['table'])
    research=dict(experiments={},report=research_report,regions=pd.read_csv(ROOT/'results/research/regional_metrics.csv').to_dict('records'))
    recent=history['dates']>=np.datetime64('2021-01-01')
    research['observations']=dict(dates=date_strings(history['dates'][recent]),tmax=history['x'][recent,:,0]);research['thresholdTable']=climate['table']
    models=dict(referenceTable=climate['table'],banks={})
    for task in ['ongoing','onset']:
        for lead in [1,3,7]:
            name=f'{task}_lead{lead}'
            z=np.load(ROOT/f'results/issued_forecasts/{name}_predictions.npz')
            cutoff=float(z['cutoff'])
            issued['experiments'][name]=dict(dates=date_strings(z['target_dates']),issueTimes=z['issue_times'].astype(str).tolist(),
                initializations=z['initializations'].astype(str).tolist(),probabilities=z['joint_error_adjusted'],cutoff=cutoff,
                predicted=z['joint_error_adjusted']>=cutoff,raw=z['raw_gfs'],truth=z['truth'])
            models['banks'][name]=dict(errors=z['development_error_bank'],cutoff=cutoff,freeze='2026-06-30T00:00:00Z')
            q=np.load(ROOT/f'results/research/{name}_predictions.npz');entry=research_report['experiments'][name]
            research['experiments'][name]=dict(dates=date_strings(q['dates']),issueTimes=q['issue_dates'].astype(str).tolist(),
                probabilities=q['probability'],truth=q['truth'],cutoff=float(q['threshold']),predicted=q['probability']>=float(q['threshold']),
                variant=entry['selected']['variant'],model='Gradient boosting')
    save('issued.json',issued);save('research.json',research);save('live-model.json',models)
    for source,target in [('researchpaper.md','manuscript.md'),('ISSUED_FORECAST_PROTOCOL.md','issued-protocol.md'),('RESEARCH_PROTOCOL.md','research-protocol.md'),
                          ('results/issued_forecasts/REPORT.md','issued-report.md'),('results/research/REPORT.md','research-report.md'),
                          ('results/issued_forecasts/comparison.csv','issued-comparison.csv'),('results/research/comparison.csv','research-comparison.csv'),
                          ('results/issued_forecasts/onset_event_summary.csv','onset-events.csv')]:shutil.copyfile(ROOT/source,OUT/'reports'/target)
    print('Exported actual saved predictions, verification, forecast trajectories, metrics and frozen error banks.')
    for p in (OUT/'data').glob('*.json'):print(f'{p.name}: {p.stat().st_size/1024:.0f} KB')

if __name__=='__main__':main()
