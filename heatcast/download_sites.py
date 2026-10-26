"""Download a fixed-model ERA5 multi-location research benchmark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

SITES = [
    ('Paris',48.85,2.35,'western'), ('London',51.51,-.13,'western'),
    ('Madrid',40.42,-3.70,'southern'), ('Rome',41.90,12.50,'southern'),
    ('Athens',37.98,23.73,'southern'), ('Berlin',52.52,13.41,'central_eastern'),
    ('Warsaw',52.23,21.01,'central_eastern'), ('Stockholm',59.33,18.07,'northern')]
VARIABLES = ['temperature_2m_max','temperature_2m_min','pressure_msl_mean',
             'soil_moisture_0_to_7cm_mean','relative_humidity_2m_mean','precipitation_sum']
URL='https://archive-api.open-meteo.com/v1/archive'


def fetch(params, path):
    if path.exists():
        data=json.loads(path.read_text())
        if 'daily' in data: return data
    for attempt in range(8):
        try:
            response=requests.get(URL,params=params,timeout=(20,180))
            if response.status_code==429 and 'Hourly' in response.text:
                raise RuntimeError('Archive hourly quota exhausted. Responses are cached; rerun after the quota resets rather than repeatedly polling.')
            if response.status_code==429 or response.status_code>=500:
                wait=min(65*(attempt+1),300)
                print(f'Retry HTTP {response.status_code}: {response.text[:200]}; waiting {wait}s',flush=True)
                time.sleep(wait);continue
            response.raise_for_status()
            data=response.json()
            if 'daily' not in data: raise ValueError(f'No daily observations: {str(data)[:300]}')
            partial=path.with_suffix('.part');partial.write_text(json.dumps(data));partial.replace(path)
            return data
        except (requests.ConnectionError,requests.Timeout) as error:
            if attempt==7: raise
            print(f'Retrying {type(error).__name__}',flush=True);time.sleep(min(15*(attempt+1),60))
    raise RuntimeError('Archive request failed after retries')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='data/europe_sites')
    p.add_argument('--diagnose',action='store_true')
    p.add_argument('--reuse-cache',default='data/europe_sites',help='Reuse identical existing response files when creating a separate expanded dataset')
    p.add_argument('--include-northern',action='store_true',help='Also download Stockholm when API quota permits; default benchmark uses seven complete sites')
    a=p.parse_args()
    if a.diagnose:
        response=requests.get(URL,params=dict(latitude=59.33,longitude=18.07,start_date='1996-01-01',end_date='2010-12-31',daily=','.join(VARIABLES),models='era5',timezone='GMT',elevation='nan'),timeout=30)
        print(response.status_code,response.text[:600]);return
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    selected_sites=SITES if a.include_northern else SITES[:-1]
    manifest=dict(source=URL,documentation='https://open-meteo.com/en/docs/historical-weather-api',
                  attribution='ERA5: Copernicus Climate Change Service / ECMWF; API: Open-Meteo, CC BY 4.0',
                  model='era5',timezone='GMT',elevation='nan (disable elevation downscaling)',
                  retrieved_utc=datetime.now(timezone.utc).isoformat(),sites=[])
    arrays=[]; all_dates=None
    # Three blocks limit individual request weight and permit resume after interruption.
    for name,lat,lon,region in selected_sites:
        frames=[];files=[]; metadata=[]
        for start,end in [(1981,1995),(1996,2010),(2011,2025)]:
            params=dict(latitude=lat,longitude=lon,start_date=f'{start}-01-01',end_date=f'{end}-12-31',
                        daily=','.join(VARIABLES),models='era5',timezone='GMT',elevation='nan',cell_selection='land')
            path=root/f'{name}_{start}_{end}.json'
            print(f'Downloading {name} {start}–{end}',flush=True)
            reusable=Path(a.reuse_cache)/path.name
            if not path.exists() and reusable.exists() and reusable.resolve()!=path.resolve():
                shutil.copyfile(reusable,path)
            response=fetch(params,path)
            frame=pd.DataFrame(response['daily']).set_index('time')
            frames.append(frame[VARIABLES]); files.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),request=params,retrieved_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()))
            metadata.append({k:response.get(k) for k in ['latitude','longitude','elevation','daily_units']})
        frame=pd.concat(frames);dates=pd.to_datetime(frame.index).to_numpy()
        expected=pd.date_range('1981-01-01','2025-12-31').to_numpy()
        if not np.array_equal(dates,expected): raise ValueError(f'Incomplete/duplicated dates for {name}')
        values=frame.to_numpy(dtype='float32')
        if not np.isfinite(values).all(): raise ValueError(f'Missing values for {name}; do not silently impute targets')
        if all_dates is not None and not np.array_equal(dates,all_dates):raise ValueError('Unaligned sites')
        all_dates=dates;arrays.append(values)
        manifest['sites'].append(dict(name=name,requested_latitude=lat,requested_longitude=lon,region=region,returned_metadata=metadata,files=files))
        (root/'manifest.partial.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(root/'daily.npz',x=np.stack(arrays,axis=1),dates=all_dates,
                        sites=np.array([s[0] for s in selected_sites]),lat=np.array([s[1] for s in selected_sites]),
                        lon=np.array([s[2] for s in selected_sites]),regions=np.array([s[3] for s in selected_sites]),variables=np.array(VARIABLES))
    manifest['shape']=list(np.stack(arrays,axis=1).shape)
    manifest['daily_sha256']=hashlib.sha256((root/'daily.npz').read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Completed {len(selected_sites)} sites and {len(all_dates)} days per site.',flush=True)

if __name__=='__main__':main()
