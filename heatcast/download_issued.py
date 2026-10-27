"""Retrieve explicit archived GFS runs and separate ERA5 verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

RUN_URL='https://single-runs-api.open-meteo.com/v1/forecast'
VERIFY_URL='https://archive-api.open-meteo.com/v1/archive'
LEADS=[1,3,7]


def target_dates():
    return dict(development=pd.date_range('2026-05-01','2026-06-22').to_numpy(),
                test=pd.date_range('2026-07-08','2026-09-30').to_numpy())


def required_runs():
    dates=np.concatenate(list(target_dates().values())).astype('datetime64[D]')
    return np.unique(np.concatenate([dates-np.timedelta64(lead+1,'D') for lead in LEADS]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(url,params,path):
    request_hash=hashlib.sha256(json.dumps(dict(url=url,params=params),sort_keys=True).encode()).hexdigest()
    meta=path.with_suffix('.meta.json')
    if path.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if record['request_sha256']!=request_hash:raise ValueError(f'Cached request differs: {path}')
        if digest(path)!=record['sha256']:raise ValueError(f'Cached response checksum differs: {path}')
        return json.loads(path.read_text()),record
    for attempt in range(5):
        try:
            response=requests.get(url,params=params,timeout=(20,150))
            if response.status_code==429:
                if 'hour' in response.text.lower() or 'day' in response.text.lower():
                    raise RuntimeError(f'Provider quota exhausted; resume later. {response.text[:200]}')
                print('Minute quota: pausing 60 seconds',flush=True);time.sleep(60);continue
            if response.status_code>=500:
                print(f'Provider HTTP {response.status_code}; retry {attempt+1}',flush=True);time.sleep(min(5*(attempt+1),30));continue
            if response.status_code>=400:
                raise RuntimeError(f'Archive HTTP {response.status_code}: {response.text[:800]}')
            data=response.json()
            partial=path.with_suffix('.part');partial.write_text(json.dumps(data));partial.replace(path)
            record=dict(url=url,request=params,request_sha256=request_hash,sha256=digest(path),
                        retrieved_utc=datetime.now(timezone.utc).isoformat(),file=path.name)
            meta.write_text(json.dumps(record,indent=2))
            return data,record
        except (requests.ConnectionError,requests.Timeout,requests.exceptions.JSONDecodeError):
            if attempt==4:raise
            time.sleep(min(5*(attempt+1),30))
    raise RuntimeError(f'Request failed: {path}')


def parse_run(data,initialization,n_sites):
    responses=data if isinstance(data,list) else [data]
    if len(responses)!=n_sites:raise ValueError('Wrong number of returned locations')
    fields=[];coordinates=[];daily_dates=None
    for item in responses:
        if item.get('utc_offset_seconds')!=0:raise ValueError('Non-UTC response')
        h=item['hourly'];times=pd.to_datetime(h['time']).to_numpy()
        expected=np.arange(initialization,initialization+np.timedelta64(12,'D'),np.timedelta64(1,'h'))
        if not np.array_equal(times,expected):raise ValueError('Run valid-hour range differs from requested initialization/horizon')
        values=np.array(h['temperature_2m'],dtype='float32')
        if not np.isfinite(values).all():raise ValueError('Missing forecast hours; do not substitute a later run')
        if item['hourly_units']['temperature_2m']!='°C':raise ValueError('Unexpected temperature unit')
        frame=pd.DataFrame(dict(date=times.astype('datetime64[D]'),temperature=values))
        grouped=frame.groupby('date').temperature
        if not (grouped.count()==24).all():raise ValueError('Incomplete UTC forecast day')
        current=grouped.max()
        current_dates=current.index.to_numpy()
        if daily_dates is not None and not np.array_equal(daily_dates,current_dates):raise ValueError('Unaligned forecast dates')
        daily_dates=current_dates;fields.append(current.to_numpy())
        coordinates.append({k:item.get(k) for k in ['latitude','longitude','elevation','utc_offset_seconds','hourly_units']})
    return daily_dates,np.stack(fields,1),coordinates


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--output',default='data/issued_forecasts');a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);runs_root=root/'runs';runs_root.mkdir(exist_ok=True)
    old=json.loads(Path('data/europe_sites/manifest.json').read_text());sites=old['sites']
    lat=[s['returned_metadata'][0]['latitude'] for s in sites];lon=[s['returned_metadata'][0]['longitude'] for s in sites]
    base=dict(latitude=','.join(map(str,lat)),longitude=','.join(map(str,lon)),timezone='GMT',elevation=','.join(['nan']*len(sites)),cell_selection='nearest')
    dates=required_runs()
    if a.probe:dates=dates[:1]
    manifest=dict(source=RUN_URL,documentation='https://open-meteo.com/en/docs/single-runs-api',model='gfs_global',
        attribution='NOAA/NCEP GFS; archive and extraction: Open-Meteo, CC BY 4.0. Verification: Copernicus/ECMWF ERA5 through Open-Meteo.',
        protocol_sha256=digest('ISSUED_FORECAST_PROTOCOL.md'),sites=[s['name'] for s in sites],verification_latitude=lat,verification_longitude=lon,
        forecast_days=12,warning_policy='00 UTC decision; previous-day 00 UTC initialization; 12-hour assumed publication allowance plus 12-hour buffer',runs=[])
    arrays=[];valid_dates=[]
    for i,date in enumerate(dates):
        date=np.datetime64(date,'ns');run=str(date.astype('datetime64[m]'))
        path=runs_root/f'{run[:10]}T00.json'
        params=dict(base,hourly='temperature_2m',models='gfs_global',run=run,forecast_days=12)
        print(f'GFS run {i+1}/{len(dates)}: {run}',flush=True)
        data,record=fetch(RUN_URL,params,path)
        try:
            vd,values,metadata=parse_run(data,date,len(sites))
            record['status']='complete'
        except ValueError as error:
            if 'Missing forecast hours' not in str(error):raise
            vd=date+np.arange(12)*np.timedelta64(1,'D')
            values=np.full((12,len(sites)),np.nan,dtype='float32')
            responses=data if isinstance(data,list) else [data]
            metadata=[{k:item.get(k) for k in ['latitude','longitude','elevation','utc_offset_seconds','hourly_units']} for item in responses]
            record['status']='incomplete_archive_run';record['reason']=str(error)
            print(f'Archive gap: {run}; affected matched targets will be excluded and counted.',flush=True)
        arrays.append(values);valid_dates.append(vd)
        record['file']='runs/'+record['file'];record['initialization']=run;record['returned_metadata']=metadata;manifest['runs'].append(record)
        (root/'manifest.partial.json').write_text(json.dumps(manifest,indent=2))
    if a.probe:
        print(f'Probe passed: {len(sites)} sites, {len(valid_dates[0])} complete forecast days.');return
    params=dict(base,daily='temperature_2m_max',models='era5',start_date='2026-04-01',end_date='2026-10-02')
    data,record=fetch(VERIFY_URL,params,root/'verification_2026.json')
    responses=data if isinstance(data,list) else [data]
    if len(responses)!=len(sites):raise ValueError('Wrong number of verification locations')
    verification=[];vd=pd.date_range('2026-04-01','2026-10-02').to_numpy()
    verification_coordinates=[]
    for i,item in enumerate(responses):
        if not np.array_equal(pd.to_datetime(item['daily']['time']).to_numpy(),vd):raise ValueError('Verification dates incomplete')
        if not np.isclose(item['latitude'],lat[i]) or not np.isclose(item['longitude'],lon[i]):raise ValueError('Verification grid differs from historical reference grid')
        values=np.asarray(item['daily']['temperature_2m_max'],dtype='float32')
        if not np.isfinite(values).all():raise ValueError('Missing verification maximum temperatures')
        if item['daily_units']['temperature_2m_max']!='°C':raise ValueError('Unexpected verification unit')
        verification.append(values);verification_coordinates.append({k:item.get(k) for k in ['latitude','longitude','elevation','daily_units']})
    record['returned_metadata']=verification_coordinates;manifest['verification']=record
    np.savez_compressed(root/'daily_runs.npz',initializations=dates.astype('datetime64[ns]'),valid_dates=np.stack(valid_dates),
        forecast_tmax=np.stack(arrays).astype('float32'),verification_dates=vd,verification_tmax=np.stack(verification,1),
        sites=np.array(manifest['sites']),latitude=np.array(lat),longitude=np.array(lon))
    manifest['prepared_sha256']=digest(root/'daily_runs.npz');manifest['run_count']=len(dates)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Completed {len(dates)} explicit runs, {len(sites)} sites and separate ERA5 verification.',flush=True)

if __name__=='__main__':main()
