"""Fixed-reference targets and observation-only features for site experiments."""
import numpy as np
import pandas as pd


def day_index(dates):
    # Feb 29 shares Feb 28's climatology; dates themselves remain consecutive.
    d=pd.DatetimeIndex(dates)
    return np.array([pd.Timestamp(2001,int(v.month),min(int(v.day),28) if v.month==2 else int(v.day)).dayofyear-1 for v in d])


def fit_climatology(temp,dates,reference,quantile=.9,radius=2):
    day=day_index(dates);table=np.empty((365,temp.shape[1]),dtype='float32')
    if len(np.unique(pd.DatetimeIndex(dates[reference]).year))!=30:
        raise ValueError('Reference must contain 30 years')
    for d in range(365):
        delta=np.abs(day-d);keep=reference & (np.minimum(delta,365-delta)<=radius)
        if keep.sum()<100:raise ValueError('Insufficient reference coverage')
        table[d]=np.quantile(temp[keep],quantile,axis=0)
    return table


def event_targets(hot,dates):
    ongoing=np.zeros_like(hot,dtype=bool);onset=np.zeros_like(hot,dtype=bool)
    gap=np.diff(dates).astype('timedelta64[D]')==np.timedelta64(1,'D')
    for t in range(2,len(hot)):
        if gap[t-2:t].all():ongoing[t]=hot[t-2:t+1].all(0)
    for t in range(1,len(hot)-2):
        if gap[t-1:t+2].all():onset[t]=hot[t:t+3].all(0)&~hot[t-1]
    return dict(ongoing=ongoing,onset=onset)


def make_features(raw,dates,threshold,lat,lon,lead,variant='all_features'):
    """Target d uses observations no later than d-lead (end of issue day)."""
    target=np.arange(lead+13,len(dates)-2)
    month=pd.DatetimeIndex(dates[target]).month
    target=target[(month>=5)&(month<=9)]
    day=day_index(dates)
    anomaly=raw[:,:,0]-threshold
    issue=target-lead
    columns=[];names=[]
    def add(name,values):columns.append(values);names.append(name)
    for lag in [0,1,2,3,6,13]:
        add(f'tmax_anomaly_lag{lag}',anomaly[issue-lag])
        add(f'tmin_lag{lag}',raw[issue-lag,:,1])
        if variant!='temperature_only':
            add(f'pressure_lag{lag}',raw[issue-lag,:,2])
            if variant!='no_soil':add(f'soil_lag{lag}',raw[issue-lag,:,3])
            add(f'humidity_lag{lag}',raw[issue-lag,:,4])
    for length in [3,7,14]:
        add(f'anomaly_mean_{length}',np.stack([anomaly[issue-k] for k in range(length)]).mean(0))
        if variant!='temperature_only':
            add(f'rain_sum_{length}',np.stack([raw[issue-k,:,5] for k in range(length)]).sum(0))
    add('anomaly_change',anomaly[issue]-anomaly[issue-1])
    add('local_target_threshold',threshold[target])
    for name,values in [('season_sin',np.sin(2*np.pi*day[target]/365)),('season_cos',np.cos(2*np.pi*day[target]/365))]:
        add(name,np.broadcast_to(values[:,None],(len(target),len(lat))))
    add('latitude',np.broadcast_to(lat[None],(len(target),len(lat))))
    add('longitude',np.broadcast_to(lon[None],(len(target),len(lat))))
    x=np.stack(columns,-1).astype('float32')
    if not np.isfinite(x).all():raise ValueError('Nonfinite features')
    return target,x,names


def macro_f1(y,p,threshold):
    from sklearn.metrics import f1_score
    return float(np.mean([f1_score(y[:,s],p[:,s]>=(threshold if np.ndim(threshold)==0 else threshold[s]),zero_division=0) for s in range(y.shape[1])]))


def validate_daily(raw,dates,sites):
    if raw.shape!=(len(dates),len(sites),6):raise ValueError('Expected six variables and aligned site/date axes')
    if not np.isfinite(raw).all():raise ValueError('Nonfinite daily values')
    if len(np.unique(sites))!=len(sites):raise ValueError('Duplicate site names')
    if not np.all(np.diff(dates).astype('timedelta64[D]')==np.timedelta64(1,'D')):raise ValueError('Duplicated, unordered or missing dates')
    bounds=[(-90,65),(-100,60),(800,1100),(0,1),(0,100),(0,1000)]
    for channel,(lower,upper) in enumerate(bounds):
        if raw[:,:,channel].min()<lower or raw[:,:,channel].max()>upper:
            raise ValueError(f'Variable {channel} outside expected physical/unit bounds')
    if (raw[:,:,1]>raw[:,:,0]+.2).any():raise ValueError('Minimum exceeds maximum temperature')
    return dict(n_days=len(dates),n_sites=len(sites),complete_daily_series=True,
                variable_minima=raw.min(axis=(0,1)).tolist(),variable_maxima=raw.max(axis=(0,1)).tolist())
