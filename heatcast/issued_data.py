"""Time-safe event trajectories and development-only forecast error adjustment."""
import numpy as np


def offsets(task):
    if task=='ongoing':return np.array([-2,-1,0])
    if task=='onset':return np.array([-1,0,1,2])
    raise ValueError(f'Unknown task {task}')


def event_rule(temperature,threshold,task):
    hot=temperature>threshold
    if task=='ongoing':return hot.all(axis=-1)
    if task=='onset':return (~hot[...,0]) & hot[...,1:].all(axis=-1)
    raise ValueError(f'Unknown task {task}')


def trajectory(archive,target_dates,lead,task):
    """Every event day uses one explicit initialization; never a later run."""
    targets=np.asarray(target_dates).astype('datetime64[ns]')
    initialization=targets-np.timedelta64(lead+1,'D')
    issues=targets-np.timedelta64(lead,'D')
    lookup={v:i for i,v in enumerate(archive['initializations'].astype('datetime64[ns]').astype('int64'))}
    rows=[];valid=[]
    for init,target in zip(initialization,targets):
        key=init.astype('int64')
        if key not in lookup:raise ValueError(f'Missing exact initialized run {init}; later runs are forbidden')
        r=lookup[key];days=target+offsets(task)*np.timedelta64(1,'D')
        run_days=archive['valid_dates'][r].astype('datetime64[ns]')
        selected=np.searchsorted(run_days,days)
        if np.any(selected>=len(run_days)) or not np.array_equal(run_days[selected],days):raise ValueError('Event days unavailable in frozen run')
        rows.append(archive['forecast_tmax'][r,selected].T);valid.append(days)
    forecast=np.stack(rows)
    if not np.isfinite(forecast).all():raise ValueError('Missing forecast trajectory')
    availability=initialization+np.timedelta64(12,'h')
    if not np.all(availability<issues):raise ValueError('Forecast not available before decision under the stated policy')
    if not np.all(targets-issues==np.timedelta64(lead,'D')):raise ValueError('Incorrect warning lead')
    return dict(forecast=forecast,valid_dates=np.stack(valid),initialization=initialization,
                assumed_available=availability,issue=issues,target=targets)


def verification(archive,valid_dates):
    dates=archive['verification_dates'].astype('datetime64[ns]')
    positions=np.searchsorted(dates,valid_dates)
    if np.any(positions>=len(dates)) or not np.array_equal(dates[positions],valid_dates):raise ValueError('Missing verification days')
    return archive['verification_tmax'][positions].transpose(0,2,1)


def adjusted_risk(forecast,threshold,error_bank,task,prediction_dates=None,bank_dates=None,exclude_days=None):
    """Enumerate empirical joint errors. Exclusion is used only for development cross-fitting."""
    bank=error_bank.transpose(1,0,2)
    result=[]
    for i,values in enumerate(forecast):
        keep=np.ones(len(error_bank),dtype=bool)
        if exclude_days is not None:
            if prediction_dates is None or bank_dates is None:raise ValueError('Cross-fitting requires bank/prediction dates')
            delta=np.abs((bank_dates-prediction_dates[i]).astype('timedelta64[D]').astype(int))
            keep=delta>exclude_days
        if keep.sum()<20:raise ValueError('Fewer than 20 eligible development error trajectories')
        scenarios=values[:,None,:]+bank[:,keep,:]
        events=event_rule(scenarios,threshold[i,:,None,:],task)
        result.append(events.mean(axis=-1))
    return np.stack(result)


def latest_usable_era5(issue_dates,delay_days=5):
    """A complete UTC day D is considered available at start of D+1+delay."""
    return np.asarray(issue_dates).astype('datetime64[ns]')-np.timedelta64(delay_days+1,'D')


def validate_timeline(development,test,freeze,verification_delay_days=5):
    last_usable=np.max(development['valid_dates']+np.timedelta64(verification_delay_days+1,'D'))
    if last_usable>freeze:raise ValueError('Development verification unavailable at calibration freeze')
    if np.any(test['issue']<=freeze):raise ValueError('Warning precedes development freeze')
    if np.any(test['assumed_available']>=test['issue']):raise ValueError('Forecast availability does not precede warning')
    return last_usable


def eligible_targets(archive,dates,leads=(1,3,7)):
    selected=[];excluded=[]
    for target in dates:
        reasons=[]
        for lead in leads:
            for task in ['ongoing','onset']:
                try:trajectory(archive,np.array([target]),lead,task)
                except ValueError as error:reasons.append(dict(lead=lead,task=task,reason=str(error)))
        if reasons:excluded.append(dict(target_date=str(target)[:10],reasons=reasons))
        else:selected.append(target)
    return np.asarray(selected,dtype='datetime64[ns]'),excluded
