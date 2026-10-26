from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def daily_data(path, cache):
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    fields = []
    counts = None
    for name in ['t2m', 'msl', 'swvl1']:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'shortName': '2t' if name == 't2m' else name}, 'indexpath': ''})
        da = ds[name].load()
        if name == 't2m':
            counts = da.resample(time='1D').count().values
            daily = (da - 273.15).resample(time='1D').max()
        else:
            daily = da.resample(time='1D').mean()
        fields.append(daily.values)
    # Exclude partial and absent UTC days; ocean cells are masked separately.
    keep = counts.max(axis=(1, 2)) == 24
    out = dict(x=np.stack(fields, axis=1)[keep].astype('float32'),
               dates=daily.time.values[keep], lat=daily.latitude.values,
               lon=daily.longitude.values)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def calendar_day(dates):
    # Fixed non-leap calendar avoids the March shift in leap years.
    return np.array([pd.Timestamp('2001-' + d.strftime('%m-%d')).dayofyear
                     for d in pd.to_datetime(dates)])


def thresholds(temp, dates, training, radius=15):
    day = calendar_day(dates)
    result = np.empty_like(temp)
    for d in np.unique(day):
        delta = np.abs(day - d)
        reference = training & (np.minimum(delta, 365-delta) <= radius)
        if reference.sum() < 10:
            raise ValueError('Insufficient training reference days')
        result[day == d] = np.nanquantile(temp[reference], .9, axis=0)
    return result


def labels(temp, threshold, dates):
    """Causal ongoing event: today and two previous calendar days exceed threshold."""
    hot = temp > threshold
    y = np.zeros_like(hot, dtype=bool)
    consecutive = np.diff(dates).astype('timedelta64[D]').astype(int) == 1
    for t in range(2, len(hot)):
        if consecutive[t-2:t].all():
            y[t] = hot[t] & hot[t-1] & hot[t-2]
    return y


def sequence_indices(dates, length=7):
    return np.array([t for t in range(length, len(dates))
                     if np.all(np.diff(dates[t-length:t+1]).astype('timedelta64[D]')
                               == np.timedelta64(1, 'D'))])
