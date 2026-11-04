"""Independent station verification from quality-controlled, provenance-backed CSVs.

This adapter never downloads or invents station observations. Grid-model
probabilities retain their original target; station-event comparisons are
explicit external-transfer diagnostics, not a refit or operational validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import metrics
from .issued_data import event_rule, offsets
from .research_data import day_index


def read_station(path):
    frame = pd.read_csv(path, dtype={'site': str, 'station_id': str, 'quality_ok': str})
    required = {'site', 'station_id', 'date', 'tmax_c', 'quality_ok'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Station CSV requires columns: {sorted(required)}')
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing station identity, date, temperature or quality flag')
    frame['date'] = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='raise')
    if frame.duplicated(['site', 'date']).any():
        raise ValueError('Duplicate site/date observations')
    if (frame.groupby('site').station_id.nunique() != 1).any():
        raise ValueError('One stable station identity per site is required')
    if frame.groupby('station_id').site.nunique().gt(1).any():
        raise ValueError('One station cannot count as multiple independent sites')
    flags = frame.quality_ok.astype(str).str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('quality_ok must be true/false or 1/0')
    frame['quality_ok'] = flags.isin(['true', '1'])
    frame['tmax_c'] = pd.to_numeric(frame.tmax_c, errors='raise')
    if not np.isfinite(frame.tmax_c).all() or not frame.tmax_c.between(-90, 65).all():
        raise ValueError('Invalid station maximum temperatures in Celsius')
    return frame.sort_values(['site', 'date'])


def station_reference(frame, minimum_coverage=.9):
    """Fixed 1981–2010, five-calendar-day local percentile, station data only."""
    valid = frame.loc[frame.quality_ok].copy()
    valid = valid.loc[valid.date.dt.year.between(1981, 2010)]
    years = valid.date.dt.year
    for year in range(1981, 2011):
        days_in_year = 366 if pd.Timestamp(year, 12, 31).dayofyear == 366 else 365
        if (years == year).sum() / days_in_year < minimum_coverage:
            raise ValueError(f'Insufficient station reference coverage for {year}')
    index = day_index(valid.date.to_numpy())
    table = np.empty(365)
    for d in range(365):
        delta = np.abs(index - d)
        keep = np.minimum(delta, 365 - delta) <= 2
        if keep.sum() < 100:
            raise ValueError(f'Insufficient station reference window at calendar day {d}')
        table[d] = np.quantile(valid.loc[keep, 'tmax_c'], .9)
    return table


def station_cases(z, frame, task, reference):
    """Matched complete trajectories only; retain exclusion reasons."""
    included, excluded = [], []
    for s, site in enumerate(z['sites']):
        local = frame.loc[(frame.site == str(site)) & frame.quality_ok].set_index('date')
        table = reference.get(str(site))
        for i, target in enumerate(z['target_dates']):
            days = target.astype('datetime64[D]') + offsets(task) * np.timedelta64(1, 'D')
            stamps = pd.DatetimeIndex(days)
            base = dict(site=str(site), target_date=str(target)[:10])
            if table is None:
                excluded.append(dict(**base, reason='No complete 1981–2010 station reference'))
                continue
            if not stamps.isin(local.index).all():
                excluded.append(dict(**base, reason='Missing or quality-rejected event trajectory'))
                continue
            actual = local.loc[stamps, 'tmax_c'].to_numpy()
            threshold = table[day_index(days)]
            predicted_temperature = z['forecast_trajectory'][i, s]
            target_position = 2 if task == 'ongoing' else 1
            included.append(dict(**base, station_id=str(local.station_id.iloc[0]),
                                 station_event=bool(event_rule(actual, threshold, task)),
                                 raw_gfs_station_threshold_event=bool(event_rule(predicted_temperature, threshold, task)),
                                 grid_model_probability=float(z['joint_error_adjusted'][i, s]),
                                 grid_model_cutoff=float(z['cutoff']),
                                 station_target_tmax_c=float(actual[target_position]),
                                 gfs_target_tmax_c=float(predicted_temperature[target_position]),
                                 target_temperature_error_c=float(predicted_temperature[target_position] - actual[target_position])))
    return included, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True, help='CSV containing reference and verification station records')
    parser.add_argument('--provenance', required=True, help='JSON: source_url, license, retrieved_utc, daily_definition, stations')
    parser.add_argument('--results', default='results/issued_forecasts')
    parser.add_argument('--output', default='results/station_validation')
    args = parser.parse_args()
    path = Path(args.observations)
    provenance = json.loads(Path(args.provenance).read_text())
    if not {'source_url', 'license', 'retrieved_utc', 'daily_definition', 'stations'}.issubset(provenance):
        raise ValueError('Station source, license, receipt time, daily definition and metadata are required')
    if provenance['daily_definition'] != 'UTC 00:00-24:00 daily maximum':
        raise ValueError('Station days must match the benchmark UTC daily maximum; do not silently mix local-day maxima')
    frame = read_station(path)
    declared = {str(row['station_id']): row for row in provenance['stations']}
    for station in frame.station_id.unique():
        if str(station) not in declared or not {'latitude', 'longitude', 'elevation_m'}.issubset(declared[str(station)]):
            raise ValueError('Every station needs coordinates/elevation in provenance')
    references, reference_errors = {}, {}
    for site, group in frame.groupby('site'):
        try:
            references[site] = station_reference(group)
        except ValueError as error:
            reference_errors[site] = str(error)
    if not references:
        raise ValueError(f'No sites have a complete station reference: {reference_errors}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(observation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  provenance=provenance, reference_exclusions=reference_errors,
                  scope='External station-target transfer diagnostic. Original empirical probabilities were fitted to ERA5 grid events, not station events.',
                  experiments={})
    all_cases, all_excluded = [], []
    for task in ['ongoing', 'onset']:
        for lead in [1, 3, 7]:
            name = f'{task}_lead{lead}'
            source = Path(args.results) / f'{name}_predictions.npz'
            z = np.load(source)
            cases, excluded = station_cases(z, frame, task, references)
            all_cases.extend(dict(experiment=name, **row) for row in cases)
            all_excluded.extend(dict(experiment=name, **row) for row in excluded)
            entry = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), matched_cases=len(cases), excluded_cases=len(excluded), sites={})
            groups = pd.DataFrame(cases).groupby('site') if cases else []
            for site, group in groups:
                y = group.station_event.to_numpy(dtype=bool)
                error = group.target_temperature_error_c.to_numpy()
                entry['sites'][site] = dict(n=len(group), positive_events=int(y.sum()),
                    target_temperature_mae_c=float(np.abs(error).mean()),
                    target_temperature_rmse_c=float(np.sqrt(np.mean(error ** 2))),
                    target_temperature_bias_c=float(error.mean()),
                    raw_gfs_station_threshold=metrics(y, group.raw_gfs_station_threshold_event.to_numpy(dtype=float), .5),
                    original_grid_probability_station_transfer=metrics(y, group.grid_model_probability.to_numpy(), float(z['cutoff'])))
            report['experiments'][name] = entry
    pd.DataFrame(all_cases).to_csv(output / 'matched_cases.csv', index=False)
    pd.DataFrame(all_excluded).to_csv(output / 'excluded_cases.csv', index=False)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f'Station diagnostics saved to {output}; no model or cutoff changed.')


if __name__ == '__main__':
    main()
