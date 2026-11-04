"""Capture timestamped, single-run forecasts before outcomes, then verify later.

Capture and verification are separate immutable files. No past capture time can
be supplied on the CLI. Summer event probabilities remain experimental.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import requests

from .download_issued import RUN_URL, VERIFY_URL, parse_run
from .issued_data import offsets, event_rule, adjusted_risk
from .research_data import day_index


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc_now():
    return datetime.now(timezone.utc)


def iso(value):
    return value.isoformat().replace('+00:00', 'Z')


def immutable_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def forecast_rows(valid_dates, forecast, sites, receipt, results):
    """Targets start at least lead*24 hours after actual receipt, never before.

    The extra calendar day is deliberate: a midday receipt is not a next-day
    24-hour warning. Inference loads no ERA5 verification arrays.
    """
    receipt_day = np.datetime64(receipt.date(), 'D')
    days = np.asarray(valid_dates).astype('datetime64[D]')
    values = np.asarray(forecast, dtype=float)
    if values.shape != (len(days), len(sites)) or not np.isfinite(values).all():
        raise ValueError('Forecast arrays are incomplete')
    rows, bundles = [], {}
    for lead in [1, 3, 7]:
        target = receipt_day + np.timedelta64(lead + 1, 'D')
        summer = int(str(target)[5:7]) in [5, 6, 7, 8, 9]
        for task in ['ongoing', 'onset']:
            name = f'{task}_lead{lead}'
            required = target + offsets(task) * np.timedelta64(1, 'D')
            positions = np.searchsorted(days, required)
            if (positions >= len(days)).any() or not np.array_equal(days[positions], required):
                raise ValueError('Requested event trajectory is not available in this single run')
            trajectory = values[positions].T[None].astype('float32')
            probability = cutoff = raw = threshold = None
            if summer:
                bundle_path = Path(results) / f'{name}_adjustment.joblib'
                bundle = joblib.load(bundle_path)
                if list(sites) != bundle['sites']:
                    raise ValueError('Site order differs from frozen adjustment')
                if np.datetime64(receipt.replace(tzinfo=None), 'ns') <= bundle['freeze']:
                    raise ValueError('Adjustment must be frozen before receipt')
                threshold = bundle['reference_table'][day_index(required)].T[None]
                probability = adjusted_risk(trajectory, threshold, bundle['error_bank'], task)[0]
                raw = event_rule(trajectory, threshold, task)[0]
                cutoff = float(bundle['cutoff'])
                bundles[name] = sha(bundle_path.read_bytes())
            target_time = datetime.fromisoformat(str(target)).replace(tzinfo=timezone.utc)
            lead_hours = (target_time - receipt).total_seconds() / 3600
            for s, site in enumerate(sites):
                rows.append(dict(site=str(site), task=task, minimum_lead_days=lead,
                                 actual_lead_hours=lead_hours, target_date=str(target),
                                 trajectory_dates=required.astype(str).tolist(),
                                 forecast_tmax_c=trajectory[0, s].tolist(),
                                 local_hot_threshold_c=threshold[0, s].tolist() if summer else None,
                                 probability=float(probability[s]) if summer else None,
                                 raw_gfs_event=bool(raw[s]) if summer else None,
                                 cutoff=cutoff, predicted_event=bool(probability[s] >= cutoff) if summer else None,
                                 scope='experimental summer event forecast' if summer else 'temperature only: outside evaluated season'))
    return rows, bundles


def capture(output='data/prospective', results='results/issued_forecasts', session=None):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    initialization = np.datetime64(started.date(), 'D') - np.timedelta64(1, 'D')
    manifest = json.loads(Path('data/issued_forecasts/manifest.json').read_text())
    sites = manifest['sites']
    params = dict(latitude=','.join(map(str, manifest['verification_latitude'])),
                  longitude=','.join(map(str, manifest['verification_longitude'])),
                  timezone='GMT', elevation=','.join(['nan'] * len(sites)), cell_selection='nearest',
                  hourly='temperature_2m', models='gfs_global', forecast_days=12,
                  run=str(initialization) + 'T00:00')
    client = session or requests
    response = client.get(RUN_URL, params=params, timeout=(20, 120))
    response.raise_for_status()
    raw = response.content
    received = utc_now()
    # Midnight rollover would change target semantics; retry as a new capture.
    if received.date() != started.date():
        raise ValueError('UTC day changed during retrieval; retry capture')
    days, temperatures, coordinates = parse_run(json.loads(raw), initialization.astype('datetime64[ns]'), len(sites))
    rows, bundles = forecast_rows(days, temperatures, sites, received, results)
    identity = received.strftime('%Y%m%dT%H%M%S%fZ')
    directory = root / identity
    directory.mkdir()
    with (directory / 'response.json').open('xb') as stream:
        stream.write(raw)
    record = dict(schema_version=1, capture_id=identity, request_started_utc=iso(started),
                  received_utc=iso(received), initialization_utc=str(initialization) + 'T00:00:00Z',
                  endpoint=RUN_URL, request=params, response_sha256=sha(raw),
                  returned_coordinates=coordinates, model='gfs_global', sites=sites,
                  adjustment_sha256=bundles, warnings=rows,
                  verification_status='pending',
                  timing='Actual receipt; targets are at least minimum_lead_days*24h after receipt.',
                  limitation='Local receipt log, not independently timestamp-certified. Experimental adjustment; no station verification.')
    immutable_json(directory / 'forecast.json', record)
    print(f'Captured {len(rows)} forecasts at {record["received_utc"]}: {directory}')
    return directory


def verify(directory, session=None):
    directory = Path(directory)
    forecast_path = directory / 'forecast.json'
    record = json.loads(forecast_path.read_text())
    if sha((directory / 'response.json').read_bytes()) != record['response_sha256']:
        raise ValueError('Saved forecast response checksum mismatch')
    required = sorted({day for row in record['warnings'] for day in row['trajectory_dates']})
    now = utc_now()
    # Keep the benchmark's conservative D+6 publication policy.
    ready = datetime.fromisoformat(required[-1]).replace(tzinfo=timezone.utc)
    if np.datetime64(now.replace(tzinfo=None), 'ns') < np.datetime64(ready.replace(tzinfo=None), 'ns') + np.timedelta64(6, 'D'):
        raise ValueError('Verification is not yet available under the D+6 policy')
    params = {key: record['request'][key] for key in ['latitude', 'longitude', 'timezone', 'elevation', 'cell_selection']}
    params.update(daily='temperature_2m_max', models='era5', start_date=required[0], end_date=required[-1])
    response = (session or requests).get(VERIFY_URL, params=params, timeout=(20, 120))
    response.raise_for_status()
    payload = response.json()
    observations = payload if isinstance(payload, list) else [payload]
    if len(observations) != len(record['sites']):
        raise ValueError('Verification location count differs')
    labels = []
    for row in record['warnings']:
        s = record['sites'].index(row['site'])
        item = observations[s]
        if item.get('utc_offset_seconds') != 0 or item['daily_units']['temperature_2m_max'] != '°C':
            raise ValueError('Unexpected verification timezone or units')
        expected_lat = float(record['request']['latitude'].split(',')[s])
        expected_lon = float(record['request']['longitude'].split(',')[s])
        if not np.isclose(item['latitude'], expected_lat) or not np.isclose(item['longitude'], expected_lon):
            raise ValueError('Verification grid differs from reference')
        dates = item['daily']['time']
        values = np.array([item['daily']['temperature_2m_max'][dates.index(day)] for day in row['trajectory_dates']], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Incomplete verification temperatures')
        # Verify against the threshold actually recorded at capture, never a
        # subsequently changed reference or fitted model.
        if row['probability'] is not None:
            threshold = np.asarray(row['local_hot_threshold_c'], dtype=float)
            if threshold.shape != values.shape or not np.isfinite(threshold).all():
                raise ValueError('Captured threshold trajectory is invalid')
            truth = bool(event_rule(values, threshold, row['task']))
        else:
            truth = None
        error = values - np.array(row['forecast_tmax_c'])
        labels.append(dict(site=row['site'], task=row['task'], minimum_lead_days=row['minimum_lead_days'],
                           target_date=row['target_date'], observed_tmax_c=values.tolist(),
                           trajectory_mae_c=float(np.abs(error).mean()), truth=truth,
                           predicted_event=row['predicted_event'], probability=row['probability'],
                           matched=truth == row['predicted_event'] if truth is not None else None))
    identity = now.strftime('%Y%m%dT%H%M%S%fZ')
    raw_name = f'verification-response-{identity}.json'
    with (directory / raw_name).open('xb') as stream:
        stream.write(response.content)
    result = dict(verified_utc=iso(now), forecast_sha256=sha(forecast_path.read_bytes()),
                  threshold_source='Original capture; no updated climatology loaded',
                  endpoint=VERIFY_URL, request=params, response_file=raw_name,
                  response_sha256=sha(response.content), source='ERA5; not independent station observations', results=labels)
    immutable_json(directory / f'verification-{identity}.json', result)
    print(f'Verified {len(labels)} recorded forecasts; original capture unchanged.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('capture')
    command.add_argument('--output', default='data/prospective')
    command.add_argument('--results', default='results/issued_forecasts')
    command = commands.add_parser('verify')
    command.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'capture':
        capture(args.output, args.results)
    else:
        verify(args.directory)


if __name__ == '__main__':
    main()
