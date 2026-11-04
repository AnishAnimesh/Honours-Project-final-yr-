"""Repeat fixed neural protocols without selecting a seed on test outcomes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import sequence_indices
from .evaluation import choose_threshold, metrics
from .models import LSTM, ConvLSTM
from .research_data import day_index, event_targets, make_features
from .research_neural import neural_comparison
from .run import train


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/neural_sensitivity')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    protocol = Path('NEURAL_SENSITIVITY_PROTOCOL.md')
    (out / 'PROTOCOL_AT_FIT.md').write_text(protocol.read_text())
    records = []

    def save_record(name, seed, y, pv, pt, vy, history, parameters, model_file=None):
        cutoff = choose_threshold(vy, pv)
        row = dict(experiment=name, seed=seed, parameter_count=parameters,
                   epochs_completed=len(history), validation=metrics(vy, pv, cutoff),
                   test=metrics(y, pt, cutoff),
                   source='saved benchmark' if seed == 42 else 'fixed-protocol retraining',
                   checkpoint_sha256=digest(model_file) if model_file else None)
        records.append(row)
        np.savez_compressed(out / f'{name}_seed{seed}_predictions.npz',
                            truth=y, probability=pt, validation_probability=pv,
                            validation_truth=vy, cutoff=cutoff)
        (out / 'metrics.partial.json').write_text(json.dumps(records, indent=2))
        print(f"COMPLETE {name} seed={seed} F1={row['test']['f1']:.4f}", flush=True)

    raw_data = np.load(next(Path('results').glob('daily_*.npz')))
    prep = np.load('results/preprocessing.npz')
    source_report = json.loads(Path('results/metrics.json').read_text())
    saved = np.load('results/test_predictions.npz')
    raw, dates, mask = raw_data['x'], raw_data['dates'], prep['mask']
    if not np.array_equal(dates, prep['dates']):
        raise ValueError('Spatial cache differs from saved preprocessing')
    years = pd.DatetimeIndex(dates).year.to_numpy()
    hot = raw[:, 0] > prep['temperature_threshold']
    y = event_targets(hot[:, mask], dates)['ongoing']
    y_grid = np.zeros((len(dates), *mask.shape), dtype=bool)
    y_grid[:, mask] = y
    x = np.nan_to_num((raw - prep['mean']) / np.maximum(prep['std'], 1e-6)).astype('float32')
    x *= mask[None, None]
    idx = sequence_indices(dates)
    splits = dict(train=idx[years[idx] <= 2022], validation=idx[(years[idx] >= 2023) & (years[idx] <= 2024)], test=idx[years[idx] >= 2025])
    if not np.array_equal(dates[splits['test']], saved['dates']) or not np.array_equal(y[splits['test']], saved['truth']):
        raise ValueError('Spatial labels or target dates differ from the benchmark')
    point = (int(np.argmin(abs(raw_data['lat'] - 48.85))), int(np.argmin(abs(raw_data['lon'] - 2.35))))
    for name, channels, location in [('lstm_Paris', [0, 1, 2], point), ('convlstm', [0, 1, 2], None), ('convlstm_no_soil', [0, 1], None)]:
        model = LSTM(len(channels)) if location else ConvLSTM(len(channels))
        params = sum(v.numel() for v in model.parameters())
        for seed in [42, 7, 21]:
            output = out / f'{name}_seed{seed}_predictions.npz'
            if seed == 42:
                from .run import predict, Windows
                from torch.utils.data import DataLoader
                checkpoint = torch.load(Path('results') / f'{name}.pt', map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['state_dict'])
                pv = predict(model, DataLoader(Windows(x, y_grid, splits['validation'], channels, location), batch_size=16))
                pt = saved[name]
                history = source_report['models'][name]['history']
            else:
                model, history, probabilities = train(x, y_grid, splits, channels, mask, location, 35, seed)
                pv, pt = probabilities['validation'], probabilities['test']
                torch.save(dict(state_dict=model.state_dict(), seed=seed), out / f'{name}_seed{seed}.pt')
            vy = y_grid[splits['validation']] if location is None else y_grid[splits['validation'], point[0], point[1]]
            truth = y_grid[splits['test']] if location is None else y_grid[splits['test'], point[0], point[1]]
            if location is None:
                vy, truth = vy[:, mask], truth[:, mask]
                if pv.ndim == 3: pv = pv[:, mask]
                if pt.ndim == 3: pt = pt[:, mask]
            checkpoint_file = Path('results') / f'{name}.pt' if seed == 42 else out / f'{name}_seed{seed}.pt'
            save_record(name, seed, truth, pv, pt, vy, history, params, checkpoint_file)

    z = np.load('data/europe_sites/daily.npz')
    climate = np.load('results/research/climatology.npz')
    threshold = climate['table'][day_index(z['dates'])]
    raw, dates = z['x'], z['dates']
    targets = event_targets(raw[:, :, 0] > threshold, dates)
    idx, _, _ = make_features(raw, dates, threshold, z['lat'], z['lon'], 1)
    years = pd.DatetimeIndex(dates).year.to_numpy()
    target_year, label_end = years[idx], years[idx + 2]
    tr = np.flatnonzero(label_end <= 2016)
    va = np.flatnonzero((target_year >= 2017) & (label_end <= 2020))
    te = np.flatnonzero((target_year >= 2021) & (label_end <= 2025))
    for task in ['ongoing', 'onset']:
        name = f'{task}_lead1_lstm'
        y = targets[task][idx]
        reference = np.load(f'results/research/{task}_lead1_comparisons.npz')
        if not np.array_equal(reference['truth'], y[te]):
            raise ValueError('Location labels differ from benchmark')
        for seed in [42, 7, 21]:
            if seed == 42:
                checkpoint_path = Path(f'results/research/{task}_lead1_lstm.pt')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                values = np.concatenate([raw, (raw[:, :, 0] - threshold)[:, :, None]], axis=-1)
                values = ((values - np.array(checkpoint['mean'])) / np.array(checkpoint['std'])).astype('float32')
                sequences = np.stack([values[idx[va] - 1 - lag] for lag in range(13, -1, -1)], axis=2).reshape(-1, 14, 7)
                model = LSTM(7)
                model.load_state_dict(checkpoint['state_dict']); model.eval()
                with torch.no_grad():
                    pv = np.concatenate([model(torch.from_numpy(sequences[start:start + 256].copy())).sigmoid().numpy()
                                         for start in range(0, len(sequences), 256)]).reshape(len(va), -1)
                pt, history = reference['lstm'], checkpoint['history']
            else:
                checkpoint, predictions = neural_comparison(raw, raw[:, :, 0] - threshold, idx, 1, y, tr, va, te, z['lat'], z['lon'], seed=seed)
                checkpoint_path = out / f'{name}_seed{seed}.pt'
                torch.save(checkpoint, checkpoint_path)
                pv, pt, history = predictions['validation'], predictions['test'], checkpoint['history']
            save_record(name, seed, y[te], pv, pt, y[va], history, sum(v.numel() for v in LSTM(7).parameters()), checkpoint_path)

    summary = []
    for name in sorted({r['experiment'] for r in records}):
        group = [r for r in records if r['experiment'] == name]
        values = np.array([r['test']['f1'] for r in group])
        summary.append(dict(experiment=name, n_seeds=len(group), mean_f1=float(values.mean()),
                            sd_f1=float(values.std(ddof=1)), min_f1=float(values.min()), max_f1=float(values.max())))
    report = dict(protocol_sha256=digest(protocol), seeds=[7, 21, 42],
                  status='Retrospective seed sensitivity; no seed selected or benchmark replaced.',
                  spatial_target_days={k:len(v) for k,v in splits.items()},
                  location_target_days=dict(train=len(tr), validation=len(va), test=len(te)),
                  experiments=records, summary=summary)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
