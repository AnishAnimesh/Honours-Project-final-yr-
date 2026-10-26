"""Run with python -m heatcast.run. Selection uses 2023–24; test is 2025–26."""
import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader, Dataset
from .data import daily_data, thresholds, labels, sequence_indices, calendar_day
from .models import LSTM, ConvLSTM
from .evaluation import choose_threshold, metrics, block_bootstrap


class Windows(Dataset):
    def __init__(self, x, y, indices, channels, point=None):
        self.x, self.y, self.indices, self.channels, self.point = x, y, indices, channels, point
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        t = self.indices[i]
        x = self.x[t-7:t, self.channels]
        y = self.y[t].astype('float32')
        if self.point is not None:
            a, b = self.point
            x, y = x[:, :, a, b], y[a, b]
        return torch.from_numpy(x.copy()), torch.as_tensor(y)


def predict(model, loader):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(x).sigmoid().numpy() for x, _ in loader])


def train(x, y, splits, channels, mask, point, epochs, seed):
    torch.manual_seed(seed)
    loaders = {s: DataLoader(Windows(x, y, idx, channels, point), batch_size=16,
                            shuffle=s == 'train') for s, idx in splits.items()}
    model = LSTM(len(channels)) if point else ConvLSTM(len(channels))
    rate = y[splits['train']][:, mask].mean() if point is None else y[splits['train'], point[0], point[1]].mean()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.sqrt((1-rate)/max(rate, 1e-6))), reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    best, state, history, stale = float('inf'), None, [], 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loaders['train']:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss = loss[:, mask].mean() if point is None else loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vals = []
            for xb, yb in loaders['validation']:
                loss = loss_fn(model(xb), yb)
                vals.append((loss[:, mask].mean() if point is None else loss.mean()).item() * len(xb))
        val = sum(vals)/len(splits['validation'])
        history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_loss=val))
        print(f'{"LSTM" if point else "ConvLSTM"} channels={channels} epoch={epoch+1} val={val:.5f}', flush=True)
        if val < best:
            best, state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(state)
    return model, history, {s: predict(model, loaders[s]) for s in ['validation', 'test']}


def tabular(x, anomaly, hot, dates, idx, mask, soil=True):
    a = []
    for lag in [1,2,3,5,7]:
        a.extend([anomaly[idx-lag][:, mask], x[idx-lag, 1][:, mask]])
        if soil:
            a.append(x[idx-lag, 2][:, mask])
    a.extend([hot[idx-1][:,mask], hot[idx-2][:,mask],
              (anomaly[idx-1]-anomaly[idx-2])[:,mask]])
    doy = calendar_day(dates)[idx]
    for f in [np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)]:
        a.append(np.broadcast_to(f[:,None], (len(idx), mask.sum())))
    return np.stack(a, -1).reshape(-1, len(a)).astype('float32')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.grib')
    parser.add_argument('--output', default='results')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    # Cache is keyed by source identity; no silent reuse for changed GRIB files.
    source = Path(args.data)
    identity = hashlib.sha256(f'{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}'.encode()).hexdigest()[:12]
    data = daily_data(source, out / f'daily_{identity}.npz')
    raw, dates = data['x'], data['dates']
    years = pd.to_datetime(dates).year.to_numpy()
    training = years <= 2022
    mask = np.isfinite(raw[training]).all(axis=(0,1)) & (raw[training,2].mean(axis=0) > .001)
    threshold = thresholds(raw[:,0], dates, training)
    y = labels(raw[:,0], threshold, dates)
    indices = sequence_indices(dates)
    splits = dict(train=indices[years[indices]<=2022], validation=indices[(years[indices]>=2023)&(years[indices]<=2024)], test=indices[years[indices]>=2025])
    if any(len(v) == 0 for v in splits.values()):
        raise ValueError('All temporal splits must have valid sequences')
    mean = np.nanmean(raw[training][:,:,mask], axis=(0,2))[None,:,None,None]
    std = np.nanstd(raw[training][:,:,mask], axis=(0,2))[None,:,None,None]
    x = np.nan_to_num((raw-mean)/np.maximum(std,1e-6)).astype('float32')
    x[:,:,:, :] *= mask[None,None]
    point = (int(np.argmin(abs(data['lat']-48.85))), int(np.argmin(abs(data['lon']-2.35))))
    report = dict(protocol='train 2020–2022; validation 2023–2024; test 2025–2026; next UTC day; 7-day contiguous input',
                  seed=args.seed, data=dict(source=str(source), size_bytes=source.stat().st_size,
                  sha256=hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(), first=str(dates[0]), last=str(dates[-1]), complete_days=len(dates), grid_shape=list(mask.shape), land_cells=int(mask.sum()), land_mask='training mean swvl1 > 0.001 m3/m3; excludes near-zero ocean encoding; proxy rather than official land fraction',
                  latitude=[float(data['lat'].min()),float(data['lat'].max())], longitude=[float(data['lon'].min()),float(data['lon'].max())]),
                  splits={s:dict(n_days=len(i), first=str(dates[i[0]]),last=str(dates[i[-1]]), prevalence=float(y[i][:,mask].mean())) for s,i in splits.items()}, models={})
    predictions = {}
    def register(name, probabilities, location=None, fixed=None):
        target = {s: y[idx][:,mask] if location is None else y[idx, location[0], location[1]] for s,idx in splits.items() if s!='train'}
        probs = {s: p[:,mask] if p.ndim==3 else p for s,p in probabilities.items()}
        t = choose_threshold(target['validation'], probs['validation']) if fixed is None else fixed
        report['models'][name] = dict(scope='grid' if location is None else 'Paris',
                                      validation=metrics(target['validation'],probs['validation'],t), test=metrics(target['test'],probs['test'],t))
        predictions[name] = probs['test']
        return t
    for location, suffix in [(None,'grid'),(point,'Paris')]:
        register('persistence_'+suffix, {s:y[i-1].astype(float) for s,i in splits.items() if s!='train'} if location is None else {s:y[i-1,point[0],point[1]].astype(float) for s,i in splits.items() if s!='train'}, location, .5)
        day = calendar_day(dates)
        climate = np.stack([y[training & (abs(day-d)<=15)].mean(0) for d in day])
        register('climatology_'+suffix, {s:climate[i] if location is None else climate[i,point[0],point[1]] for s,i in splits.items() if s!='train'}, location)
    for name, channels, location in [('lstm_Paris',[0,1,2],point), ('convlstm',[0,1,2],None), ('convlstm_no_soil',[0,1],None)]:
        model, history, probs = train(x,y,splits,channels,mask,location,args.epochs,args.seed)
        t = register(name,probs,location)
        report['models'][name]['history'] = history
        torch.save(dict(state_dict=model.state_dict(), channels=channels, point=location, threshold=t,
                        mean=mean.tolist(),std=std.tolist(),seed=args.seed),out/f'{name}.pt')
    anomaly = raw[:,0]-threshold
    hot = anomaly>0
    features = {s:tabular(x,anomaly,hot,dates,i,mask) for s,i in splits.items()}
    targets = {s:y[i][:,mask].ravel() for s,i in splits.items()}
    rng = np.random.default_rng(args.seed)
    subset = rng.choice(len(targets['train']), min(250000,len(targets['train'])),replace=False)
    candidates = []
    for leaves in [7,15,31]:
        model = HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=leaves, learning_rate=.07,l2_regularization=10,early_stopping=False, random_state=args.seed)
        model.fit(features['train'][subset],targets['train'][subset])
        pv = model.predict_proba(features['validation'])[:,1]
        t = choose_threshold(targets['validation'],pv)
        score = metrics(targets['validation'],pv,t)['f1']
        print(f'Boosting leaves={leaves} validation F1={score:.4f}',flush=True)
        candidates.append((score,model,pv,leaves))
    _, model, pv, leaves = max(candidates,key=lambda v:v[0])
    shape = (len(splits['test']),int(mask.sum()))
    register('gradient_boosting',dict(validation=pv.reshape(len(splits['validation']),-1),test=model.predict_proba(features['test'])[:,1].reshape(shape)))
    joblib.dump(model,out/'gradient_boosting.joblib')
    report['models']['gradient_boosting']['selected_leaves'] = leaves
    # Select separately within matched spatial scopes, without test ranking.
    report['selected_grid_model'] = max((n for n,v in report['models'].items() if v['scope']=='grid'),key=lambda n:report['models'][n]['validation']['f1'])
    report['selected_Paris_model'] = max((n for n,v in report['models'].items() if v['scope']=='Paris'),key=lambda n:report['models'][n]['validation']['f1'])
    selected = report['selected_grid_model']
    report['selected_model_uncertainty'] = block_bootstrap(y[splits['test']][:,mask],predictions[selected],report['models'][selected]['test']['threshold'],predictions['persistence_grid'],dates[splits['test']],args.seed)
    np.savez_compressed(out/'test_predictions.npz', dates=dates[splits['test']], truth=y[splits['test']][:,mask], mask=mask,lat=data['lat'],lon=data['lon'], threshold=threshold[splits['test']], **predictions)
    np.savez_compressed(out/'preprocessing.npz',mean=mean,std=std,mask=mask,dates=dates,temperature_threshold=threshold)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    from .plots import create_report
    create_report(report, predictions, y[splits['test']][:,mask], dates[splits['test']],mask,data,out)
    print(json.dumps({n:v['test'] for n,v in report['models'].items()},indent=2),flush=True)

if __name__ == '__main__':
    main()
