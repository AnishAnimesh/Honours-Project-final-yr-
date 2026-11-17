"""Export HeatCast evaluation figures from saved results, without refitting models.

Run: .venv-model-search/bin/python scripts/plot_evaluation_metrics.py
"""
from pathlib import Path
import os
import json
import shutil
import zipfile
import html

os.environ.setdefault('MPLCONFIGDIR', '/tmp/heatcast-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/evaluation_metrics'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 11, 'figure.facecolor': 'white',
                     'axes.prop_cycle': plt.cycler(color=['#167d9a', '#e27b36', '#7b57a1', '#27936c', '#c44e68', '#727c8c'])})
FIGURES = []
TASKS = [f'{t}_lead{d}' for t in ['ongoing', 'onset'] for d in [1, 3, 7]]


def read(name):
    return pd.read_csv(ROOT / name)


def title(name):
    return name.replace('_lead', ' · lead ').replace('_', ' ').title()


def save(fig, name, caption):
    fig.suptitle(caption, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, .015, 1, .94))
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)
    FIGURES.append((name, caption))


def bars(frame, category, metrics, name, caption):
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(11, 4 * len(metrics)), max(4.5, len(frame) * .31)))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        vals = frame[metric].to_numpy()
        ax.barh(np.arange(len(frame)), vals)
        ax.set(yticks=np.arange(len(frame)), yticklabels=frame[category], xlabel=metric.replace('_', ' ').title())
        ax.invert_yaxis()
        ax.set_xlim(0, max(1 if metric != 'brier' else .01, np.nanmax(vals) * 1.2))
        for i, v in enumerate(vals):
            ax.text(v + ax.get_xlim()[1] * .012, i, f'{v:.3f}', va='center', fontsize=8)
        ax.grid(axis='x', alpha=.15)
    save(fig, name, caption)


def heatmap(frame, row, col, metric, name, caption):
    matrix = frame.pivot(index=row, columns=col, values=metric)
    fig, ax = plt.subplots(figsize=(12, max(4.7, len(matrix) * .48)))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set(xticks=range(len(matrix.columns)), xticklabels=[title(str(x)) for x in matrix.columns],
           yticks=range(len(matrix)), yticklabels=matrix.index)
    ax.tick_params(axis='x', rotation=25)
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            v = matrix.iloc[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', color='white' if v > .65 else '#142735', fontsize=9)
    fig.colorbar(im, ax=ax, label=metric.replace('_', ' ').upper())
    save(fig, name, caption)


def diagnostics(records, prefix, caption):
    """Each record is one matched task, with a frozen cutoff and test probabilities."""
    for kind in ['roc', 'precision_recall', 'confusion', 'calibration', 'probability_distribution']:
        cols = 3 if len(records) > 2 else len(records)
        rows = int(np.ceil(len(records) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.1 * rows), squeeze=False)
        for ax, (label, y, p, cutoff) in zip(axes.flat, records):
            y, p = np.asarray(y).ravel().astype(int), np.asarray(p).ravel()
            assert y.shape == p.shape and np.isfinite(p).all()
            assert ((p >= 0) & (p <= 1)).all()
            ax.set_title(label)
            if kind == 'roc':
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y, p):.3f}')
                ax.plot([0, 1], [0, 1], '--', c='gray', lw=1)
                ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.02))
            elif kind == 'precision_recall':
                precision, recall, _ = precision_recall_curve(y, p)
                ax.plot(recall, precision, label=f'AP = {average_precision_score(y, p):.3f}')
                ax.axhline(y.mean(), ls='--', c='gray', label=f'Prevalence = {y.mean():.3f}')
                ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
            elif kind == 'confusion':
                cm = confusion_matrix(y, p >= cutoff, labels=[0, 1])
                ax.imshow(cm, cmap='Blues')
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f'{cm[i,j]:,}\n{["TN", "FP", "FN", "TP"][i*2+j]}', ha='center', va='center', color='white' if cm[i,j] > cm.max() * .55 else '#142735')
                ax.set(xticks=[0, 1], xticklabels=['No event', 'Event'], yticks=[0, 1], yticklabels=['No event', 'Event'], xlabel=f'Predicted · frozen cutoff {cutoff:.3f}', ylabel='Observed')
            elif kind == 'calibration':
                obs, pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
                ax.plot(pred, obs, 'o-', label='Equal-count bins')
                ax.plot([0, 1], [0, 1], '--', c='gray', label='Perfect reliability')
                ax.set(xlabel='Mean predicted probability', ylabel='Observed event frequency', xlim=(0, 1), ylim=(0, 1))
            else:
                for value, lab in [(0, 'No event'), (1, 'Event')]:
                    ax.hist(p[y == value], bins=np.linspace(0, 1, 26), weights=np.full((y == value).sum(), 1 / (y == value).sum()), alpha=.55, label=lab)
                ax.axvline(cutoff, c='#333', ls='--', label='Frozen cutoff')
                ax.set(xlabel='Predicted event probability', ylabel='Fraction within observed class', xlim=(0, 1))
            if kind != 'confusion':
                ax.legend(fontsize=8)
                ax.grid(alpha=.15)
        for ax in list(axes.flat)[len(records):]:
            ax.remove()
        save(fig, f'{prefix}_{kind}', f'{caption} | {kind.replace("_", " ").title()}')


def main():
    research = json.loads((ROOT / 'results/research/metrics.json').read_text())
    grid = json.loads((ROOT / 'results/metrics.json').read_text())
    issued = json.loads((ROOT / 'results/issued_forecasts/metrics.json').read_text())
    grid_frame = read('results/comparison.csv')
    for scope in ['grid', 'Paris']:
        bars(grid_frame[grid_frame.scope == scope], 'model', ['accuracy', 'precision', 'recall', 'f1'], f'01_original_{scope}_classification', f'Original {scope} experiment · test 2025–2026')
    z = np.load(ROOT / 'results/test_predictions.npz')
    records = [(title(m), z['truth'], z[m], grid['models'][m]['test']['threshold']) for m in ['gradient_boosting', 'convlstm', 'convlstm_no_soil']]
    diagnostics(records, '02_original_grid', 'Original France land grid · 2025–2026')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, ['lstm_Paris', 'convlstm', 'convlstm_no_soil']):
        history = pd.DataFrame(grid['models'][m]['history'])
        ax.plot(history.epoch, history.train_loss, label='Training')
        ax.plot(history.epoch, history.validation_loss, label='Validation')
        ax.set(title=title(m), xlabel='Epoch', ylabel='Weighted binary cross entropy'); ax.legend(); ax.grid(alpha=.15)
    save(fig, '03_learning_curves', 'Neural training and validation loss · original experiments')
    daily = read('results/daily_performance.csv')
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    dates = pd.to_datetime(daily.date)
    axes[0].plot(dates, daily.actual_cells, label='Observed'); axes[0].plot(dates, daily.predicted_cells, label='Predicted')
    axes[0].set(ylabel='Event land cells'); axes[0].legend()
    axes[1].scatter(dates, daily.iou, s=9); axes[1].set(ylabel='Spatial intersection over union', xlabel='Target date', ylim=(0, 1.02))
    save(fig, '04_daily_spatial_performance', 'Original selected model · daily extent and spatial overlap')
    for family, report, probkey, cutoffkey, caption in [
        ('research', research, 'probability', 'threshold', 'Seven-location research · 2021–2025'),
        ('issued_forecasts', issued, 'joint_error_adjusted', 'cutoff', 'Archived GFS replay · July 8–September 30, 2026')]:
        records = []
        for task in TASKS:
            a = np.load(ROOT / f'results/{family}/{task}_predictions.npz')
            records.append((title(task), a['truth'], a[probkey], float(a[cutoffkey])))
        diagnostics(records, f'05_{family}', caption)
        frame = read(f'results/{family}/comparison.csv')
        primary = frame[frame.model == 'gradient_boosting'] if family == 'research' else frame[frame.method == 'joint_error_adjusted']
        bars(primary, 'experiment', ['accuracy', 'precision', 'recall', 'f1'], f'06_{family}_classification', caption + ' | primary model')
        bars(primary, 'experiment', ['roc_auc', 'average_precision', 'brier'], f'07_{family}_probability_metrics', caption + ' | Brier: lower is better')
        regional = read(f'results/{family}/regional_metrics.csv')
        if family != 'research': regional = regional[regional.method == 'joint_error_adjusted']
        heatmap(regional, 'site', 'experiment', 'f1', f'08_{family}_regional_f1', caption + ' | regional F1')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, task in zip(axes, ['ongoing', 'onset']):
            if family == 'research':
                series = {'Gradient boosting': [report['experiments'][f'{task}_lead{d}']['test']['f1'] for d in [1, 3, 7]]}
                for m in ['persistence', 'climatology']:
                    series[title(m)] = [report['experiments'][f'{task}_lead{d}']['baselines'][m]['test']['f1'] for d in [1, 3, 7]]
            else:
                series = {title(m): [report['experiments'][f'{task}_lead{d}']['methods'][m]['test']['f1'] for d in [1, 3, 7]] for m in ['raw_gfs', 'joint_error_adjusted', 'delayed_era5_persistence', 'historical_climatology']}
            for label, values in series.items(): ax.plot([1, 3, 7], values, 'o-', label=label)
            ax.set(title=title(task), xlabel='Lead (days)', ylabel='Test F1', ylim=(0, 1), xticks=[1, 3, 7]); ax.legend(fontsize=8); ax.grid(alpha=.2)
        save(fig, f'09_{family}_lead_time', caption + ' | lead-time and baseline comparison')
    # Matched comparisons and validation-only feature ablations.
    matched = read('results/research/comparison.csv')
    matched = matched[matched.experiment.isin(['ongoing_lead1', 'onset_lead1'])].copy()
    matched['label'] = matched.experiment + ' / ' + matched.model
    bars(matched, 'label', ['f1', 'average_precision', 'brier'], '10_matched_models', 'Matched one-day research models · same sites and test dates')
    features = read('results/research/feature_comparisons.csv')
    heatmap(features.groupby(['experiment', 'variant'], as_index=False).validation_macro_f1.max(), 'variant', 'experiment', 'validation_macro_f1', '11_feature_ablation', 'Feature ablation · best validation macro-site F1 across leaf settings')
    annual = read('results/research/yearly_metrics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, task in zip(axes, ['ongoing', 'onset']):
        for lead in [1, 3, 7]:
            f = annual[annual.experiment == f'{task}_lead{lead}']
            ax.plot(f.year, f.f1, 'o-', label=f'{lead}-day')
        ax.set(title=title(task), xlabel='Test year', ylabel='F1', ylim=(0, 1)); ax.legend(); ax.grid(alpha=.2)
    save(fig, '12_annual_robustness', 'Research benchmark · annual performance')
    heatmap(read('results/research/spatial_transfer.csv'), 'held_out_site', 'experiment', 'f1', '13_spatial_transfer', 'Research · location excluded from supervised development; local climatology retained')
    sensitivity = read('results/neural_sensitivity/summary.csv')
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(sensitivity.mean_f1, range(len(sensitivity)), xerr=sensitivity.sd_f1, fmt='o', capsize=5)
    ax.set(yticks=range(len(sensitivity)), yticklabels=sensitivity.experiment, xlabel='Test F1 · mean ± sample standard deviation across 3 seeds', xlim=(0, 1))
    save(fig, '14_neural_seed_sensitivity', 'Neural seed sensitivity · rows belong to their respective experiments')
    unc = read('results/issued_forecasts/uncertainty.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, low, high, label in [(axes[0], 'adjusted_f1_low', 'adjusted_f1_high', 'Adjusted F1 · 95% block-bootstrap interval'), (axes[1], 'difference_vs_raw_low', 'difference_vs_raw_high', 'Paired F1 difference vs raw GFS · 95% interval')]:
        for i, row in unc.iterrows(): ax.plot([row[low], row[high]], [i, i], 'o-')
        ax.set(yticks=range(len(unc)), yticklabels=unc.experiment, xlabel=label); ax.grid(axis='x', alpha=.2)
    axes[1].axvline(0, c='gray', ls='--')
    save(fig, '15_warning_uncertainty', 'Archived GFS replay · conditional uncertainty')
    issued_frame = read('results/issued_forecasts/comparison.csv')
    counts = issued_frame[issued_frame.method.isin(['raw_gfs', 'joint_error_adjusted'])].copy()
    counts['label'] = counts.experiment + ' / ' + counts.method
    bars(counts, 'label', ['event_hits', 'missed_event_days', 'false_alerts'], '16_warning_outcomes', 'Archived GFS replay · event-day hits, misses and false alerts (counts)')
    temp = read('results/issued_forecasts/temperature_metrics.csv').groupby('experiment', as_index=False)[['raw_gfs_mae', 'raw_gfs_rmse']].mean()
    bars(temp, 'experiment', ['raw_gfs_mae', 'raw_gfs_rmse'], '17_temperature_error', 'Raw GFS temperature · equal-site mean MAE and RMSE (°C; lower is better)')
    search = read('results/model_search/comparison.csv')
    champions = search[(search.policy == 'f1') & search.f1_champion].copy()
    assert len(champions) == 6
    champions['label'] = champions.experiment + ' / ' + champions.candidate
    bars(champions, 'label', ['accuracy', 'precision', 'recall', 'f1'], '18_model_search_champions', 'Exploratory expanded model search · development-selected F1 champions')
    diagnostics_records = []
    for row in champions.itertuples():
        a = np.load(ROOT / f'results/model_search/{row.experiment}_predictions.npz')
        idx = list(a['candidates']).index(row.candidate)
        p = a['probability'][idx]
        assert np.isclose(roc_auc_score(a['truth'].ravel(), p.ravel()), row.roc_auc)
        diagnostics_records.append((title(row.experiment) + '\n' + row.candidate, a['truth'], p, row.threshold))
    diagnostics(diagnostics_records, '19_model_search', 'Exploratory model search · frozen development selection; test 2021–2025')
    heatmap(search[search.policy == 'f1'], 'candidate', 'experiment', 'f1', '20_all_search_candidates', 'Exploratory expanded search · all candidate test F1 scores; F1 cutoff policy')
    amendments = read('results/improvement_study/comparison.csv')
    amendments = amendments[amendments.selected | amendments.method.isin(['raw_gfs', 'joint_error_adjusted'])].copy()
    amendments['label'] = amendments.experiment + ' / ' + amendments.method
    bars(amendments, 'label', ['f1', 'brier'], '21_forecast_amendment', 'Forecast amendment · development-selected alternatives and original baselines')
    # Keep source tables with the figures for traceable report preparation.
    tables = OUT / 'source_tables'; tables.mkdir(exist_ok=True)
    for family in ['', 'research', 'issued_forecasts', 'model_search', 'improvement_study', 'neural_sensitivity']:
        for p in (ROOT / 'results' / family).glob('*.csv'):
            shutil.copy2(p, tables / f'{family or "original"}__{p.name}')
    manifest = {'generator': 'scripts/plot_evaluation_metrics.py', 'figures': [{'file': n + '.png', 'caption': c} for n, c in FIGURES], 'note': 'Saved results only. No model training, cutoff optimization, or synthetic values.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    notes = '''# HeatCast evaluation graphs

Open `index.html` to browse every figure. PNG files are exported at 220 dpi; SVG files are scalable for papers and presentations. `source_tables/` contains the saved CSV results.

## Interpretation

- Original France land-grid and Paris results: test 2025–2026. Paris and full-grid methods have separate comparison figures.
- Seven-location research benchmark: test 2021–2025, 1/3/7-day ongoing-event and onset tasks.
- Archived GFS replay: common test targets July 8–September 30, 2026. Verifies against ERA5 under the documented publication assumption.
- Expanded model search and forecast amendments are exploratory studies. Champions use development selection, not test ranking. Search test years were already inspected during project development.
- Accuracy, precision, recall, F1, ROC AUC and average precision: higher is better. Brier, MAE and RMSE: lower is better. AP is average precision, rather than trapezoidal PR AUC.
- Confusion matrices use saved validation/development cutoffs. ROC/PR curves use continuous scores; dashed PR lines show task prevalence. Probability histograms normalize each observed class separately.
- Feature ablations report validation macro-site F1, maximized across the pre-existing leaf settings for each feature variant. They do not measure causal feature effects.
- Confidence intervals are conditional block-bootstrap estimates, while neural error bars are standard deviations across three seeds. These are different quantities.
- Event warning counts are event-day counts, not independent heatwave episode counts. MAE/RMSE are equal-site averages of saved site-level metrics.
- Performance values from different domains, labels and test periods should be interpreted within their own study. Rare onset events make accuracy insufficient on its own.

Reproduce from the project root:
`.venv-model-search/bin/python scripts/plot_evaluation_metrics.py`
'''
    (OUT / 'README.md').write_text(notes)
    cards = '\n'.join(f'<article><h2>{html.escape(c)}</h2><a href="{n}.png"><img loading="lazy" src="{n}.png" alt="{html.escape(c)}"></a><p><a href="{n}.png">PNG</a> · <a href="{n}.svg">SVG</a></p></article>' for n, c in FIGURES)
    (OUT / 'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>HeatCast evaluation graphs</title><style>body{font:16px system-ui;margin:32px auto;max-width:1300px;padding:0 20px;background:#f3f6f8;color:#152b39}article{background:white;padding:20px;margin:24px 0;border-radius:10px}img{max-width:100%;height:auto}h2{font-size:19px}a{color:#126d8a}</style><h1>HeatCast evaluation graphs</h1><p>' + str(len(FIGURES)) + ' figures · saved project results · <a href="README.md">Interpretation and study boundaries</a></p>' + cards + '</html>')
    archive = OUT.parent / 'HeatCast_Evaluation_Graphs.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): zf.write(p, p.relative_to(OUT.parent))
    print(f'Exported {len(FIGURES)} figures as PNG + SVG to {OUT}')
    print(f'Archive: {archive}')


if __name__ == '__main__':
    main()
