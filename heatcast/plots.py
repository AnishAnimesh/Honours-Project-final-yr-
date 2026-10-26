import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'heatcast-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def create_report(report, predictions, truth, dates, mask, data, out):
    rows = []
    for name,v in report['models'].items():
        rows.append(dict(model=name,scope=v['scope'],validation_f1=v['validation']['f1'],**v['test']))
    frame = pd.DataFrame(rows)
    frame.drop(columns=['confusion_matrix']).to_csv(out/'comparison.csv',index=False)
    fig, ax = plt.subplots(figsize=(8,6))
    for name,v in report['models'].items():
        if v['scope'] == 'grid':
            precision,recall,_ = precision_recall_curve(truth.ravel(),predictions[name].ravel())
            ax.plot(recall,precision,label=f"{name} (AP={v['test']['average_precision']:.3f})")
    ax.axhline(truth.mean(),color='grey',ls='--')
    ax.set(xlabel='Recall',ylabel='Precision',title='Held-out grid: precision–recall',xlim=(0,1),ylim=(0,1)); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'precision_recall.png',dpi=160); plt.close(fig)
    selected=report['selected_grid_model']; p=predictions[selected]; t=report['models'][selected]['test']['threshold']
    daily=pd.DataFrame(dict(date=dates,actual_cells=truth.sum(1),predicted_cells=(p>=t).sum(1),intersection=((p>=t)&truth).sum(1),union=((p>=t)|truth).sum(1)))
    daily['iou']=daily.intersection/daily.union.replace(0,np.nan)
    daily.to_csv(out/'daily_performance.csv',index=False)
    august=(pd.to_datetime(dates).year==2026)&(pd.to_datetime(dates).month==8)
    if august.any():
        k=np.where(august)[0][np.argmax(truth[august].sum(1))]
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,vals,title in zip(axes,[truth[k],p[k],p[k]>=t],['Observed ongoing event','Forecast probability','Forecast event']):
            grid=np.full(mask.shape,np.nan); grid[mask]=vals
            im=ax.imshow(grid,vmin=0,vmax=1,extent=[data['lon'].min(),data['lon'].max(),data['lat'].min(),data['lat'].max()]); ax.set(title=title,xlabel='Longitude',ylabel='Latitude'); fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f'{selected}: {str(dates[k])[:10]} (largest observed August 2026 extent)')
        fig.tight_layout(); fig.savefig(out/'august_case_study.png',dpi=160); plt.close(fig)
    from sklearn.metrics import roc_curve, confusion_matrix
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fpr, tpr, _ = roc_curve(truth.ravel(), p.ravel())
    axes[0].plot(fpr, tpr); axes[0].plot([0,1],[0,1], '--', color='grey')
    axes[0].set(title='Held-out ROC', xlabel='False positive rate', ylabel='True positive rate')
    observed, predicted = calibration_curve(truth.ravel(), p.ravel(), n_bins=10, strategy='quantile')
    axes[1].plot(predicted, observed, 'o-'); axes[1].plot([0,1],[0,1], '--', color='grey')
    axes[1].set(title='Probability calibration', xlabel='Predicted probability', ylabel='Observed frequency')
    matrix=confusion_matrix(truth.ravel(), (p>=t).ravel(), labels=[0,1])
    axes[2].imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[2].text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',color='red')
    axes[2].set(title=f'Confusion matrix (cutoff {t:.3f})', xlabel='Predicted', ylabel='Observed',xticks=[0,1],yticks=[0,1])
    fig.suptitle(selected); fig.tight_layout(); fig.savefig(out/'selected_diagnostics.png',dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(13,4))
    for ax,name in zip(axes,['lstm_Paris','convlstm','convlstm_no_soil']):
        history=report['models'][name]['history']
        ax.plot([v['epoch'] for v in history],[v['train_loss'] for v in history],label='Training')
        ax.plot([v['epoch'] for v in history],[v['validation_loss'] for v in history],label='Validation')
        ax.set(title=name,xlabel='Epoch',ylabel='Weighted BCE'); ax.legend()
    fig.tight_layout(); fig.savefig(out/'learning_curves.png',dpi=160); plt.close(fig)
    text=['# HeatCast evaluation report','',f"Protocol: {report['protocol']}.",'',f"Validation-selected grid model: **{selected}**.",'',markdown_table(frame[['model','scope','validation_f1','f1','average_precision','roc_auc','brier']]),'',
          'Thresholds, neural checkpoints, boosting complexity, and model selection use validation data only. Test results are reported once after selection. All grid methods share the same days and land cells. Paris methods form a separate comparison.', '',
          'Labels indicate the third and subsequent consecutive hot days. They do not retrospectively label the first two days. Thresholds are local 90th percentiles in a ±15-calendar-day window using 2020–2022 only. This short reference is an experimental threshold, not a 30-year climate normal.', '',
          f"Selected model F1 95% block-bootstrap interval: {report['selected_model_uncertainty']['f1_95_interval']}. Paired F1 difference from persistence interval: {report['selected_model_uncertainty']['f1_difference_from_persistence_95_interval']}. These intervals are conditional on this fitted model and test period.", '',
          'The case study selects the day with greatest observed August 2026 extent for descriptive inspection only. It does not establish that the event was unprecedented or validate a causal climate explanation.', '',
          'Reanalysis inputs are assumed available at the end of each UTC day. ERA5 publication latency prevents this experiment from demonstrating an operational real-time warning service. Results cover the available regional land grid, not all Europe. One seed and one held-out period do not establish robustness.']
    (out/'REPORT.md').write_text('\n'.join(text))


def markdown_table(frame):
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(rows)
