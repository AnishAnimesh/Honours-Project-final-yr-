import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                            confusion_matrix, f1_score, precision_recall_curve,
                            precision_score, recall_score, roc_auc_score)


def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y.ravel(), p.ravel())
    score = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1]+recall[:-1], 1e-12)
    return float(thresholds[np.argmax(score)])


def metrics(y, p, threshold):
    y, p = y.ravel(), p.ravel()
    pred = p >= threshold
    return dict(threshold=float(threshold), n=int(y.size), prevalence=float(y.mean()),
                accuracy=float(accuracy_score(y, pred)), f1=float(f1_score(y, pred, zero_division=0)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)), confusion_matrix=confusion_matrix(y, pred, labels=[0,1]).tolist())


def block_bootstrap(y, p, threshold, baseline, dates, seed=42, repeats=500):
    """Paired whole-day, seven-day block resampling; spatial cells stay together."""
    pred = p >= threshold
    def counts(prediction):
        return np.stack([(prediction & y).sum(1), (prediction & ~y).sum(1),
                         (~prediction & y).sum(1)],axis=1)
    a, b = counts(pred), counts(baseline >= .5)
    blocks = []
    for start in range(len(dates)):
        end = min(start+7,len(dates))
        segment = np.arange(start,end)
        if np.all(np.diff(dates[segment]).astype('timedelta64[D]') == np.timedelta64(1,'D')):
            blocks.append(segment)
    rng = np.random.default_rng(seed)
    scores, differences = [], []
    def f1(c):
        tp,fp,fn=c
        return 2*tp/max(2*tp+fp+fn,1)
    for _ in range(repeats):
        sample = []
        while len(sample)<len(dates):
            sample.extend(blocks[rng.integers(len(blocks))])
        sample = np.asarray(sample[:len(dates)])
        score = f1(a[sample].sum(0)); ref=f1(b[sample].sum(0))
        scores.append(score); differences.append(score-ref)
    return dict(method='500 paired 7-day moving-block resamples, whole spatial fields',
                f1_95_interval=np.quantile(scores,[.025,.975]).tolist(),
                f1_difference_from_persistence_95_interval=np.quantile(differences,[.025,.975]).tolist())
