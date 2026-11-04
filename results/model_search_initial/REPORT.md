# Exploratory model and feature search

Selection uses chronological development folds. Test years were already inspected; these are exploratory results requiring independent confirmation. All values below are held-out-period measurements, not promises of future skill.

| Task | Policy | Development-selected model | Accuracy | F1 | Recall | Original accuracy | Original F1 |
|---|---|---|---:|---:|---:|---:|---:|
| ongoing_lead1 | f1 | xgboost_0 | 0.9645 | 0.8307 | 0.9376 | 0.9636 | 0.8199 |
| ongoing_lead1 | accuracy | xgboost_0 | 0.9645 | 0.8307 | 0.9376 | 0.9636 | 0.8199 |
| onset_lead1 | f1 | random_forest_1 | 0.9554 | 0.2364 | 0.2387 | 0.9458 | 0.2961 |
| onset_lead1 | accuracy | hist_0 | 0.7847 | 0.1897 | 0.8710 | 0.9458 | 0.2961 |

The accuracy policy requires at least 85% pooled recall on development/validation; that is not a guarantee of 85% test recall. F1 cutoffs maximize equal-site F1. Raw versus sigmoid calibration is chosen by validation Brier, not test outcomes.

The six original-feature boosting settings and fixed-parameter engineered comparator allow feature ablation; all other family trials use engineered features. Tree ensembles use date-aligned out-of-fold predictions, with equal weights. OOF/validation results are selection scores.

See `comparison.csv` for every candidate under both policies, `regional_metrics.csv` for champion location scores, and `metrics.json` for confusion matrices, calibration decisions and paired seasonal block intervals versus the original model. Champion identities were frozen in `*_SELECTION.json` before test scoring. Inference bundles reproduce saved champion probabilities without test labels.

## Reproduce

```bash
python3 -m pip install -r requirements-model-search.txt
python3 -m heatcast.model_search
python3 -m unittest discover -s tests -p test_model_search.py -v
```

No original benchmark artifact was replaced. This uses retrospective ERA5 history; future station verification and real-time input availability are not established.
