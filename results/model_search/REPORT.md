# Exploratory model and feature search

Selection uses chronological development folds. Test years were already inspected; these are exploratory results requiring independent confirmation. All values below are held-out-period measurements, not promises of future skill.

| Task | Policy | Development-selected model | Accuracy | F1 | Recall | Original accuracy | Original F1 |
|---|---|---|---:|---:|---:|---:|---:|
| ongoing_lead1 | f1 | xgboost_0 | 0.9645 | 0.8307 | 0.9376 | 0.9636 | 0.8199 |
| ongoing_lead1 | accuracy | xgboost_0 | 0.9645 | 0.8307 | 0.9376 | 0.9636 | 0.8199 |
| onset_lead1 | f1 | ensemble_top3 | 0.9533 | 0.2647 | 0.2903 | 0.9458 | 0.2961 |
| onset_lead1 | accuracy | hist_0 | 0.7847 | 0.1897 | 0.8710 | 0.9458 | 0.2961 |
| ongoing_lead3 | f1 | original_all_features_7 | 0.8771 | 0.4553 | 0.5533 | 0.8627 | 0.4623 |
| ongoing_lead3 | accuracy | ensemble_top6 | 0.7378 | 0.3948 | 0.9215 | 0.8627 | 0.4623 |
| onset_lead3 | f1 | lightgbm_1 | 0.7931 | 0.0705 | 0.2710 | 0.8883 | 0.0597 |
| onset_lead3 | accuracy | xgboost_1 | 0.3556 | 0.0675 | 0.8065 | 0.8883 | 0.0597 |
| ongoing_lead7 | f1 | ensemble_top3 | 0.4161 | 0.2037 | 0.8048 | 0.7544 | 0.2196 |
| ongoing_lead7 | accuracy | catboost_2 | 0.3455 | 0.1948 | 0.8531 | 0.7544 | 0.2196 |
| onset_lead7 | f1 | extra_trees_1 | 0.5600 | 0.0636 | 0.5161 | 0.7455 | 0.0658 |
| onset_lead7 | accuracy | lightgbm_0 | 0.1707 | 0.0581 | 0.8839 | 0.7455 | 0.0658 |

## Interpretation

The clearest candidate is next-day ongoing-event XGBoost: accuracy rises from
0.9636 to 0.9645 and F1 from 0.8199 to 0.8307 while recall rises to 0.9376.
The paired 95% intervals for its differences from the original model are
[-0.0027, 0.0047] for accuracy and [-0.0061, 0.0296] for F1, so this reused-test
experiment does not establish a repeatable improvement.

None of the other development-selected F1 models improves both test accuracy
and F1. One-day onset accuracy rises to 0.9533 while F1 falls to 0.2647.
Three-day ongoing accuracy rises to 0.8771 while F1 falls to 0.4553. The
three- and seven-day onset tasks remain weak. Seven-day ongoing and onset
models deteriorate substantially in accuracy. The engineered features and
larger model family therefore do not provide a route to 98–99% accuracy with
useful heatwave detection on this dataset.

The accuracy policy requires at least 85% pooled recall on development/validation; that is not a guarantee of 85% test recall. F1 cutoffs maximize equal-site F1. Raw versus sigmoid calibration is chosen by validation Brier, not test outcomes.

The six original-feature boosting settings and fixed-parameter engineered comparator allow feature ablation; all other family trials use engineered features. Tree ensembles use date-aligned out-of-fold predictions, with equal weights. OOF/validation results are selection scores.

See `comparison.csv` for every candidate under both policies, `regional_metrics.csv` for champion location scores, and `metrics.json` for confusion matrices, calibration decisions and paired seasonal block intervals versus the original model. Champion identities were frozen in `*_SELECTION.json` before test scoring. Inference bundles reproduce saved champion probabilities without test labels.

## Reproduce

```bash
python3 -m venv --system-site-packages .venv-model-search
.venv-model-search/bin/python -m pip install -r requirements-model-search.txt
.venv-model-search/bin/python -m heatcast.model_search
.venv-model-search/bin/python -m unittest discover -s tests -p test_model_search.py -v
PYTHONPATH=. .venv-model-search/bin/python tests/check_model_search_artifacts.py
```

No original benchmark artifact was replaced. This uses retrospective ERA5 history; future station verification and real-time input availability are not established.
