# Exploratory forecast-refinement study

Exploratory amendment after inspecting original test; not prospective or an untouched test.

The new methods were designed after reviewing original test weaknesses. Selection uses development only, but this reused test is not untouched evidence. Original published benchmarks remain unchanged.

| Task / lead | Development-selected method | Selected test F1 | Original adjusted F1 | Raw test F1 | Selected macro-site F1 | Test positives |
| --- | --- | --- | --- | --- | --- | --- |
| ongoing_lead1 | error_scale_0.25 | 0.7564 | 0.7744 | 0.7715 | 0.7305 | 105 |
| onset_lead1 | error_scale_0.75 | 0.4478 | 0.3896 | 0.3774 | 0.4286 | 23 |
| ongoing_lead3 | logistic_C0.1_balanced | 0.6978 | 0.7132 | 0.7068 | 0.6053 | 105 |
| onset_lead3 | error_scale_1 | 0.2143 | 0.3409 | 0.3077 | 0.1786 | 23 |
| ongoing_lead7 | logistic_C1_balanced | 0.5706 | 0.5581 | 0.6259 | 0.4439 | 105 |
| onset_lead7 | error_scale_0.25 | 0.0488 | 0.0833 | 0.1154 | 0.0286 | 23 |

## Interpretation

- Selection and probability cutoffs use five development blocks, with seven-day purging and whole locations kept together.
- Development scores are used for tuning and are not unbiased estimates. Every candidate test result is retained in `comparison.csv`.
- Classification and probability quality are separate: see Brier, reliability bins, bin support, and per-location metrics in `metrics.json`.
- No candidate is promoted because it wins on test. A selected method can still worsen the original result.
- Original adjusted scores are copied unchanged for context. The new error_scale_1 candidate uses different purged folds and macro-site cutoff tuning, so it is not the original adjustment.
- The empirical error scales are sensitivity/shrinkage experiments, not a physical ensemble. Logistic probabilities are not guaranteed calibrated.
- Per-site development event frequency is the frozen probability baseline for Brier skill; this differs from the original historical-climatology baseline.
- Paired 3/7/14-day block intervals compare the selected method with raw GFS; they condition on this fitted model and one season.
- Seven grid locations, ERA5 verification, assumed historic receipt timing, and seasonal transfer remain limitations. New independent seasons and station observations are required.

## Reproduction and inference

```bash
python3 -m heatcast.improvement_study
python3 -m heatcast.improvement_study --check
python3 -m unittest discover -s tests -p test_improvement_study.py -v
python3 -m heatcast.improvement_study --issue 2026-07-25 --lead 7 --task onset --forecast-output /tmp/improved-warning.csv
```

Saved bundles contain development-only fitted quantities and historical thresholds. The inference path supplies forecast arrays only; test verification arrays never enter prediction. Archived inference is a reconstruction, not proof the new model issued a warning in 2026.
