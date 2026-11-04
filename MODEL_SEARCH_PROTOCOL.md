# Exploratory model and feature search

This amendment was designed after the 2021–2025 test results were inspected.
It is exploratory, not an untouched or prospective evaluation. Original research,
neural, and issued-forecast studies remain unchanged. No 98–99% result is promised.

## Data and scope

Use the existing seven-location ERA5 daily dataset, fixed 1981–2010 reference,
May–September targets, and original ongoing/onset definitions at 1/3/7-day lead.
Predictors end at the issue day. This remains a retrospective observation-history
experiment; it does not establish real-time ERA5 availability. No new weather data.

Compare original all-weather features against added exact 14-day temperature
margins, consecutive hot-day counts (capped at 30), temperature trends/extrema/
variability, nighttime temperature and diurnal range, 7/14/30-day rainfall totals,
soil-moisture changes, and location indicators. No normalization uses future data.
The original model's temperature-only/no-soil candidates remain available as
comparators, but new families use the engineered full-weather feature set.

## Selection, calibration and thresholds

Search HistGradientBoosting, XGBoost, CatBoost, LightGBM, Extra Trees and Random
Forest. Use three seeded sampled parameter settings per family, plus a matched
fixed-parameter HistGradientBoosting feature ablation and original-feature
comparators. Exact candidate settings are written before fitting. Three expanding
folds validate 2011–2012, 2013–2014 and 2015–2016 (all after the fixed climate
reference); training always ends before the
validation year and excludes labels whose verification crosses that boundary.
Whole date/location fields stay together. Input history may overlap earlier
training dates: it contains only information available before each target.

After the initial run demonstrated excessive dense Random Forest fitting time,
the completed initial results were retained in `results/model_search_initial`.
The final bounded search uses 100 Random Forest trees, depths 6/8/10,
minimum leaf sizes 20/50/100, and square-root or half-feature split searches.
Other model families retain their original settings. Cache transfer requires
identical data, package versions, candidate parameters, and AST-equivalent
feature, fold, fitting and scoring functions. The transfer provenance is saved.
The final report is a separate full run, not a replacement of the initial results.

Rank by equal-site F1 with a pooled cutoff chosen on concatenated chronological
out-of-fold predictions. These are tuning scores, not unbiased estimates. Take
the best candidate per family and compare equal-weight ensembles of the top
three families and all six. Select one F1 champion and one accuracy champion
(pooled recall at least 0.85) using only these development predictions.

Refit on targets whose verification ends by 2016. Fit sigmoid calibration on
out-of-fold training probabilities only. Choose raw versus calibrated probability
using Brier on 2017–2020 validation; tune both final threshold policies on that
validation period. Use all locations together when selecting thresholds. Keep
validation scores explicitly identified as selection scores.

Record champion identities before accessing test labels. Evaluate all candidates
and ensembles on 2021–2025, reporting accuracy, F1, precision, recall, balanced
accuracy, macro-site F1, AP, Brier, confusion counts and per-location scores.
Never choose a new champion from these reused-test scores. Report both threshold
policies, all-negative baseline, changes against saved original predictions, and
paired whole-location seasonal block intervals for champion differences.

Save reproducible parameters, source/data hashes, feature names, predictions,
champion inference bundles, and an honest report including negative findings.
Independent future-season and station confirmation remain outstanding.

## Run

Install `requirements-model-search.txt`, then run:

```bash
python3 -m heatcast.model_search
python3 -m unittest discover -s tests -p test_model_search.py -v
```

The default output is `results/model_search`. Existing completed candidate-fold
probabilities are reused only when their full data/config/source fingerprint
matches. Use a different output directory for a different experiment. Default
three trials per family can be changed explicitly with `--trials`; this changes
the recorded candidate family and should be reported as a different search.
