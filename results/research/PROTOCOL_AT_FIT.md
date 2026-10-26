# HeatCast research extension protocol

This protocol defines the expanded retrospective study before its final model fits. The earlier France-grid experiment remains a separate exploratory study; its 2025–2026 results have already been inspected.

## Revised objectives

1. Establish a repeatable multi-location European benchmark with substantially longer climate history.
2. Estimate local hot-day thresholds from a fixed, historical 30-year reference rather than three summers.
3. Distinguish ongoing-event prediction from advance first-day-onset warning.
4. Quantify degradation at 1-, 3- and 7-day lead, using only weather observed by the end of the issue day.
5. Evaluate geographic failures with site-wise metrics, equal-site-wseven macro F1, event counts and supervised location holdout.
6. Compare temperature-only, no-soil and full meteorology feature sets and test matched linear and recurrent baselines.
7. Report temporal backtesting, annual results, training-sample sensitivity and dependence-aware uncertainty.
8. Deliver reproducible data provenance, trained models, forecast exports and a research report that preserves negative findings.

## Data and fixed choices

ERA5 through the Open-Meteo archive, model explicitly set to `era5`, UTC days, `elevation=nan` to disable elevation downscaling, and land-cell selection. Seven requested locations are Paris, London, Madrid, Rome, Athens, Berlin and Warsaw. These represent a set of grid cells; they do not constitute full country or continent coverage, city station measurements, or objectively labelled climate zones. Broad geographic groups are descriptive.

Download dates: 1981-01-01 through 2025-12-31, six daily variables: maximum/minimum 2m temperature, mean sea-level pressure, mean top-layer soil moisture, mean relative humidity and total precipitation. Keep JSON responses, actual returned coordinates/units, request settings, retrieval timestamp and SHA-256 hashes. Fail on missing dates or values rather than silently filling targets.

The 90th percentile threshold uses a five-calendar-day window and the 1981–2010 reference. This is deliberately earlier than every validation/test period. It is not the latest 1991–2020 normal. Feb 29 uses the Feb 28 threshold. Targets are evaluated May–September; preceding observations and subsequent target verification can extend outside these months.

An ongoing label is the third or subsequent consecutive hot day. An onset label marks the first hot day of a run lasting at least three days, with a non-hot preceding day. Onset's subsequent two days enter verification only. The three-day definition is project-specific and is not the six-day ETCCDI WSDI index. Labels that could cross a chronological split are embargoed.

## Temporal evaluation and model choice

Primary train: 1981–2016; validation: 2017–2020; test: 2021–2025. Earlier backtest at one-day lead: train 1981–2012, validation 2013–2016, test 2017–2020. Data overlap means the two temporal experiments are not independent replications.

For each task and lead, compare histogram gradient boosting with 7 or 15 leaves and temperature-only, no-soil or full features. Fix 160 iterations, learning rate 0.07 and L2 penalty 10. Fit on training data and choose complexity/feature set by validation macro-site F1. Classification cutoffs maximize pooled validation F1. Do not rank or tune with test results.

At one-day lead, also fit a standardized logistic regression (C=1) and a seven-input, 14-day LSTM (32 hidden units, seed 42, at most 20 epochs, training-only scaling, validation-loss early stopping). Report these on identical dates and sites. Identify the best method using validation macro F1, preserving all test comparisons. The primary gradient-boosting result and its dedicated ablations remain separately identifiable.

Temperature persistence assumes that unobserved future daily maximum temperatures equal the last observed daily maximum, evaluated against the known reference thresholds and event rule. It does not use retrospective onset labels as predictors. Seasonal climatology uses training-only event frequencies in a ±15-day window. Baseline cutoffs also use validation only.

Per-site cutoffs are a predeclared secondary diagnostic: use local validation cutoffs only with at least 20 validation positives; otherwise use the pooled cutoff. Report both; do not select between them using test F1. Local cutoff tuning does not guarantee improved future performance.

## Robustness and geographical transfer

For one-day tasks, omit each site from supervised training, feature/complexity selection and cutoff tuning, then evaluate that site's 2021–2025 observations. Local 1981–2010 climate thresholds are still available; this is supervised-location transfer with known climatology, not a location with no historical data. Sites can be geographically/climatologically related, so this is not equivalent to buffered continent-scale spatial generalization.

Training sensitivity resamples whole training days, retaining all seven sites together, with seeds 7, 21 and 42. This measures sample sensitivity rather than merely changing a deterministic tree algorithm's seed. Use paired seven-day block bootstrap of entire spatial fields, 500 replicates, for conditional test F1 and difference from temperature persistence. Report annual results and sparse-event support. These procedures do not establish prospective real-time performance or remove reanalysis uncertainty.

## Primary outcomes and limitations

Report pooled/macro F1, precision, recall, average precision, ROC AUC, Brier score, confusion matrices, and regional/annual distributions. Longer-lead and onset results must remain visible even when poor. The original regional grid is audited cell by cell and by fixed geographic quadrants, without changing its test-selected behaviour.

The study is retrospective and revised after earlier exploratory work. There is no claim of untouched prospective validation, globally optimal models, station-level accuracy, complete European coverage, or operational warnings from delayed ERA5. A defensible research contribution is a transparent comparison and characterization of failures, not a guaranteed higher headline score.

## Research basis

- ERA5 data source and fixed-model guidance: https://open-meteo.com/en/docs/historical-weather-api
- Climate reference periods: https://wmo.int/wmo-climatological-normals
- Percentile-window and WSDI definitions: https://climate-scenarios.canada.ca/?page=climdex-indices
- Structured validation: Roberts et al. (2017), https://doi.org/10.1111/ecog.02881
- Probabilistic verification: Dimitriadis et al. (2024), https://doi.org/10.1016/j.ijforecast.2023.09.007

## Data-availability amendment before final model fitting

The archive hourly quota prevented completion of Stockholm after seven locations had complete 1981–2025 series. The final benchmark was frozen to those seven locations based on data availability, before its final fits, with no missing-year filling or mixed-source substitution. Northern Europe beyond these locations remains unvalidated. Cached Stockholm responses are retained; `python -m heatcast.download_sites --include-northern --output data/europe_sites_northern` can acquire a separate expanded dataset when quota permits. That expansion would require a new complete evaluation; no northern-coverage claim is made here.
