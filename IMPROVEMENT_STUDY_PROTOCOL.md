# Exploratory issued-forecast improvement amendment

This protocol is written after the original 2026 test results were inspected.
It is an exploratory amendment, not a historically frozen June 2026 protocol
and not a new untouched prospective evaluation. Original study files and model
bundles remain unchanged. A positive result needs confirmation on a new season.

## Fixed scope

Use the same seven locations, fixed 1981–2010 reference, frozen GFS trajectories,
three-day event definitions, 1/3/7-day leads and matched development/test dates
as the original issued-warning study. Never use later model runs or recent
verification measurements as prediction features. Enforce the original timing
and source checks. No new downloads or geographic performance claims.

## Candidate set and selection

For each task/lead, compare exactly these nine methods:

1. Raw deterministic GFS event rule, cutoff 0.5.
2. Empirical joint-error adjustment at scales 0.25, 0.5, 0.75 and 1.0. Scale
   multiplies the entire development error vector, retaining dependence among
   days. The scale is not a physical uncertainty ensemble or calibration proof.
3. Four pooled logistic models: C in {0.1, 1.0}, with and without balanced
   class weights. Inputs are only the forecast-minus-local-threshold margins
   for the relevant event days, their minimum, maximum, mean, standard
   deviation and consecutive changes. Standardization is fitted on each
   training fold only. No site or future verification features are supplied.

Split the ordered development dates into five contiguous blocks. For each
validation block, exclude every training date within seven calendar days of
any date in that block. Whole location fields stay together. At least twenty
training target dates are required. Train residual banks and logistic models
only from the remaining development dates, then score the block. This is
offline blocked cross-fitting at the end of development, not rolling-origin
forecast delivery within development.

Each probabilistic candidate's cutoff maximizes macro-site F1 on its combined
out-of-fold development probabilities. Ties use the higher cutoff, favoring
fewer alerts. Choose the candidate with highest development macro-site F1;
ties prefer lower development Brier, then the declared candidate order. Both
candidate and cutoff tuning use development labels; these development scores
are selection scores, not unbiased performance estimates.

Write selection metadata before accessing test verification. Refit the chosen
candidate (and all diagnostic alternatives) using all development dates, and
retain its unchanged cutoff on the existing July–September test period.
Do not choose a second winner based on test scores. Store an inference bundle
with development-only residuals or logistic estimator, historical thresholds,
site order, exact source/protocol hashes and the retrospective status.

## Reporting

Report every candidate's pooled and macro-site F1, precision, recall, Brier,
average precision, confusion matrix and event support. For the selected model,
report per-location results, equal-width reliability bins and Brier skill
against a frozen per-site development event-frequency baseline. Reliability
is descriptive; class weighting or an empirical scenario fraction does not
establish calibration. Retain zero-positive locations and empty bins.

Report paired whole-location 3/7/14-day moving-block uncertainty against raw
GFS, conditional on this fitted model and this one season. Report deterioration
as openly as improvement. Do not merge these scores with the observation-only
research benchmark. Independent station validation, prospective delivery,
additional seasons, new regions and unseen-test confirmation remain open.

## Reproduction

```bash
python3 -m heatcast.improvement_study
python3 -m heatcast.improvement_study --check
python3 -m unittest discover -s tests -p test_improvement_study.py -v
python3 -m heatcast.improvement_study --issue 2026-07-25 --lead 7 --task onset --forecast-output /tmp/improved-warning.csv
```

Inference is a reconstruction using a newly designed model and archived runs.
It must never be described as a warning actually issued in July 2026.
