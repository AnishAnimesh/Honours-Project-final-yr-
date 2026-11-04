# Strengthening HeatCast's models and evidence

This work improves research tooling and adds a separate model amendment. It does not change the UI or overwrite the original experiments. A new model can improve a metric without removing uncertainty, geographic limits or the need for independent observations.

## What can be measured immediately

Run the additional audit against the actual saved predictions:

```bash
python3 -m heatcast.validation_audit
```

Read [the audit report](results/validation_audit/REPORT.md). The audit adds:

- Probability reliability with ten fixed bins and per-bin sample counts, including empty bins.
- Brier skill relative to the saved historical-climatology forecast on the same cases, rather than judging small Brier scores without considering event rarity.
- Ongoing-event episode detection, so a long heatwave does not masquerade as many independent successful events. Episodes contain third-and-later hot days; they are not onset detections. Episodes touching the evaluated season boundary may be truncated.
- Exact-onset hits and false alerts, retained separately from ongoing episodes.
- Paired uncertainty intervals with 3-, 7- and 14-day full-length blocks within each contiguous season. Whole location fields stay together, seasonal sample sizes are retained and blocks never cross winter gaps.

These intervals condition on the existing models and sampled years. They do not measure the full uncertainty from retraining, observations, climate change or new geographic regions. Reliability-bin errors are descriptive and are not evidence of causal model quality.

The first audit finds 149 of 159 labelled ongoing episodes detected by the one-day research model, compared with 61 of 155 exact onset events. Seven-day issued-onset Brier skill is negative against historical climatology. Onset, ongoing-event detection and probability quality therefore need separate conclusions.

## Collect genuinely prospective evidence

```bash
python3 -m heatcast.prospective capture
python3 -m heatcast.prospective verify data/prospective/CAPTURE_ID
```

`capture` requests one explicitly initialized GFS run, records actual UTC receipt time, preserves the raw response and its SHA-256 hash, and saves forecasts before outcomes. It does not accept a historical receipt timestamp. Repeating capture creates a new directory rather than replacing prior forecasts. The original frozen empirical adjustment is used only for May–September targets; other months record temperature forecasts without event probabilities.

Every target starts at least 24, 72 or 168 hours after actual receipt. Targets are deliberately later than the old replay's assumed 00 UTC issue when capture happens during the day. `minimum_lead_days` identifies the model bank, and `actual_lead_hours` records the real horizon. This timing difference, future GFS updates and seasonal shift mean that prospective skill must be assessed separately from the 2026 replay.

Summer captures preserve the threshold trajectory and adjustment-file hashes. Verification uses the captured thresholds, not a newly fitted model or changed reference. It waits until the last required observation day plus six UTC days, then saves a separate ERA5 response, verification results and source hashes. The original capture is unchanged; re-verification creates a separate record, allowing later ERA5 revisions to be inspected.

Run capture regularly through a future summer to build evidence. No recurring OS job is installed by these changes. Local timestamps and exclusive-create files are useful provenance but are not independent timestamp certification or a guarantee against manual file edits. Forecast logs contain no retrospective future outcomes at capture time. ERA5 verification remains distinct from independent station verification.

The [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) preserves initialization-specific forecasts; the ordinary latest Forecast API can stitch updates. [Provider availability metadata](https://open-meteo.com/en/docs/model-updates) also distinguishes initialization, completion and actual availability. Actual response receipt is recorded here rather than inferred from initialization.

## Validate against independent stations

```bash
python3 -m heatcast.station_validation \
  --observations /path/to/station_daily.csv \
  --provenance /path/to/station_provenance.json
```

The station CSV must contain real reference and verification observations:

| Column | Meaning |
| --- | --- |
| `site` | Existing HeatCast location name, such as Paris |
| `station_id` | Stable source station identity |
| `date` | UTC date, YYYY-MM-DD |
| `tmax_c` | Daily maximum temperature in Celsius |
| `quality_ok` | Source-quality decision, true/false or 1/0 |

The provenance JSON requires `source_url`, `license`, `retrieved_utc`, `daily_definition`, and a `stations` array. Each station entry requires `station_id`, `latitude`, `longitude`, and `elevation_m`. `daily_definition` must be `UTC 00:00-24:00 daily maximum`. Do not relabel a local-day or morning-to-morning maximum as a UTC day; obtain compatible hourly observations or explicitly design another evaluation first.

The adapter rejects duplicate site-days, mixed station identities, invalid units/ranges, and insufficient reference coverage. Station-specific 1981–2010 thresholds require at least 90% quality-accepted daily coverage in every reference year and at least 100 observations per five-calendar-day percentile window. Verification requires every event-trajectory day; missing/quality-rejected cases and reference exclusions are reported.

Outputs include matched-case/exclusion CSVs, per-site temperature MAE/RMSE/bias, raw-GFS event skill using station-local thresholds and original grid-model probability comparisons against station events. The latter is explicitly an external target-transfer diagnostic: a probability trained for an ERA5 grid event is not automatically a station-event probability. No station observation is fabricated or downloaded by this adapter, and no independent station result is claimed until real data are supplied. Station proximity, elevation mismatch, relocations and homogenization remain research considerations.

## Improve the model without tuning to known test failures

The separate [improvement-study protocol](IMPROVEMENT_STUDY_PROTOCOL.md) defines a small forecast-based candidate family, purged development folds and validation-only choice. The original research and archived-warning benchmarks stay intact. See [the study report](results/improvement_study/REPORT.md) for measured outcomes, including negative findings.

The 2026 test outcomes have already been inspected, so this amendment is exploratory even when its fitting and selection code never reads test labels. It must be confirmed on future forecasts or another untouched season. A method is not promoted because it happens to have the best reused-test score.

In the first run, the development-selected one-day onset method (0.75-scaled joint errors) reaches F1 0.4478, compared with original adjusted F1 0.3896 and raw-GFS F1 0.3774. The seven-day paired interval for its F1 difference against raw GFS is approximately [-0.031, 0.196], so improvement is not established under that uncertainty analysis. Its probability Brier score also does not improve on the original adjustment. Most other selected methods do not outperform raw GFS. The study is retained as a reproducible result and a diagnostic of weak seasonal transfer, rather than replacing the original models with a claim of universal improvement.

## Evidence still needed

1. A complete prospective summer with actual receipt logs and separately collected verification.
2. Quality-controlled, temporally compatible independent station observations and station reference data.
3. Additional locations, including a northern-region holdout, with equal-site metrics and sparse-event support retained.
4. Further independent seasons and model-update tracking to test whether early-summer adjustments transfer.
5. New model families or ensemble inputs evaluated under a separately frozen protocol if the bounded amendment remains weak.

For geographic expansion, the existing downloader supports `python3 -m heatcast.download_sites --include-northern --output data/europe_sites_northern`. This acquires a separate dataset subject to provider quotas; it does not validate a northern region by itself. Do not overwrite the original dataset or fit on its already evaluated test failures. Run a complete new, documented experiment after acquisition.

## Checks

```bash
python3 -m unittest discover -s tests -v
python3 -m heatcast.improvement_study --check
python3 tests/check_research_artifacts.py
python3 tests/check_issued_artifacts.py
```

New tests cover seasonal gaps, probability-bin endpoints, paired bootstrap comparisons, minimum actual warning lead, off-season abstention, capture-file preservation, verification-time reference isolation, station quality checks and exclusion of future years from station climatology.

## Neural sensitivity and case-level error diagnosis

The fixed supplementary protocol in `NEURAL_SENSITIVITY_PROTOCOL.md` records seeds 7, 21 and 42 for five neural experiments. The seed-42 benchmarks are reused; ten additional fits supply three-seed variation without selecting a favourable test seed. Results and predictions are in `results/neural_sensitivity/`. Run `python3 -m heatcast.neural_sensitivity` to reproduce the study.

`python3 -m heatcast.error_analysis` reconstructs six deterministic post hoc trajectory examples, the changes in false alerts and hits after empirical adjustment, the onset cutoff diagnostic and spatial prevalence/checkpoint context. Its files are in `results/error_analysis/`. These are explanations of existing predictions, not new model selection or causal attribution.

The revised report includes these completed diagnostics, sample and parameter counts, heat-specific literature, equations, module interfaces, a published observatory screenshot and explicit independent-confirmation criteria. The public DOI identity audit is retained in `results/reference_audit/`. Independent station evidence, common-input architecture comparisons and prospective summer verification remain outstanding.
