# Archived issued-forecast evaluation protocol

## Question and scope

Could HeatCast have produced useful 1-, 3-, and 7-day heatwave warnings using forecasts initialized before the event? This extension replays explicit archived NOAA GFS Global runs at the seven existing ERA5 verification locations. The provider documents non-ECMWF single-run archives from 2026-04-02. Its earlier ECMWF archive is described as hindcasts, so 2024–2025 ECMWF runs are not used as evidence of contemporaneously issued forecasts.

The experiment is frozen before evaluation. It is a retrospective archive replay for one 2026 summer, not a prospective deployment or proof of historical API delivery uptime. The earlier project has already inspected some 2026 reanalysis events; this is not an untouched prospective test.

## Issue time, initialization and availability

- Warning decisions are at 00:00 UTC on issue date I.
- Select one GFS run initialized at 00:00 UTC on I−1 day. No later initialization may contribute to that warning.
- Allow 12 hours after initialization for publication/ingestion, and retain a further 12-hour buffer before the warning decision. This is an explicit conservative availability assumption. Historical receipt timestamps are unavailable; a real outage or publication delay beyond 24 hours cannot be ruled out.
- Target dates are I+1, I+3 and I+7 days, meaning exactly 24, 72 and 168 hours from decision to the target day's start. Because the selected run is 24 hours old, initialization-to-target lead is 48, 96 and 192 hours.
- Request 12 days of hourly 2m temperature from each explicit run. Only complete 24-hour UTC days are aggregated to daily maximum. Do not concatenate later model runs to complete an event trajectory.
- The complete three/four-day event trajectory comes from that same frozen run, including previous event days needed by the ongoing/onset rules. GFS forecasts provide those values; retrospective ERA5 values must not fill trajectory inputs.

This deliberate 24-hour-old run policy makes the timing unambiguous and conservative; it does not measure the best achievable skill from the freshest operational run.

## Verification and reference

Use ERA5 daily maximum temperature as retrospective verification only, extracted at the actual ERA5 grid coordinates recorded in the existing data manifest. Reuse the fixed 1981–2010 local 90th-percentile thresholds and five-calendar-day window. Daily dates are UTC. Forecast grid coordinates may differ from ERA5 and are preserved in the archive manifest.

Targets retain the existing definitions: ongoing means the third or subsequent consecutive hot day; onset means the first of at least three hot days following a non-hot day. Onset verification uses target through target+2 only after the event. Reanalysis is not station truth, so this experiment establishes agreement against ERA5 rather than direct station/health-outcome verification.

## Development and test dates

- Development targets: 2026-05-01 through 2026-06-22 for all tasks and leads.
- Final development verification ends by 2026-06-24 for onset.
- Assume a complete ERA5 day is usable at the start of day D+6 (five-day publication delay after that UTC day finishes).
- Freeze correction/error distributions and cutoffs at 2026-06-30 00:00 UTC, when even the last required development verification is allowed to be available.
- Common test targets for all three leads: 2026-07-08 through 2026-09-30. The first seven-day warning is issued on July 1, after development is frozen. All test issue times must be after the freeze.
- Targets from June 23 through July 7 are excluded from model selection and test scoring. They form a temporal buffer and keep all leads on the same test target dates.
- Download ERA5 verification from April 1 through October 2, 2026, and archived runs from April 23 through September 28 at 00 UTC. The precise run set is generated from the development/test target dates and lead values.

## Methods fixed before test evaluation

1. Raw GFS event forecast: apply the event rule directly to the same-run predicted maximum temperatures and fixed local thresholds. This is a deterministic event forecast, not a calibrated probability.
2. Joint-error adjusted GFS: estimate vectors of ERA5-minus-GFS errors over the three/four-day event trajectory using development targets only. At each site/lead/task, add these development error vectors to the frozen forecast trajectory and report the fraction satisfying the event rule. Joint vectors preserve error dependence between the event days; this is not an ensemble of independent NWP model runs.
3. Choose a single pooled classification cutoff per task/lead from development predictions, with each development target's error bank excluding targets within seven calendar days of it. Test error banks contain development records only. No cutoff is chosen from test labels.
4. Delayed-ERA5 temperature persistence: hold the latest usable ERA5 daily maximum constant, using only days whose assumed publication time is at/before the warning decision. This intentionally respects ERA5 latency.
5. Historical climatology: local event frequencies within ±15 calendar days from 1981–2016, using only historical verification. Choose its cutoff from development only.

Do not plug recent ERA5 observations into the existing observation-history ML models at issue time: those recent ERA5 fields would have been unavailable under the publication-delay policy. The earlier reanalysis-history scores remain a different experiment.

## Outcomes and audit

Report pooled/site F1, precision, recall, average precision, ROC AUC, Brier score, raw temperature MAE/RMSE, contingency counts, event hits/misses and false alerts per 100 location-days. Evaluate all methods on exactly the same target/site records. Save run initialization, assumed availability, warning time, target dates, verification dates, and delayed-observation availability for every prediction.

Use 500 paired seven-day moving blocks of whole seven-location fields for conditional uncertainty and F1 differences from raw GFS. Show skill by location and lead. Keep sparse-event counts and zero-skill results visible. Missing archived runs or incomplete UTC days must be reported; never substitute a later run. Save request settings, raw responses and hashes, provider metadata, data preparation settings, and a protocol hash.

## Limits

One season and seven locations limit generalization. Early-summer development and later-summer testing may have different error distributions. The archive is retrieved retrospectively, without individual historical delivery logs. GFS/ERA5 grid mismatch, model updates, reanalysis error and the small development sample remain. The joint error adjustment assumes development errors remain representative in test; it does not establish probability calibration. This is a warning-feasibility replay against ERA5, not a public safety warning service.

## Sources

- Explicit model-run semantics, archive coverage, initialization versus publication, and ECMWF hindcasts: https://open-meteo.com/en/docs/single-runs-api
- GFS forecast variables/horizon: https://open-meteo.com/en/docs/gfs-api
- Provider availability metadata and distributed delivery caveats: https://open-meteo.com/en/docs/model-updates
- ERA5 publication delay: https://climate.copernicus.eu/climate-reanalysis
