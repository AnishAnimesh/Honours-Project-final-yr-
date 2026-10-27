# HeatCast: Regional Robustness and Heatwave Early Warning

The expanded research project downloads **45 years of ERA5 daily data (1981–2025) for seven European locations** and evaluates both ongoing events and first-day onset at 1-, 3- and 7-day lead. It uses a fixed 30-year historical reference, validation-only selection, regional failure analysis, earlier temporal backtesting, nested location holdout and matched one-day linear/LSTM comparisons.

Start with the [expanded research report](results/research/REPORT.md), [research manuscript](researchpaper.md), and [predeclared protocol](RESEARCH_PROTOCOL.md). The original France-grid experiment is preserved below and in [its report](results/REPORT.md); its scores describe a different task/domain and cannot be compared directly with the expanded benchmark.

## Interactive HeatCast Observatory

Private hosted UI: https://heatcast-observatory.abloomknoll1026.chatgpt.site

The project now includes a responsive presentation UI in [web/heatcast-observatory](web/heatcast-observatory/README.md). It connects to the actual saved probabilities, ERA5 outcomes, GFS temperature trajectories and regional metrics, with selectors for location, date, lead and event type. It includes an interactive European map, keyboard-accessible temperature chart, warning timeline, regional atlas, research guide and CSV exports.

```bash
python3 -m http.server 8787 --bind 127.0.0.1 --directory web/heatcast-observatory/dist
```

Open http://127.0.0.1:8787. The archived GFS and climate-model studies are clearly labelled as historical. A separate live tab fetches the latest GFS weather feed, records retrieval time and never displays a future observed outcome. Outside the validated May–September season, live mode shows temperatures without an event probability. The latest-feed adjustment remains experimental and differs from the frozen-run benchmark.

The UI’s data exporter and verification commands are documented in its README. All 35,700 exported model outputs match the source artifacts; the browser implementation reproduces all 3,570 archived empirical-model calculations exactly.

## Archived forecasts issued before the event

The [issued-forecast evaluation](results/issued_forecasts/REPORT.md) now replays **150 explicitly initialized GFS runs from 2026** at the seven locations. Warnings use a previous-day 00 UTC run at a 00 UTC decision, giving a conservative 24-hour buffer after initialization. Actual warning leads are 24, 72 and 168 hours. Every event trajectory comes from the same run. Historical receipt times are unavailable, so the replay states its publication assumption explicitly.

Development targets are May 1–June 22, with error corrections and cutoffs frozen June 30 after the assumed ERA5 verification delay. Common test targets are July 8–September 30. A June 11 archive gap excludes three development target dates across all leads; all 85 test target dates are retained. Scores cover this separate 2026 experiment and are not directly comparable with the 2021–2025 reanalysis-history scores.

| Warning lead | Raw GFS ongoing-event F1 | Raw GFS onset F1 | Adjusted ongoing F1 | Adjusted onset F1 |
| --- | --- | --- | --- | --- |
| 1 day | 0.772 | 0.377 | 0.774 | 0.390 |
| 3 days | 0.707 | 0.308 | 0.713 | 0.341 |
| 7 days | 0.626 | 0.115 | 0.558 | 0.083 |

The adjustment uses development-only joint forecast errors. It increases false alerts in several cases and worsens seven-day ongoing F1 in this experiment. Full hits, misses, regional scores, uncertainty and warning timestamps are saved. This demonstrates an archived warning replay against ERA5 under stated timing assumptions; it does not establish prospective service delivery or station-level verification.

```bash
python -m heatcast.download_issued
python -m heatcast.issued_evaluation
python -m unittest discover -s tests -p test_issued_protocol.py -v
python tests/check_issued_artifacts.py
python -m heatcast.issued_forecast --issue 2026-07-25 --lead 7 --task onset --output results/issued_forecasts/example_warning.csv
```

See the [issued-forecast timing protocol](ISSUED_FORECAST_PROTOCOL.md), [data provenance](data/issued_forecasts/README.md), and [all warning decisions](results/issued_forecasts/warning_audit.csv). Warning inference uses only frozen forecast arrays, historical thresholds and the earlier development error bank. Recent ERA5 fields are not inserted into the original ML inputs as if they were already published.

## Expanded project objectives

1. Acquire and verify a complete long-history, multi-location weather dataset with reproducible provenance.
2. Define local hot-day thresholds from 1981–2010, and distinguish ongoing events from advance onset warning.
3. Compare one-, three- and seven-day lead forecasts using only observations available by the issue day.
4. Give locations equal weight in model selection and expose weak regions with support counts and local metrics.
5. Compare temperature-only, no-soil and full-weather predictors; evaluate matched linear and LSTM baselines at one-day lead.
6. Test earlier years, omit locations from all supervised model development, and quantify uncertainty and training-sample sensitivity.
7. Deliver reproducible models, geographic diagnostics, forecast CSV exports and a research write-up that retains poor results.

## Reproduce the expanded study

```bash
pip install -r requirements.txt
python -m heatcast.download_sites
python -m unittest discover -s tests -v
python -m heatcast.research
python tests/check_research_artifacts.py
python -m heatcast.regional_audit
python -m heatcast.research_forecast --date 2025-07-15 --task ongoing --lead 1 --output results/research/example_forecast.csv
```

Downloads resume from cached JSON responses. API rate limits are respected; complete responses and the prepared dataset are saved under `data/europe_sites/`. The included [manifest](data/europe_sites/manifest.json) records exact requests, returned coordinates/units and hashes. The primary benchmark contains Paris, London, Madrid, Rome, Athens, Berlin and Warsaw. Stockholm was requested but could not be completed under the archive's hourly quota; it was excluded on data-availability grounds before final model fitting. Northern Europe remains unvalidated.

The API model is explicitly ERA5, UTC days, with elevation downscaling disabled. Six daily variables are maximum/minimum temperature, mean sea-level pressure, mean top-layer soil moisture, mean humidity and precipitation. This is a representative grid-cell study, not station verification or complete European spatial coverage. Data attribution: Copernicus Climate Change Service/ECMWF ERA5, served by Open-Meteo under CC BY 4.0. See [archive documentation](https://open-meteo.com/en/docs/historical-weather-api).

Final train/validation/test periods are 1981–2016 / 2017–2020 / 2021–2025. The three-day event rule is project-specific; it is not the six-day ETCCDI WSDI. The onset target's next two observed days enter verification only. All predictors end at the issue day. The fixed 1981–2010 reference precedes validation and testing; it is not the latest climate normal. This is a revised retrospective study, not untouched prospective validation or an operational warning service.

The primary six-candidate gradient-boosting comparison uses validation **macro-site F1**, so each location counts equally. Linear and LSTM comparisons are matched on the one-day tasks. The best method is identified by validation performance; test outcomes remain reported for all methods. Site-specific cutoff policies are evaluated as a separate predeclared experiment, with a minimum validation support requirement. A local cutoff is not guaranteed to improve future skill.

Use `--method champion` with `research_forecast` to export the validation-selected method at one-day lead. The default `primary` exports the primary boosting model. Both reconstruct forecasts after discarding observations after the issue date. The CSV includes issue date, target date, lead, task, model, location, probability and the fixed cutoff.

The original grid's [regional audit](results/regional_audit/REPORT.md) reports cell F1, false alarms, recall, positive counts and four fixed geographic quadrants. Low-support cells remain identified; the audit does not retune the original model on its test errors.

## Original France-grid experiment

Reproducible next-day forecasting of ongoing regional heatwaves from ERA5 temperature, mean sea-level pressure and surface soil moisture. This project compares a Paris LSTM, a spatial ConvLSTM, a no-soil ConvLSTM ablation, gradient boosting, persistence and seasonal climatology.

The authoritative evaluation is [results/REPORT.md](results/REPORT.md), with machine-readable metrics in [results/metrics.json](results/metrics.json). Original notebook scores and root-level model files are retained as exploratory artifacts; they used test-tuned cutoffs and are superseded by the corrected pipeline.

## Objective and scope

Predict the probability that the next UTC day is the third or a later consecutive hot day at each evaluated grid cell. A hot day exceeds its local training-derived 90th-percentile daily maximum temperature. The available grid covers 42–51°N, 5°W–8°E at 0.25° spacing; this is a France-centred region, not all Europe.

The local GRIB contains May–August observations, 2020–2026. The pipeline keeps complete 24-hour days, including the last complete day on August 10, 2026. Partial days and sequences crossing seasonal or missing-day gaps are excluded. Temperature is aggregated to daily maximum in Celsius; pressure and soil moisture to daily mean.

## Run

Python 3.13 was used for the completed experiment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m heatcast.run --data data.grib --output results --epochs 35 --seed 42
```

Daily preprocessing is cached against the input file's path, size and modification time. Core package versions used locally are recorded in `requirements-lock.txt`.

Outputs include model checkpoints, gradient-boosting estimator, validation/test metrics, precision–recall and case-study plots, daily spatial extent/IoU, preprocessing parameters, and saved held-out probabilities. A run log can be captured with `> results/run.log 2>&1` after creating `results/`.

Export a saved held-out forecast:

```bash
python -m heatcast.predict --date 2026-08-04 --output results/forecast_20260804.csv
```

Forecast the next day from a prepared daily NPZ (replace the cache path with the generated filename):

```bash
python -m heatcast.forecast --daily results/daily_SOURCE_ID.npz --output results/next_day_forecast.csv
```

The daily file must contain `x` (day, three channels, latitude, longitude), `dates`, `lat` and `lon`, the original training reference, and seven complete consecutive recent days. The channel order is temperature maximum in Celsius, mean pressure in Pa, mean volumetric soil moisture. This command demonstrates inference; ERA5 latency means it is not an operational real-time warning service.

## Data acquisition

The supplied `data.grib` is sufficient to reproduce the experiment; another download is unnecessary. Raw data and large intermediate arrays are excluded from Git.

Optional replacement download:

```bash
python -m heatcast.download --start 2020 --end 2026
python -m heatcast.download --start 2020 --end 2026 --execute
```

The first command writes exact monthly requests. Execution requires your own CDS credentials and acceptance of the dataset terms. See the [official CDS setup instructions](https://cds.climate.copernicus.eu/how-to-api) and [ERA5 single-level dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels). Requests use all 24 hourly times, May–August, the regional box above, and the three variables. To combine downloaded GRIB messages into one input, without altering their contents:

```python
from pathlib import Path
import shutil
with open('downloaded.grib', 'wb') as destination:
    for source in sorted(Path('data/era5').glob('*.grib')):
        with source.open('rb') as stream:
            shutil.copyfileobj(stream, destination)
```

Then run with `--data downloaded.grib`. Avoid combining overlapping downloads. A longer reference period requires changing and validating the chronological split design before reporting results.

## Evaluation protocol

- Train: 2020–2022. Validation: 2023–2024. Held-out test: 2025–2026.
- Thresholds: per-cell 90th percentile within ±15 calendar days, using training years only; a fixed non-leap calendar aligns dates across years.
- Labels: causal third-and-later hot days. The first two hot days are not retrospectively relabelled as ongoing events.
- Inputs: seven contiguous observed days, ending the day before the target; no target-day weather enters any predictor.
- Scaling and land proxy: fitted from training only. Grid evaluation retains finite cells with training mean soil moisture above 0.001 m³/m³, excluding ocean values encoded near zero. This is a documented proxy, not an official land-fraction mask.
- Checkpoints: copied at the lowest validation loss, with early stopping. Model cutoffs maximize validation F1. Gradient boosting tests 7, 15 and 31 leaves on validation only, using at most 250,000 uniformly sampled training cell-days.
- Selection: highest validation F1 within grid and Paris scopes separately. The selected model may be a simple baseline if it performs best.
- Report: F1, precision, recall, ROC AUC, average precision, Brier score, accuracy and confusion matrices. Grid models share the same cells and dates; Paris results are a separate comparison.
- Uncertainty: paired 7-day moving-block bootstrap of entire daily spatial fields, with 500 resamples. Reported intervals are conditional on the trained model and this test period.

## Project files

| Path | Purpose |
|---|---|
| `heatcast/data.py` | GRIB aggregation, thresholds, causal labels, contiguous windows |
| `heatcast/models.py` | LSTM and ConvLSTM |
| `heatcast/run.py` | Training, validation selection, baselines, ablation and evaluation |
| `heatcast/evaluation.py` | Cutoffs, classification metrics and uncertainty |
| `heatcast/forecast.py` | Next-day model inference |
| `heatcast/predict.py` | Export saved test forecasts |
| `heatcast/download.py` | Reproducible optional CDS requests |
| `tests/test_protocol.py` | Leakage, date gaps, calendar alignment and model checks |
| `results/` | Corrected run artifacts and report |
| `01_data_exploration.ipynb` | Legacy exploratory work |

## Interpretation and limitations

Three training years do not constitute a 30-year climate normal. Forecast skill here concerns a regional experimental event definition and a short reference period. Spatial cell-days are correlated and cannot be treated as independent evidence. One seed, one held-out interval and a small hyperparameter search limit robustness claims. Reanalysis assumes retrospectively available observations, with a publication delay in practice. Skill at one-day lead does not establish longer-range warning skill.

The August 2026 case study selects the available day with greatest labelled event extent for descriptive inspection. The dataset and model alone cannot establish that the event was unprecedented, attribute a model error to climate change, or prove causal value from soil moisture. The ablation reports an association under one controlled training setup.
