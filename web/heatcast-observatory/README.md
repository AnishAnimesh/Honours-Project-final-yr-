# HeatCast Observatory UI

A responsive, buildless research interface connected to the actual saved project models and results. Fonts and Natural Earth geometry are hosted locally. No JavaScript framework or npm installation is needed.

## Preview locally

From the honours-project root:

```bash
python3 -m http.server 8787 --bind 127.0.0.1 --directory web/heatcast-observatory/dist
```

Open http://127.0.0.1:8787. A normal browser is required; opening `index.html` directly through `file://` cannot load the JSON assets.

## Views and data

- **Forecast explorer:** location, date, 1/3/7-day lead and ongoing/onset selectors; actual saved model probabilities; observed ERA5 outcome; interactive temperature chart; exact warning chronology; CSV export.
- **Regional atlas:** per-location precision/recall/F1 and event support, method comparison, selectable metric and lead comparison. Scores always describe the full historical test period.
- **Research guide:** experimental definitions, model limitations, data provenance, report/CSV/manuscript downloads.
- **Archived warnings:** 2026 frozen GFS runs and the empirical joint-error correction. Includes 85 common held-out target dates, July 8–September 30.
- **Climate model study:** actual saved gradient-boosting predictions for May–September 2021–2025. This assumes recent observations and is not an operational warning replay.
- **Live weather:** a fresh, timestamped GFS feed requested directly from Open-Meteo. No future observed outcome is supplied. The empirical summer correction is available only in May–September and remains experimental because this latest feed differs from the frozen-run benchmark. Outside that season, the interface displays temperatures without an event probability. Failed network calls display an explicit error; they never produce placeholder weather.

The static site displays exported outputs from real Python models. It does not execute the gradient-boosting estimator in the browser. The empirical joint-error model is implemented in `dist/model.mjs`; all 3,570 archived browser calculations are checked against the saved Python results.

## Refresh the dataset

After regenerating Python experiment artifacts, run from the honours-project root:

```bash
python3 web/heatcast-observatory/scripts/export_data.py
python3 web/heatcast-observatory/scripts/check_export.py
node web/heatcast-observatory/scripts/check_data.mjs
node --check web/heatcast-observatory/dist/app.js
```

The exporter includes predictions, observed outcomes, regional metrics, actual forecast temperature trajectories, fixed thresholds and development-only error banks. No mock probabilities or synthetic weather are used. `dist/data/manifest.json` records source hashes and export time. Copied research reports and CSVs are under `dist/reports/`.

The Python export/check scripts expect this folder inside the honours-project repository. The built `dist/` directory is self-contained and can be served independently.

## Access and presentation

The hosted Site starts private. Historical experiments and live weather have separate tabs and distinct verification labels. Exact scores and timestamps are retained in exported forecast CSVs. A live probability must not be interpreted as a verified or operational public safety alert.

Use the archived-warning view for the viva, then compare regions in the atlas. The source tabs explain the difference between the reanalysis-history experiment and the archived issued-forecast replay.
