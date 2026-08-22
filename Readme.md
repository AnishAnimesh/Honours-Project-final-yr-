# HeatCast: Deep Learning-Based Forecasting of European Heat Waves Using ERA5 Reanalysis Data

A 4-month research project applying deep learning (LSTM and ConvLSTM) to predict heat wave occurrence across Europe using historical climate reanalysis data.

---

## Problem Statement

Europe has experienced increasingly frequent, intense, and prolonged heat waves in recent years — 2026 alone has seen five separate heat wave events, with temperatures exceeding 42°C and triggering wildfires across multiple countries. This project explores whether a deep learning model, trained on historical weather data, can predict when and where a heat wave is likely to occur — using temperature, pressure, and soil moisture as inputs.

**Core question:** Can a deep learning model, trained on historical ERA5 reanalysis data, predict heat wave occurrence across Europe with meaningful skill, and how does it perform when tested against a real, unprecedented recent event (the August 2026 heat wave)?

---

## Objectives

1. Build a data pipeline to acquire and preprocess ERA5 reanalysis data for a defined European region
2. Define a clear, reproducible heat wave metric (3+ consecutive days above the local 90th percentile for that calendar day)
3. Build and evaluate a single-location LSTM baseline model
4. Build and evaluate a full-grid ConvLSTM model for spatial heat wave prediction
5. Compare both models against simple baselines (persistence, climatology)
6. Conduct an ablation study to test the contribution of individual input features
7. Test the trained model against the real August 2026 European heat wave as a case study
8. Present findings with honest discussion of strengths and limitations

---

## Dataset

- **Primary source:** [ERA5 hourly reanalysis data](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) via the Copernicus Climate Data Store (CDS)
- **Variables used:** 2m temperature (`t2m`), mean sea level pressure (`msl`), volumetric soil water layer 1 (`swvl1`)
- **Region:** Europe/France bounding box
- **Time period:** Summer months (June–August), 2020–2026
- **Format:** GRIB, downloaded via the `cdsapi` Python package

**Note:** Raw data files are not included in this repository due to size. To reproduce this project, register for a free CDS account and download the data using the variable/region/date settings described above.

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python |
| Data access | `cdsapi` |
| Data handling | `xarray`, `cfgrib`, `pandas`, `numpy` |
| Deep learning | `PyTorch` |
| Baseline ML / metrics | `scikit-learn` |
| Visualization | `matplotlib` |
| Compute | MacBook Air M5 (local) |

---

## Methodology

1. **Data pipeline:** Downloaded ERA5 data, converted to daily aggregates, and labeled heat wave days using a 90th-percentile, 3-day-consecutive rule.
2. **Baselines:** Persistence ("today = yesterday") and climatology (historical average per calendar day).
3. **LSTM model:** Single-location (Paris) time-series model using a 14-day input window.
4. **ConvLSTM model:** Full-grid spatiotemporal model using a 7-day input window, predicting a heat wave probability map across the entire region.
5. **Ablation study:** Retrained the ConvLSTM without soil moisture to test its contribution.
6. **Case study:** Evaluated the trained ConvLSTM against the real August 2026 European heat wave.

---

## Results

### Model Comparison

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| Persistence baseline | 91.2% | 0.811 | — |
| Climatology baseline | 62.4% | 0.000 | — |
| LSTM (Paris only) | 62.7% | 0.497 | 0.729 |
| **ConvLSTM (full grid)** | **87.5%** | **0.488** | **0.851** |
| ConvLSTM ablation (no soil moisture) | 86.3% | 0.454 | 0.832 |

**Key findings:**
- The persistence baseline is a strong benchmark, as expected — heat waves last multiple consecutive days by definition, making "today = yesterday" a naturally strong heuristic.
- The climatology baseline fails on rare-event prediction (F1 = 0), highlighting why accuracy alone is a misleading metric for this problem.
- The ConvLSTM substantially outperforms the single-location LSTM in AUC (0.851 vs. 0.729), demonstrating the value of spatial context.
- Removing soil moisture measurably reduced performance (AUC 0.851 → 0.832), confirming its predictive value — consistent with its role in evaporative cooling and its use in the feature-selection literature this project is based on.

### Case Study: August 2026 Heat Wave

The trained ConvLSTM was tested against the real August 1–11, 2026 heat wave event (one of the most extreme on record for the region):

- The model **underpredicted the most extreme day** (August 4th, 468 affected grid cells) — its predicted probability map showed a diffuse, muted signal that was also spatially offset from the actual event, and never crossed the classification threshold.
- The model performed **noticeably better on smaller, more localized event days** (August 7–8), correctly concentrating its predicted probability in the same region as the actual heat wave, and partially crossing the threshold in the correct location.

**Interpretation:** The model has learned meaningful spatial patterns for typical-to-moderate heat wave conditions, but its confidence calibration breaks down for unprecedented, large-scale events that fall outside the range represented in its training data (2020–2024). This is a well-documented limitation in applying historical climate data to an accelerating climate, and is consistent with gaps identified in the literature reviewed for this project.

---

## Repository Structure

```
heatwave-prediction/
├── 01_data_exploration.ipynb    # Main notebook: data pipeline, models, evaluation
├── lstm_baseline.pt              # Trained LSTM model weights
├── lstm_baseline_results.json    # LSTM evaluation metrics
├── convlstm_fullgrid.pt          # Trained ConvLSTM model weights
├── convlstm_results.json         # ConvLSTM evaluation metrics
├── ablation_results.json         # Ablation study results
├── case_study_20260804.png       # Case study visualization (Aug 4)
├── case_study_20260807.png       # Case study visualization (Aug 7)
├── case_study_20260808.png       # Case study visualization (Aug 8)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## How to Reproduce

1. Clone this repository
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Register for a free account at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) and set up your API key
4. Download ERA5 data for the variables, region, and date range described above
5. Place the downloaded `.grib` file in the project root and update the filename in the notebook
6. Run `01_data_exploration.ipynb` from top to bottom

---

## Limitations

- Training data is limited to 2020–2024 (summer months only), which constrains the model's exposure to extreme/unprecedented events
- The validation and test sets have very different heat wave base rates due to the unusually extreme 2025–2026 season, making direct comparison across splits imperfect
- Sea surface temperature was excluded due to lack of coverage for the inland/regional grid used
- Compute constraints limited the scope of hyperparameter tuning and model size

## Future Work

- Expand training data to a longer historical period and additional years
- Incorporate sea surface temperature using a coastal-inclusive region
- Explore transformer-based architectures as an alternative to ConvLSTM
- Extend the case study to additional 2026 heat wave events for a broader evaluation

---