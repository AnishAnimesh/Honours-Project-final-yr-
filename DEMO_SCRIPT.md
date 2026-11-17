# HeatCast: presentation and technical demonstration script

Suggested title: **HeatCast — Regional Heatwave Forecasting and Early-Warning Evaluation**

Allow 15–18 minutes plus questions. Quoted passages are speaking lines; actions tell you what to show. Timings are approximate, so rehearse once and shorten explanations to fit your allotted slot. All numerical examples below come from this repository's saved results.

## 1. Prepare before the presentation

Run from the project root:

```bash
python3 -m http.server 8787 --bind 127.0.0.1 --directory web/heatcast-observatory/dist
```

Keep this terminal running. Open the following browser tabs in order:

1. Successful warning: <http://127.0.0.1:8787/?source=issued&site=paris&date=2026-07-08&lead=1&event=ongoing>
2. False alert: <http://127.0.0.1:8787/?source=issued&site=paris&date=2026-07-17&lead=1&event=ongoing>
3. Lead-time comparison: <http://127.0.0.1:8787/?source=issued&site=paris&date=2026-08-01&lead=7&event=ongoing>
4. Climate study: <http://127.0.0.1:8787/?source=research&site=paris&date=2025-08-12&lead=1&event=ongoing>

The third tab is intentionally a failed seven-day forecast. The fourth is a navigation example, not a preselected successful prediction; describe its displayed outcome as it appears.

Keep these files open in your editor:

- `heatcast/research_data.py`: thresholds, event labels and features.
- `heatcast/research.py`: chronological splits and model selection.
- `heatcast/models.py`: LSTM and ConvLSTM implementation.
- `heatcast/issued_data.py`: frozen-run timing and joint-error adjustment.
- `heatcast/issued_forecast.py`: inference without future verification inputs.
- `results/research/REPORT.md`: main climate-study results.
- `results/issued_forecasts/REPORT.md`: archived warning results.

Use a second terminal for inference. Do not retrain models or download datasets during the presentation. The saved artifacts are enough for the demonstration.

Optional technical check before the audience arrives:

```bash
node web/heatcast-observatory/scripts/check_data.mjs
```

This check was run while preparing this script. It verified 3,570 browser-model predictions against the Python outputs, research classifications, calendar behavior and required assets. The inference command in section 10 was also run successfully. This does not mean the entire repository test suite was rerun.

## 2. Opening and problem statement — 0:00–1:00

**Show:** the Forecast explorer, with the successful-warning tab already loaded.

**Say:**

> Good morning. Our project is HeatCast, a research system for forecasting regional heatwave events and evaluating whether warnings can be reconstructed before those events occur.
>
> A temperature forecast answers how hot a day may be. Our project asks a more specific question: will temperatures satisfy a locally defined, multi-day heatwave condition, and how does forecast performance change across locations and warning lead times?
>
> I will demonstrate an actual historical warning, explain the data and machine-learning pipeline, compare regional performance, and show both a successful prediction and a failure case.

Use “our” for a team presentation or replace it with “my.” Do not introduce the interface as a deployed public warning service.

## 3. Define the prediction task — 1:00–2:15

**Show:** select **The research** in the sidebar and point to **Three hot days. Two different questions.** Then return to **Forecast explorer**.

**Say:**

> We first define a hot day relative to the local historical climate. In the expanded study, a day is hot when its maximum temperature exceeds the local 90th-percentile threshold for that calendar date, estimated from the fixed 1981–2010 reference period.
>
> The threshold uses a five-calendar-day window, pooling the reference years. This makes the threshold location-specific and season-specific, instead of applying one absolute temperature to every city.
>
> We predict two different targets. Ongoing heatwave means the target day is the third or a later consecutive hot day. Onset means the first hot day of a run lasting at least three days, preceded by a non-hot day.
>
> Onset is harder because it requires anticipating the beginning of a sustained event. The following two observed days are used only to verify the onset label; they are not supplied as observed inputs to the forecasting model.

**If asked for equations:** let `H(t)` mean maximum temperature exceeds the local threshold.

```text
Ongoing(t) = H(t−2) AND H(t−1) AND H(t)
Onset(t)   = NOT H(t−1) AND H(t) AND H(t+1) AND H(t+2)
```

This is the project's custom three-day definition, not the six-day WSDI definition.

## 4. Demonstrate a successful warning — 2:15–3:45

**Set:** Archived warnings → Paris → 8 July 2026 → 1 day → Ongoing heatwave. The first prepared tab already has these settings.

**Show in order:** probability, warning status, observed outcome, temperature chart, local threshold, timing panel.

**Say:**

> This example comes from the archived 2026 GFS warning replay. The selected target is Paris on 8 July, with a one-day warning lead.
>
> The adjusted event probability is 100%, which exceeds the study's frozen cutoff of approximately 22.7%. The warning therefore predicts an ongoing heatwave. ERA5 verification later confirms the event, so this is a hit.
>
> The chart helps explain the event: we compare the frozen forecast temperature trajectory, temperatures observed afterward, and the local hot-day threshold. A heatwave decision depends on the sequence of days, not just one high temperature.
>
> A 100% empirical score means all retained development-error scenarios satisfied this event rule. It is not a guarantee that an event must occur.

Point to the map:

> The map lets us inspect seven representative locations. These are grid-cell predictions associated with the cities, not a continuous heat map or complete country-wide coverage.

Click a second location briefly, then return to Paris. State its displayed result rather than assuming it matches Paris.

## 5. Explain timing and data leakage — 3:45–5:00

**Show:** **What was known, and when?** Keep the successful-warning example selected.

**Say:**

> Forecast accuracy is only useful as early-warning evidence if the inputs existed before the warning decision.
>
> For this target, the GFS run was initialized on 6 July at 00 UTC. The warning decision is 7 July at 00 UTC, and the target day starts on 8 July at 00 UTC. That gives a 24-hour warning lead, while the forecast initialization is 48 hours before the target.
>
> The replay assumes publication within 12 hours of initialization and waits 24 hours before issuing the warning. Historical receipt timestamps are unavailable, so this is an explicit availability assumption, not a measured delivery record.
>
> Every temperature needed for an event comes from the same frozen run. We do not replace inconvenient forecast days with a later run or insert future ERA5 observations into inference.

**Technical follow-up:** open `trajectory()` in `heatcast/issued_data.py` and point to the exact initialization lookup and missing-run error. Explain that initialization time, assumed availability, warning issue time and target time are separate fields.

## 6. Show a false alert and lead-time sensitivity — 5:00–6:30

**Set:** second tab, Paris → 17 July 2026 → 1 day → Ongoing heatwave.

**Say:**

> This example shows why probability and classification must be distinguished. The model gives a 30% event probability. Since the frozen cutoff is approximately 22.7%, it issues a warning. The observed event is absent, making this a false alert.
>
> A warning cutoff does not have to be 50%. It is selected during development according to the evaluation objective. Lower cutoffs can detect more events while also producing more false alerts.

**Set:** third tab, Paris → 1 August 2026 → Ongoing heatwave. Start with 7 days, then click 3 days and 1 day.

| Lead | Adjusted probability | Classification | ERA5 event |
|---|---:|---|---|
| 7 days | 96% | Warning | Absent |
| 3 days | 16% | No warning | Absent |
| 1 day | 12% | No warning | Absent |

**Say:**

> Here the seven-day forecast produces a high-confidence false alert, while the three-day and one-day forecasts correctly predict no event. Changing the lead selects a different earlier frozen initialization; this is not one forecast trajectory being edited.
>
> This example illustrates why we must evaluate longer leads separately. A shorter-lead model's score cannot establish seven-day warning skill.

This is a descriptive case study. Use the full-period results in the next section for performance claims.

## 7. Regional atlas and performance — 6:30–8:00

**Show:** **Regional atlas**. Keep Archived warnings selected. Choose one-day ongoing events, compare **Adjusted GFS** and **Raw GFS**, and change the metric between F1, recall and precision. Point to observed event counts.

**Say:**

> A single successful forecast does not establish model quality. The regional atlas summarizes the complete held-out period and shows F1, precision, recall and event support for every location.
>
> Precision asks: of the warnings issued, how many were correct? Recall asks: of the observed events, how many were detected? F1 combines precision and recall. We report these because accuracy alone can be misleading when event days are uncommon.
>
> The archived test covers 85 target dates at seven locations, giving 595 location-days for each task and lead. Its results show useful short-lead ongoing-event detection, weaker onset prediction, and deterioration at longer leads.

Use these rounded **pooled test F1** values from `results/issued_forecasts/REPORT.md`:

| Warning lead | Raw ongoing | Adjusted ongoing | Raw onset | Adjusted onset |
|---|---:|---:|---:|---:|
| 1 day | 0.772 | 0.774 | 0.377 | 0.390 |
| 3 days | 0.707 | 0.713 | 0.308 | 0.341 |
| 7 days | 0.626 | 0.558 | 0.115 | 0.083 |

> Adjustment gives small point-score gains at short leads, but it worsens seven-day performance. The short-lead uncertainty intervals include no improvement over raw GFS, so I would not claim a proven overall benefit from adjustment.

The atlas's per-location scores, pooled F1 and macro-site F1 are different summaries. Say which one you are pointing to. The date selector does not restrict atlas evaluation to that single date.

## 8. Explain the system architecture — 8:00–9:15

**Show:** a slide or this diagram in your notes.

```text
CLIMATE-HISTORY STUDY
ERA5 daily weather → quality checks → fixed local thresholds and labels
 → historical features → chronological training / validation / testing
 → saved classifiers, probabilities, metrics and reports

ARCHIVED WARNING REPLAY
Frozen GFS runs + historical thresholds + development forecast-error bank
 → empirical event probabilities → frozen cutoff → timestamped warnings
 → later ERA5 verification → regional and lead-time evaluation

PRESENTATION INTERFACE
Exported JSON and reports → static HTML / CSS / JavaScript observatory
Latest GFS provider feed → separate experimental Live weather tab
```

**Say:**

> The system separates data preparation, model development, inference, evaluation and presentation.
>
> The Python layer uses NumPy and pandas for weather processing, scikit-learn for statistical models, and PyTorch for the neural baselines. Saved arrays and model artifacts make the outputs reproducible.
>
> The observatory uses HTML, CSS and JavaScript with exported JSON. The climate-study tab displays saved Python classifier outputs. The empirical GFS adjustment also has a JavaScript implementation checked against the saved Python results.
>
> The local HTTP server serves static files. It is not a Python prediction API, and the browser does not execute the gradient-boosting estimator.

## 9. Technical explanation of the climate models — 9:15–12:00

**Show:** fourth prepared tab, **Climate model study**, then `results/research/REPORT.md` and selected source functions.

**Say:**

> The expanded climate study uses 45 years of ERA5 daily data, from 1981 through 2025, at Paris, London, Madrid, Rome, Athens, Berlin and Warsaw. Evaluation focuses on May through September.
>
> The six weather variables are daily maximum and minimum temperature, mean sea-level pressure, top-layer soil moisture, humidity and precipitation.
>
> Features include temperature anomalies relative to the local threshold; weather lags at zero, one, two, three, six and thirteen days before or on the input day; rolling three-, seven- and fourteen-day summaries; seasonal sine and cosine terms; and geographic coordinates. All observed weather predictors stop at the issue day. The target day's historical threshold is already known from the fixed reference.

**Point to:** `make_features()` in `heatcast/research_data.py`.

> We use chronological splits: training from 1981–2016, validation from 2017–2020 and testing from 2021–2025. Randomly mixing dates would create an unrealistic evaluation of forecasting and could mix closely related weather episodes across splits.
>
> The primary search evaluates three feature sets—temperature only, no soil moisture, and all variables—at two tree complexities. This gives six histogram gradient-boosting candidates per task and lead. Candidate selection uses validation macro-site F1, giving each location equal weight. Classification cutoffs also use validation data.

**Point to:** `select_model()` in `heatcast/research.py`.

> Histogram gradient boosting combines successive decision trees to improve predictions. It can capture nonlinear interactions in these engineered weather features. We compare it with simpler baselines and with a matched LSTM rather than assuming a neural model must be better.

**Show this matched one-day table:**

| Method | Ongoing test F1 | Onset test F1 |
|---|---:|---:|
| Logistic regression | 0.7250 | 0.1137 |
| LSTM | 0.7701 | 0.2181 |
| Gradient boosting | 0.8199 | 0.2961 |

> Gradient boosting was selected by validation performance for both one-day tasks and achieved the strongest test F1 among these matched methods. Its one-day ongoing pooled F1 is approximately 0.820; the macro-site F1 is approximately 0.799.
>
> Onset remains much weaker at approximately 0.296. Also, the paired uncertainty interval for the ongoing model's advantage over temperature persistence includes zero. The comparison supports a measured conclusion, not a claim that machine learning conclusively beats every simple baseline.

**If asked about LSTM:** the expanded comparison uses 14-day sequences, seven input channels including the temperature anomaly, a 32-unit LSTM, and a linear output head. Training uses weighted binary cross-entropy, AdamW, gradient clipping and validation-loss early stopping. The original France-grid experiment uses a different seven-day setup.

**If asked about ConvLSTM:**

> The earlier France-grid experiment contains a ConvLSTM that applies convolutional gates to preserve spatial structure while processing the time sequence. It maintains hidden and cell states over the weather grid and outputs a probability map. It is a separate experiment, not the model behind the archived GFS adjustment.

Its corrected grid F1 values are 0.5671 for ConvLSTM, 0.5705 without soil moisture and 0.8505 for gradient boosting. These are a different domain, period and task setup; do not combine them with the expanded-study scores.

**If asked how GFS probabilities work:**

> We store multi-day forecast-error vectors from the development period. For a new frozen forecast, we add each historical error vector to create an empirical temperature scenario, preserving the errors together across event days. The fraction of scenarios meeting the event rule is the adjusted probability.

```text
Scenario k = frozen forecast trajectory + development error vector k
Probability = scenarios satisfying the event rule / eligible scenarios
```

This is an empirical error-scenario model, not an independent dynamical weather-model ensemble. Development adjustments and cutoffs freeze on 30 June 2026; test targets run from 8 July through 30 September.

## 10. Demonstrate inference and traceability — 12:00–13:15

**Show:** second terminal. From the project root, run:

```bash
.venv-model-search/bin/python -m heatcast.issued_forecast \
  --issue 2026-07-16 \
  --lead 1 \
  --task ongoing \
  --output /tmp/heatcast-demo-warning.csv
```

This existing environment and command were checked successfully during script preparation. If using another machine, use that machine's prepared Python environment.

**Say:**

> This reconstructs the warning issued on 16 July for the following day at all seven locations. The output includes forecast initialization, assumed availability, issue time, target date, adjusted probability, fixed cutoff and the warning decision.
>
> Paris has a probability of 0.30 and a cutoff of approximately 0.2273, reproducing the false alert we inspected in the interface.
>
> The inference function receives forecast arrays and a frozen adjustment bundle; it does not require future verification observations to produce the warning.

**Show:** `heatcast/issued_forecast.py`, particularly the `run_only` dictionary and frozen bundle.

Return to the browser and click **Export forecast** on the false-alert example. Open the CSV if time permits.

> The interface export records the selected forecast, its timing, source and verification status. This makes a displayed result traceable and available for independent analysis.

The browser export contains the selected location; the Python command outputs all seven. Keep that distinction clear.

## 11. Live weather and research limitations — 13:15–14:30

**Show:** Live weather briefly if the network is available. Point to the retrieval timestamp and verification label. Return to the archived view afterward.

**Say:**

> The live tab requests the latest GFS provider feed and records when it was retrieved. Future observed outcomes are unavailable, so it cannot display a verified future success.
>
> The live feed differs from the frozen-run benchmark. Its summer adjustment is experimental. Outside May through September, the system displays temperature forecasts without an event probability.

**For a November presentation:** an absent live heatwave probability is expected. Do not change historical dates and present them as today's weather.

If the feed fails:

> The provider request is unavailable, and the interface reports that explicitly. The historical demonstration remains available from local saved artifacts.

Then state the boundaries:

> The main limitations are seven representative grid locations, ERA5 rather than independent station verification, a single season for the archived replay, assumed historical publication timing, and weaker onset and longer-lead performance. The research is retrospective; it does not establish prospective service delivery.

## 12. Closing — 14:30–15:15

**Show:** The research page or the one-day ongoing regional atlas.

**Say:**

> HeatCast delivers a reproducible pipeline from weather data to event definitions, model predictions, timestamped warning replay and an interactive research interface.
>
> The strongest results are for short-lead ongoing-event detection. Onset and seven-day prediction remain difficult, and empirical adjustment does not consistently improve raw weather forecasts.
>
> The next steps are to collect forecasts prospectively, verify against independent stations, expand geographic coverage and investigate methods that improve onset and longer-lead performance. Thank you. I am ready for questions.

## Viva answers to rehearse

**What is your contribution beyond showing a weather API?**

The API supplies weather fields. The project implements local event definitions, historical feature engineering, model comparisons, chronological selection, forecast-error scenarios, timing checks, regional diagnostics, uncertainty analysis, reproducible artifacts and an interface tied to those outputs. Present the contribution as the system and evaluation; do not claim to have invented LSTM, gradient boosting or GFS.

**Why Europe when we are presenting in India?**

The validated dataset covers seven European locations and the earlier France-centered grid. Applying the method to India requires local historical thresholds, representative data and independent validation under Indian climate conditions.

**Why a local percentile instead of a fixed Celsius threshold?**

An unusual temperature depends on local climate and season. Percentiles express local extremeness. Absolute public-health thresholds are a different design choice and would require separate validation.

**Why gradient boosting over LSTM?**

It won the validation comparison and performed better on the matched test tasks here. Engineered tabular features can suit tree models well. This result does not establish that boosting always outperforms neural models.

**Why include pressure, moisture, humidity and rainfall?**

They may provide additional information about atmospheric conditions and surface dryness. Feature-set comparisons assess predictive usefulness. They do not prove a causal contribution from soil moisture or another variable.

**How do you prevent leakage?**

Observed features end at the issue day; historical thresholds use a fixed reference preceding validation and testing; model and cutoff selection use development/validation data; onset's future observations enter labels only; archived event days come from one exact earlier run; inference excludes future verification arrays.

**Are the experiments completely untouched holdouts?**

No. The expanded benchmark is revised retrospective research. The later model search explicitly reuses already inspected test years and is exploratory. Within the implemented selection procedures, development and test roles are separated, but that does not erase earlier research exposure.

**What does F1 = 0.820 mean?**

It is the harmonic mean of precision and recall, `2PR/(P+R)`. It is not 82% accuracy, an 82% event probability, or a guarantee about future forecasts.

**What are pooled and macro-site F1?**

Pooled F1 aggregates all location-days before computing the score. Macro-site F1 computes each location's F1 separately and averages the seven scores, giving locations equal weight. The expanded candidate selection uses validation macro-site F1.

**How do you assess uncertainty?**

The main reported intervals use 500 paired seven-day moving-block bootstrap resamples, retaining whole spatial fields to respect temporal and cross-location dependence. Intervals condition on the fitted model and observed evaluation period; they do not capture every source of uncertainty.

**Is the system real-time?**

The interface can fetch live weather, but the evaluated ML study uses retrospective ERA5 data and the issued-warning experiment is a historical replay. An operational warning claim would require prospective timestamped delivery and later verification.

**Why can a 30% probability generate a warning?**

Classification compares the probability with the frozen study cutoff, not automatically with 50%. Paris on 17 July 2026 is a concrete example: 30% exceeds approximately 22.7%, producing a false alert.

**Did the larger model search solve the problem?**

No. Exploratory one-day ongoing XGBoost reached F1 0.8307 and accuracy 0.9645 versus original F1 0.8199 and accuracy 0.9636. Paired difference intervals include zero. Other tasks remain mixed or weak, and this reused-test experiment does not establish a repeatable improvement. These later models are not the saved primary classifier displayed in the climate-study UI.

**What happens when data are missing?**

Preparation validates continuity and finite values. The archived pipeline requires the exact frozen initialization and excludes unavailable matched targets instead of substituting a later run. A June archive gap excludes three development target dates; all 85 test target dates remain present. The live UI reports failed requests explicitly.

## If you have only five minutes

1. **0:00–0:40:** introduce the problem and distinguish onset from ongoing events.
2. **0:40–1:40:** show Paris, 8 July 2026, one-day ongoing warning and its timing.
3. **1:40–2:20:** show Paris, 17 July, and explain the 30% false alert.
4. **2:20–3:15:** show the regional atlas, support counts and lead-time deterioration.
5. **3:15–4:20:** explain the two pipeline branches and the climate study's chronological splits; report one-day boosting F1 0.820 and onset F1 0.296.
6. **4:20–5:00:** export a forecast, state the retrospective and geographic limits, and close.

## Final presentation checklist

- Keep the demo sequence predictable: successful warning → timing → false alert → lead comparison → regional results → architecture → code/inference → limitations.
- Read displayed probabilities and cutoffs separately. Identify the dataset, task and lead whenever quoting a score.
- Use “ERA5 verification” when discussing observed outcomes; do not imply station measurements.
- Explain one function per code screen. Avoid scrolling through whole training files.
- Preserve the distinction between the original grid study, expanded climate study, exploratory model search and archived GFS replay.
- Keep the live request optional so network availability does not determine whether the presentation succeeds.
- Rehearse transitions and terminology aloud. Pause after each visual change before explaining it.

## Evidence used for this script

- `Readme.md` and `web/heatcast-observatory/README.md`: scope, setup and UI behavior.
- `results/research/REPORT.md`: expanded-study metrics and uncertainty.
- `results/issued_forecasts/REPORT.md`: issued-warning timing and performance.
- `results/issued_forecasts/warning_audit.csv`: exact Paris examples.
- `results/REPORT.md`: corrected original grid results.
- `results/model_search/REPORT.md`: exploratory model-search outcomes.
- `heatcast/research_data.py`, `research.py`, `research_neural.py`, `models.py`, `issued_data.py` and `issued_forecast.py`: implementation details.
- `web/heatcast-observatory/dist/index.html` and `app.js`: actual labels, controls, exports and live behavior.
