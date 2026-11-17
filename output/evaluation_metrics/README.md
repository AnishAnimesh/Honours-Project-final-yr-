# HeatCast evaluation graphs

Open `index.html` to browse every figure. PNG files are exported at 220 dpi; SVG files are scalable for papers and presentations. `source_tables/` contains the saved CSV results.

## Interpretation

- Original France land-grid and Paris results: test 2025–2026. Paris and full-grid methods have separate comparison figures.
- Seven-location research benchmark: test 2021–2025, 1/3/7-day ongoing-event and onset tasks.
- Archived GFS replay: common test targets July 8–September 30, 2026. Verifies against ERA5 under the documented publication assumption.
- Expanded model search and forecast amendments are exploratory studies. Champions use development selection, not test ranking. Search test years were already inspected during project development.
- Accuracy, precision, recall, F1, ROC AUC and average precision: higher is better. Brier, MAE and RMSE: lower is better. AP is average precision, rather than trapezoidal PR AUC.
- Confusion matrices use saved validation/development cutoffs. ROC/PR curves use continuous scores; dashed PR lines show task prevalence. Probability histograms normalize each observed class separately.
- Feature ablations report validation macro-site F1, maximized across the pre-existing leaf settings for each feature variant. They do not measure causal feature effects.
- Confidence intervals are conditional block-bootstrap estimates, while neural error bars are standard deviations across three seeds. These are different quantities.
- Event warning counts are event-day counts, not independent heatwave episode counts. MAE/RMSE are equal-site averages of saved site-level metrics.
- Performance values from different domains, labels and test periods should be interpreted within their own study. Rare onset events make accuracy insufficient on its own.

Reproduce from the project root:
`.venv-model-search/bin/python scripts/plot_evaluation_metrics.py`
