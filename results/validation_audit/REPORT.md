# Additional validation audit

Descriptive audit of already inspected held-out results; no model selection.

Brier skill compares with each experiment’s saved historical climatology probabilities. Negative skill means worse squared-probability error than that baseline.

| Experiment | Brier skill | Event support | Detection |
| --- | --- | --- | --- |
| research/ongoing_lead1 | 0.718 | 159 | 149 |
| research/ongoing_lead3 | 0.219 | 159 | 114 |
| research/ongoing_lead7 | 0.035 | 159 | 68 |
| research/onset_lead1 | 0.097 | 155 | 61 |
| research/onset_lead3 | 0.005 | 155 | 19 |
| research/onset_lead7 | 0.002 | 155 | 48 |
| issued_forecasts/ongoing_lead1 | 0.712 | 25 | 23 |
| issued_forecasts/ongoing_lead3 | 0.586 | 25 | 23 |
| issued_forecasts/ongoing_lead7 | 0.405 | 25 | 23 |
| issued_forecasts/onset_lead1 | 0.141 | 23 | 15 |
| issued_forecasts/onset_lead3 | 0.160 | 23 | 15 |
| issued_forecasts/onset_lead7 | -0.080 | 23 | 4 |

Ongoing support counts episodes of labelled ongoing days. Onset support counts exact first days; these are different outcomes.

Intervals for 3-, 7- and 14-day blocks and site/bin support are in metrics.json and reliability_bins.csv. They do not measure retraining uncertainty or establish station accuracy.

The source benchmark cutoffs and artifacts are unchanged. These diagnostics are not a new untouched test.
