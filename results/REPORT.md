# HeatCast evaluation report

Protocol: train 2020–2022; validation 2023–2024; test 2025–2026; next UTC day; 7-day contiguous input.

Validation-selected grid model: **gradient_boosting**.

| model | scope | validation_f1 | f1 | average_precision | roc_auc | brier |
| --- | --- | --- | --- | --- | --- | --- |
| persistence_grid | grid | 0.5838 | 0.7822 | 0.6438 | 0.8721 | 0.0633 |
| climatology_grid | grid | 0.0359 | 0.2541 | 0.1432 | 0.4982 | 0.1385 |
| persistence_Paris | Paris | 0.0000 | 0.8485 | 0.7436 | 0.9102 | 0.0474 |
| climatology_Paris | Paris | 0.0171 | 0.2705 | 0.1565 | 0.4630 | 0.1502 |
| lstm_Paris | Paris | 0.5000 | 0.5505 | 0.5564 | 0.8875 | 0.0994 |
| convlstm | grid | 0.3643 | 0.5671 | 0.6045 | 0.8977 | 0.0858 |
| convlstm_no_soil | grid | 0.3687 | 0.5705 | 0.6438 | 0.9156 | 0.0798 |
| gradient_boosting | grid | 0.7389 | 0.8505 | 0.8812 | 0.9842 | 0.0390 |

Thresholds, neural checkpoints, boosting complexity, and model selection use validation data only. Test results are reported once after selection. All grid methods share the same days and land cells. Paris methods form a separate comparison.

Labels indicate the third and subsequent consecutive hot days. They do not retrospectively label the first two days. Thresholds are local 90th percentiles in a ±15-calendar-day window using 2020–2022 only. This short reference is an experimental threshold, not a 30-year climate normal.

Selected model F1 95% block-bootstrap interval: [0.7732664275344862, 0.9021948414542944]. Paired F1 difference from persistence interval: [0.015200647034950396, 0.15140337745154372]. These intervals are conditional on this fitted model and test period.

The case study selects the day with greatest observed August 2026 extent for descriptive inspection only. It does not establish that the event was unprecedented or validate a causal climate explanation.

Reanalysis inputs are assumed available at the end of each UTC day. ERA5 publication latency prevents this experiment from demonstrating an operational real-time warning service. Results cover the available regional land grid, not all Europe. One seed and one held-out period do not establish robustness.