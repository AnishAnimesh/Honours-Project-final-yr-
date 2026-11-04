# Neural initialization sensitivity supplement

This diagnostic was specified after inspecting the benchmark results. It tests initialization and training-order sensitivity, not a new untouched test or a search for a winning seed. Preserve all benchmark artifacts and all completed seeds. No hyperparameters, epochs, masks, channels or cutoffs are chosen by test scores.

Use seeds 7, 21 and 42 for the spatial Paris LSTM, full ConvLSTM, no-soil ConvLSTM and the seven-location one-day ongoing and onset LSTM. Seed 42 reuses the saved benchmark checkpoint and predictions under their recorded conditions. Seeds 7 and 21 retrain with the same loss, optimizer, early stopping, split and validation-only cutoff procedure. Maximum epochs remain 35 for spatial models and 20 for location models. All available runs are retained; do not select a seed using test results.

Report parameter counts, eligible target days, dependent location/cell-day counts, event prevalence, cutoff, precision, recall, F1, Brier and completed epochs. Summarize mean, sample standard deviation and range across the three seeds. These descriptive across-seed statistics are conditional on the fixed data and design, not climate-sampling confidence intervals. The point and grid architectures retain different input information; seed replication does not resolve that confound.
