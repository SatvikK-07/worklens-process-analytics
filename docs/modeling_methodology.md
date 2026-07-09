# Modelling methodology

## Synthetic product demonstration

Prediction occurs after the first five observed events. Inputs combine intake
attributes, statistics computed from those five events, and provider/team
history from cases closed before the current case was created.

Cases are sorted by creation time and split 70%/15%/15%. Candidate selection
and the SLA operating threshold use the validation window. The chosen pipeline
is refit on train plus validation and evaluated on the later test window.
Random-split results are diagnostic only.

SLA candidates are majority, logistic regression, random forest, and XGBoost.
Completion candidates are median, linear regression, random forest, and
XGBoost. Metrics include ROC-AUC, PR-AUC, precision, recall, F1, Brier score,
confusion matrix, threshold trade-offs, MAE, RMSE, median absolute error, and
R².

## Independent public-log validation

The Sepsis Cases log is evaluated separately at prefixes N=3 and N=5. The
classification target candidates are:

1. duration above the training-window 75th percentile;
2. duration above the training-window median;
3. still open after 7, 14, or 30 days.

The target/model pair is selected using validation PR-AUC lift, balanced
accuracy, and ROC-AUC—in that order. Test results do not change the selected
task. Remaining-time candidates use raw median/mean baselines and log1p target
transforms for ridge, random forest, and robust histogram boosting.

Only observed-prefix features enter pipelines: activity sequence/path, elapsed
time, observed gaps, repeats, transitions, calendar fields, rarity/delay flags,
and resource fields. Historical duration/rate encodings are fit on train only,
use leave-one-out values on fitting rows, and default unseen categories to
training statistics.

The full validation-selected N=3 classifier (`long_case_q75`, random forest)
produced ROC-AUC 0.444 and PR-AUC 0.161 against 0.171 prevalence. It did not
beat the ranking baseline. A diagnostic N=5 `open_after_30_days` variant
produced ROC-AUC 0.586 and PR-AUC 0.187 against 0.150 prevalence, but was not
the validation-selected primary task. For remaining time, the median baseline
was correctly retained at 354.93 hours MAE; the best validation-selected
non-baseline candidate, log1p ridge, reached 362.21 hours on test and did not
beat it.

This negative result is evidence of temporal drift, heavy-tailed durations, and
sparse paths—not a reason to select on test or report the random split.

## Explainability

Global transformed-feature importance is saved for supported models. Local
explanations measure prediction sensitivity when a feature is replaced with a
reference value. They are labelled “associated drivers” and are not causal.
