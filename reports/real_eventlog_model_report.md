# Real event-log model report

Generated from machine-readable results by `scripts/generate_real_eventlog_report.py`.

## Dataset

- Dataset: Sepsis Cases Event Log
- Run type: `full_public_event_log`
- Cases/events: 1,050 / 15,214
- Activities/variants: 16 / 846
- Date range: 2013-11-07T07:18:29+00:00 to 2015-06-05T10:25:11+00:00
- Validation: chronological 70/15/15 primary; random split diagnostic only

## Classification task

- Prediction moment: after the first 3 events
- Validation-selected target: `long_case_q75`
- Definition: Duration exceeds the training-window 75th percentile
- Selected model: Random Forest
- Selection used validation metrics only

| model | roc_auc | pr_auc | precision | recall | f1 | balanced_accuracy | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Majority Baseline | 0.5000 | 0.1709 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1458 |
| Logistic Regression | 0.4012 | 0.1452 | 0.1000 | 0.0370 | 0.0541 | 0.4842 | 0.2246 |
| Random Forest | 0.4436 | 0.1615 | 0.1538 | 0.0741 | 0.1000 | 0.4951 | 0.1702 |
| Histogram Gradient Boosting | 0.3866 | 0.1363 | 0.0000 | 0.0000 | 0.0000 | 0.4656 | 0.2096 |

### Target variants

| target | selected model | prevalence | ROC-AUC | PR-AUC | PR lift |
| --- | --- | --- | --- | --- | --- |
| long_case_q75 | Random Forest | 0.1709 | 0.4436 | 0.1615 | 0.9449 |
| long_case_median | Logistic Regression | 0.4367 | 0.4739 | 0.4580 | 1.0487 |
| open_after_7_days | Logistic Regression | 0.3608 | 0.4942 | 0.3702 | 1.0262 |
| open_after_14_days | Random Forest | 0.2342 | 0.4557 | 0.2623 | 1.1200 |
| open_after_30_days | Logistic Regression | 0.1456 | 0.4841 | 0.1566 | 1.0760 |

The primary temporal test PR-AUC lift over prevalence is
0.945. A value above 1 beats the majority/prevalence ranking
baseline; a value at or below 1 does not. Target-variant test results are
diagnostic and were not used to change the validation-selected primary task.

## Remaining-time regression

- Prediction moment: after the first 3 events
- Target: remaining hours after the prefix timestamp
- Non-baseline target transform: log1p during fitting, expm1 after prediction
- Validation-selected model: Median Baseline

| model | mae | rmse | median_absolute_error | r2 |
| --- | --- | --- | --- | --- |
| Median Baseline | 354.9254 | 788.9131 | 129.0936 | -0.1194 |
| Mean Baseline | 693.0403 | 820.6327 | 648.8171 | -0.2112 |
| Log1p Ridge Regression | 362.2103 | 802.4958 | 90.1769 | -0.1583 |
| Log1p Random Forest | 422.6265 | 775.2696 | 241.2633 | -0.0810 |
| Log1p Histogram Gradient Boosting | 367.9128 | 815.2138 | 85.6739 | -0.1953 |

Selected MAE: 354.93 hours. Median-baseline MAE:
354.93 hours. **Beats baseline:
no**.

## Leakage controls

- Prefix fields contain only the first N observed events.
- Duration, remaining time, final event count, end timestamp, and future events
  are forbidden model inputs.
- Historical rate/duration encodings are fitted on the training window only.
- Fitting rows use leave-one-out encodings.
- Unseen validation/test categories use training-window defaults.

## Interpretation

The full experiment is evidence for reproducible, leakage-controlled predictive
monitoring—not clinical usefulness. Temporal performance remains weak and
unstable across target definitions. The richer feature layer and log target
reduce remaining-time MAE substantially relative to the earlier implementation,
but the report retains the median baseline whenever it is better.

## Limitations and next work

The public log is hospital care, not claims operations. Prefix paths are sparse,
duration is heavy-tailed, and resource/context fields are limited. Next work
would use rolling-origin backtests, censor-aware survival methods, richer
resource/case attributes, and calibration monitoring on a governed enterprise
event stream.
