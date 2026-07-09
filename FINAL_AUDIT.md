# WorkLens AI final audit

Audit date: 2026-06-27

## 1. Changelog

- Added an enforceable repository hygiene scanner and `make check-hygiene`.
- Added dependency, project-structure, sample-generation, API, and model-import
  checks through `make verify`.
- Added explicit sample/full real-event-log Make targets.
- Expanded real prefix features with paths, activity counts, transitions,
  repeats, timing distributions, calendar fields, rarity/delay flags, resources,
  and historical encodings.
- Implemented train-only historical encoders, leave-one-out fitting-row values,
  smoothing, and unseen-category defaults.
- Added median/Q75 and open-after-7/14/30-day classification targets.
- Added balanced accuracy and strengthened target/model selection on validation.
- Added median/mean regression baselines and log1p/expm1 ridge, random forest,
  and robust histogram-gradient-boosting candidates.
- Changed regression selection to retain the baseline when trained models do
  not improve chronological validation MAE.
- Expanded the real-log Streamlit page into a methodology/evidence page.
- Added generated real model reports and seven reproducible real-result plots.
- Expanded hygiene, encoding-leakage, dashboard-helper, API-docs, and
  end-to-end real modelling tests.
- Raised the enforced coverage floor from 70% to 80%.
- Updated CI, README, artifact policy, methodology, reproducibility,
  limitations, and interview guidance.

## 2. Commands run and observed outcomes

| Command | Observed result |
|---|---|
| `make setup` | PASS; all runtime/dev requirements already satisfied |
| `make clean` | PASS; caches, databases, binaries, raw log, temporary files, and legacy figures removed |
| `make check-hygiene` | PASS after clean; no forbidden artifacts |
| `make verify` | PASS; 14 dependencies, 15 folders, 6 key files, 7 core imports, 12 API paths, sample generation |
| `make lint` | Initial final-gate attempt found one unformatted file; after `ruff format real_eventlog_experiments/scripts/run_all_real_experiments.py`, rerun PASS |
| `make test` | PASS; 73 tests, 85.67% coverage, enforced floor 80% |
| `make generate-sample-data` | PASS; 250 cases and 2,081 events |
| `make train-sample` | PASS; all four sample model workflows trained and predictions written |
| `make run-real-eventlog-sample` | PASS; sample code path completed and wrote only to `/tmp/worklens-real-eventlog-sample/` |
| `make run-real-eventlog-full` | PASS before final clean; checksum-verified 1,050-case/15,214-event public log |
| `make reports` | PASS; leakage, synthetic, and real reports generated from saved JSON/CSV |
| Streamlit browser check | PASS; evidence page rendered all required sections with zero browser-console errors |
| FastAPI tests | PASS; health, OpenAPI/docs, predictions, metadata, leakage audit, cases, explanations, and validation errors covered |

The only failure during the final sequence was the formatting check described
above. It was corrected and the exact lint command then passed.

## 3. Final repository hygiene status

- No `__pycache__`, `.pyc`, pytest/Ruff caches, Mac metadata, local database,
  raw public log, model binary, temporary file, or archive is permitted.
- CSV files above 5 MB are rejected outside `data/sample/`.
- Small sample CSVs, source, metric JSON/CSV, generated report figures, and
  documentation are intentional artifacts.
- `scripts/check_repo_hygiene.py` has tests proving it rejects cache, database,
  binary-model, and oversized-CSV fixtures.
- `.gitignore` and `docs/artifact_policy.md` match the scanner.
- Final post-clean workspace size: **2.1 MB**; no file exceeds 5 MB.

## 4. Final test and coverage status

- Tests: **73 passed**
- Warnings: one Starlette/TestClient deprecation warning from the installed
  dependency combination
- Coverage: **85.67%**
- Enforced minimum: **80%**
- Critical real feature/encoder modules: above 90% coverage

## 5. Final lint status

- `ruff check .`: PASS
- `ruff format --check .`: PASS after formatting the one reported file

## 6. Full real-event-log modelling results

Dataset: Sepsis Cases Event Log, 1,050 cases, 15,214 events, 16 activities,
846 variants. Chronological 70/15/15 is primary.

### N=3 validation-selected primary classification

- Target: `long_case_q75`
- Model: Random Forest
- ROC-AUC: 0.4436
- PR-AUC: 0.1615
- Positive prevalence: 0.1709
- PR-AUC lift over prevalence: 0.9449
- Precision: 0.1538
- Recall: 0.0741
- F1: 0.1000
- Balanced accuracy: 0.4951
- Brier score: 0.1702

This primary task does **not** beat the prevalence/majority ranking baseline.

### Diagnostic survival-style result

The N=5 `open_after_30_days` variant produced ROC-AUC 0.5856 and PR-AUC
0.1867 against prevalence 0.1503 (1.2416x lift). This is a secondary diagnostic
test result and was not used to replace the validation-selected primary task.

### N=3 remaining-time regression

- Validation-selected model: Median Baseline
- Baseline/test MAE: 354.93 hours
- RMSE: 788.91 hours
- Median absolute error: 129.09 hours
- R²: -0.1194
- Best validation-selected non-baseline candidate: Log1p Ridge
- Log1p Ridge test MAE: 362.21 hours
- Non-baseline improvement over median: -2.05%

No learned regression model beats the meaningful median baseline. The pipeline
therefore retains the baseline instead of presenting a worse model as selected.

## 7. Leakage assessment

- Full duration, remaining time, case end, final event count/status, and future
  activities/transitions are forbidden prefix inputs.
- Q75/median thresholds use training data only.
- Historical encoders fit on train only for model/target selection.
- Training rows use leave-one-out target/duration encodings.
- Unseen categories use training-window defaults.
- Final test encoders refit on train plus validation only after selection.
- Tests prove validation target mutation cannot change encoded values.

## 8. Remaining limitations

- Synthetic claims results validate a product demonstration, not operations.
- The Sepsis log is hospital care, not claims processing.
- The primary real classifier is weak and unstable over time.
- Learned remaining-time models do not beat the median baseline.
- No clinical, causal, fraud, payment, or audited-savings claim is supported.
- Local files/SQLite are development choices.
- Authentication, authorization, PHI controls, migrations, live feature jobs,
  registry/approval workflows, drift/outcome monitoring, observability,
  rollback, and human review remain production work.

## 9. Final honest score

**9.3/10 as an internship portfolio repository.**

The engineering, reproducibility, leakage control, testing, reporting, API, and
dashboard evidence are strong. It is not scored 10/10 because the
validation-selected real model is not operationally useful, no learned
remaining-time model beats baseline, and production controls are intentionally
not implemented.

## 10. Files changed in this hardening pass

- `.github/workflows/ci.yml`
- `.gitignore`
- `FINAL_AUDIT.md`
- `Makefile`
- `README.md`
- `app/pages/14_Real_Event_Log_Validation.py`
- `app/real_eventlog_evidence.py`
- `docs/artifact_policy.md`
- `docs/interview_guide.md`
- `docs/modeling_methodology.md`
- `docs/reproducibility.md`
- `models/sample/.gitkeep`
- `pyproject.toml`
- `real_eventlog_experiments/README.md`
- `real_eventlog_experiments/scripts/run_all_real_experiments.py`
- `real_eventlog_experiments/src/baselines.py`
- `real_eventlog_experiments/src/evaluate_classification.py`
- `real_eventlog_experiments/src/evaluate_regression.py`
- `real_eventlog_experiments/src/prefix_features.py`
- `real_eventlog_experiments/src/preprocess_eventlog.py`
- `real_eventlog_experiments/src/temporal_split.py`
- `real_eventlog_experiments/src/train_long_case_classifier.py`
- `real_eventlog_experiments/src/train_remaining_time_regressor.py`
- `requirements.txt`
- `scripts/__init__.py`
- `scripts/check_repo_hygiene.py`
- `scripts/generate_model_reports.py`
- `scripts/generate_real_eventlog_report.py`
- `scripts/verify_clean_setup.py`
- `tests/test_api.py`
- `tests/test_feature_registry_and_anomalies.py`
- `tests/test_real_eventlog_experiments.py`
- `tests/test_repo_hygiene.py`

Generated evidence updated by committed scripts:

- `docs/leakage_audit.md`
- `reports/leakage_audit.csv`
- `reports/real_eventlog_model_report.md`
- `reports/figures/real_long_case_roc_curve.png`
- `reports/figures/real_long_case_pr_curve.png`
- `reports/figures/real_long_case_calibration_curve.png`
- `reports/figures/real_long_case_confusion_matrix.png`
- `reports/figures/real_long_case_feature_importance.png`
- `reports/figures/real_remaining_time_residuals.png`
- `reports/figures/real_remaining_time_error_by_bucket.png`
- `real_eventlog_experiments/results/experiment_comparison.csv`
- `real_eventlog_experiments/results/experiment_summary.json`
- `real_eventlog_experiments/results/long_case_n3_metrics.json`
- `real_eventlog_experiments/results/long_case_n3_predictions.csv`
- `real_eventlog_experiments/results/long_case_n3_feature_importance.csv`
- `real_eventlog_experiments/results/long_case_n5_metrics.json`
- `real_eventlog_experiments/results/long_case_n5_predictions.csv`
- `real_eventlog_experiments/results/long_case_n5_feature_importance.csv`
- `real_eventlog_experiments/results/remaining_time_n3_metrics.json`
- `real_eventlog_experiments/results/remaining_time_n3_predictions.csv`
- `real_eventlog_experiments/results/remaining_time_n5_metrics.json`
- `real_eventlog_experiments/results/remaining_time_n5_predictions.csv`
