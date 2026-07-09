# Initial repository audit

Audit date: 2026-06-27

## Structure and entry points

- Streamlit entry point: `app/main.py`
- Streamlit workspaces: `app/pages/`
- FastAPI entry point: `backend/main.py`
- Synthetic generator: `src/data_generation/generate_claims_event_log.py`
- Database loader/schema: `src/ingestion/load_to_db.py`, `database/schema.sql`
- Synthetic ML orchestration: `src/ml/train_all.py`
- Existing external validation: `src/external_validation/sepsis.py`
- Tests: 10 files under `tests/`
- CI: `.github/workflows/ci.yml`
- Lint/coverage configuration: `pyproject.toml`

## Strengths found

- Synthetic claims product demo and 13 Streamlit workspaces were functional.
- Early-case feature engineering already used the first five observed events.
- Provider/team historical features used as-of joins against prior closed cases.
- Temporal model validation, baselines, threshold analysis, and calibration existed.
- FastAPI had typed prediction, case, and analytics routes.
- A real 4TU Sepsis log parser and process-mining validation existed.
- The suite had 45 passing tests and measured above 80% coverage.
- README already separated synthetic product simulation from external validation.

## Gaps against the new acceptance specification

- The real event log was used only for process-algorithm validation, not independent
  prefix-based classification and remaining-time modelling.
- No central feature registry with per-task safety metadata existed.
- Anomaly detection mixed early and retrospective concepts.
- Model-report figures and generated Markdown reports were incomplete.
- API lacked early/retrospective anomaly, model metadata, explanation, and leakage routes.
- Dashboard lacked dedicated real-validation, monitoring, leakage, methodology, and
  split anomaly workspaces.
- Missing generated artifacts caused several pages to fail instead of showing setup steps.
- Detailed architecture, data dictionary, methodology, API, reproducibility,
  productionization, and interview documents were missing.
- Make targets did not yet match the final required command names.

## Generated artifacts found

The working directory contained approximately 226 MB, primarily:

- 144 MB local SQLite database plus WAL files
- 78 MB generated synthetic CSVs
- 3 MB trained pickle artifacts
- Python and pytest caches

These are reproducible runtime artifacts, not source. The artifact policy and
`make clean` now define how they are excluded and regenerated.

