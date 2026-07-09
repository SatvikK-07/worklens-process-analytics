# Architecture

WorkLens is a portfolio-scale modular monolith. The separation is by evidence
track and responsibility, not by unnecessary services.

```mermaid
flowchart LR
  SG["Deterministic synthetic generator"] --> SC["Synthetic CSVs"]
  SC --> DB[("Local SQLite")]
  SC --> SF["First-five-event features"]
  DB --> AN["Process and operational analytics"]
  SF --> SM["Synthetic temporal ML"]
  RL["Public Sepsis XES log"] --> PF["First-N-event features"]
  PF --> RM["Independent temporal ML"]
  AN --> UI["Streamlit"]
  SM --> UI
  RM --> UI
  AN --> API["FastAPI services"]
  SM --> API
  SM --> SR["Generated synthetic report"]
  RM --> RR["Generated real-log report"]
```

## Boundaries

- `src/data_generation/`: deterministic synthetic product data.
- `src/analytics/` and `src/process_mining/`: reusable calculations.
- `src/ml/`: synthetic early-prediction and split anomaly pipelines.
- `real_eventlog_experiments/`: independent public-log experiments.
- `backend/`: typed HTTP routes calling services, which call core modules.
- `app/`: Streamlit presentation; no model fitting occurs in pages.
- `database/`: SQLite schema and views for local demonstration.

Generated CSVs, SQLite files, raw public logs, and pickle artifacts are local
runtime outputs. JSON metrics and reports are reproducible evidence artifacts.

