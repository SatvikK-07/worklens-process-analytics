# API

Run `make run-api`, then open `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health |
| POST | `/predict/sla` | Prefix-safe synthetic SLA probability and associated drivers |
| POST | `/predict/duration` | Prefix-safe synthetic completion-time estimate |
| POST | `/predict/anomaly/early` | First-five-event anomaly risk |
| POST | `/predict/anomaly/retrospective` | Completed-case investigation score |
| GET | `/cases/{case_id}` | Synthetic case and ordered events |
| GET | `/cases/{case_id}/explanation` | Associated risk drivers, not causal reasons |
| GET | `/analytics/bottlenecks` | Ranked synthetic bottlenecks |
| GET | `/analytics/anomalies` | Synthetic anomaly queue |
| GET | `/model/metrics` | Saved synthetic metrics and evidence-track labels |
| GET | `/model/metadata` | Backward-compatible model metadata alias |
| GET | `/model/features/leakage-audit` | Feature availability and task safety |

Pydantic request models use `extra="forbid"`. Submitting final duration, actual
outcome, or any unknown field to an early endpoint returns HTTP 422 before a
model artifact is loaded. Missing model artifacts return HTTP 503.

The retrospective schema intentionally accepts only completed-case fields and
the response labels its mode explicitly.

