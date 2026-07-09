# Productionization plan

This repository is not production-ready. A real deployment would require:

1. Governed event contracts, immutable raw storage, idempotent ingestion, and
   quality alerts for lateness, duplicates, schema drift, and chronology.
2. PostgreSQL or a warehouse for operational data, with migrations and
   incremental feature computation replacing local SQLite rebuilds.
3. A feature store or versioned feature jobs that enforce point-in-time
   correctness and preserve training/serving parity.
4. A model registry with dataset, code, feature, metric, approval, and rollback
   metadata; scheduled retraining only after monitored evidence supports it.
5. Live outcome capture, calibration and performance monitoring, drift checks,
   subgroup analysis, alert thresholds, and rollback criteria.
6. Authentication, role-based authorization, audit logging, encryption,
   secrets management, rate limits, and privacy/compliance review.
7. Human review for high-impact predictions and anomalies, with reason codes,
   disposition capture, and an appeal/escalation path.
8. Structured logs, traces, service metrics, SLOs, incident response, backup,
   disaster recovery, and failure-mode testing.
9. Controlled operational pilots to estimate intervention effects before any
   savings or outcome claim.

