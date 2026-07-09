# Limitations

- Synthetic claims metrics measure generator behavior, not real claims
  performance.
- The real hospital log validates a generic prefix-method implementation, not
  claims-domain validity. Current real-log holdout performance is weak.
- Long-case classification depends on a training-window percentile definition;
  it is not a clinical or contractual SLA.
- Remaining-time errors are large and variant-dependent. The current model
  should not support operational decisions.
- Associated feature drivers are not causal explanations.
- Anomaly scores prioritize review and do not establish fraud, error, or
  misconduct.
- ROI and what-if outputs use configured assumptions and are not audited
  savings or causal intervention estimates.
- New providers/resources use neutral cold-start defaults.
- Local SQLite, files, and in-memory artifact loading are development choices.
- The project has no authentication, authorization, PHI controls, production
  monitoring, approval workflow, or deployment hardening.

