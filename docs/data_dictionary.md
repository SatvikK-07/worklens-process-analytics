# Data dictionary

## Synthetic product-demo tables

### `cases`

| Field | Meaning |
|---|---|
| `case_id` | Stable synthetic workflow identifier |
| `claim_type`, `priority`, `region` | Intake attributes available at case start |
| `provider_id`, `member_id` | Synthetic entity identifiers |
| `diagnosis_group`, `procedure_group` | Synthetic intake categories |
| `created_at`, `closed_at` | Case start and completion timestamps |
| `outcome` | Final synthetic outcome; retrospective only |
| `sla_threshold_hours` | Configured service threshold |
| `total_duration_hours` | Final duration target; never an early-model input |
| `sla_breached` | Classification target; never an early-model input |
| `total_cost` | Generated completed-case cost |
| `rework_count`, `anomaly_label` | Full-case generated labels for retrospective analytics |

### `events`

| Field | Meaning |
|---|---|
| `event_id`, `case_id` | Event and parent-case identifiers |
| `activity`, `timestamp`, `duration_minutes` | Observed workflow step |
| `user_id`, `team` | Synthetic resource ownership |
| `application_used`, `screen_name` | Product-demo system context |
| `event_type`, `status` | Event classification and state |

### `providers` and `users`

Providers contain type, region, and generator parameters. Users contain team,
role, region, and experience level. Historical provider/team model features are
recomputed from cases completed strictly before each prediction timestamp; the
generator parameters themselves are not model inputs.

### `predictions`

Contains early SLA probability, predicted total completion hours, risk band,
associated drivers, intervention text, and early anomaly score. The generated
CSV also contains explicitly named retrospective anomaly outputs. Explanations
are associations, not causal effects.

## Feature groups

- Intake: known at case creation.
- Early prefix: computed from at most the first five synthetic events.
- Historical: computed only from prior completed cases.
- Post-completion: evaluation targets or retrospective investigation fields.

The complete machine-readable registry is generated at
`reports/leakage_audit.csv`.

## Real event-log schema

The normalized public log uses `event_id`, `case_id`, `activity`, `timestamp`,
and `resource`. Prefix features include the first N activity labels, last
observed resource, elapsed prefix time, activity/resource counts, repetition,
inter-event deltas, start hour, and weekday. Targets—full duration, long-case
label, and remaining time—are joined only after prefix construction.

