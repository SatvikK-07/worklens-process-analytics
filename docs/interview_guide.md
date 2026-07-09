# Interview guide

## Sixty-second explanation

WorkLens is an end-to-end workflow-intelligence portfolio project. A
deterministic synthetic claims log demonstrates the product without implying
access to real claims. A separate public hospital event log tests whether the
first-N-event predictive-monitoring method generalizes outside that generator.
Point-in-time feature controls, temporal validation, baselines, typed APIs,
tests, hygiene gates, and generated reports make every claim inspectable.

## What problem does this solve?

Operations teams often know aggregate outcomes but not where work waits, loops,
or becomes risky. WorkLens reconstructs event paths, ranks bottlenecks/rework,
estimates early risk, supports case review, and models automation scenarios from
one governed event-log interface.

## Why synthetic data?

Real claims events are private, organization-specific, and not available for
this portfolio. Synthetic data permits a safe, deterministic, full-stack
product demonstration. Its metrics describe generator behavior only.

## Why add real event-log validation?

Without independent data, feature engineering and model evaluation could simply
learn generator rules. The public Sepsis log tests generic process-mining and
prefix-prediction code on 1,050 real cases and 15,214 events. It is
healthcare-adjacent, not claims validation.

## Why were the real model results weak?

The later temporal window differs from training, full durations are
heavy-tailed, 846 paths make pattern encodings sparse, the first three/five
events carry limited outcome information, and resource/context fields are
limited. The primary N=3 classifier failed to beat prevalence on PR-AUC. Log1p
regression reduced the previous error substantially, but the median baseline
still won on validation and test. The baseline was retained as the selected
model rather than hiding the negative result.

## What is prefix-safe prediction?

Each prediction row is frozen immediately after event N. Features use only
activities, resources, timestamps, gaps, repeats, transitions, and calendar
information observed by that timestamp. Full duration, remaining time, final
event count, end timestamp, final status, and future activities remain targets
or retrospective fields only.

## How was leakage prevented?

- Synthetic tasks use a central per-task feature registry.
- Real tasks maintain an explicit forbidden-prefix set.
- Target thresholds come from the training window.
- Historical encoders fit on train only during selection.
- Fitting rows use leave-one-out target encodings.
- Unseen validation/test categories use training defaults.
- Final encoders may refit on train plus validation only after model selection.
- Tests mutate validation targets and confirm encoded values do not change.

## Why temporal splitting?

Deployment predicts future cases from past cases. Chronological 70/15/15
splitting exposes drift and path changes that random splitting mixes away. A
random split is retained only as a diagnostic comparison.

## Why baselines?

A model is useful only relative to a simple policy. Classification compares
against prevalence/majority predictions. Regression compares against
training-window median and mean. The real regression pipeline selects the
median baseline when trained models fail to improve validation MAE.

## Why calibration and threshold analysis?

ROC-AUC alone does not specify an intervention policy. PR-AUC matters under
imbalance, Brier/calibration check probability quality, and threshold tables
show the precision/recall cost trade-off. Thresholds are selected on validation,
not the final test window.

## How does FastAPI fit in?

FastAPI exposes typed prediction, anomaly, case, analytics, model-metadata, and
leakage-audit endpoints. Pydantic rejects unknown or post-completion fields
before model loading. Routes call services, which call shared core modules; the
backend does not execute notebook code.

## How would this be productionized?

Use governed event contracts, immutable ingestion, PostgreSQL/warehouse
storage, point-in-time feature jobs, a model registry, rolling backtests,
outcome/drift monitoring, authentication and authorization, audit logs,
privacy controls, human review, SLOs, rollback, and controlled intervention
pilots. See `docs/productionization_plan.md`.

## What would improve with real enterprise data?

Richer case attributes, queue/resource capacity, business calendars, reliable
status/censoring, operational SLA definitions, disposition feedback, and
intervention outcomes would enable rolling survival models, subgroup analysis,
and impact evaluation.

## Biggest limitations

Synthetic claims are not operational validation. The public log is hospital
care, not claims. Real holdout performance is weak. Explanations are
associations, anomaly scores are review priorities, ROI is scenario-based, and
the local application lacks production security/observability controls.

## What parts were built?

The repository contains the generator, schema/loader, analytics and process
mining, point-in-time features, classification/regression/anomaly pipelines,
train-only encoders, FastAPI, Streamlit, Docker/Make/CI, reports, hygiene and
verification scripts, and meaningful tests. In an interview, describe only
work personally completed and distinguish reused libraries from original
application logic.

## Decisions worth defending

- Reporting weak real results instead of choosing a favorable test target.
- Selecting the regression baseline when learned models lose on validation.
- Separating synthetic product evidence from real modelling evidence.
- Separating early and retrospective anomaly products.
- Using a modular monolith instead of premature microservices.
- Calling explanations “associated drivers,” never causal reasons.
