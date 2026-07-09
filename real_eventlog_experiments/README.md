# Real event-log modelling track

This package is independent from the synthetic claims product demo. It evaluates
whether WorkLens modelling methodology generalizes to a real public event log.

Primary dataset:

Mannhardt, Felix (2016), *Sepsis Cases - Event Log*, Version 1,
4TU.ResearchData, DOI `10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460`.

Prefix-safe tasks are implemented:

1. Compare long-case median/Q75 and open-after-7/14/30-day targets.
2. Predict remaining time with median/mean baselines and log1p robust models.

The default prefix is N=3; N=5 is also evaluated. Only the observed prefix is
used as model input. Temporal 70/15/15 validation is primary.

```bash
make external-data
make run-real-eventlog-full
```

If network download is unavailable, use the committed sample:

```bash
make run-real-eventlog-sample
```

The sample validates code paths only. README model claims use the full public
log and are written only after the full experiment actually runs.
