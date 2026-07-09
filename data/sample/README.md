# Public sample data

`sepsis_events_sample.csv` is a small derived sample from:

Mannhardt, Felix (2016), *Sepsis Cases - Event Log*, Version 1,
4TU.ResearchData. DOI:
`10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460`.

The sample is included only to exercise process-mining code in CI. Download and
verify the complete public log with:

```bash
python scripts/download_external_log.py
python scripts/validate_external_log.py
```

The source file is excluded from version control. Its published MD5 checksum is
validated before analysis.

