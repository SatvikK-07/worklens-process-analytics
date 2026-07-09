# Reproducibility

Use Python 3.11.

```bash
make setup
make verify
make check-hygiene
make generate-sample-data
make train-sample
make test
make lint
```

For the full synthetic product demo:

```bash
make data
make db
make train
make reports
make run-streamlit
```

For independent validation:

```bash
make external-data
make run-real-eventlog-full
```

If download is unavailable:

```bash
make run-real-eventlog-sample
```

The sample run validates code paths only and must not be cited as modelling
evidence. The downloader verifies the published MD5 checksum. Synthetic
generation uses `WORKLENS_RANDOM_SEED` (default 42). `make clean` removes local
databases, binary models, raw downloads, full generated CSVs, and caches while
preserving source, small samples, JSON metrics, and generated reports.
