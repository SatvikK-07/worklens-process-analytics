from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external_validation.sepsis import (
    EXPECTED_MD5,
    external_validation_report,
    file_md5,
    parse_sepsis_xes,
)


def main() -> None:
    source = Path("data/real/raw/sepsis_cases.xes.gz")
    if not source.exists():
        raise FileNotFoundError("Run `python scripts/download_external_log.py` before validation.")
    if file_md5(source) != EXPECTED_MD5:
        raise ValueError("External dataset checksum does not match the published file.")
    events = parse_sepsis_xes(source)
    report = external_validation_report(events)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "external_validation.json").write_text(json.dumps(report, indent=2))

    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_cases = events["case_id"].drop_duplicates().head(35)
    events[events["case_id"].isin(sample_cases)].to_csv(
        sample_dir / "sepsis_events_sample.csv", index=False
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
