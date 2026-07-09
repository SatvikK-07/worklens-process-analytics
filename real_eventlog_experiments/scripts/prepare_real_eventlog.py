from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_eventlog_experiments.src.load_eventlog import load_eventlog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/real/raw/sepsis_cases.xes.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/processed/sepsis_events.csv"),
    )
    args = parser.parse_args()
    events = load_eventlog(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)
    print(f"Prepared {len(events):,} events across {events['case_id'].nunique():,} cases")


if __name__ == "__main__":
    main()
