from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_eventlog_experiments.src.load_eventlog import load_eventlog
from real_eventlog_experiments.src.train_remaining_time_regressor import (
    run_remaining_time_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prefix-length", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("real_eventlog_experiments/results")
    )
    args = parser.parse_args()
    result = run_remaining_time_experiment(
        load_eventlog(args.input), args.prefix_length, args.output_dir
    )
    print(json.dumps(result["models"][result["selected_model"]], indent=2))


if __name__ == "__main__":
    main()
