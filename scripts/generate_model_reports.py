from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_leakage_audit import main as generate_leakage_audit
from scripts.generate_real_eventlog_report import generate as generate_real_report
from scripts.generate_synthetic_model_report import main as generate_synthetic_report


def main() -> None:
    generate_leakage_audit()
    generate_synthetic_report()
    generate_real_report()
    print("All model reports generated from saved metrics")


if __name__ == "__main__":
    main()
