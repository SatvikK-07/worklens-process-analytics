from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.external_validation.sepsis import DOWNLOAD_URL, EXPECTED_MD5, file_md5


def download(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DOWNLOAD_URL, destination)
    actual_md5 = file_md5(destination)
    if actual_md5 != EXPECTED_MD5:
        destination.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch: expected {EXPECTED_MD5}, got {actual_md5}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public Sepsis XES log.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/raw/sepsis_cases.xes.gz"),
    )
    args = parser.parse_args()
    print(download(args.output))


if __name__ == "__main__":
    main()
