from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_DIRECTORY_NAMES = {
    "__MACOSX",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
}
FORBIDDEN_FILE_NAMES = {".DS_Store", ".coverage"}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".db-shm",
    ".db-wal",
    ".joblib",
    ".pyc",
    ".pyo",
    ".pkl",
    ".tmp",
}
IGNORED_ROOT_DIRECTORIES = {".git", ".venv", "venv"}
MAX_NON_SAMPLE_CSV_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class HygieneIssue:
    path: Path
    reason: str


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def find_hygiene_issues(root: Path) -> list[HygieneIssue]:
    root = root.resolve()
    sample_root = root / "data" / "sample"
    issues: list[HygieneIssue] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in IGNORED_ROOT_DIRECTORIES:
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIRECTORY_NAMES:
            issues.append(HygieneIssue(relative, "forbidden generated directory"))
            continue
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_FILE_NAMES:
            issues.append(HygieneIssue(relative, "forbidden generated file"))
            continue
        suffix = next(
            (candidate for candidate in FORBIDDEN_SUFFIXES if path.name.endswith(candidate)),
            None,
        )
        if suffix:
            issues.append(HygieneIssue(relative, f"forbidden `{suffix}` artifact"))
            continue
        if path.suffix.lower() in {".zip", ".tar"} or path.name.endswith(".tar.gz"):
            issues.append(HygieneIssue(relative, "archive/duplicate unpack artifact"))
            continue
        if (
            path.suffix.lower() == ".csv"
            and path.stat().st_size > MAX_NON_SAMPLE_CSV_BYTES
            and not _is_inside(path, sample_root)
        ):
            issues.append(
                HygieneIssue(
                    relative,
                    f"CSV exceeds {MAX_NON_SAMPLE_CSV_BYTES // (1024 * 1024)} MB outside data/sample",
                )
            )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail when repository junk is present.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    issues = find_hygiene_issues(args.root)
    if issues:
        print(f"Repository hygiene check failed with {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue.path}: {issue.reason}")
        return 1
    print("Repository hygiene check passed: no forbidden artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
