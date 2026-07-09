from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/worklens-matplotlib")

REQUIRED_DEPENDENCIES = {
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "numpy": "numpy",
    "pandas": "pandas",
    "plotly": "plotly",
    "pytest": "pytest",
    "ruff": "ruff",
    "sklearn": "scikit-learn",
    "sqlalchemy": "SQLAlchemy",
    "streamlit": "streamlit",
    "uvicorn": "uvicorn",
}
REQUIRED_DIRECTORIES = [
    "app/pages",
    "backend/routes",
    "backend/services",
    "data/sample",
    "data/synthetic",
    "data/real/raw",
    "data/real/processed",
    "database",
    "docs",
    "models",
    "real_eventlog_experiments/results",
    "reports",
    "scripts",
    "src/ml",
    "tests",
]
REQUIRED_FILES = [
    "app/main.py",
    "app/pages/14_Real_Event_Log_Validation.py",
    "backend/main.py",
    "database/schema.sql",
    "real_eventlog_experiments/scripts/run_all_real_experiments.py",
    "real_eventlog_experiments/src/prefix_features.py",
]
REQUIRED_PROJECT_IMPORTS = [
    "backend.main",
    "src.ml.train_sla_model",
    "src.ml.train_completion_model",
    "real_eventlog_experiments.src.load_eventlog",
    "real_eventlog_experiments.src.prefix_features",
    "real_eventlog_experiments.src.train_long_case_classifier",
    "real_eventlog_experiments.src.train_remaining_time_regressor",
]


def verify_dependencies() -> list[str]:
    missing = [
        distribution
        for module, distribution in REQUIRED_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]
    return sorted(missing)


def verify_structure() -> list[str]:
    missing = [
        path for path in [*REQUIRED_DIRECTORIES, *REQUIRED_FILES] if not (ROOT / path).exists()
    ]
    return sorted(missing)


def verify_sample_generation() -> tuple[int, int]:
    from src.data_generation.generate_claims_event_log import (
        GenerationConfig,
        generate_dataset,
    )

    tables = generate_dataset(GenerationConfig(case_count=12, seed=101))
    case_count = len(tables["cases"])
    event_count = len(tables["events"])
    if case_count != 12 or event_count < case_count:
        raise RuntimeError("Synthetic sample generation returned invalid row counts")
    return case_count, event_count


def main() -> int:
    missing_dependencies = verify_dependencies()
    if missing_dependencies:
        print("Missing dependencies: " + ", ".join(missing_dependencies))
        print("Run `make setup` and retry.")
        return 1
    missing_paths = verify_structure()
    if missing_paths:
        print("Missing required project paths: " + ", ".join(missing_paths))
        return 1
    for module in REQUIRED_PROJECT_IMPORTS:
        importlib.import_module(module)
    from backend.main import app

    route_paths = app.openapi().get("paths", {})
    if "/health" not in route_paths:
        print("FastAPI app imported but /health is missing.")
        return 1
    cases, events = verify_sample_generation()
    print(f"Dependencies: {len(REQUIRED_DEPENDENCIES)} required imports available")
    print(f"Project structure: {len(REQUIRED_DIRECTORIES)} folders and {len(REQUIRED_FILES)} files")
    print(f"Core imports: {len(REQUIRED_PROJECT_IMPORTS)} modules imported")
    print(f"FastAPI: imported with {len(route_paths)} documented paths")
    print(f"Sample generation: {cases} cases and {events} events")
    print("Clean setup verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
