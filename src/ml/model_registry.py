from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.utils.config import settings


def save_model(name: str, artifact: Any, metadata: dict[str, Any]) -> tuple[Path, Path]:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.model_dir / f"{name}.pkl"
    metadata_path = settings.model_dir / f"{name}_metrics.json"
    joblib.dump(artifact, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def load_model(name: str) -> Any:
    return joblib.load(settings.model_dir / f"{name}.pkl")


def load_metadata(name: str) -> dict[str, Any]:
    return json.loads((settings.model_dir / f"{name}_metrics.json").read_text())
