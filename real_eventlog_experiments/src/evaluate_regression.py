from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(
    target: pd.Series,
    prediction: np.ndarray,
    baseline_mae: float | None = None,
) -> dict[str, float]:
    mae = float(mean_absolute_error(target, prediction))
    result = {
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "median_absolute_error": float(median_absolute_error(target, prediction)),
        "r2": float(r2_score(target, prediction)),
    }
    if baseline_mae is not None:
        result["baseline_improvement_pct"] = float(
            100 * (baseline_mae - mae) / max(baseline_mae, 1e-9)
        )
    return result


def error_by_duration_bucket(
    target: pd.Series, prediction: np.ndarray
) -> list[dict[str, float | str | int]]:
    frame = pd.DataFrame(
        {
            "actual": target.to_numpy(),
            "prediction": prediction,
        }
    )
    frame["absolute_error"] = (frame["actual"] - frame["prediction"]).abs()
    frame["duration_bucket"] = pd.qcut(
        frame["actual"],
        q=min(4, frame["actual"].nunique()),
        duplicates="drop",
    ).astype(str)
    return (
        frame.groupby("duration_bucket", as_index=False, observed=True)
        .agg(
            cases=("actual", "size"),
            mean_actual_hours=("actual", "mean"),
            mae=("absolute_error", "mean"),
        )
        .to_dict("records")
    )


def residual_summary(target: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    residuals = target.to_numpy() - prediction
    return {
        "mean_residual_hours": float(np.mean(residuals)),
        "median_residual_hours": float(np.median(residuals)),
        "residual_std_hours": float(np.std(residuals)),
        "p90_absolute_error_hours": float(np.quantile(np.abs(residuals), 0.90)),
        "max_absolute_error_hours": float(np.max(np.abs(residuals))),
    }
