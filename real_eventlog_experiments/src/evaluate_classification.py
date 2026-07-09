from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    target: pd.Series,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = probability >= threshold
    prevalence = float(target.mean())
    pr_auc = float(average_precision_score(target, probability)) if target.sum() > 0 else 0.0
    roc_auc = float(roc_auc_score(target, probability)) if target.nunique() == 2 else None
    balanced_accuracy = (
        float(balanced_accuracy_score(target, prediction)) if target.nunique() == 2 else None
    )
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": balanced_accuracy,
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(target, probability)),
        "positive_prevalence": prevalence,
        "pr_auc_lift_over_prevalence": (float(pr_auc / prevalence) if prevalence else 0.0),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }


def threshold_table(target: pd.Series, probability: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        prediction = probability >= threshold
        precision = precision_score(target, prediction, zero_division=0)
        recall = recall_score(target, prediction, zero_division=0)
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision),
                "recall": float(recall),
                "f2": float(f2),
            }
        )
    return rows


def choose_threshold(rows: list[dict[str, float]]) -> float:
    eligible = [row for row in rows if row["precision"] >= 0.30]
    return float(max(eligible or rows, key=lambda row: row["f2"])["threshold"])


def lift_table(
    target: pd.Series, probability: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    frame = pd.DataFrame({"target": target.to_numpy(), "probability": probability})
    frame["decile"] = pd.qcut(
        frame["probability"].rank(method="first"),
        q=min(bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    prevalence = max(float(frame["target"].mean()), 1e-9)
    result = (
        frame.groupby("decile", as_index=False)
        .agg(cases=("target", "size"), positive_rate=("target", "mean"))
        .sort_values("decile", ascending=False)
    )
    result["lift"] = result["positive_rate"] / prevalence
    return result.to_dict("records")
