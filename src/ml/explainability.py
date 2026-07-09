from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _shap_values(
    artifact: dict,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(frame[artifact["features"]])
    background = transformed[: min(100, len(transformed))]
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values, preprocessor.get_feature_names_out()


def global_feature_importance(
    artifact: dict,
    top_n: int = 15,
    sample: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    if sample is not None and not sample.empty:
        values, feature_names = _shap_values(artifact, sample)
        importance = np.abs(values).mean(axis=0)
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_)).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def local_shap_explanation(
    artifact: dict,
    row: pd.Series,
    background: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    analysis_frame = pd.concat(
        [background[artifact["features"]].head(100), row[artifact["features"]].to_frame().T],
        ignore_index=True,
    )
    values, feature_names = _shap_values(artifact, analysis_frame)
    local_values = values[-1]
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": local_values,
            "absolute_impact": np.abs(local_values),
        }
    )
    result["direction"] = np.where(result["shap_value"] >= 0, "Increases risk", "Decreases risk")
    return result.nlargest(top_n, "absolute_impact").reset_index(drop=True)


def case_risk_factors(row: pd.Series) -> list[str]:
    factors = []
    if row["missing_document_flag"]:
        factors.append("Missing or inconsistent documents")
    if row["queue_wait_time_total"] > 24:
        factors.append("High accumulated queue wait")
    if row["provider_clarification_count"] > 0:
        factors.append("Provider clarification loop")
    if row["rework_count"] > 0:
        factors.append("Repeated workflow activity")
    if row["priority"] in {"High", "Urgent"}:
        factors.append("Compressed priority SLA")
    if row["claim_type"] in {"Prior Authorization", "High-Cost Specialty Drug"}:
        factors.append(f"Complex {row['claim_type'].lower()} workflow")
    return factors[:5] or ["No dominant operational risk factor"]


def recommended_action(row: pd.Series, probability: float) -> str:
    if probability >= 0.75:
        if row["missing_document_flag"]:
            return "Escalate document resolution and reserve senior-review capacity."
        return "Escalate to the queue lead for immediate prioritization."
    if probability >= 0.50:
        return "Move to the priority queue and review the next handoff."
    if probability >= 0.25:
        return "Monitor at the next activity transition."
    return "Continue standard processing."
