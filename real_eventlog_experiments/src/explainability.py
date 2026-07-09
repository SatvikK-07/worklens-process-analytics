from __future__ import annotations

import numpy as np
import pandas as pd


def transformed_feature_importance(pipeline, top_n: int = 20) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_)).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .nlargest(top_n, "importance")
        .reset_index(drop=True)
    )


def local_linear_contributions(pipeline, row: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "contribution", "absolute_impact"])
    transformed = preprocessor.transform(row)
    coefficients = np.asarray(model.coef_).ravel()
    contributions = np.asarray(transformed).ravel() * coefficients
    result = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "contribution": contributions,
            "absolute_impact": np.abs(contributions),
        }
    )
    return result.nlargest(top_n, "absolute_impact").reset_index(drop=True)


def local_associated_drivers(
    pipeline,
    row: pd.DataFrame,
    reference: pd.DataFrame,
    feature_columns: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Model-agnostic local sensitivity; associations are not causal effects."""
    original = float(pipeline.predict_proba(row[feature_columns])[0, 1])
    rows = []
    for feature in feature_columns:
        modified = row[feature_columns].copy().astype(object)
        if pd.api.types.is_numeric_dtype(reference[feature]):
            replacement = reference[feature].median()
        else:
            replacement = reference[feature].mode().iloc[0]
        modified.loc[modified.index[0], feature] = replacement
        changed = float(pipeline.predict_proba(modified)[0, 1])
        rows.append(
            {
                "feature": feature,
                "associated_risk_impact": original - changed,
                "absolute_impact": abs(original - changed),
            }
        )
    return pd.DataFrame(rows).nlargest(top_n, "absolute_impact").reset_index(drop=True)
