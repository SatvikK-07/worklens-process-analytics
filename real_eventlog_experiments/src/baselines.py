from __future__ import annotations

import os

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def classification_models(seed: int = 42) -> dict:
    return {
        "Majority Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=24,
            learning_rate=0.06,
            random_state=seed,
        ),
    }


def regression_models(seed: int = 42) -> dict:
    return {
        "Median Baseline": DummyRegressor(strategy="median"),
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Log1p Ridge Regression": TransformedTargetRegressor(
            regressor=Ridge(alpha=4.0),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Log1p Random Forest": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=seed,
                n_jobs=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Log1p Histogram Gradient Boosting": TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=220,
                max_leaf_nodes=20,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=seed,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }
