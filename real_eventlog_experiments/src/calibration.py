from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def calibration_data(
    target: pd.Series, probability: np.ndarray, bins: int = 10
) -> dict[str, list[float]]:
    fraction_positive, mean_predicted = calibration_curve(
        target,
        probability,
        n_bins=min(bins, max(2, len(target) // 10)),
        strategy="quantile",
    )
    return {
        "mean_predicted_probability": mean_predicted.tolist(),
        "fraction_positive": fraction_positive.tolist(),
    }
