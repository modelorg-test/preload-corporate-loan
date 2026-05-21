"""Weekly batch inference for the Corporate Credit Grading model.

Scores the full wholesale portfolio, maps PD estimates to the internal
15-grade rating scale, and applies sector-specific committee overlays.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features import build_feature_matrix
from train import CorporateGradingNet, map_to_grade

logger = logging.getLogger(__name__)

# Sectors under structural stress — overlay shifts PD by these grade notches
STRESS_OVERLAYS: dict[str, int] = {
    "RealEstate": 2,
    "Hospitality": 1,
}


class CorporateGradingPredictor:
    """Weekly batch scoring engine for corporate obligors."""

    def __init__(self, model_path: str, scaler_path: str) -> None:
        self._model = CorporateGradingNet()
        self._model.load_state_dict(torch.load(model_path, weights_only=True))
        self._model.eval()
        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)
        logger.info("Loaded corporate grading model from %s", model_path)

    def score(self, obligors: pd.DataFrame) -> pd.DataFrame:
        """Score the corporate portfolio.

        Args:
            obligors: DataFrame with financial and sector columns.

        Returns:
            DataFrame with pd_estimate, lgd_estimate, and internal_grade.
        """
        X, _ = build_feature_matrix(obligors, scaler=self._scaler, fit=False)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            pd_estimates = self._model(X_tensor).numpy()

        result = obligors.copy()
        result["pd_estimate"] = pd_estimates
        # LGD estimated as function of seniority (simplified)
        result["lgd_estimate"] = 0.45  # Regulatory LGD floor for unsecured corporate

        grades = [map_to_grade(float(p)) for p in pd_estimates]

        # Apply committee overlays for stressed sectors
        sectors = obligors.get("sector", pd.Series(["Unknown"] * len(obligors)))
        adjusted_grades = [
            min(g + STRESS_OVERLAYS.get(s, 0), 15)
            for g, s in zip(grades, sectors)
        ]

        result["internal_grade"] = adjusted_grades
        return result[["pd_estimate", "lgd_estimate", "internal_grade"]]
