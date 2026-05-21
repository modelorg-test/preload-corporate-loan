"""Financial ratio feature engineering for the Corporate Credit Grading model.

Normalises balance sheet inputs, winsorises extreme ratios,
and encodes GICS sector classifications for the neural network.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FINANCIAL_FEATURES = [
    "ebitda_ratio",
    "leverage_ratio",
    "interest_coverage",
    "current_ratio",
    "revenue_growth_yoy",
    "net_margin",
    "total_assets_log",
    "debt_to_equity",
    "working_capital_ratio",
    "cash_flow_coverage",
    "sector_code",
    "years_in_operation",
    "sovereign_support_flag",
    "external_rating_notch",
    "concentration_index",
]

GICS_CODES: dict[str, int] = {
    "Energy": 10, "Materials": 15, "Industrials": 20,
    "ConsumerDiscretionary": 25, "ConsumerStaples": 30,
    "HealthCare": 35, "Financials": 40, "IT": 45,
    "CommunicationServices": 50, "Utilities": 55, "RealEstate": 60,
}


class FinancialRatioTransformer(BaseEstimator, TransformerMixin):
    """Winsorise and scale financial ratios for the corporate grading model.

    Parameters
    ----------
    winsorise_pct : float
        Percentile for winsorisation (applied symmetrically).
    """

    def __init__(self, winsorise_pct: float = 0.01) -> None:
        self.winsorise_pct = winsorise_pct

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None,
    ) -> FinancialRatioTransformer:
        """Compute percentile bounds and fit the internal scaler."""
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.scaler_ = StandardScaler()

        numeric = X.select_dtypes(include=[np.number])
        for col in numeric.columns:
            lo = float(numeric[col].quantile(self.winsorise_pct))
            hi = float(numeric[col].quantile(1 - self.winsorise_pct))
            self.bounds_[col] = (lo, hi)

        winsorised = self._winsorise(numeric)
        self.scaler_.fit(winsorised)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Winsorise and scale features."""
        numeric = X.select_dtypes(include=[np.number])
        winsorised = self._winsorise(numeric)
        return self.scaler_.transform(winsorised)

    def _winsorise(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in result.columns:
                result[col] = result[col].clip(lo, hi)
        return result.fillna(result.median())
