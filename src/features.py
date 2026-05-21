"""Financial ratio feature engineering for the Corporate Credit Grading model.

Normalises balance sheet inputs, handles missing values for sovereign-
linked entities, and encodes GICS sector classifications.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FINANCIAL_FEATURES = [
    "ebitda_ratio",
    "leverage_ratio",
    "interest_coverage",
    "current_ratio",
    "revenue_growth_yoy",
    "net_margin",
]

# GICS sector encoding (subset)
GICS_CODES: dict[str, int] = {
    "Energy": 10, "Materials": 15, "Industrials": 20,
    "ConsumerDiscretionary": 25, "ConsumerStaples": 30,
    "HealthCare": 35, "Financials": 40, "IT": 45,
    "CommunicationServices": 50, "Utilities": 55, "RealEstate": 60,
}


def encode_sector(sector: str) -> int:
    """Map a GICS sector name to an integer code."""
    return GICS_CODES.get(sector, 0)


def build_feature_matrix(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, StandardScaler]:
    """Build the feature matrix for the corporate grading neural network.

    Args:
        df: Obligor-level DataFrame with financial and sector columns.
        scaler: Optional pre-fitted StandardScaler. Required when fit=False.
        fit: If True, fit a new scaler on df (training mode).

    Returns:
        Tuple of (feature_matrix, fitted_scaler).
    """
    out = df[FINANCIAL_FEATURES].copy()

    # Winsorise extreme ratios at the 1st and 99th percentile
    for col in FINANCIAL_FEATURES:
        lo, hi = out[col].quantile([0.01, 0.99])
        out[col] = out[col].clip(lo, hi)

    out = out.fillna(out.median())
    out["sector_code"] = df["sector_code"].apply(encode_sector).astype(float)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(out)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        X = scaler.transform(out)

    return X, scaler
