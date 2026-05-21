"""Corporate credit grading model using sklearn MLPClassifier.

Mirrors the architecture of the PyTorch-based production model
(3 hidden layers: 64→32→16) but uses sklearn for accessibility.
Includes Platt scaling calibration and internal grade mapping.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.features.engineering import FinancialRatioTransformer

logger = logging.getLogger(__name__)

INTERNAL_GRADES = 15  # 15-grade internal rating scale


def build_grading_pipeline(
    hidden_layers: tuple[int, ...] = (64, 32, 16),
    alpha: float = 1e-4,
    max_iter: int = 500,
    calibrate: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """Build the corporate credit grading pipeline.

    Parameters
    ----------
    hidden_layers : tuple
        Sizes of hidden layers for the MLPClassifier.
    alpha : float
        L2 regularisation strength.
    max_iter : int
        Maximum training iterations.
    calibrate : bool
        If True, wrap with CalibratedClassifierCV for Platt scaling.
    random_state : int
        Random seed.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )

    if calibrate:
        classifier = CalibratedClassifierCV(
            classifier,
            method="sigmoid",
            cv=3,
        )

    pipeline = Pipeline(
        steps=[
            ("financial_ratios", FinancialRatioTransformer(winsorise_pct=0.01)),
            ("classifier", classifier),
        ]
    )

    logger.info(
        "Built grading pipeline: layers=%s, alpha=%.1e, calibrated=%s",
        hidden_layers, alpha, calibrate,
    )
    return pipeline


def map_to_grade(pd_estimate: float) -> int:
    """Map a PD estimate to an internal 1–15 grade scale.

    Uses logarithmically spaced breakpoints between 0.1% and 30% PD.

    Parameters
    ----------
    pd_estimate : float
        Probability of default.

    Returns
    -------
    int
        Internal grade (1 = best, 15 = worst).
    """
    breakpoints = np.logspace(np.log10(0.001), np.log10(0.30), INTERNAL_GRADES - 1)
    return int(np.searchsorted(breakpoints, pd_estimate) + 1)


def grade_distribution(pd_estimates: np.ndarray) -> dict[int, int]:
    """Compute the distribution of internal grades.

    Parameters
    ----------
    pd_estimates : np.ndarray
        Array of PD estimates.

    Returns
    -------
    dict[int, int]
        Grade → count mapping.
    """
    grades = [map_to_grade(p) for p in pd_estimates]
    return {g: grades.count(g) for g in sorted(set(grades))}
