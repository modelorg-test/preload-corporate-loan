"""Training pipeline for the corporate credit grading model.

Uses sklearn's make_classification to generate synthetic obligor data,
trains the MLPClassifier pipeline with Platt scaling, and evaluates
using banking-standard metrics (AUC, accuracy ratio, grade migration).

Usage::

    python -m src.pipelines.train
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.features.engineering import FINANCIAL_FEATURES
from src.models.grading import (
    build_grading_pipeline,
    grade_distribution,
)

logger = logging.getLogger(__name__)


def generate_corporate_dataset(
    n_samples: int = 15000,
    default_rate: float = 0.04,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic corporate obligor dataset.

    Parameters
    ----------
    n_samples : int
        Number of obligor-year observations.
    default_rate : float
        Target default rate (~4% for investment-grade portfolio).
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (X, y)
        X: DataFrame with financial ratio features.
        y: Series of binary default indicators.
    """
    n_features = len(FINANCIAL_FEATURES)

    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 3,
        n_redundant=2,
        n_classes=2,
        weights=[1 - default_rate, default_rate],
        flip_y=0.01,
        random_state=random_state,
    )

    X = pd.DataFrame(X_raw, columns=FINANCIAL_FEATURES)

    scalers = {
        "ebitda_ratio": (0.0, 0.40),
        "leverage_ratio": (0.1, 5.0),
        "interest_coverage": (0.5, 20.0),
        "current_ratio": (0.3, 4.0),
        "revenue_growth_yoy": (-0.30, 0.50),
        "net_margin": (-0.20, 0.30),
        "total_assets_log": (6.0, 12.0),
        "debt_to_equity": (0.0, 8.0),
        "working_capital_ratio": (-0.5, 1.5),
        "cash_flow_coverage": (0.1, 5.0),
        "sector_code": (10, 60),
        "years_in_operation": (1, 100),
        "sovereign_support_flag": (0, 1),
        "external_rating_notch": (1, 22),
        "concentration_index": (0.0, 1.0),
    }

    for col, (lo, hi) in scalers.items():
        if col in X.columns:
            col_min, col_max = X[col].min(), X[col].max()
            X[col] = lo + (X[col] - col_min) / (col_max - col_min + 1e-8) * (hi - lo)

    for int_col in ["sector_code", "years_in_operation",
                     "sovereign_support_flag", "external_rating_notch"]:
        if int_col in X.columns:
            X[int_col] = X[int_col].round().clip(lower=0).astype(int)

    y = pd.Series(y, name="default_indicator")

    logger.info(
        "Generated corporate dataset: %d obligors, default_rate=%.2f%%",
        n_samples, y.mean() * 100,
    )
    return X, y


def train(
    n_samples: int = 15000,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Run the corporate grading model training.

    Returns
    -------
    dict
        Training results.
    """
    X, y = generate_corporate_dataset(n_samples=n_samples, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    pipeline = build_grading_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    # CV evaluation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y_prob_cv = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba",
    )[:, 1]

    cv_auc = roc_auc_score(y_train, y_prob_cv)
    cv_ar = 2 * cv_auc - 1

    # OOT evaluation
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_ar = 2 * test_auc - 1

    # Grade distribution
    grades = grade_distribution(y_prob_test)

    metrics = {
        "cv_auc_roc": cv_auc,
        "cv_accuracy_ratio": cv_ar,
        "test_auc_roc": test_auc,
        "test_accuracy_ratio": test_ar,
    }

    logger.info("CV AUC=%.4f AR=%.4f | Test AUC=%.4f AR=%.4f", cv_auc, cv_ar, test_auc, test_ar)
    logger.info("Grade distribution: %s", grades)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "grade_distribution": grades,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train corporate credit grading model")
    parser.add_argument("--samples", type=int, default=15000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    results = train(n_samples=args.samples)
    m = results["metrics"]

    print("\n" + "=" * 60)
    print("CORPORATE GRADING MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"  CV AUC-ROC:          {m['cv_auc_roc']:.4f}")
    print(f"  CV Accuracy Ratio:   {m['cv_accuracy_ratio']:.4f}")
    print(f"  Test AUC-ROC:        {m['test_auc_roc']:.4f}")
    print(f"  Test Accuracy Ratio: {m['test_accuracy_ratio']:.4f}")
    print(f"  Grade distribution:  {results['grade_distribution']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
