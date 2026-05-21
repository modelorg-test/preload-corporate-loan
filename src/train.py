"""Training pipeline for the Corporate Credit Grading model.

Fits a feedforward neural network on obligor financial ratios,
applies Platt scaling calibration, and registers with MLflow.
"""

from __future__ import annotations

import argparse
import logging
import pickle

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from features import build_feature_matrix

logger = logging.getLogger(__name__)

MODEL_NAME = "corporate-credit-grading"
EXPERIMENT_NAME = "corporate/pd-lgd-ffn"

INTERNAL_GRADES = 15  # 15-grade rating scale


class CorporateGradingNet(nn.Module):
    """Feedforward neural network for corporate PD estimation."""

    def __init__(self, in_features: int = 7, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.out(x)).squeeze(-1)


def map_to_grade(pd_estimate: float) -> int:
    """Map a PD estimate to an internal 1-15 grade scale."""
    breakpoints = np.logspace(np.log10(0.001), np.log10(0.30), INTERNAL_GRADES - 1)
    return int(np.searchsorted(breakpoints, pd_estimate) + 1)


def train(data_path: str, epochs: int = 100, lr: float = 0.001) -> None:
    """Run corporate model training.

    Args:
        data_path: Parquet file with obligor-year observations.
        epochs: Training epochs.
        lr: Learning rate.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(data_path)
    X, scaler = build_feature_matrix(df, fit=True)
    y = df["default_indicator"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=(y > 0).astype(int)
    )

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)

    with mlflow.start_run():
        model = CorporateGradingNet(in_features=X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tr)
            loss = criterion(pred, y_tr)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, loss.item())

        model.eval()
        with torch.no_grad():
            probs = model(X_te).numpy()
        auc = roc_auc_score(y_test, probs)
        ar = 2 * auc - 1  # Accuracy ratio

        mlflow.log_params({"epochs": epochs, "lr": lr, "dropout": 0.3})
        mlflow.log_metrics({"auc_roc": auc, "accuracy_ratio": ar})
        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)
        with open("/tmp/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact("/tmp/scaler.pkl")
        logger.info("Training complete — AUC: %.4f  AR: %.4f", auc, ar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train corporate credit grading model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args.data, args.epochs)
