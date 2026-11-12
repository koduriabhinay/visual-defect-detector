"""Classifier training and evaluation.

Model choice: RandomForestClassifier, selected after comparing RBF-SVM,
logistic regression, and random forest with 3-fold cross-validation on the
training split (see scripts/compare_models.py). Random forests need no
feature scaling and give calibrated-enough probabilities via predict_proba.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def build_model(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def train(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    model = build_model(random_state=random_state)
    model.fit(X, y)
    return model


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute classification metrics on a held-out set."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "n_samples": int(len(y)),
        "y_true": [int(v) for v in y],
        "y_pred": [int(v) for v in y_pred],
    }
