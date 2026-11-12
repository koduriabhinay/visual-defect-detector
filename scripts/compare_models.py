"""Reproducible model comparison (3-fold CV F1 on the training split).

Run from the repo root after generating data:
    python scripts/train.py --n-per-class 600   # generates data/ first
    python scripts/compare_models.py
"""

import csv
import os
import sys

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import extract_features


def load_features(data_dir="data"):
    X, y = [], []
    with open(os.path.join(data_dir, "labels.csv")) as f:
        for row in csv.DictReader(f):
            img = cv2.imread(os.path.join(data_dir, row["path"]), cv2.IMREAD_GRAYSCALE)
            X.append(extract_features(img))
            y.append(int(row["label"]))
    return np.stack(X), np.array(y)


CANDIDATES = {
    "rbf-svm": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=2.0, class_weight="balanced", random_state=42),
    ),
    "logistic-regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    ),
    "random-forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}


def main() -> None:
    X, y = load_features()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    print(f"{'model':<20}{'mean F1':>10}   fold F1s")
    for name, model in CANDIDATES.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        print(f"{name:<20}{scores.mean():>10.4f}   {np.round(scores, 4)}")


if __name__ == "__main__":
    main()
