"""End-to-end training: generate data -> HOG features -> train -> evaluate.

Saves the trained model to models/model.joblib and metrics to metrics.json.
"""

import argparse
import json
import os
import sys

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_gen import generate_dataset
from src.features import extract_features
from src.model import evaluate, train


def load_features(data_dir: str):
    import csv

    X, y = [], []
    with open(os.path.join(data_dir, "labels.csv")) as f:
        for row in csv.DictReader(f):
            img = cv2.imread(os.path.join(data_dir, row["path"]), cv2.IMREAD_GRAYSCALE)
            X.append(extract_features(img))
            y.append(int(row["label"]))
    return np.stack(X), np.array(y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the visual defect detector.")
    parser.add_argument("--n-per-class", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--metrics-out", default="metrics.json")
    args = parser.parse_args()

    print(f"[1/4] generating {2 * args.n_per_class} images (seed={args.seed}) ...")
    generate_dataset(args.data_dir, n_per_class=args.n_per_class, seed=args.seed)

    print("[2/4] extracting HOG features ...")
    X, y = load_features(args.data_dir)
    print(f"      feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print("[3/4] training random forest (selected via scripts/compare_models.py) ...")
    model = train(X_train, y_train, random_state=args.seed)

    print("[4/4] evaluating on held-out test set ...")
    metrics = evaluate(model, X_test, y_test)
    metrics["train_size"] = int(len(y_train))
    metrics["seed"] = args.seed

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    joblib.dump(model, args.model_out)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nmodel saved to {args.model_out}")
    print(f"metrics saved to {args.metrics_out}")
    print(
        f"\nTest accuracy: {metrics['accuracy']:.4f} | precision: {metrics['precision']:.4f} "
        f"| recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}"
    )
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
