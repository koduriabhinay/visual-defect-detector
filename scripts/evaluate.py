"""Pretty-print the metrics produced by scripts/train.py."""

import json
import os
import sys

METRICS = os.path.join(os.path.dirname(__file__), "..", "metrics.json")


def main() -> None:
    with open(METRICS) as f:
        m = json.load(f)
    cm = m["confusion_matrix"]
    print("Visual Defect Detector — held-out test metrics")
    print("=" * 48)
    print(f"test samples : {m['n_samples']} (train: {m.get('train_size')}, seed: {m.get('seed')})")
    print(f"accuracy     : {m['accuracy']:.4f}")
    print(f"precision    : {m['precision']:.4f}")
    print(f"recall       : {m['recall']:.4f}")
    print(f"F1           : {m['f1']:.4f}")
    print(f"ROC-AUC      : {m['roc_auc']:.4f}")
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"  {cm[0]}")
    print(f"  {cm[1]}")


if __name__ == "__main__":
    main()
