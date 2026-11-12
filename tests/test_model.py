"""Tests for model training and evaluation."""

import numpy as np

from src.data_gen import make_image
from src.features import extract_features
from src.model import evaluate, train


def _tiny_xy(n_per_class=30, seed=11):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per_class):
        X.append(extract_features(make_image(rng, defective=False)))
        y.append(0)
        X.append(extract_features(make_image(rng, defective=True)))
        y.append(1)
    return np.stack(X), np.array(y)


def test_train_and_predict():
    X, y = _tiny_xy()
    model = train(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    # Must do clearly better than chance on its own training data.
    assert (preds == y).mean() > 0.8


def test_evaluate_metrics_shape():
    X, y = _tiny_xy()
    model = train(X, y)
    m = evaluate(model, X, y)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= m[key] <= 1.0, key
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y)
    assert m["n_samples"] == len(y)
