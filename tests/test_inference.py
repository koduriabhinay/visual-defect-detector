"""Tests for single-image inference."""

import cv2
import numpy as np

from src.data_gen import make_image
from src.features import extract_features
from src.inference import load_model, predict_image
from src.model import train


def _trained_model_path(tmp_path):
    rng = np.random.default_rng(21)
    X, y = [], []
    for _ in range(25):
        X.append(extract_features(make_image(rng, defective=False)))
        y.append(0)
        X.append(extract_features(make_image(rng, defective=True)))
        y.append(1)
    model = train(np.stack(X), np.array(y))
    import joblib

    path = str(tmp_path / "model.joblib")
    joblib.dump(model, path)
    return path


def test_predict_image_returns_valid_result(tmp_path):
    model_path = _trained_model_path(tmp_path)
    model = load_model(model_path)
    rng = np.random.default_rng(22)
    img_path = str(tmp_path / "sample.png")
    cv2.imwrite(img_path, make_image(rng, defective=True))
    out = predict_image(model, img_path)
    assert out["label"] in ("non-defective", "defective")
    assert out["label_id"] in (0, 1)
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6


def test_predict_missing_file_raises(tmp_path):
    import joblib
    from src.model import build_model

    path = str(tmp_path / "m.joblib")
    joblib.dump(build_model(), path)
    try:
        predict_image(load_model(path), str(tmp_path / "nope.png"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
