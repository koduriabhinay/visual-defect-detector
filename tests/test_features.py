"""Tests for preprocessing and HOG feature extraction."""

import numpy as np

from src.data_gen import make_image
from src.features import FEATURE_DIM, extract_features, preprocess


def test_feature_vector_shape():
    rng = np.random.default_rng(3)
    img = make_image(rng, defective=True)
    feats = extract_features(img)
    assert feats.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(feats))


def test_preprocess_output():
    rng = np.random.default_rng(4)
    img = make_image(rng, defective=False)
    proc = preprocess(img)
    assert proc.shape == (128, 128)
    assert proc.dtype == np.uint8


def test_features_differ_between_classes():
    rng = np.random.default_rng(5)
    f_good = extract_features(make_image(rng, defective=False))
    f_bad = extract_features(make_image(rng, defective=True))
    assert not np.allclose(f_good, f_bad)
