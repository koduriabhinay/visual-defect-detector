"""Tests for synthetic data generation."""

import numpy as np

from src.data_gen import generate_dataset, make_image


def test_deterministic_for_fixed_seed():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    a = make_image(rng1, defective=True)
    b = make_image(rng2, defective=True)
    assert np.array_equal(a, b)


def test_image_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = make_image(rng, defective=False)
    assert img.shape == (128, 128)
    assert img.dtype == np.uint8


def test_generate_dataset_layout(tmp_path):
    rows = generate_dataset(str(tmp_path), n_per_class=10, seed=123)
    assert len(rows) == 20
    labels = [label for _, label in rows]
    assert labels.count(0) == 10 and labels.count(1) == 10
    assert (tmp_path / "labels.csv").exists()
    assert (tmp_path / "good").is_dir()
    assert (tmp_path / "defective").is_dir()


def test_defective_differs_from_good():
    rng = np.random.default_rng(99)
    good = make_image(rng, defective=False).astype(float)
    bad = make_image(rng, defective=True).astype(float)
    # A defect must change the image noticeably.
    assert np.abs(good - bad).mean() > 0.5
