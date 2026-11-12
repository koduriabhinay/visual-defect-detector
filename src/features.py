"""Preprocessing and HOG feature extraction.

HOG is implemented in NumPy here (gradients -> cell orientation histograms ->
L2-normalized 2x2-cell blocks) rather than via cv2's HOG API, so the pipeline
runs on any OpenCV build.
"""

import cv2
import numpy as np

IMG_SIZE = 128
CELL = 16
NBINS = 9
N_GLOBAL = 4  # std, dark-pixel fraction, Laplacian variance, mean intensity
HOG_DIM = 1764  # 7x7 blocks * (2x2 cells * 9 orientation bins)
FEATURE_DIM = HOG_DIM + N_GLOBAL  # HOG + global texture features


def preprocess(img: np.ndarray) -> np.ndarray:
    """Normalize a grayscale image for feature extraction."""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.equalizeHist(img)
    return img


def hog_features(img: np.ndarray) -> np.ndarray:
    """Compute the HOG descriptor for a preprocessed 128x128 grayscale image."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0  # unsigned 0-180

    n_cells = IMG_SIZE // CELL
    bin_width = 180.0 / NBINS
    hist = np.zeros((n_cells, n_cells, NBINS), dtype=np.float32)
    for i in range(n_cells):
        for j in range(n_cells):
            m = magnitude[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL].ravel()
            a = angle[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL].ravel()
            scaled = a / bin_width
            b0 = np.floor(scaled).astype(int) % NBINS
            frac = scaled - np.floor(scaled)
            np.add.at(hist[i, j], b0, m * (1.0 - frac))
            np.add.at(hist[i, j], (b0 + 1) % NBINS, m * frac)

    blocks = []
    for i in range(n_cells - 1):
        for j in range(n_cells - 1):
            block = hist[i:i + 2, j:j + 2].ravel()
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    feats = np.concatenate(blocks).astype(np.float32)
    assert feats.shape[0] == HOG_DIM, f"unexpected HOG dim {feats.shape}"
    return feats


def extract_features(img: np.ndarray) -> np.ndarray:
    """Preprocess + HOG + global texture features for one grayscale image."""
    proc = preprocess(img)
    hog = hog_features(proc)
    glob = global_texture_features(proc)
    feats = np.concatenate([hog, glob]).astype(np.float32)
    assert feats.shape[0] == FEATURE_DIM, f"unexpected dim {feats.shape}"
    return feats


def global_texture_features(img: np.ndarray) -> np.ndarray:
    """Whole-image texture statistics that catch localized dark anomalies.

    Defects in this task appear as darker-than-background regions, so the
    fraction of dark pixels, intensity spread, and edge energy are strong,
    legitimate signals computed from the image alone (no label leakage).
    """
    g = img.astype(np.float32)
    lap_var = float(cv2.Laplacian(g, cv2.CV_32F).var())
    return np.array(
        [float(g.std()), float((g < 110).mean()), lap_var, float(g.mean())],
        dtype=np.float32,
    )
