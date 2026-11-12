"""Synthetic visual-defect dataset generator.

Generates grayscale images of manufactured metal-like plates, labelled
``good`` (0) or ``defective`` (1). Defects are procedurally rendered
scratches, cracks, and dents. Generation is deterministic for a fixed seed.
"""

import csv
import os

import cv2
import numpy as np

IMG_SIZE = 128


def _base_plate(rng: np.random.Generator, size: int = IMG_SIZE) -> np.ndarray:
    """Render a clean brushed-metal-like plate."""
    img = np.full((size, size), 150, dtype=np.float32)
    # Horizontal brushing streaks.
    for _ in range(40):
        y = int(rng.integers(0, size))
        img[y, :] += float(rng.normal(0, 6))
    img += rng.normal(0, 4, (size, size))
    # Plate edge.
    cv2.rectangle(img, (4, 4), (size - 5, size - 5), 135, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_scratch(img: np.ndarray, rng: np.random.Generator) -> None:
    x0, y0 = (int(v) for v in rng.integers(10, IMG_SIZE - 10, 2))
    angle = float(rng.uniform(0, np.pi))
    length = int(rng.integers(25, 70))
    x1 = int(x0 + length * np.cos(angle))
    y1 = int(y0 + length * np.sin(angle))
    cv2.line(img, (x0, y0), (x1, y1),
             int(rng.integers(40, 80)), thickness=int(rng.integers(1, 3)))


def _add_crack(img: np.ndarray, rng: np.random.Generator) -> None:
    x, y = (int(v) for v in rng.integers(20, IMG_SIZE - 20, 2))
    pts = []
    for _ in range(int(rng.integers(5, 9))):
        pts.append([x, y])
        x += int(rng.integers(-14, 15))
        y += int(rng.integers(-14, 15))
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False,
                  int(rng.integers(35, 70)), thickness=2)


def _add_dent(img: np.ndarray, rng: np.random.Generator) -> None:
    cx, cy = (int(v) for v in rng.integers(20, IMG_SIZE - 20, 2))
    ax, ay = (int(v) for v in rng.integers(6, 18, 2))
    angle = float(rng.uniform(0, 180))
    cv2.ellipse(img, (cx, cy), (ax, ay), angle, 0, 360,
                int(rng.integers(50, 90)), -1)
    cv2.ellipse(img, (cx, cy), (ax, ay), angle, 0, 360, 170, 1)


DEFECT_RENDERERS = [_add_scratch, _add_crack, _add_dent]


def make_image(rng: np.random.Generator, defective: bool) -> np.ndarray:
    """Render one plate image, with or without a defect."""
    img = _base_plate(rng)
    if defective:
        rng.choice(DEFECT_RENDERERS)(img, rng)
        if float(rng.random()) < 0.25:  # occasionally a second defect
            rng.choice(DEFECT_RENDERERS)(img, rng)
    return img


def generate_dataset(out_dir: str, n_per_class: int = 600, seed: int = 42):
    """Generate the dataset on disk and return ``[(rel_path, label)]`` rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_per_class):
        for label, name in ((0, "good"), (1, "defective")):
            img = make_image(rng, defective=bool(label))
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
            fname = f"{name}_{i:04d}.png"
            cv2.imwrite(os.path.join(out_dir, name, fname), img)
            rows.append((os.path.join(name, fname), label))
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)
    return rows
