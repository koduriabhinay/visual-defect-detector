"""Single-image defect inference."""

import cv2
import joblib
import numpy as np

from .features import extract_features

LABELS = {0: "non-defective", 1: "defective"}


def load_model(path: str):
    return joblib.load(path)


def predict_image(model, image_path: str) -> dict:
    """Classify one image file; returns label, confidence, and probabilities."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    feats = extract_features(img).reshape(1, -1)
    proba = model.predict_proba(feats)[0]
    label_id = int(np.argmax(proba))
    return {
        "label": LABELS[label_id],
        "label_id": label_id,
        "confidence": float(proba[label_id]),
        "probabilities": {
            LABELS[i]: float(proba[i]) for i in range(len(proba))
        },
    }
