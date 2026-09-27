"""One scoring path, shared by training, the API and the demo.

Keeping this in a single place matters because raw scores and calibrated
probabilities are used for two different jobs and must not be mixed up:

* the **raw model score** carries the ranking, and the band thresholds are set
  on it against a false-positive budget;
* the **calibrated probability** is what a customer is told, and is only
  meaningful as a probability.

Banding a calibrated probability against a raw-space threshold silently
mis-bands everything, and a Platt calibrator's ``.predict`` returns class
labels rather than probabilities - both easy mistakes to make twice.
"""
from __future__ import annotations

import numpy as np


def predict(booster, calibrator, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_score, calibrated_probability)."""
    raw = np.asarray(booster.predict(X), dtype=float)
    if calibrator is None:
        return raw, raw
    if hasattr(calibrator, "predict_proba"):
        prob = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    else:  # isotonic and friends
        prob = np.asarray(calibrator.predict(raw), dtype=float)
    return raw, np.clip(prob, 0.0, 1.0)


def band(raw_score: float, thresholds: dict) -> str:
    """Bands are decided on the raw ranking, per the FPR budget."""
    if raw_score >= thresholds["high_risk"]:
        return "high_risk"
    if raw_score >= thresholds["caution"]:
        return "caution"
    return "safe"
