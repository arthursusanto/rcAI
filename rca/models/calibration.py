"""Probability calibration fitted on the validation fold, and calibration error.

Stages 1 and 2 are binary, so their raw scores go through isotonic regression when the
validation fold carries enough positives and through Platt scaling (a logistic fit on the
raw score) otherwise -- isotonic overfits badly on a few dozen positives. Stage 3 is
multiclass and uses temperature scaling over the class probability vector.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

MIN_ISOTONIC_POSITIVES = 50
EPS = 1e-12


class BinaryCalibrator:
    """Monotone map from a raw stage score to a calibrated probability."""

    def __init__(self, method: str, model=None, constant: float = 0.5, nan_fill: float = 0.0):
        self.method = method
        self.model = model
        self.constant = constant
        self.nan_fill = nan_fill

    def transform(self, scores) -> np.ndarray:
        clean = _finite(scores, self.nan_fill)
        if self.method == "constant":
            return np.full(clean.shape, self.constant)
        if self.method == "isotonic":
            return np.clip(self.model.predict(clean), 0.0, 1.0)
        return self.model.predict_proba(clean.reshape(-1, 1))[:, 1]


def fit_binary_calibrator(
    scores, labels, min_positives: int = MIN_ISOTONIC_POSITIVES
) -> BinaryCalibrator:
    """Isotonic when the fold has >= ``min_positives`` positives, Platt otherwise."""
    labels = np.asarray(labels).astype(bool)
    nan_fill = _nan_fill(scores)
    clean = _finite(scores, nan_fill)
    base = float(labels.mean()) if labels.size else 0.5
    if labels.size == 0 or labels.all() or not labels.any() or np.ptp(clean) == 0.0:
        return BinaryCalibrator("constant", constant=base, nan_fill=nan_fill)
    if int(labels.sum()) >= min_positives:
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(clean, labels.astype(float))
        return BinaryCalibrator("isotonic", model=model, nan_fill=nan_fill)
    model = LogisticRegression(max_iter=1000)
    model.fit(clean.reshape(-1, 1), labels.astype(int))
    return BinaryCalibrator("platt", model=model, nan_fill=nan_fill)


def fit_temperature(probs, labels, classes, grid=None) -> float:
    """Temperature minimising the validation NLL of ``probs ** (1 / T)`` renormalised."""
    probs = np.asarray(probs, dtype=float)
    index = {name: position for position, name in enumerate(classes)}
    target = np.array([index.get(str(value), -1) for value in labels])
    keep = target >= 0
    if probs.size == 0 or not keep.any():
        return 1.0
    probs, target = probs[keep], target[keep]
    grid = np.geomspace(0.05, 20.0, 200) if grid is None else np.asarray(grid, dtype=float)
    losses = [
        -np.log(np.clip(apply_temperature(probs, t)[np.arange(len(target)), target], EPS, 1.0)).mean()
        for t in grid
    ]
    return float(grid[int(np.argmin(losses))])


def apply_temperature(probs, temperature: float) -> np.ndarray:
    """Sharpen (T < 1) or soften (T > 1) a probability vector, then renormalise."""
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return probs
    scaled = np.clip(probs, EPS, 1.0) ** (1.0 / max(temperature, EPS))
    return scaled / scaled.sum(axis=1, keepdims=True)


def reliability_bins(confidence, correct, n_bins: int = 10) -> list[dict]:
    """Equal-width bins over [0, 1] with their mean confidence and empirical accuracy."""
    confidence = np.clip(np.asarray(confidence, dtype=float), 0.0, 1.0)
    correct = np.asarray(correct).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize puts exact 1.0 in an extra bin; fold it back into the last one.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        bins.append({
            "low": float(edges[b]),
            "high": float(edges[b + 1]),
            "count": int(mask.sum()),
            "confidence": float(confidence[mask].mean()) if mask.any() else float("nan"),
            "accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
        })
    return bins


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """Sum over bins of |accuracy - confidence| weighted by bin occupancy."""
    total = len(np.asarray(confidence))
    if total == 0:
        return float("nan")
    error = 0.0
    for b in reliability_bins(confidence, correct, n_bins):
        if b["count"]:
            error += b["count"] / total * abs(b["accuracy"] - b["confidence"])
    return float(error)


def _finite(scores, fill: float) -> np.ndarray:
    """NaN -> ``fill`` (the lowest score seen while fitting); infinities -> float bounds."""
    values = np.asarray(scores, dtype=float)
    if np.isfinite(values).all():
        return values
    return np.nan_to_num(values, nan=fill)


def _nan_fill(scores) -> float:
    values = np.asarray(scores, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.min()) if finite.size else 0.0
