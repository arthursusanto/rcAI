"""The three-stage model interface shared by every baseline and learned model.

The class is still named ``TwoStageModel``: it began as detect + localize, and the
name is baked into every model pickled by ``save``. Renaming it would break loading
existing artifacts for no behavioural gain, so the name is historical and the stage
count below is authoritative.

Three stages over the window table (``rca.features.schema``):

1. **detect** -- one calibrated probability per (experiment, window), scored from the
   system-level aggregate (``rca.models.aggregate``) so it never sees which service is
   failing.
2. **localize** -- a binary "is this row the root" score for every service row, trained
   on fault windows only, turned into a ranking within each window. The stage sees only
   the row's own features and its graph-relative features; a service-identity column
   would make held-out-service evaluation meaningless, so ``fit`` refuses one.
3. **classify** -- the fault family, trained on true-root rows and applied at inference
   to the row of the predicted root service.

Subclasses implement six hooks (``fit_detect``/``score_detect`` and friends) returning
*raw* scores; calibration, threshold selection and the assembly of the prediction frame
live here so every model is compared on the same footing.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rca.features.schema import feature_columns
from rca.models.aggregate import KEYS, aggregate_feature_columns, aggregate_windows
from rca.models.augment import DEFAULT_DROPOUT, augment_training, dropout_spec
from rca.models.calibration import (
    apply_temperature,
    fit_binary_calibrator,
    fit_temperature,
)

PREDICTION_COLUMNS = [
    "experiment_id", "window_idx", "window_start_ns", "window_end_ns",
    "detect_prob", "detect",
    "ranked_services", "root_scores", "root_top1", "root_margin",
    "fault_type_pred", "fault_type_probs", "fault_type_conf",
]

# Structural, not identity: how far the service sits from the entry point. Every other
# feature must vary within a service, or it is a disguised service one-hot.
STRUCTURAL_FEATURES = {"f_graph_depth_from_entry"}


def sustained(experiment_ids, flags, windows_required: int) -> np.ndarray:
    """``flags`` filtered to runs of at least ``windows_required`` consecutive positives.

    Rows must already be in (experiment, window) order. Used both by a model whose own
    alarm rule is burn-rate style and by the evaluation's ``--debounce``.
    """
    flags = pd.Series(np.asarray(flags, dtype=bool))
    if windows_required <= 1:
        return flags.to_numpy(bool)
    group = pd.Series(np.asarray(experiment_ids)).to_numpy()
    keep = flags.to_numpy(bool)
    for lag in range(1, windows_required):
        keep = keep & flags.groupby(group).shift(lag, fill_value=False).to_numpy(bool)
    return keep


class TwoStageModel:
    """Detect -> localize -> classify, with calibrated confidence for each stage."""

    name = "base"
    # Consecutive positive windows the model itself requires before it calls an alarm.
    # 1 for every learned model; the rule baseline uses a burn-rate style 2.
    detect_sustain = 1

    def __init__(
        self,
        target_false_alarm_rate: float | None = None,
        modality_dropout: float | tuple[float, list[str]] = DEFAULT_DROPOUT,
        params: dict[str, dict] | None = None,
    ):
        # When set, the detection threshold targets this false-positive rate on the
        # validation fold instead of maximising F1.
        self.target_false_alarm_rate = target_false_alarm_rate
        # Fraction of training windows re-added with degraded telemetry, either as a
        # bare ``p`` or as ``(p, modalities)``. Ignored by the untrained baselines.
        self.modality_dropout = modality_dropout
        # ``{stage: {hyperparameter: value}}`` overriding the estimator defaults, as
        # written by ``rca tune``. Ignored by the untrained baselines.
        self.params = params or {}

    def stage_params(self, stage: str) -> dict:
        """Hyperparameter overrides for one stage; ``{}`` means "use the defaults"."""
        return dict(self.params.get(stage, {}))

    # --- stage hooks ----------------------------------------------------------------
    def fit_detect(self, aggregate: pd.DataFrame, y: np.ndarray) -> None:
        """Train stage 1 on the aggregated training windows."""

    def score_detect(self, aggregate: pd.DataFrame, windows: pd.DataFrame) -> np.ndarray:
        """Raw, uncalibrated detection score per aggregated window (higher = faultier).

        ``windows`` is the matching per-service table; the rule baseline needs raw
        per-service features that the aggregate does not carry.
        """
        raise NotImplementedError

    def fit_root(self, rows: pd.DataFrame, y: np.ndarray) -> None:
        """Train stage 2 on the service rows of fault windows."""

    def score_root(self, rows: pd.DataFrame) -> np.ndarray:
        """Raw "this row is the root" score per service row."""
        raise NotImplementedError

    def fit_fault(self, rows: pd.DataFrame, y: np.ndarray) -> None:
        """Train stage 3 on true-root rows; ``self.classes_`` is already set."""

    def score_fault(self, rows: pd.DataFrame) -> np.ndarray:
        """Uncalibrated class probabilities, shape (len(rows), len(self.classes_))."""
        raise NotImplementedError

    # --- fitting --------------------------------------------------------------------
    def fit(self, windows: pd.DataFrame, val: pd.DataFrame) -> TwoStageModel:
        """Fit the three stages on ``windows`` and calibrate them on ``val``.

        Every stage trains on ``augment_training(windows)``; calibration and the
        detection threshold always come from the untouched validation fold, so the
        reported probabilities still describe clean telemetry.
        """
        self.features_ = feature_columns(windows.columns)
        assert_no_service_identity(windows, self.features_)
        clean_rows = len(windows)
        windows = self.augment_training(windows)
        self.training_rows_ = (clean_rows, len(windows))

        train_aggregate = aggregate_windows(windows)
        val_aggregate = aggregate_windows(val)
        self.agg_features_ = aggregate_feature_columns(train_aggregate)

        self.fit_detect(train_aggregate, train_aggregate["is_fault_window"].to_numpy(bool))
        raw = self.score_detect(val_aggregate, val)
        y = val_aggregate["is_fault_window"].to_numpy(bool)
        self.detect_calibrator_ = fit_binary_calibrator(raw, y)
        probability = self.detect_calibrator_.transform(raw)
        self.detect_threshold_ = self.choose_threshold(y, probability, raw)
        above = raw[probability >= self.detect_threshold_]
        self.detect_threshold_raw_ = float(above.min()) if above.size else float("inf")

        train_fault = windows[windows["is_fault_window"]]
        self.fit_root(train_fault, train_fault["is_root"].to_numpy(bool))
        val_fault = val[val["is_fault_window"]]
        self.root_calibrator_ = fit_binary_calibrator(
            self.score_root(val_fault), val_fault["is_root"].to_numpy(bool)
        )

        train_root = windows[windows["is_root"]]
        self.classes_ = sorted(str(v) for v in train_root["fault_type"].unique()) or ["none"]
        self.fit_fault(train_root, train_root["fault_type"].astype(str).to_numpy())
        val_root = val[val["is_root"]]
        self.temperature_ = fit_temperature(
            self.score_fault(val_root), val_root["fault_type"].astype(str), self.classes_
        )
        return self

    def augment_training(self, windows: pd.DataFrame) -> pd.DataFrame:
        """Training table for the three stages; untrained baselines take it as it is."""
        return windows

    def choose_threshold(self, y: np.ndarray, probability: np.ndarray, raw: np.ndarray) -> float:
        if self.target_false_alarm_rate is not None:
            return threshold_for_target_fpr(y, probability, self.target_false_alarm_rate)
        return best_f1_threshold(y, probability)

    # --- inference ------------------------------------------------------------------
    def detect_probabilities(self, windows: pd.DataFrame) -> pd.DataFrame:
        """Aggregated windows with a calibrated ``detect_prob`` column."""
        aggregate = aggregate_windows(windows)
        aggregate["detect_prob"] = self.detect_calibrator_.transform(
            self.score_detect(aggregate, windows)
        )
        return aggregate

    def root_probabilities(self, rows: pd.DataFrame) -> np.ndarray:
        """Calibrated P(this row is the root) for each service row."""
        return self.root_calibrator_.transform(self.score_root(rows))

    def classify(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Calibrated fault-family probabilities for arbitrary service rows."""
        probabilities = apply_temperature(self.score_fault(rows), self.temperature_)
        return pd.DataFrame(probabilities, columns=self.classes_, index=rows.index)

    def predict(self, windows: pd.DataFrame) -> pd.DataFrame:
        """One row per (experiment_id, window_idx); see ``PREDICTION_COLUMNS``."""
        aggregate = self.detect_probabilities(windows)
        ranking = pd.DataFrame({
            "experiment_id": windows["experiment_id"].to_numpy(),
            "window_idx": windows["window_idx"].to_numpy(),
            "service": windows["service"].astype(str).to_numpy(),
            "score": self.root_probabilities(windows),
            "position": np.arange(len(windows)),
        }).sort_values(
            ["experiment_id", "window_idx", "score", "service"],
            ascending=[True, True, False, True], kind="mergesort",
        )
        grouped = ranking.groupby(KEYS, sort=True)
        order = pd.MultiIndex.from_arrays(
            [aggregate["experiment_id"], aggregate["window_idx"]], names=KEYS
        )
        ranked = grouped["service"].apply(list).reindex(order)
        scores = grouped["score"].apply(list).reindex(order)
        top = grouped["position"].first().reindex(order).to_numpy()

        classes = self.classify(windows.iloc[top]).to_numpy()
        best = classes.argmax(axis=1)

        return pd.DataFrame({
            "experiment_id": aggregate["experiment_id"].to_numpy(),
            "window_idx": aggregate["window_idx"].to_numpy(),
            "window_start_ns": aggregate["window_start_ns"].to_numpy(),
            "window_end_ns": aggregate["window_end_ns"].to_numpy(),
            "detect_prob": aggregate["detect_prob"].to_numpy(),
            "detect": sustained(
                aggregate["experiment_id"].to_numpy(),
                aggregate["detect_prob"].to_numpy() >= self.detect_threshold_,
                self.detect_sustain,
            ),
            "ranked_services": ranked.to_numpy(),
            "root_scores": scores.to_numpy(),
            "root_top1": [names[0] for names in ranked],
            "root_margin": [
                values[0] - values[1] if len(values) > 1 else values[0] for values in scores
            ],
            "fault_type_pred": [self.classes_[i] for i in best],
            "fault_type_probs": [dict(zip(self.classes_, row)) for row in classes],
            "fault_type_conf": classes.max(axis=1),
        }, columns=PREDICTION_COLUMNS)

    # --- persistence ----------------------------------------------------------------
    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "model.joblib")
        return directory

    @staticmethod
    def load(directory: Path | str) -> TwoStageModel:
        return joblib.load(Path(directory) / "model.joblib")


class LearnedTwoStage(TwoStageModel):
    """Skeleton for the learned models: one estimator per stage, built by a hook."""

    def make_estimator(self, stage: str, y: np.ndarray):
        """Return an unfitted sklearn-style estimator for ``detect``/``root``/``fault``."""
        raise NotImplementedError

    def augment_training(self, windows):
        """Add degraded copies so a missing or late modality is not out of distribution."""
        fraction, modalities = dropout_spec(self.modality_dropout)
        return augment_training(windows, fraction, modalities)

    def fit_detect(self, aggregate, y):
        self.detect_model_ = self.make_estimator("detect", y)
        self.detect_model_.fit(aggregate[self.agg_features_], y.astype(int))

    def score_detect(self, aggregate, windows):
        return self.detect_model_.predict_proba(aggregate[self.agg_features_])[:, 1]

    def fit_root(self, rows, y):
        self.root_model_ = self.make_estimator("root", y)
        self.root_model_.fit(rows[self.features_], y.astype(int))

    def score_root(self, rows):
        if rows.empty:
            return np.zeros(0)
        return self.root_model_.predict_proba(rows[self.features_])[:, 1]

    def fit_fault(self, rows, y):
        encoded = np.array([self.classes_.index(value) for value in y])
        if len(self.classes_) < 2:
            self.fault_model_ = None
            return
        self.fault_model_ = self.make_estimator("fault", encoded)
        self.fault_model_.fit(rows[self.features_], encoded)

    def score_fault(self, rows):
        width = max(len(self.classes_), 1)
        if self.fault_model_ is None or rows.empty:
            return np.full((len(rows), width), 1.0 / width)
        return self.fault_model_.predict_proba(rows[self.features_])


# --- helpers -------------------------------------------------------------------------
def assert_no_service_identity(windows: pd.DataFrame, features: list[str]) -> None:
    """Raise if a feature column encodes *which* service the row is.

    A service one-hot is a binary column that is constant inside every service and set
    for some but not all of them. ``f_graph_depth_from_entry`` is constant per service
    too but describes the topology, not the identity, so it is allowed through.
    """
    services = windows["service"]
    for name in features:
        if name in STRUCTURAL_FEATURES:
            continue
        column = windows[name]
        observed = column.dropna()
        # An identity encoding is defined on every row; a sparse 0/1 fraction that is
        # constant per service only because it is rarely observed is not one.
        if len(observed) < len(column):
            continue
        if observed.empty or not set(np.unique(observed.to_numpy())) <= {0.0, 1.0}:
            continue
        per_service = column.groupby(services, observed=True).agg(["nunique", "max"])
        if (per_service["nunique"] > 1).any():
            continue
        hot = int((per_service["max"] == 1).sum())
        if 0 < hot < len(per_service):
            raise ValueError(
                f"feature {name!r} is a service-identity one-hot; stage 2 must stay "
                "service-agnostic for held-out-service evaluation to mean anything"
            )


def best_f1_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Lowest threshold maximising F1 of ``probability >= threshold`` on this fold."""
    y = np.asarray(y).astype(bool)
    if not y.any() or y.size == 0:
        return float("inf")
    best, best_f1 = float("inf"), -1.0
    for threshold in _candidates(probability):
        predicted = probability >= threshold
        true_positive = int((predicted & y).sum())
        if true_positive == 0:
            continue
        precision = true_positive / int(predicted.sum())
        recall = true_positive / int(y.sum())
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return best


def threshold_for_target_fpr(y: np.ndarray, probability: np.ndarray, target: float) -> float:
    """Lowest threshold whose false-positive rate on this fold stays within ``target``."""
    y = np.asarray(y).astype(bool)
    negatives = probability[~y]
    if negatives.size == 0:
        return best_f1_threshold(y, probability)
    for threshold in sorted(_candidates(probability)):
        if float((negatives >= threshold).mean()) <= target:
            return float(threshold)
    return float("inf")


def _candidates(probability: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(probability, dtype=float))
