"""The five models compared in the evaluation.

``rules`` and ``stat`` are the untrained references from ``docs/MODELS.md``; ``logreg``,
``rf`` and ``xgb`` plug an estimator into the shared three-stage skeleton. All five go
through the same calibration and threshold selection on the validation fold, so their
probabilities and their ``detect`` flags mean the same thing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from rca.features.schema import FEATURE_PREFIX
from rca.models.aggregate import AGGREGATE_PREFIX, KEYS
from rca.models.base import LearnedTwoStage, TwoStageModel

# Stage-1/2 ``scale_pos_weight`` standing for "the class ratio of this training fold",
# which is what XGBModel computes for itself. Naming it makes it a value the tuner can
# put in a parameter file and compare against a fixed weight.
AUTO_POS_WEIGHT = "auto"

# --- rules ---------------------------------------------------------------------------
# feature column -> threshold. Four service-level rules any monitoring stack ships with,
# plus two dependency-side rules any service mesh gives you for free. Latency is compared
# as a *ratio* to the experiment's own warm-up rather than as a z-score: a z threshold
# fires on any tight baseline, which made this baseline a straw man.
RULES: dict[str, tuple[str, float]] = {
    "latency": ("f_traces_latency_ratio", 2.0),
    "error_rate": ("f_traces_error_rate", 0.05),
    "cpu": ("f_metrics_cpu_util_mean", 0.90),
    "memory": ("f_metrics_mem_frac_mean", 0.90),
    "inbound_errors": ("f_graph_inbound_client_error_rate", 0.05),
    "inbound_unanswered": ("f_graph_inbound_unanswered_frac", 0.20),
}
# Consecutive windows a rule must hold before the alarm is raised -- the burn-rate
# pattern real alerting stacks use to stop single-window noise.
RULE_SUSTAIN = 2
RULE_FAULT = {
    "latency": "network_latency",
    "error_rate": "error_rate",
    "cpu": "cpu_saturation",
    "memory": "memory_leak",
    "inbound_errors": "dependency_failure",
    "inbound_unanswered": "dependency_failure",
}

# --- signature baseline ----------------------------------------------------------------
# component -> (feature, the value at which it counts as fully anomalous). Each component
# is value/scale clipped to [0, 1], so their mean is a normalised anomaly score and every
# signal contributes on the same footing whatever its natural units.
SIGNATURE_COMPONENTS: dict[str, tuple[str, float]] = {
    "latency": ("f_traces_latency_z", 3.0),
    "inbound_latency": ("f_graph_inbound_client_latency_z_max", 3.0),
    "error_rate": ("f_traces_error_rate", 0.05),
    "cpu": ("f_metrics_cpu_util_z", 3.0),
    "memory": ("f_metrics_mem_frac_z", 3.0),
    "queue": ("f_metrics_queue_depth_z", 3.0),
    "log_errors": ("f_logs_error_z", 3.0),
    "unanswered": ("f_graph_inbound_unanswered_frac", 0.20),
}
# Largest component -> fault family. Log errors map to packet_loss rather than to
# error_rate because the trace-level error rate is the direct evidence for an error_rate
# fault, so elevated log errors *without* it is the retry/loss signature.
SIGNATURE_FAULT = {
    "latency": "network_latency",
    "inbound_latency": "cache_slowdown",
    "error_rate": "error_rate",
    "cpu": "cpu_saturation",
    "memory": "memory_leak",
    "queue": "queue_backlog",
    "log_errors": "packet_loss",
    "unanswered": "dependency_failure",
}
# Same z scale as the components, so the downstream discount sits on the [0, 1] footing
# the score is on instead of swamping it with a raw z.
DOWNSTREAM_SCALE = 3.0

# --- statistical detector ------------------------------------------------------------
STAT_Z = [
    "f_traces_latency_z", "f_logs_error_z", "f_metrics_cpu_util_z",
    "f_metrics_mem_frac_z", "f_metrics_queue_depth_z",
]
# Fault family -> the feature whose deviation is evidence for it, and a scale that puts
# rates on the same footing as z-scores.
STAT_EVIDENCE: dict[str, tuple[str, float]] = {
    "cpu_saturation": ("f_metrics_cpu_util_z", 1.0),
    "memory_leak": ("f_metrics_mem_frac_z", 1.0),
    "network_latency": ("f_traces_latency_z", 1.0),
    "queue_backlog": ("f_metrics_queue_depth_z", 1.0),
    "error_rate": ("f_logs_error_z", 1.0),
    "packet_loss": ("f_traces_error_rate", 20.0),
    "dependency_failure": ("f_traces_client_error_rate_max", 20.0),
    "cache_slowdown": ("f_traces_client_latency_z_max", 1.0),
}


class RulesModel(TwoStageModel):
    """Industry-style static thresholds, burn-rate alarmed. Nothing is trained."""

    name = "rules"
    detect_sustain = RULE_SUSTAIN

    def score_detect(self, aggregate, windows):
        counts = pd.DataFrame({
            "experiment_id": windows["experiment_id"].to_numpy(),
            "window_idx": windows["window_idx"].to_numpy(),
            "n": _fired(windows).sum(axis=1).to_numpy(),
        })
        return (counts.groupby(KEYS, sort=True)["n"].max() / len(RULES)).to_numpy(float)

    def choose_threshold(self, y, probability, raw):
        """Reproduce "detection = any rule fires" on the calibrated scale."""
        positive = raw > 0.0
        return float(probability[positive].min()) if positive.any() else float("inf")

    def score_root(self, rows):
        """Rules fired, tie-broken towards the most downstream service."""
        depth = _column(rows, "f_graph_depth_from_entry").fillna(0.0).clip(0.0, 9.0)
        return (_fired(rows).sum(axis=1) + 0.09 * depth).to_numpy(float)

    def score_fault(self, rows):
        """Rule -> family; a family with several rules takes the strongest evidence."""
        fired = _fired(rows)
        evidence = np.zeros((len(rows), len(self.classes_)))
        for position, name in enumerate(self.classes_):
            for rule, family in RULE_FAULT.items():
                if family == name and rule in fired:
                    evidence[:, position] = np.maximum(
                        evidence[:, position], fired[rule].to_numpy(float)
                    )
        return _softmax(evidence)


class StatModel(TwoStageModel):
    """Robust z-score detector; ``k`` is the validation-chosen detection threshold.

    The threshold is picked on the calibrated probability, which is a monotone map of the
    raw max-z score, so it *is* the F1-optimal ``k`` on the validation fold;
    ``detect_threshold_raw_`` reports it back in z units.
    """

    name = "stat"

    def score_detect(self, aggregate, windows):
        columns = [
            AGGREGATE_PREFIX + name[len(FEATURE_PREFIX):] + "_max" for name in STAT_Z
        ]
        return _max_over(aggregate, columns)

    def score_root(self, rows):
        """Largest deviation, minus what a failing downstream dependency explains."""
        own = _max_over(rows, STAT_Z)
        explained = _column(rows, "f_graph_downstream_explained").fillna(0.0).to_numpy(float)
        return np.nan_to_num(own, nan=0.0) - explained

    def score_fault(self, rows):
        evidence = np.zeros((len(rows), len(self.classes_)))
        for position, name in enumerate(self.classes_):
            column, scale = STAT_EVIDENCE.get(name, (None, 1.0))
            if column is not None:
                evidence[:, position] = (
                    _column(rows, column).fillna(0.0).to_numpy(float) * scale
                )
        return _softmax(np.clip(evidence, -10.0, 10.0))


class SignatureModel(TwoStageModel):
    """Normalised multi-signal anomaly score, untrained; the threshold comes from val.

    The score is what an engineer draws on a dashboard: put every signal on a 0-1 scale
    against the level at which it is clearly abnormal, then average them. It is a fairer
    reference than a single-signal detector, because a fault that moves three signals a
    little scores as high as one that moves a single signal a lot.
    """

    name = "signature"

    def score_detect(self, aggregate, windows):
        score = pd.DataFrame({
            "experiment_id": windows["experiment_id"].to_numpy(),
            "window_idx": windows["window_idx"].to_numpy(),
            "score": _signature_score(windows),
        })
        return score.groupby(KEYS, sort=True)["score"].max().to_numpy(float)

    def score_root(self, rows):
        """Most anomalous service, discounted by what a failing dependency explains."""
        explained = _column(rows, "f_graph_downstream_explained").fillna(0.0).to_numpy(float)
        discount = np.clip(explained / DOWNSTREAM_SCALE, 0.0, 1.0)
        return _signature_score(rows) - discount

    def score_fault(self, rows):
        components = _signature_components(rows)
        evidence = np.zeros((len(rows), len(self.classes_)))
        for position, name in enumerate(self.classes_):
            for component, family in SIGNATURE_FAULT.items():
                if family == name and component in components:
                    evidence[:, position] = np.maximum(
                        evidence[:, position], np.nan_to_num(components[component])
                    )
        # x4 so a component at full scale is a decisive vote rather than a shrug.
        return _softmax(evidence * 4.0)


class LogRegModel(LearnedTwoStage):
    """Median imputation with a missingness indicator, standardisation, logistic model."""

    name = "logreg"

    def make_estimator(self, stage, y):
        weight = "balanced" if stage in ("detect", "root") else None
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(**{"max_iter": 2000, "class_weight": weight,
                                  **self.stage_params(stage)}),
        )


class RandomForestModel(LearnedTwoStage):
    name = "rf"

    def make_estimator(self, stage, y):
        weight = "balanced" if stage in ("detect", "root") else None
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(**{
                "n_estimators": 300, "min_samples_leaf": 2, "class_weight": weight,
                "random_state": 0, "n_jobs": -1, **self.stage_params(stage),
            }),
        )


class XGBModel(LearnedTwoStage):
    """Gradient boosting; XGBoost splits on NaN natively, so no imputation."""

    name = "xgb"

    def make_estimator(self, stage, y):
        params = {
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "tree_method": "hist", "random_state": 0, "n_jobs": -1,
        }
        if stage in ("detect", "root"):
            params["scale_pos_weight"] = AUTO_POS_WEIGHT
            params["eval_metric"] = "logloss"
        else:
            # XGBClassifier picks binary vs multi:softprob from the label count itself;
            # forcing multi:softprob breaks a two-family training set. The eval metric has
            # to follow the same rule -- mlogloss is only defined for the multiclass
            # objective and rejects the first round of any eval_set under a binary one.
            params["eval_metric"] = "mlogloss" if len(np.unique(y)) > 2 else "logloss"
        params.update(self.stage_params(stage))
        if params.get("scale_pos_weight") == AUTO_POS_WEIGHT:
            positive = max(int(np.sum(y == 1)), 1)
            params["scale_pos_weight"] = float(len(y) - positive) / positive
        return XGBClassifier(**params)


MODELS: dict[str, type[TwoStageModel]] = {
    "rules": RulesModel,
    "signature": SignatureModel,
    "stat": StatModel,
    "logreg": LogRegModel,
    "rf": RandomForestModel,
    "xgb": XGBModel,
}


# --- helpers -------------------------------------------------------------------------
def _column(rows: pd.DataFrame, name: str) -> pd.Series:
    if name in rows.columns:
        return rows[name]
    return pd.Series(np.nan, index=rows.index, dtype="float64")


def _fired(rows: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame of the static rules; an unobservable feature does not fire."""
    return pd.DataFrame(
        {rule: _column(rows, name) > threshold for rule, (name, threshold) in RULES.items()},
        index=rows.index,
    )


def _signature_components(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each signal as value/scale clipped to [0, 1]; NaN where it is not observable."""
    out = {}
    for name, (column, scale) in SIGNATURE_COMPONENTS.items():
        if column in rows.columns:
            out[name] = np.clip(rows[column].to_numpy(float) / scale, 0.0, 1.0)
    return out


def _signature_score(rows: pd.DataFrame) -> np.ndarray:
    """Mean of the observable components; 0 for a row with no signal at all."""
    components = _signature_components(rows)
    if not components:
        return np.zeros(len(rows))
    stacked = np.vstack(list(components.values()))
    with np.errstate(invalid="ignore"):
        score = np.nanmean(stacked, axis=0)
    return np.nan_to_num(score, nan=0.0)


def _max_over(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Row-wise max over the columns that exist; NaN where none is observable."""
    present = [name for name in columns if name in frame.columns]
    if not present:
        return np.zeros(len(frame))
    return frame[present].max(axis=1, skipna=True).to_numpy(float)


def _softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)
