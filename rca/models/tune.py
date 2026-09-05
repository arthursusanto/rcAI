"""Cross-validated hyperparameter search for the learned models.

The search runs a grouped K-fold over the TRAIN + VAL experiments of a split -- the test
experiments are never handed to it -- with the folds stratified by fault family so no
family disappears from a fold. Grouping is by experiment, so every window of an incident
stays on one side of every fold and a configuration is never scored on windows whose
neighbours it was fitted on.

Each stage is selected on its own metric, because the three stages answer different
questions and one blended number would hide which of them a configuration actually moved:

* stage 1 ``detect`` -- AUROC over the aggregated windows (average precision recorded too)
* stage 2 ``root``   -- oracle top-1: rank the services of every fault window by the raw
  stage-2 score and ask whether the true root came first
* stage 3 ``fault``  -- macro-F1 over the fault families on true-root rows

Fold matrices are built once and reused by every trial, so a trial costs exactly ``K``
estimator fits and no pandas work. Configurations are turned into estimators through the
model's own ``make_estimator``, and the winners are written as ``{model: {stage: params}}``
for ``TwoStageModel(params=...)`` to read back -- so what was measured here is what
``rca train`` fits.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from rca.features.schema import feature_columns
from rca.models.aggregate import aggregate_feature_columns, aggregate_windows
from rca.models.augment import DEFAULT_DROPOUT, augment_training, dropout_spec
from rca.models.baselines import MODELS

STAGES = ("detect", "root", "fault")
# stage -> the metric a configuration is selected on. The others are recorded, not used.
SELECTION = {"detect": "auroc", "root": "top1", "fault": "macro_f1"}

MLFLOW_EXPERIMENT = "rca-tune"
MLFLOW_DIR = "mlruns"

# --- search spaces ---------------------------------------------------------------------
# name -> ("int", lo, hi) | ("float", lo, hi) | ("logfloat", lo, hi) | ("choice", values)
Space = dict[str, tuple]

XGB_SPACE: Space = {
    "n_estimators": ("int", 100, 800),
    "max_depth": ("int", 3, 8),
    # log-uniform: 0.02 -> 0.2 is a factor of ten, and a uniform draw would spend most of
    # its budget in the top half of it.
    "learning_rate": ("logfloat", 0.02, 0.2),
    "min_child_weight": ("int", 1, 10),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
    "reg_lambda": ("float", 0.5, 5.0),
}
# Stage 2 is the badly imbalanced stage -- one root among every window's services -- so
# the positive weight is worth searching there. "auto" is the class ratio XGBModel
# computes for itself, which is the current default.
XGB_ROOT_SPACE: Space = {**XGB_SPACE, "scale_pos_weight": ("choice", ["auto", 1, 3, 10])}

RF_SPACE: Space = {
    "n_estimators": ("choice", [200, 400, 800]),
    "max_depth": ("choice", [None, 8, 16]),
    "min_samples_leaf": ("choice", [1, 2, 5]),
    "max_features": ("choice", ["sqrt", 0.5]),
}
LOGREG_SPACE: Space = {"C": ("choice", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])}

SPACES: dict[str, dict[str, Space]] = {
    "xgb": {"detect": XGB_SPACE, "root": XGB_ROOT_SPACE, "fault": XGB_SPACE},
    "rf": dict.fromkeys(STAGES, RF_SPACE),
    "logreg": dict.fromkeys(STAGES, LOGREG_SPACE),
}
# Configurations per stage, including the defaults trial. These are budget numbers, not
# a statement about how many the search deserves: a trial is K estimator fits, and on the
# 400-experiment simulated set one XGBoost trial measured ~20 s, so 8 configurations over
# three stages plus the early-stopping re-runs is about ten minutes. ``--trials`` raises
# it; the per-stage cap the study was specified against is 40, which costs ~45 minutes.
DEFAULT_TRIALS = {"xgb": 8, "rf": 8, "logreg": 9}
# logreg's grid is 8 values of C, so 9 trials enumerates it exactly (defaults + the grid).


def sample_config(space: Space, rng: np.random.Generator) -> dict:
    """One draw from ``space``."""
    config = {}
    for name, (kind, *bounds) in space.items():
        if kind == "int":
            config[name] = int(rng.integers(bounds[0], bounds[1] + 1))
        elif kind == "float":
            config[name] = float(rng.uniform(bounds[0], bounds[1]))
        elif kind == "logfloat":
            config[name] = float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))
        elif kind == "choice":
            config[name] = bounds[0][int(rng.integers(len(bounds[0])))]
        else:
            raise ValueError(f"unknown distribution {kind!r} for {name!r}")
    return config


def configurations(space: Space, n: int, rng: np.random.Generator) -> list[dict]:
    """Up to ``n`` distinct configurations.

    A small all-discrete space is enumerated -- sampling it at random would just draw the
    same points twice -- and anything larger is a seeded random search.
    """
    if n <= 0:
        return []
    if all(kind == "choice" for kind, *_ in space.values()):
        grid = [dict(zip(space, values, strict=True))
                for values in itertools.product(*(spec[1] for spec in space.values()))]
        if len(grid) <= n:
            return grid
    seen, out = set(), []
    for _ in range(100 * n):
        if len(out) >= n:
            break
        config = sample_config(space, rng)
        key = json.dumps(config, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(config)
    return out


# --- folds -------------------------------------------------------------------------------
def grouped_folds(
    experiment_ids: Iterable[str], strata: dict[str, str], k: int = 4, seed: int = 0
) -> list[list[str]]:
    """``k`` disjoint experiment-id lists covering ``experiment_ids``.

    Each stratum is shuffled and dealt round-robin, so every fault family is spread as
    evenly over the folds as its count allows. The running offset means a stratum with
    fewer than ``k`` members does not always land in fold 0.
    """
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    members: dict[str, list[str]] = defaultdict(list)
    for experiment_id in sorted(set(experiment_ids)):
        members[strata.get(experiment_id, "")].append(experiment_id)
    offset = 0
    for stratum in sorted(members):
        ids = members[stratum]
        rng.shuffle(ids)
        for position, experiment_id in enumerate(ids):
            folds[(offset + position) % k].append(experiment_id)
        offset += len(ids)
    return [sorted(fold) for fold in folds]


def subsample(
    experiment_ids: Sequence[str], strata: dict[str, str], n: int, seed: int = 0
) -> list[str]:
    """About ``n`` experiment ids keeping the fault-family mix; used to fit a time budget."""
    ids = sorted(set(experiment_ids))
    if n <= 0 or n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    members: dict[str, list[str]] = defaultdict(list)
    for experiment_id in ids:
        members[strata.get(experiment_id, "")].append(experiment_id)
    picked: list[str] = []
    for stratum in sorted(members):
        group = members[stratum]
        rng.shuffle(group)
        picked += group[:max(1, round(n * len(group) / len(ids)))]
    return sorted(picked)


@dataclass
class StageData:
    """Design matrices for one stage of one fold.

    ``fit`` and ``eval`` are a second, also grouped, split of the fold's training
    experiments: ``eval`` is what XGBoost early stopping watches, so the stopping round is
    never chosen on the rows the trial is scored on. ``all`` is the two together, which is
    what a trial without early stopping trains on. Only the training side is augmented --
    the fold's validation rows stay clean, exactly as the validation fold does in
    ``TwoStageModel.fit``.
    """

    x_all: np.ndarray
    y_all: np.ndarray
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_eval: np.ndarray
    y_eval: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    window: np.ndarray | None = None    # stage 2: the window each validation row belongs to


@dataclass
class Fold:
    val_ids: list[str]
    train_ids: list[str]
    stages: dict[str, StageData]


def stage_frame(windows: pd.DataFrame, stage: str) -> pd.DataFrame:
    """The rows one stage trains on: aggregated windows, fault rows, true-root rows."""
    if stage == "detect":
        return aggregate_windows(windows)
    if stage == "root":
        return windows[windows["is_fault_window"]]
    return windows[windows["is_root"]]


def _stage_xy(frame: pd.DataFrame, stage: str) -> tuple[np.ndarray, np.ndarray]:
    if stage == "detect":
        columns = aggregate_feature_columns(frame)
        y = frame["is_fault_window"].to_numpy(bool).astype(int)
    else:
        columns = feature_columns(frame.columns)
        y = (frame["is_root"].to_numpy(bool).astype(int) if stage == "root"
             else frame["fault_type"].astype(str).to_numpy())
    return frame[columns].to_numpy("float32"), y


def _stage_data(frames: dict[str, pd.DataFrame], stage: str) -> StageData:
    parts = {name: stage_frame(rows, stage) for name, rows in frames.items()}
    if stage == "root":
        # Sorted so that ranking ties fall to the alphabetically first service, which is
        # how TwoStageModel.predict breaks them.
        parts["val"] = parts["val"].sort_values(
            ["experiment_id", "window_idx", "service"], kind="mergesort"
        )
    xy = {name: _stage_xy(rows, stage) for name, rows in parts.items()}
    window = None
    if stage == "root":
        window = parts["val"].groupby(
            ["experiment_id", "window_idx"], sort=False
        ).ngroup().to_numpy()
    return StageData(
        x_all=xy["all"][0], y_all=xy["all"][1],
        x_fit=xy["fit"][0], y_fit=xy["fit"][1],
        x_eval=xy["eval"][0], y_eval=xy["eval"][1],
        x_val=xy["val"][0], y_val=xy["val"][1],
        window=window,
    )


def build_folds(
    windows: pd.DataFrame,
    strata: dict[str, str],
    k: int = 4,
    seed: int = 0,
    eval_folds: int = 5,
    modality_dropout: float | tuple[float, list[str]] = DEFAULT_DROPOUT,
) -> list[Fold]:
    """Grouped K-fold matrices for all three stages, built once and reused by every trial.

    ``windows`` must already be restricted to the CV pool (the split's train + val
    experiments); nothing here looks at a manifest, so a test experiment cannot leak in
    unless the caller puts it in.
    """
    fraction, modalities = dropout_spec(modality_dropout)
    ids = sorted(windows["experiment_id"].unique())
    folds = []
    for index, val_ids in enumerate(grouped_folds(ids, strata, k, seed)):
        held_out = set(val_ids)
        train_ids = [i for i in ids if i not in held_out]
        eval_ids = grouped_folds(train_ids, strata, eval_folds, seed + index + 1)[0]
        fit_ids = [i for i in train_ids if i not in set(eval_ids)]

        def rows(subset: Sequence[str]) -> pd.DataFrame:
            return windows[windows["experiment_id"].isin(set(subset))]

        frames = {
            "all": augment_training(rows(train_ids), fraction, modalities),
            "fit": augment_training(rows(fit_ids), fraction, modalities),
            "eval": rows(eval_ids),
            "val": rows(val_ids),
        }
        folds.append(Fold(
            val_ids=list(val_ids), train_ids=train_ids,
            stages={stage: _stage_data(frames, stage) for stage in STAGES},
        ))
    return folds


# --- one trial ----------------------------------------------------------------------------
@dataclass
class Trial:
    model: str
    stage: str
    index: int
    params: dict
    early_stopping: int | None
    metrics: dict[str, float]       # mean over the folds
    spread: dict[str, float]        # standard deviation over the folds
    per_fold: list[float]           # the selection metric on each fold
    seconds: float
    rounds: int | None              # median stopping round, when early stopping was on

    @property
    def score(self) -> float:
        value = self.metrics.get(SELECTION[self.stage], float("nan"))
        return -math.inf if math.isnan(value) else value


def _encode(labels: np.ndarray, classes: list[str] | None) -> np.ndarray:
    return labels if classes is None else np.array([classes.index(v) for v in labels])


def fit_stage(
    model_name: str, stage: str, config: dict, data: StageData, early_stopping: int | None
):
    """Fit one stage estimator on one fold, through the model's own ``make_estimator``.

    Stage 3 labels are fault-family names and XGBoost needs 0..K-1, so they are encoded
    against the classes present in *this* fold's training rows and decoded after
    predicting; a family the fold never trained on simply cannot be predicted.
    """
    x, y = ((data.x_fit, data.y_fit) if early_stopping else (data.x_all, data.y_all))
    classes = sorted(set(y.tolist())) if stage == "fault" else None
    target = _encode(y, classes)

    params = dict(config)
    fit_kwargs: dict = {}
    if early_stopping:
        params["early_stopping_rounds"] = int(early_stopping)
        keep = np.isin(data.y_eval, classes) if classes else np.ones(len(data.y_eval), bool)
        fit_kwargs = {"eval_set": [(data.x_eval[keep], _encode(data.y_eval[keep], classes))],
                      "verbose": False}
    estimator = MODELS[model_name](params={stage: params}).make_estimator(stage, target)
    estimator.fit(x, target, **fit_kwargs)
    return estimator, classes


def score_stage(stage: str, estimator, classes: list[str] | None, data: StageData) -> dict:
    """The stage's selection metric on the fold's validation rows, plus its companions."""
    keys = {"detect": ("auroc", "average_precision"), "root": ("top1", "top3"),
            "fault": ("macro_f1", "accuracy")}[stage]
    if len(data.x_val) == 0:
        return dict.fromkeys(keys, float("nan"))

    if stage == "fault":
        predicted = np.asarray(classes)[estimator.predict(data.x_val)]
        return {
            "macro_f1": float(f1_score(data.y_val, predicted, average="macro",
                                       zero_division=0)),
            "accuracy": float((predicted == data.y_val).mean()),
        }

    score = estimator.predict_proba(data.x_val)[:, 1]
    if stage == "detect":
        y = data.y_val.astype(bool)
        if y.all() or not y.any():
            return dict.fromkeys(keys, float("nan"))
        return {"auroc": float(roc_auc_score(y, score)),
                "average_precision": float(average_precision_score(y, score))}

    frame = pd.DataFrame({"window": data.window, "score": score,
                          "is_root": data.y_val.astype(bool)})
    rank = frame.groupby("window")["score"].rank(ascending=False, method="first")
    truth = rank[frame["is_root"].to_numpy()]
    if truth.empty:
        return dict.fromkeys(keys, float("nan"))
    return {"top1": float((truth == 1).mean()), "top3": float((truth <= 3).mean())}


def run_trial(
    model_name: str,
    stage: str,
    config: dict,
    folds: Sequence[Fold],
    index: int,
    early_stopping: int | None = None,
) -> Trial:
    """Fit and score one configuration on every fold."""
    start = time.perf_counter()
    per_fold, rounds = [], []
    for fold in folds:
        data = fold.stages[stage]
        estimator, classes = fit_stage(model_name, stage, config, data, early_stopping)
        per_fold.append(score_stage(stage, estimator, classes, data))
        stopped = getattr(estimator, "best_iteration", None)
        if stopped is not None:
            rounds.append(int(stopped) + 1)
    return Trial(
        model=model_name, stage=stage, index=index, params=config,
        early_stopping=early_stopping,
        metrics={key: float(np.nanmean([f[key] for f in per_fold])) for key in per_fold[0]},
        spread={key: float(np.nanstd([f[key] for f in per_fold])) for key in per_fold[0]},
        per_fold=[float(f[SELECTION[stage]]) for f in per_fold],
        seconds=time.perf_counter() - start,
        rounds=int(np.median(rounds)) if rounds else None,
    )


# --- the search ----------------------------------------------------------------------------
def search_stage(
    model_name: str,
    stage: str,
    folds: Sequence[Fold],
    trials: int,
    seed: int,
    early_stopping: int = 0,
    early_stopping_top: int = 0,
) -> list[Trial]:
    """Random search over one stage's space, then an early-stopping A/B on the best few.

    Trial 0 is always the model's own defaults, so the search is measured against the
    incumbent on exactly the same folds and can never report an improvement it did not
    make. The early-stopping trials re-run the top configurations unchanged except for
    the stopping rule, which is the only way to attribute a difference to it.
    """
    rng = np.random.default_rng(seed)
    configs = [{}] + configurations(SPACES[model_name][stage], trials - 1, rng)
    log = [run_trial(model_name, stage, config, folds, index)
           for index, config in enumerate(configs)]
    if early_stopping and early_stopping_top:
        best = sorted(log, key=lambda trial: -trial.score)[:early_stopping_top]
        log += [run_trial(model_name, stage, trial.params, folds, len(log) + offset,
                          early_stopping)
                for offset, trial in enumerate(best)]
    return log


def select(log: Sequence[Trial]) -> Trial:
    """Highest CV mean of the stage's selection metric, ties going to the earlier trial.

    Trial 0 is the defaults and the early-stopping re-runs come last, so a tie keeps the
    incumbent: a configuration is only adopted when it is strictly better. Tie-breaking on
    anything measured (wall time, say) would make the selection nondeterministic, and CV
    means tie often enough -- a saturated metric, a stage the data makes easy -- to matter.
    """
    return max(log, key=lambda trial: (trial.score, -trial.index))


def final_params(trial: Trial) -> dict:
    """The winner's overrides, as ``rca train`` will apply them.

    An early-stopping winner cannot carry an eval set into training, so the round it
    stopped at (median over the folds) becomes its ``n_estimators`` instead.
    """
    params = dict(trial.params)
    if trial.early_stopping and trial.rounds:
        params["n_estimators"] = int(trial.rounds)
    return params


def tune_model(
    model_name: str,
    folds: Sequence[Fold],
    trials: int = 0,
    seed: int = 0,
    early_stopping: int = 50,
    early_stopping_top: int = 5,
) -> tuple[list[Trial], dict[str, dict]]:
    """Search all three stages of one model; returns every trial and the winning params."""
    trials = trials or DEFAULT_TRIALS[model_name]
    # Early stopping is an XGBoost facility; the other two models have no such knob.
    rounds = early_stopping if model_name == "xgb" else 0
    log, best = [], {}
    for offset, stage in enumerate(STAGES):
        stage_log = search_stage(model_name, stage, folds, trials, seed + offset,
                                 rounds, early_stopping_top)
        log += stage_log
        best[stage] = final_params(select(stage_log))
    return log, best


# --- persistence -----------------------------------------------------------------------------
def trials_frame(log: Sequence[Trial]) -> pd.DataFrame:
    """One row per trial: params, CV mean and standard deviation, wall time."""
    rows = []
    for trial in log:
        metric = SELECTION[trial.stage]
        rows.append({
            "model": trial.model, "stage": trial.stage, "trial": trial.index,
            "selection_metric": metric,
            "cv_mean": trial.metrics.get(metric, float("nan")),
            "cv_std": trial.spread.get(metric, float("nan")),
            "seconds": round(trial.seconds, 2),
            "early_stopping": trial.early_stopping or 0,
            "rounds": trial.rounds if trial.rounds is not None else "",
            **{f"cv_{key}": value for key, value in trial.metrics.items()},
            **{f"fold{index}": value for index, value in enumerate(trial.per_fold)},
            **{f"p_{key}": value for key, value in trial.params.items()},
        })
    return pd.DataFrame(rows)


def trials_records(log: Sequence[Trial]) -> list[dict]:
    return [{
        "model": trial.model, "stage": trial.stage, "trial": trial.index,
        "params": trial.params, "early_stopping": trial.early_stopping or 0,
        "rounds": trial.rounds, "selection_metric": SELECTION[trial.stage],
        "cv_mean": trial.metrics.get(SELECTION[trial.stage], float("nan")),
        "cv_std": trial.spread.get(SELECTION[trial.stage], float("nan")),
        "cv": trial.metrics, "cv_std_all": trial.spread, "per_fold": trial.per_fold,
        "seconds": round(trial.seconds, 3),
    } for trial in log]


def write_tuning(
    out_dir: Path | str, log: Sequence[Trial], best: dict[str, dict], meta: dict | None = None
) -> Path:
    """Write ``trials-<model>.csv`` / ``.json`` per model, and merge ``best.json``.

    ``best.json`` is merged rather than replaced: the three models cost very different
    amounts to search, so tuning them in separate runs is the normal way to keep one run
    inside a time budget, and each should be able to land in the file the other one is
    already in. A model searched again simply replaces its own entry.
    """
    from rca.eval.report import jsonable

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, winners in best.items():
        trials = [trial for trial in log if trial.model == model]
        trials_frame(trials).to_csv(
            out_dir / f"trials-{model}.csv", index=False, lineterminator="\n"
        )
        payload = {"meta": meta or {}, "best": winners, "trials": trials_records(trials)}
        (out_dir / f"trials-{model}.json").write_text(
            json.dumps(jsonable(payload), indent=2) + "\n", newline="\n"
        )

    path = out_dir / "best.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(best)
    path.write_text(json.dumps(jsonable(merged), indent=2) + "\n", newline="\n")
    return path


def read_params(path: Path | str | None, model: str) -> dict[str, dict]:
    """Tuned ``{stage: params}`` for ``model`` from a ``rca tune`` ``best.json``.

    Reads the file ``rca tune`` writes (``{model: {stage: params}}``) and also a bare
    ``{stage: params}`` mapping, for a hand-written one. A model with no entry gets ``{}``,
    which is "use the defaults" -- so one file can be passed to a run comparing all five
    models without the untuned ones failing.
    """
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if any(stage in payload for stage in STAGES):
        return {stage: dict(payload[stage]) for stage in STAGES if stage in payload}
    return {stage: dict(params) for stage, params in payload.get(model, {}).items()}


def log_trials(
    log: Sequence[Trial],
    meta: dict | None = None,
    artifacts: Iterable[Path] = (),
    experiment: str = MLFLOW_EXPERIMENT,
    tracking_dir: Path | str = MLFLOW_DIR,
) -> None:
    """One MLflow run per trial, in a single tracking session; never fatal to a search."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    tracking = Path(tracking_dir).absolute()
    tracking.mkdir(parents=True, exist_ok=True)
    try:
        mlflow.set_tracking_uri(tracking.as_uri())
        mlflow.set_experiment(experiment)
        for trial in log:
            name = f"{trial.model}-{trial.stage}-{trial.index}"
            with mlflow.start_run(run_name=name):
                mlflow.log_params({
                    **{key: str(value) for key, value in (meta or {}).items()},
                    "model": trial.model, "stage": trial.stage, "trial": trial.index,
                    "early_stopping": trial.early_stopping or 0,
                    **{key: str(value) for key, value in trial.params.items()},
                })
                mlflow.log_metrics({
                    "cv_mean": trial.score, "seconds": trial.seconds,
                    "cv_std": trial.spread.get(SELECTION[trial.stage], 0.0),
                    **{f"cv_{key}": value for key, value in trial.metrics.items()
                       if math.isfinite(value)},
                    **{f"fold{index}": value for index, value in enumerate(trial.per_fold)
                       if math.isfinite(value)},
                })
        with mlflow.start_run(run_name=f"{experiment}-artifacts"):
            for artifact in artifacts:
                if Path(artifact).exists():
                    mlflow.log_artifact(str(artifact))
    except Exception as error:                                  # noqa: BLE001
        print(f"warning: MLflow logging failed ({error})", file=sys.stderr)
