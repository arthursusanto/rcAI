"""Evaluate on the real OpenTelemetry-Demo campaign with grouped cross-validation.

Three regimes, all scored on real experiments only:
  zero-shot  - the simulator-trained artifact predicts every real experiment (no real
               data in training at all);
  real-only  - grouped 5-fold CV over real experiments; each fold's model trains on the
               other folds (with a validation carve-out for calibration and threshold)
               and predicts the held-out fold; out-of-fold predictions are pooled;
  sim+real   - the same folds, but the training set also contains every simulated
               experiment; calibration/threshold still come from real validation rows.
Debounce K is pre-registered from the simulated study (K = 1 and K = 2 are both
reported); nothing is chosen on the real test predictions.

    .venv/Scripts/python scripts/real_eval.py --real data/real --sim data/experiments \
        --real-windows data/real_windows.parquet --sim-windows data/windows.parquet \
        --sim-model artifacts/xgb --out artifacts/eval/real

The --sim inputs are the simulated dataset and the simulator-trained model from the
README's quickstart; pass --skip-sim to run only the zero-shot and real-only regimes.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "passive")
warnings.filterwarnings("ignore")

import pandas as pd

from rca.data import schema, splits
from rca.eval import metrics as M
from rca.models.base import TwoStageModel
from rca.models.baselines import MODELS
from rca.models.tune import grouped_folds


class Pooled:
    """Out-of-fold predictions from several fold models, presented as one model."""

    def __init__(self, fold_models: dict[str, TwoStageModel], fold_of: dict[str, str]):
        self.fold_models = fold_models
        self.fold_of = fold_of

    def _by_fold(self, frame: pd.DataFrame):
        for fold, ids in _group(self.fold_of).items():
            part = frame[frame["experiment_id"].isin(ids)]
            if len(part):
                yield self.fold_models[fold], part

    def predict(self, windows: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([m.predict(part) for m, part in self._by_fold(windows)],
                         ignore_index=True)

    def classify(self, rows: pd.DataFrame) -> pd.DataFrame:
        parts = [m.classify(part) for m, part in self._by_fold(rows)]
        return pd.concat(parts).reindex(rows.index)


def _group(fold_of: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for exp, fold in fold_of.items():
        out.setdefault(fold, []).append(exp)
    return out


def strata_of(manifests: list[schema.Manifest]) -> dict[str, str]:
    return {m.experiment_id: (m.faults[0].fault_type if m.faults else "none")
            for m in manifests}


def summary(result: dict) -> dict:
    d, fa, loc, det, cls = (result["detection_delay"], result["false_alarms"],
                            result["localization"], result["detection"],
                            result["classification"])
    ci = result.get("bootstrap", {})

    def band(key):
        v = ci.get(key)
        return [round(v["lo"], 3), round(v["hi"], 3)] if v else None

    return {
        "incidents": d["n_incidents"],
        "incident_recall": round(1 - d["undetected_fraction"], 3),
        "time_to_alarm_median_s": round(d["median_s"], 1),
        "time_to_alarm_p90_s": round(d["p90_s"], 1),
        "fa_per_hour": round(fa["all_fault_free"]["per_hour"], 2),
        "fa_per_hour_ci": band("fa_all"),
        "fa_normal": round(fa["normal"]["per_hour"], 2),
        "fa_spike": round(fa["traffic_spike"]["per_hour"], 2),
        "top1_incident": round(loc["top1_incident"], 3),
        "top1_incident_ci": band("top1_incident"),
        "top1_oracle": round(loc["oracle"]["top1"], 3),
        "top1_oracle_ci": band("top1_oracle"),
        "top3_oracle": round(loc["oracle"]["top3"], 3),
        "top3_detected": round(loc["detected"]["top3"], 3),
        "fault_macro_f1": round(cls["predicted_root"]["macro_f1"], 3),
        "fault_macro_f1_ci": band("fault_macro_f1"),
        "detect_precision": round(det["precision"], 3),
        "detect_recall": round(det["recall"], 3),
        "auroc": round(det["auroc"], 3),
        "per_family_top1": {k: round(v["top1"], 2)
                            for k, v in loc["per_fault_type"].items()},
    }


def cv_regime(name, real_w, real_mans, sim_w, folds, model_names, debounces, bootstrap):
    fold_of = {exp: f"f{i}" for i, ids in enumerate(folds) for exp in ids}
    man_by_id = {m.experiment_id: m for m in real_mans}
    rows = {}
    for model_name in model_names:
        fold_models = {}
        for i, test_ids in enumerate(folds):
            train_ids = [e for e in fold_of if fold_of[e] != f"f{i}"]
            carve = splits.split_by_experiment(
                [man_by_id[e] for e in train_ids], seed=i, fracs=(0.8, 0.2, 0.0))
            tr = real_w[real_w["experiment_id"].isin(carve["train"])]
            va = real_w[real_w["experiment_id"].isin(carve["val"])]
            if sim_w is not None:
                tr = pd.concat([sim_w, tr], ignore_index=True)
            model = MODELS[model_name]()
            model.fit(tr, va)
            fold_models[f"f{i}"] = model
        pooled = Pooled(fold_models, fold_of)
        predictions = pooled.predict(real_w)
        for k in debounces:
            result = M.evaluate(pooled, real_w, real_mans, predictions=predictions,
                                profile=False, debounce=k, bootstrap=bootstrap)
            rows[f"{name}/{model_name}/K={k}"] = summary(result)
            print(f"{name:9s} {model_name:6s} K={k}: {json.dumps(rows[f'{name}/{model_name}/K={k}'])[:220]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="data/real")
    ap.add_argument("--real-windows", default="data/real_windows.parquet")
    ap.add_argument("--sim", default="data/experiments")
    ap.add_argument("--sim-windows", default="data/windows.parquet")
    ap.add_argument("--sim-model", default="artifacts/xgb")
    ap.add_argument("--out", default="artifacts/eval/real")
    ap.add_argument("--models", default="rules,stat,rf,xgb")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--skip-sim", action="store_true", help="only zero-shot and real-only")
    args = ap.parse_args()

    real_w = pd.read_parquet(args.real_windows)
    real_mans = [schema.read_manifest(p) for p in schema.list_experiments(Path(args.real))]
    sim_w = pd.read_parquet(args.sim_windows)
    model_names = args.models.split(",")
    folds = grouped_folds([m.experiment_id for m in real_mans], strata_of(real_mans),
                          k=args.folds, seed=0)
    rows: dict[str, dict] = {}

    sim_model = TwoStageModel.load(Path(args.sim_model))
    predictions = sim_model.predict(real_w)
    for k in (1, 2):
        result = M.evaluate(sim_model, real_w, real_mans, predictions=predictions,
                            profile=False, debounce=k, bootstrap=args.bootstrap)
        rows[f"zero-shot/xgb/K={k}"] = summary(result)
        print(f"zero-shot xgb    K={k}: {json.dumps(rows[f'zero-shot/xgb/K={k}'])[:220]}")

    rows.update(cv_regime("real-only", real_w, real_mans, None, folds, model_names,
                          (1, 2), args.bootstrap))
    if not args.skip_sim:
        learned = [m for m in model_names if m in ("rf", "xgb", "logreg")]
        rows.update(cv_regime("sim+real", real_w, real_mans, sim_w, folds, learned,
                              (1, 2), args.bootstrap))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"folds": folds, "rows": rows}, indent=2), newline="\n")
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
