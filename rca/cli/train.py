"""``rca train`` and ``rca eval``."""
from __future__ import annotations

import os
from pathlib import Path

# XGBoost's OpenMP pool spins while it waits for work, which on a machine with anything
# else running turns a seconds-long fit into a minutes-long one. Set before importing
# anything that loads libomp.
os.environ.setdefault("OMP_WAIT_POLICY", "passive")

import pandas as pd
import typer

from rca.data import splits as split_module
from rca.data.schema import list_experiments, read_manifest
from rca.eval.ablation import ablation_report
from rca.eval.metrics import evaluate
from rca.eval.report import (
    comparison_display,
    comparison_table,
    flat_metrics,
    log_mlflow,
    markdown_table,
    nested_table,
    write_markdown,
    write_results,
)
from rca.eval.robustness import robustness_report
from rca.models.augment import DEFAULT_DROPOUT
from rca.models.baselines import MODELS
from rca.models.tune import read_params

SPLITS = [
    "by-experiment", "holdout-service", "holdout-intensity-high", "holdout-intensity-low",
    "holdout-traffic", "by-time",
]
LEARNED = ["logreg", "rf", "xgb"]

train_app = typer.Typer(help="Train a three-stage model on a leakage-free split.")
eval_app = typer.Typer(help="Evaluate and compare models on a leakage-free split.")


def load_manifests(experiments: Path) -> list:
    return [read_manifest(path) for path in list_experiments(experiments)]


def resolve_split(
    name: str, manifests: list, seed: int = 0, holdout_services: str = "cart,recommendation",
    holdout_shapes: str = "bursty",
) -> dict[str, list[str]]:
    """Named split -> {"train": [...], "val": [...], "test": [...]} of experiment ids."""
    if name == "by-experiment":
        return split_module.split_by_experiment(manifests, seed=seed)
    if name == "holdout-service":
        return split_module.split_holdout_service(
            manifests, [s for s in holdout_services.split(",") if s], seed=seed
        )
    if name == "holdout-intensity-high":
        return split_module.split_holdout_intensity(manifests, (0.8, 1.0), seed=seed)
    if name == "holdout-intensity-low":
        return split_module.split_holdout_intensity(manifests, (0.0, 0.3), seed=seed)
    if name == "holdout-traffic":
        return split_module.split_holdout_traffic(
            manifests, [s for s in holdout_shapes.split(",") if s], seed=seed
        )
    if name == "by-time":
        return split_module.split_by_time(manifests)
    raise typer.BadParameter(f"unknown split {name!r}; choose one of {SPLITS}")


def split_frames(windows: pd.DataFrame, split: dict) -> dict[str, pd.DataFrame]:
    return {
        name: windows[windows["experiment_id"].isin(ids)]
        for name, ids in split.items()
    }


@train_app.callback(invoke_without_command=True)
def train(
    windows: Path = typer.Option(..., help="Window table parquet."),
    experiments: Path = typer.Option(Path("data/experiments"), help="Experiment directory."),
    split: str = typer.Option("by-experiment", help=f"One of {SPLITS}."),
    model: str = typer.Option("xgb", help=f"One of {sorted(MODELS)}."),
    out: Path = typer.Option(..., help="Directory to write the fitted model to."),
    seed: int = typer.Option(0, help="Split seed."),
    holdout_services: str = typer.Option("cart,recommendation"),
    holdout_shapes: str = typer.Option("bursty"),
    target_false_alarm_rate: float | None = typer.Option(
        None, help="Pick the detection threshold for this validation FPR instead of F1."
    ),
    modality_dropout: float = typer.Option(
        DEFAULT_DROPOUT, help="Fraction of training windows re-added degraded; 0 disables."
    ),
    debounce: int = typer.Option(1, help="Consecutive positive windows required to alarm."),
    params: Path = typer.Option(
        None, help="Tuned hyperparameters from `rca tune` (artifacts/tune/<split>/best.json)."
    ),
) -> None:
    """Fit one model and save it, then report its test metrics."""
    if model not in MODELS:
        raise typer.BadParameter(f"unknown model {model!r}; choose one of {sorted(MODELS)}")
    tuned = read_params(params, model)
    table = pd.read_parquet(windows)
    manifests = load_manifests(experiments)
    ids = resolve_split(split, manifests, seed, holdout_services, holdout_shapes)
    frames = split_frames(table, ids)
    typer.echo(f"split {split}: " + ", ".join(f"{k}={len(v)} exp" for k, v in ids.items()))

    if tuned:
        typer.echo(f"tuned params from {params}: {tuned}")
    fitted = MODELS[model](target_false_alarm_rate=target_false_alarm_rate,
                           modality_dropout=modality_dropout, params=tuned)
    fitted.fit(frames["train"], frames["val"])
    fitted.save(out)
    clean, augmented = fitted.training_rows_
    typer.echo(
        f"training rows {clean} -> {augmented} (modality dropout {modality_dropout}); "
        f"detection threshold {fitted.detect_threshold_:.4f} "
        f"(raw {fitted.detect_threshold_raw_:.4f}), stage-3 temperature "
        f"{fitted.temperature_:.3f}"
    )

    result = evaluate(fitted, frames["test"], manifests, debounce=debounce)
    write_results({model: result}, out)
    write_markdown({model: result}, out, f"{model} on the {split} split")
    log_mlflow(f"train-{model}-{split}",
               {"model": model, "split": split, "seed": seed,
                "threshold": fitted.detect_threshold_, "debounce": debounce,
                "modality_dropout": modality_dropout, "params": tuned or "default"},
               flat_metrics(result), [out / "results.json"])
    typer.echo(markdown_table(comparison_table({model: result})))
    typer.echo(f"wrote model and results to {out}")


@eval_app.callback(invoke_without_command=True)
def evaluate_models(
    windows: Path = typer.Option(..., help="Window table parquet."),
    experiments: Path = typer.Option(Path("data/experiments"), help="Experiment directory."),
    split: str = typer.Option("by-experiment", help=f"One of {SPLITS}."),
    models: str = typer.Option("rules,stat,logreg,rf,xgb", help="Comma-separated model list."),
    out: Path = typer.Option(..., help="Directory to write the comparison to."),
    seed: int = typer.Option(0, help="Split seed."),
    holdout_services: str = typer.Option("cart,recommendation"),
    holdout_shapes: str = typer.Option("bursty"),
    robustness: bool = typer.Option(False, help="Also evaluate degraded telemetry."),
    ablation: bool = typer.Option(False, help="Also retrain on modality subsets (learned)."),
    modality_dropout: float = typer.Option(
        DEFAULT_DROPOUT, help="Fraction of training windows re-added degraded; 0 disables."
    ),
    debounce: int = typer.Option(1, help="Consecutive positive windows required to alarm."),
    bootstrap: int = typer.Option(
        500, help="Cluster-bootstrap replicates behind the comparison CIs; 0 disables."
    ),
    params: Path = typer.Option(
        None, help="Tuned hyperparameters from `rca tune` (artifacts/tune/<split>/best.json)."
    ),
) -> None:
    """Train every requested model on the same split and compare them on the test set."""
    names = [name.strip() for name in models.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODELS]
    if unknown:
        raise typer.BadParameter(f"unknown models {unknown}; choose from {sorted(MODELS)}")
    tuned = {name: read_params(params, name) for name in names}
    if params is not None:
        typer.echo(f"tuned params from {params}: "
                   f"{ {name: sorted(stages) for name, stages in tuned.items() if stages} }")

    table = pd.read_parquet(windows)
    manifests = load_manifests(experiments)
    ids = resolve_split(split, manifests, seed, holdout_services, holdout_shapes)
    frames = split_frames(table, ids)
    typer.echo(f"split {split}: " + ", ".join(f"{k}={len(v)} exp" for k, v in ids.items()))

    results, sections, extra = {}, {}, {}
    for name in names:
        def build(name=name):
            return MODELS[name](modality_dropout=modality_dropout, params=tuned[name])

        model = build().fit(frames["train"], frames["val"])
        results[name] = evaluate(model, frames["test"], manifests, debounce=debounce,
                                 bootstrap=bootstrap)
        typer.echo(f"{name}: fitted, threshold {model.detect_threshold_:.4f}")
        if robustness:
            extra.setdefault("robustness", {})[name] = robustness_report(
                model, frames["test"], manifests, debounce=debounce
            )
        if ablation and name in LEARNED:
            extra.setdefault("ablation", {})[name] = ablation_report(
                build, frames["train"], frames["val"], frames["test"], manifests,
                debounce=debounce,
            )
        log_mlflow(f"eval-{name}-{split}",
                   {"model": name, "split": split, "seed": seed,
                    "threshold": model.detect_threshold_, "debounce": debounce,
                    "modality_dropout": modality_dropout,
                    "params": tuned[name] or "default"},
                   flat_metrics(results[name]))

    for kind, report in extra.items():
        for name, table_data in report.items():
            sections[f"{kind}: {name}"] = nested_table(table_data)

    payload = {"split": split, "debounce": debounce, "bootstrap": bootstrap,
               "modality_dropout": modality_dropout,
               "params": {name: stages for name, stages in tuned.items() if stages},
               "split_experiments": ids, "models": results, **extra}
    path = write_results(payload, out)
    title = (f"Model comparison on the {split} split "
             f"(debounce K={debounce}, modality dropout {modality_dropout}"
             f"{', tuned params' if any(tuned.values()) else ''})")
    write_markdown(results, out, title, sections)
    log_mlflow(f"eval-{split}", {"split": split, "models": models, "seed": seed,
                                 "debounce": debounce, "params": str(params),
                                 "modality_dropout": modality_dropout}, {}, [path])
    typer.echo(markdown_table(comparison_display(results)))
    typer.echo(f"wrote comparison to {out}")
