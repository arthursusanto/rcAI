"""``rca tune`` -- cross-validated hyperparameter search over a split's train + val set."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import typer

from rca.cli.train import SPLITS, load_manifests, resolve_split
from rca.data.splits import fault_type_of
from rca.eval.report import markdown_table
from rca.models.augment import DEFAULT_DROPOUT
from rca.models.tune import (
    DEFAULT_TRIALS,
    SELECTION,
    STAGES,
    build_folds,
    final_params,
    log_trials,
    search_stage,
    select,
    subsample,
    trials_frame,
    write_tuning,
)

app = typer.Typer(help="Cross-validated hyperparameter search for the learned models.")


@app.callback(invoke_without_command=True)
def tune(
    windows: Path = typer.Option(..., help="Window table parquet."),
    experiments: Path = typer.Option(Path("data/experiments"), help="Experiment directory."),
    split: str = typer.Option("by-experiment", help=f"One of {SPLITS}."),
    models: str = typer.Option("xgb", help=f"Comma-separated, from {sorted(DEFAULT_TRIALS)}."),
    out: Path = typer.Option(Path("artifacts/tune"), help="Root; results go in <out>/<split>."),
    trials: int = typer.Option(0, help=f"Configs per stage; 0 uses {DEFAULT_TRIALS}."),
    folds: int = typer.Option(4, help="Grouped CV folds over the train + val experiments."),
    seed: int = typer.Option(0, help="Split and search seed."),
    early_stopping: int = typer.Option(
        50, help="XGBoost early-stopping rounds for the A/B re-run; 0 disables it."
    ),
    early_stopping_top: int = typer.Option(
        2, help="How many of the best configs are re-run with early stopping."
    ),
    max_experiments: int = typer.Option(
        0, help="Subsample the CV pool to about this many experiments; 0 uses all of them."
    ),
    modality_dropout: float = typer.Option(
        DEFAULT_DROPOUT, help="Training-side degraded-telemetry fraction, as in `rca train`."
    ),
    holdout_services: str = typer.Option("cart,recommendation"),
    holdout_shapes: str = typer.Option("bursty"),
) -> None:
    """Search each stage's hyperparameters and write the winners to <out>/<split>/best.json."""
    # XGBoost's OpenMP threads spin-wait between boosting rounds. On a machine whose cores
    # are already busy that turns a two-second fit into minutes, which is the difference
    # between this command fitting its runtime budget and not; sleeping instead of
    # spinning costs nothing when the machine is idle.
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")

    names = [name.strip() for name in models.split(",") if name.strip()]
    unknown = [name for name in names if name not in DEFAULT_TRIALS]
    if unknown:
        raise typer.BadParameter(
            f"cannot tune {unknown}; the learned models are {sorted(DEFAULT_TRIALS)}"
        )

    table = pd.read_parquet(windows)
    manifests = load_manifests(experiments)
    ids = resolve_split(split, manifests, seed, holdout_services, holdout_shapes)
    strata = {manifest.experiment_id: fault_type_of(manifest) for manifest in manifests}

    # The CV pool is the split's train + val experiments. The test experiments are not
    # even loaded into it, so nothing downstream can select on them.
    pool = sorted(set(ids["train"]) | set(ids["val"]))
    pool = subsample(pool, strata, max_experiments, seed)
    typer.echo(
        f"split {split}: cv pool {len(pool)} experiments "
        f"(train {len(ids['train'])} + val {len(ids['val'])}), "
        f"test {len(ids['test'])} held out"
    )

    planned = {name: (trials or DEFAULT_TRIALS[name]) for name in names}
    total = sum(count + (early_stopping_top if name == "xgb" and early_stopping else 0)
                for name, count in planned.items()) * len(STAGES)
    typer.echo(
        f"plan: {planned} configurations per stage x {len(STAGES)} stages "
        f"(+{early_stopping_top} early-stopping re-runs per xgb stage) = {total} trials, "
        f"{folds} fits each"
    )

    start = time.perf_counter()
    cv = build_folds(table[table["experiment_id"].isin(set(pool))], strata, folds, seed,
                     modality_dropout=modality_dropout)
    typer.echo(
        f"{folds} grouped folds ({[len(fold.val_ids) for fold in cv]} experiments) "
        f"built in {time.perf_counter() - start:.1f}s"
    )

    log, best = [], {}
    for name in names:
        for offset, stage in enumerate(STAGES):
            stage_log = search_stage(name, stage, cv, planned[name], seed + offset,
                                     early_stopping if name == "xgb" else 0,
                                     early_stopping_top)
            log += stage_log
            winner = select(stage_log)
            best.setdefault(name, {})[stage] = final_params(winner)
            typer.echo(
                f"[{time.perf_counter() - start:6.1f}s] {name}/{stage}: "
                f"best {SELECTION[stage]} {winner.score:.4f} "
                f"+/- {winner.spread[SELECTION[stage]]:.4f} "
                f"(trial {winner.index} of {len(stage_log)}, "
                f"early stopping {winner.early_stopping or 0}) {best[name][stage]}"
            )

    elapsed = time.perf_counter() - start
    meta = {"split": split, "seed": seed, "folds": folds, "trials": trials,
            "modality_dropout": modality_dropout, "early_stopping": early_stopping,
            "cv_experiments": len(pool), "seconds": round(elapsed, 1)}
    directory = out / split
    path = write_tuning(directory, log, best, {**meta, "cv_pool": pool})
    log_trials(log, meta, [path, *directory.glob("trials-*.csv")])
    typer.echo(f"{path} now holds tuned params for {sorted(json.loads(path.read_text()))}")

    frame = trials_frame(log)
    typer.echo(markdown_table(
        frame.sort_values(["model", "stage", "cv_mean"], ascending=[True, True, False])
        .groupby(["model", "stage"]).head(3)
        .set_index("model")[["stage", "trial", "cv_mean", "cv_std", "seconds",
                             "early_stopping"]],
        decimals=4,
    ))
    typer.echo(f"{len(log)} trials in {elapsed:.1f}s; wrote {path}")
