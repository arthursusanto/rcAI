"""``rca features`` commands."""
from __future__ import annotations

from pathlib import Path

import typer

from rca.features.build import build_dataset
from rca.features.windows import DEFAULT_WINDOW_NS

app = typer.Typer(help="Window and feature construction.")


@app.command("build")
def build(
    experiments: Path = typer.Option(..., help="Directory of canonical experiments."),
    out: Path = typer.Option(..., help="Parquet file to write."),
    window_ns: int = typer.Option(DEFAULT_WINDOW_NS, help="Window width in nanoseconds."),
    stride_ns: int | None = typer.Option(None, help="Window stride; defaults to the width."),
    limit: int | None = typer.Option(None, help="Only window the first N experiments."),
) -> None:
    """Build the window table for every experiment under ``--experiments``."""
    windows = build_dataset(experiments, out, window_ns=window_ns, stride_ns=stride_ns,
                            limit=limit)
    typer.echo(
        f"wrote {len(windows)} windows "
        f"({windows['experiment_id'].nunique()} experiments) to {out}"
    )
