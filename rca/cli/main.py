"""``rca`` command line entry point."""
from __future__ import annotations

from pathlib import Path

import typer

from rca.cli.bench import app as bench_app
from rca.cli.features import app as features_app
from rca.cli.serve import replay_app, serve_app
from rca.cli.train import eval_app, train_app
from rca.cli.tune import app as tune_app
from rca.data.schema import write_experiment
from rca.sim import FaultSpec, generate_dataset, generate_experiment, sample_traffic

app = typer.Typer(help="Cloud-outage root-cause AI", no_args_is_help=True)
sim_app = typer.Typer(help="Telemetry simulator", no_args_is_help=True)
app.add_typer(sim_app, name="sim")
app.add_typer(bench_app, name="bench")
app.add_typer(features_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(tune_app, name="tune")
app.add_typer(serve_app, name="serve")
app.add_typer(replay_app, name="replay")


@sim_app.command("generate")
def sim_generate(
    out: Path = typer.Option(Path("data/experiments"), "--out", help="dataset root"),
    n: int = typer.Option(50, "--n", help="number of experiments"),
    seed: int = typer.Option(0, "--seed"),
    duration_min: float = typer.Option(120.0, "--duration-min"),
    duration_max: float = typer.Option(240.0, "--duration-max"),
    normal_fraction: float = typer.Option(0.20, "--normal-fraction"),
    spike_fraction: float = typer.Option(0.10, "--spike-fraction",
                                         help="traffic-only spikes (hard negatives)"),
    warmup: float = typer.Option(60.0, "--warmup", help="fault-free warm-up seconds"),
    gap: float = typer.Option(60.0, "--gap", help="seconds between consecutive runs"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Generate a randomised simulated dataset."""
    paths = generate_dataset(out, n, seed=seed, duration_range=(duration_min, duration_max),
                             normal_fraction=normal_fraction,
                             traffic_only_spike_fraction=spike_fraction,
                             warmup_s=warmup, gap_s=gap, progress=not quiet)
    typer.echo(f"wrote {len(paths)} experiments to {out}")


@sim_app.command("one")
def sim_one(
    out: Path = typer.Option(Path("data/experiments"), "--out"),
    seed: int = typer.Option(0, "--seed"),
    duration: float = typer.Option(180.0, "--duration"),
    fault: str = typer.Option("", "--fault", help="fault type, empty for a normal run"),
    target: str = typer.Option("", "--target"),
    intensity: float = typer.Option(0.8, "--intensity"),
    start: float = typer.Option(75.0, "--start", help="fault start, seconds"),
    fault_duration: float = typer.Option(60.0, "--fault-duration"),
    warmup: float = typer.Option(60.0, "--warmup"),
) -> None:
    """Generate a single experiment with an explicit fault."""
    import numpy as np

    spec = None
    if fault:
        if not target:
            raise typer.BadParameter("--target is required with --fault")
        spec = FaultSpec(fault, target, start, fault_duration, intensity)
    traffic = sample_traffic(np.random.default_rng(seed), duration, warmup)
    exp = generate_experiment(seed, duration, spec, traffic, warmup_s=warmup)
    typer.echo(str(write_experiment(exp, out)))


if __name__ == "__main__":
    app()
