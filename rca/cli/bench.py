"""``rca bench`` -- drive the OpenTelemetry Demo benchmark and ingest its capture."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from rca.benchmark import ingest as ingest_mod
from rca.benchmark import runner
from rca.benchmark.injector import FaultInjector
from rca.data import schema

app = typer.Typer(help="OpenTelemetry Demo benchmark: fault campaigns and OTLP ingest.")


@app.command("run-campaign")
def run_campaign(
    n: int = typer.Option(..., help="Number of experiments to run."),
    seed: int = typer.Option(0, help="Campaign seed; the same seed replays the plan."),
    out_root: Path = typer.Option(..., help="Directory to write experiment dirs into."),
    capture_dir: Path = typer.Option(..., help="Host dir the collector writes *.jsonl to."),
    flags_path: Path = typer.Option(None, help="Demo's src/flagd/demo.flagd.json."),
    prefix: str = typer.Option("otel", help="Experiment id prefix."),
    netem_mode: str = typer.Option("sidecar", help="netem via 'sidecar' or 'exec'."),
    warmup_s: float = typer.Option(
        runner.CampaignConfig.warmup_s,
        help="Recorded fault-free baseline at the start of every window, seconds. Also "
             "the period the feature baseline is fitted on, so keep it >= 30 s."),
    fault_min_s: float = typer.Option(
        runner.CampaignConfig.hold_s_range[0], help="Shortest fault hold, seconds."),
    fault_max_s: float = typer.Option(
        runner.CampaignConfig.hold_s_range[1], help="Longest fault hold, seconds."),
    cooldown_s: float = typer.Option(
        runner.CampaignConfig.cooldown_s,
        help="Recorded period after the fault is cleared, seconds."),
    settle_s: float = typer.Option(
        runner.CampaignConfig.settle_s,
        help="Unrecorded pause after changing the traffic level, before the window starts."),
    offset_max_s: float = typer.Option(
        runner.CampaignConfig.extra_offset_s_range[1],
        help="Fault start is warmup + uniform(0, this), seconds."),
    dry_run: bool = typer.Option(False, help="Print the plan and the commands; run nothing."),
) -> None:
    """Run a randomized fault campaign against a running demo stack."""
    config = runner.CampaignConfig(warmup_s=warmup_s, cooldown_s=cooldown_s,
                                   hold_s_range=(fault_min_s, fault_max_s),
                                   settle_s=settle_s,
                                   extra_offset_s_range=(0.0, offset_max_s))
    specs = runner.plan_campaign(n, seed, config, prefix)
    wall = runner.campaign_wall_seconds(specs)
    typer.echo(f"{len(specs)} experiments, estimated wall time {wall / 60:.1f} min")

    injector = FaultInjector(flags_path=flags_path, dry_run=dry_run, netem_mode=netem_mode)
    if dry_run:
        for spec in specs:
            typer.echo(json.dumps({"experiment_id": spec.experiment_id, "kind": spec.kind,
                                   "base_vus": spec.base_vus, "spike_vus": spec.spike_vus,
                                   "start_offset_s": spec.start_offset_s,
                                   "hold_s": spec.hold_s,
                                   "fault": spec.fault.__dict__ if spec.fault else None}))
            if spec.fault is not None:
                f = schema.Fault(fault_type=spec.fault.fault_type, target=spec.fault.target,
                                 start_ns=0, end_ns=int(spec.hold_s * 1_000_000_000),
                                 intensity=spec.fault.intensity)
                for command in injector.plan(f) + injector.plan_clear(f):
                    typer.echo(f"    {command}")
        return

    paths, failed = runner.run_campaign(n, seed, out_root, injector, capture_dir,
                                        config, prefix)
    for path in paths:
        typer.echo(str(path))
    if failed:
        # A chunk that loses an experiment still keeps the rest, but the caller has to
        # know: a silent success here would hide a stack that needs attention.
        typer.echo(f"{len(failed)} of {len(paths) + len(failed)} experiments failed: "
                   f"{', '.join(failed)}", err=True)
        raise typer.Exit(1)


@app.command("reset")
def reset(
    flags_path: Path = typer.Option(None, help="Demo's src/flagd/demo.flagd.json."),
    vus: int = typer.Option(None, help="loadGeneratorVUs to restore; keeps the file's "
                                       "current value when omitted."),
    dry_run: bool = typer.Option(False, help="Print the commands; run nothing."),
) -> None:
    """Force the stack back to its fault-free baseline.

    Unpauses every container, deletes every netem qdisc, restores the 1.0 CPU baseline
    and puts every feature flag back to off. Idempotent and safe to run at any time. Run
    it after any interrupted campaign: an experiment killed mid-hold leaves its fault in
    place, and every experiment after it is then labelled as something it is not.
    """
    injector = FaultInjector(flags_path=flags_path, dry_run=dry_run)
    injector.reset(vus=vus)
    for command in injector.commands:
        typer.echo(command)
    typer.echo(f"reset: {len(injector.commands)} commands")


@app.command("ingest")
def ingest(
    capture_dir: Path = typer.Option(..., help="Host dir holding traces/metrics/logs.jsonl."),
    out_root: Path = typer.Option(..., help="Directory to write the experiment dir into."),
    experiment_id: str = typer.Option(..., help="Experiment id / output directory name."),
    start_ns: int = typer.Option(..., help="Window start, ns since epoch."),
    end_ns: int = typer.Option(..., help="Window end, ns since epoch."),
    seed: int = typer.Option(0),
    warmup_s: float = typer.Option(60.0, help="Fault-free baseline period inside the window."),
    ip_map: Path = typer.Option(
        None, help="JSON {ip: service}, as a campaign writes to manifest.extra['ip_map']. "
                   "Without it, client spans whose peer attribute is a container IP "
                   "(every gRPC call out of checkout) get no peer_service."),
) -> None:
    """Parse one capture window into a canonical experiment directory."""
    manifest = schema.Manifest(
        experiment_id=experiment_id,
        source="otel-demo",
        seed=seed,
        start_ns=start_ns,
        end_ns=end_ns,
        traffic=schema.TrafficProfile(base_rps=0.0, shape="steady"),
        faults=[],
        services=list(schema.SERVICES),
        edges=list(schema.DEPENDENCY_EDGES),
        metric_interval_ns=runner.METRIC_INTERVAL_NS,
        warmup_ns=int(warmup_s * 1_000_000_000),
        extra={"ip_map": json.loads(ip_map.read_text())} if ip_map else {},
    )
    path, report = ingest_mod.ingest(capture_dir, manifest, out_root)
    typer.echo(str(path))
    typer.echo(json.dumps(report.to_dict(), indent=2))
