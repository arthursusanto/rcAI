"""``rca serve`` (incident API + UI) and ``rca replay`` (batch streaming replay)."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from rca.data.schema import read_manifest
from rca.models.base import TwoStageModel
from rca.serve.api import create_app
from rca.serve.store import IncidentStore
from rca.serve.stream import DEFAULT_WINDOW_NS, ReplaySource, StreamingDetector

serve_app = typer.Typer(help="Serve the incident API and UI over a live or replayed feed.")
replay_app = typer.Typer(help="Replay one experiment through the streaming detector.")

NS_PER_S = 1e9


@serve_app.callback(invoke_without_command=True)
def serve(
    model: Path = typer.Option(..., help="Directory holding model.joblib."),
    replay: Path | None = typer.Option(None, help="Experiment directory to replay."),
    speed: float = typer.Option(1.0, help="Replay speed; 0 is as fast as possible."),
    capture: Path | None = typer.Option(None, help="Collector file-exporter directory (live)."),
    out: Path | None = typer.Option(None, help="Directory for incident/window JSONL."),
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Bind port."),
    window: float = typer.Option(10.0, help="Window width, seconds."),
    warmup: float | None = typer.Option(
        None, help="Warm-up seconds; defaults to the manifest (replay) or 120 s (live)."
    ),
    debounce_open: int = typer.Option(1, help="Consecutive positives that open an incident."),
    debounce_close: int = typer.Option(3, help="Consecutive negatives that close one."),
    refresh_windows: int | None = typer.Option(
        None, help="Re-fit the baseline every N windows; 0 disables. "
                   "Default: 60 for --capture (live), off for --replay."
    ),
) -> None:
    """Run the incident API; the UI is at ``/``."""
    import uvicorn

    if replay is not None and capture is not None:
        raise typer.BadParameter("--replay and --capture are mutually exclusive")
    app = create_app(
        TwoStageModel.load(model), replay=replay, speed=speed, capture=capture, out_dir=out,
        window_ns=int(window * NS_PER_S),
        warmup_ns=None if warmup is None else int(warmup * NS_PER_S),
        debounce_open=debounce_open, debounce_close=debounce_close,
        refresh_windows=refresh_windows,
    )
    typer.echo(f"serving on http://{host}:{port}/  (model {model})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


@replay_app.callback(invoke_without_command=True)
def replay(
    model: Path = typer.Option(..., help="Directory holding model.joblib."),
    experiment: Path = typer.Option(..., help="Experiment directory to replay."),
    out: Path | None = typer.Option(None, help="Directory for the replay artefacts."),
    window: float = typer.Option(10.0, help="Window width, seconds."),
    warmup: float | None = typer.Option(None, help="Warm-up seconds; default the manifest."),
    debounce_open: int = typer.Option(1, help="Consecutive positives that open an incident."),
    debounce_close: int = typer.Option(3, help="Consecutive negatives that close one."),
) -> None:
    """Run the streaming detector over one experiment without a server and report.

    This is the check that the streaming path reproduces the offline evaluation: same
    features, same model, same verdicts, plus the latency the online path actually costs.
    """
    summary = replay_experiment(
        TwoStageModel.load(model), experiment, out_dir=out,
        window_ns=int(window * NS_PER_S),
        warmup_ns=None if warmup is None else int(warmup * NS_PER_S),
        debounce_open=debounce_open, debounce_close=debounce_close,
    )
    typer.echo(format_replay(summary))
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "summary.json").write_text(
            json.dumps(summary, indent=2), newline="\n")
        typer.echo(f"wrote replay artefacts to {out}")


def replay_experiment(model, exp_dir: Path | str, out_dir: Path | str | None = None,
                      window_ns: int = DEFAULT_WINDOW_NS, warmup_ns: int | None = None,
                      debounce_open: int = 1, debounce_close: int = 3) -> dict:
    """Replay one experiment as fast as possible and summarise what the system said."""
    source = ReplaySource(exp_dir, speed=0.0)
    store = IncidentStore(out_dir)
    detector = StreamingDetector(
        model, source.manifest, window_ns=window_ns, warmup_ns=warmup_ns, store=store,
        debounce_open=debounce_open, debounce_close=debounce_close,
    )
    records: list[dict] = []
    for frame in source.frames():
        records.extend(detector.push(frame))
    return replay_summary(detector, records, read_manifest(Path(exp_dir)))


def replay_summary(detector: StreamingDetector, records: list[dict], manifest) -> dict:
    """Detection timing against ground truth, the two rankings, and latency."""
    fault = manifest.faults[0] if manifest.faults else None
    positives = [r for r in records if r["detect"]]
    during = [r for r in positives
              if fault is not None and r["window_end_ns"] > fault.start_ns
              and r["window_start_ns"] < fault.end_ns]
    first = during[0] if during else (positives[0] if positives else None)
    lag_ns = detector.lag_windows * detector.window_ns

    detection = {"detected": first is not None}
    if fault is not None and first is not None:
        detection.update({
            # Same convention as rca.eval.metrics.detection_delay.
            "detection_delay_s": max(0.0, (first["window_start_ns"] - fault.start_ns) / NS_PER_S),
            # When the streaming path could actually alert: window end plus its grace.
            "alert_delay_s": (first["window_end_ns"] + lag_ns - fault.start_ns) / NS_PER_S,
            "root_at_detection": first["root_top1"],
            "ranking_at_detection": list(zip(first["ranked_services"][:3],
                                             first["root_scores"][:3])),
            "fault_type_at_detection": first["fault_type_pred"],
            "detect_prob_at_detection": first["detect_prob"],
            "correct_at_detection": first["root_top1"] == _owner_of(fault.target),
        })
    final = positives[-1] if positives else (records[-1] if records else None)
    if final is not None:
        detection.update({
            "final_root": final["root_top1"],
            "final_ranking": list(zip(final["ranked_services"][:3], final["root_scores"][:3])),
            "final_fault_type": final["fault_type_pred"],
            "final_margin": final["root_margin"],
        })
    return {
        "experiment_id": manifest.experiment_id,
        "ground_truth": None if fault is None else {
            "fault_type": fault.fault_type, "target": _owner_of(fault.target),
            "start_ns": fault.start_ns, "end_ns": fault.end_ns,
            "duration_s": (fault.end_ns - fault.start_ns) / NS_PER_S,
            "intensity": fault.intensity,
        },
        "windows_scored": len(records),
        "positive_windows": len(positives),
        "detection": detection,
        "incidents": [incident.summary() for incident in detector.incidents],
        "latency": detector.latency_stats(),
    }


def _owner_of(service: str) -> str:
    from rca.data.schema import INFRA_OWNER
    return INFRA_OWNER.get(service, service)


def _mark(what: str, correct: bool | None) -> str:
    """`` [root correct]`` / `` [root WRONG]``; empty when there is no ground truth."""
    if correct is None:
        return ""
    return f"  [{what} correct]" if correct else f"  [{what} WRONG]"


def format_replay(summary: dict) -> str:
    truth, detection = summary["ground_truth"], summary["detection"]
    header = (f"experiment {summary['experiment_id']}: {summary['windows_scored']} windows "
              f"scored, {summary['positive_windows']} positive")
    lines = [header]
    if truth is None:
        lines.append("ground truth: no fault (this is a negative experiment)")
    else:
        lines.append(f"ground truth: {truth['fault_type']} on {truth['target']} "
                     f"for {truth['duration_s']:.0f}s at intensity {truth['intensity']:.2f}")
    if not detection["detected"]:
        lines.append("detection: never flagged")
    else:
        if "detection_delay_s" in detection:
            lines.append(
                f"detection: {detection['detection_delay_s']:.1f}s after fault start "
                f"(alertable at +{detection['alert_delay_s']:.1f}s), "
                f"p={detection['detect_prob_at_detection']:.3f}")
            # The root and the fault class are graded separately. A single marker after
            # both used to read as a verdict on the fault, which it never was.
            lines.append(
                "ranking at detection: " + _ranking(detection["ranking_at_detection"])
                + _mark("root", None if truth is None
                        else detection["correct_at_detection"])
                + f"  -> fault {detection['fault_type_at_detection']}"
                + _mark("fault", None if truth is None
                        else detection["fault_type_at_detection"] == truth["fault_type"]))
        lines.append(
            "final ranking:        " + _ranking(detection["final_ranking"])
            + _mark("root", None if truth is None
                    else detection["final_root"] == truth["target"])
            + f"  -> fault {detection['final_fault_type']}"
            + _mark("fault", None if truth is None
                    else detection["final_fault_type"] == truth["fault_type"])
            + f", margin {detection['final_margin']:.3f}")
    latency = summary["latency"]
    if latency.get("n"):
        for key in ("feature_ms", "predict_ms", "total_ms"):
            stat = latency[key]
            lines.append(f"latency {key:<11} mean {stat['mean']:7.2f} ms  p50 {stat['p50']:7.2f}"
                         f"  p95 {stat['p95']:7.2f}  max {stat['max']:7.2f}  (n={latency['n']})")
    for incident in summary["incidents"]:
        lines.append(f"incident {incident['incident_id']}: {incident['n_windows']} windows, "
                     f"root {incident['root_top1']}, "
                     f"{'open' if incident['is_open'] else 'closed'}")
    return "\n".join(lines)


def _ranking(pairs) -> str:
    return ", ".join(f"{name} {score:.3f}" for name, score in pairs)
