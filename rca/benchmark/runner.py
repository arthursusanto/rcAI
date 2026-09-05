"""Experiment and campaign orchestration against a running OpenTelemetry Demo.

One experiment is a single window of the collector's append-only capture files:

    set traffic level
    settle_s          -- not recorded; lets the stack reach steady state
    ---- start_ns ----
    warmup_s          -- recorded, fault-free, the per-experiment baseline
    ... start_offset_s (>= warmup_s) ...
    inject fault
    hold_s
    clear fault
    cooldown_s
    ---- end_ns ------
    ingest [start_ns, end_ns] -> canonical experiment directory

Normal experiments run the same shape with no fault; traffic-only experiments swap the
fault for a load-generator VU spike, so "high load" is available as a hard negative.

``plan_campaign`` is pure and seeded: the same seed always yields the same list of
specs, which is what makes a campaign reproducible and what the tests exercise.
"""
from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rca.benchmark import ingest as ingest_mod
from rca.benchmark.injector import (
    DEFAULT_CPU_MECHANISM,
    VU_VARIANTS,
    FaultInjector,
    allowed_targets,
    container_ip_map,
)
from rca.data import schema

log = logging.getLogger(__name__)

CAPTURE_FILES = ("traces.jsonl", "metrics.jsonl", "logs.jsonl")

# docker_stats collection_interval in deploy/compose/otelcol-config-extras.yml.
METRIC_INTERVAL_NS = 5_000_000_000
_S = 1_000_000_000

# ExperimentSpec.kind -> manifest.extra["category"], the vocabulary rca.eval and
# rca.sim share.
_CATEGORY = {"fault": "fault", "normal": "normal", "traffic-only": "traffic_spike"}


@dataclass
class FaultSpec:
    """What happens during the hold. *When* it happens is the experiment's business."""

    fault_type: str
    target: str
    intensity: float


@dataclass
class ExperimentSpec:
    experiment_id: str
    seed: int
    kind: str                   # "fault" | "normal" | "traffic-only"
    base_vus: int
    settle_s: float = 30.0
    warmup_s: float = 60.0
    cooldown_s: float = 60.0
    # The recorded window has the same shape for every kind: a fault-free warm-up, then
    # something happens at start_offset_s (>= warmup_s) and lasts hold_s, then a
    # cool-down. A "normal" experiment is one where that something is nothing.
    #
    # Uniform on purpose. While only fault experiments held for hold_s, a normal
    # experiment recorded warm-up + cool-down and no more -- about 80 s, 8 windows -- so
    # the negatives were a small fraction of the evaluation time and the false-alarm
    # rate was being estimated from very little fault-free data.
    start_offset_s: float = 60.0
    hold_s: float = 60.0
    fault: FaultSpec | None = None
    spike_vus: int | None = None

    @property
    def duration_s(self) -> float:
        """Length of the recorded window; the same shape for every kind."""
        return self.start_offset_s + self.hold_s + self.cooldown_s


def run_experiment(
    spec: ExperimentSpec,
    out_root: Path | str,
    injector: FaultInjector,
    capture_dir: Path | str,
    sleep: Callable[[float], None] = time.sleep,
    now_ns: Callable[[], int] = time.time_ns,
) -> tuple[Path, ingest_mod.IngestReport]:
    """Run one experiment end to end and write its canonical directory."""
    # Container IPs are stable while the stack is up but change on every recreate, so
    # the map is captured per experiment rather than per campaign.
    ip_map = {} if injector.dry_run else container_ip_map(injector.docker)
    injector.set_traffic(vus=spec.base_vus, enabled=True)
    sleep(spec.settle_s)

    start_ns = now_ns()
    # Byte position of each capture file now, so ingest parses only this window's bytes
    # instead of the whole campaign's file. Recorded after the settle so nothing from
    # the previous experiment's tail is in range.
    offsets = _capture_offsets(capture_dir)
    faults: list[schema.Fault] = []

    sleep(spec.start_offset_s)
    if spec.fault is not None:
        injected_ns = now_ns()
        # end_ns is planned up front because the injector derives self-limiting
        # in-container loop durations from it; it is corrected to the real clear time
        # once the fault is cleared.
        fault = schema.Fault(
            fault_type=spec.fault.fault_type,
            target=spec.fault.target,
            start_ns=injected_ns,
            end_ns=injected_ns + int(spec.hold_s * _S),
            intensity=spec.fault.intensity,
        )
        faults.append(fault)
        try:
            injector.inject(fault)
            sleep(spec.hold_s)
        finally:
            # Ctrl-C or a failure anywhere in the hold must still take the fault out:
            # a netem qdisc or a paused container left behind would silently contaminate
            # every experiment after this one.
            injector.clear(fault)
            fault.end_ns = now_ns()
    elif spec.spike_vus is not None:
        try:
            injector.set_traffic(vus=spec.spike_vus)
            sleep(spec.hold_s)
        finally:
            injector.set_traffic(vus=spec.base_vus)
    else:
        # A normal experiment holds for exactly as long, doing nothing.
        sleep(spec.hold_s)

    sleep(spec.cooldown_s)
    end_ns = now_ns()

    manifest = schema.Manifest(
        experiment_id=spec.experiment_id,
        source="otel-demo",
        seed=spec.seed,
        start_ns=start_ns,
        end_ns=end_ns,
        traffic=schema.TrafficProfile(
            base_rps=0.0,
            shape="bursty" if spec.spike_vus is not None else "steady",
            params={"vus": spec.base_vus, "spike_vus": spec.spike_vus,
                    "start_offset_s": spec.start_offset_s, "hold_s": spec.hold_s},
        ),
        faults=faults,
        services=list(schema.SERVICES),
        edges=list(schema.DEPENDENCY_EDGES),
        metric_interval_ns=METRIC_INTERVAL_NS,
        warmup_ns=int(spec.warmup_s * _S),
        # rca.eval.metrics.experiment_categories reads extra["category"]; use the
        # simulator's vocabulary ("fault" | "normal" | "traffic_spike") so real and
        # simulated runs land in the same buckets.
        extra={"category": _CATEGORY[spec.kind], "spec": asdict(spec),
               # ingest resolves IP-valued peer attributes through this.
               "ip_map": ip_map,
               # ... and starts reading each capture file here.
               "capture_offsets": offsets,
               # Non-empty means a clear command failed and the stack may have been
               # dirty for part of this experiment.
               "clear_failures": [f for fault in faults
                                  for f in fault.params.get("clear_failures", [])]},
    )
    experiment, report = ingest_mod.build_experiment(capture_dir, manifest)
    manifest.traffic.base_rps = _measured_rps(experiment)
    manifest.extra["ingest_report"] = report.to_dict()
    return schema.write_experiment(experiment, Path(out_root)), report


def _capture_offsets(capture_dir: Path | str) -> dict[str, int]:
    """Current size of each capture file, i.e. where this experiment's window starts."""
    capture = Path(capture_dir)
    return {name: os.path.getsize(capture / name)
            for name in CAPTURE_FILES if (capture / name).exists()}


def _measured_rps(experiment: schema.Experiment) -> float:
    """Entry-point request rate, from frontend-proxy server spans."""
    seconds = (experiment.manifest.end_ns - experiment.manifest.start_ns) / _S
    spans = experiment.spans
    if seconds <= 0 or spans.empty:
        return 0.0
    entry = spans[(spans["service"] == "frontend-proxy") & (spans["kind"] == "server")]
    return round(len(entry) / seconds, 3)


# --- campaign -------------------------------------------------------------------------
@dataclass
class CampaignConfig:
    """Randomization ranges shared by every experiment in a campaign."""

    warmup_s: float = 60.0
    settle_s: float = 30.0
    cooldown_s: float = 60.0
    hold_s_range: tuple[float, float] = (60.0, 180.0)
    extra_offset_s_range: tuple[float, float] = (0.0, 60.0)
    intensity_range: tuple[float, float] = (0.15, 1.0)
    fault_fraction: float = 0.70
    normal_fraction: float = 0.15      # remainder is traffic-only
    fault_types: list[str] = field(default_factory=lambda: list(schema.FAULT_TYPES))
    vus_choices: tuple[int, ...] = VU_VARIANTS
    # Must match the injector's: it decides which services cpu_saturation can reach.
    cpu_mechanism: str = DEFAULT_CPU_MECHANISM


def plan_campaign(n: int, seed: int, config: CampaignConfig | None = None,
                  prefix: str = "otel") -> list[ExperimentSpec]:
    """Deterministically randomize n experiment specs.

    Randomized per experiment: traffic level (the load generator's ``loadGeneratorVUs``
    flag), experiment kind, fault type, target service (only from the services the
    family can actually be injected on), intensity, start offset and hold time.

    The offset and the hold are drawn for *every* kind, so a normal experiment records
    as much wall time as a fault one and the fault-free half of the evaluation is not a
    rounding error.
    """
    config = config or CampaignConfig()
    rng = random.Random(seed)
    specs: list[ExperimentSpec] = []
    for i in range(n):
        sub_seed = rng.getrandbits(32)
        sub = random.Random(sub_seed)
        roll = sub.random()
        kind = ("fault" if roll < config.fault_fraction
                else "normal" if roll < config.fault_fraction + config.normal_fraction
                else "traffic-only")
        base_vus = sub.choice(config.vus_choices)
        spec = ExperimentSpec(
            experiment_id=f"{prefix}-{seed}-{i:04d}",
            seed=sub_seed,
            kind=kind,
            base_vus=base_vus,
            settle_s=config.settle_s,
            warmup_s=config.warmup_s,
            cooldown_s=config.cooldown_s,
            start_offset_s=config.warmup_s + round(
                sub.uniform(*config.extra_offset_s_range), 1),
            hold_s=round(sub.uniform(*config.hold_s_range), 1),
        )
        if kind == "fault":
            fault_type = sub.choice(config.fault_types)
            spec.fault = FaultSpec(
                fault_type=fault_type,
                target=sub.choice(allowed_targets(fault_type, config.cpu_mechanism)),
                intensity=round(sub.uniform(*config.intensity_range), 3),
            )
        elif kind == "traffic-only":
            # The spike must be upward, so never start from the highest VU level.
            spec.base_vus = base_vus = sub.choice(config.vus_choices[:-1])
            spec.spike_vus = sub.choice([v for v in config.vus_choices if v > base_vus])
        specs.append(spec)
    return specs


def run_campaign(
    n: int,
    seed: int,
    out_root: Path | str,
    injector: FaultInjector,
    capture_dir: Path | str,
    config: CampaignConfig | None = None,
    prefix: str = "otel",
    sleep: Callable[[float], None] = time.sleep,
    now_ns: Callable[[], int] = time.time_ns,
) -> tuple[list[Path], list[str]]:
    """Run a whole campaign.

    Returns the experiment directories written and the ids that failed. One experiment
    blowing up -- a docker hiccup, an ingest error -- must not cost the rest of the
    chunk, so each is isolated and followed by a reset; the caller decides what a
    non-empty failure list means.
    """
    out: list[Path] = []
    failed: list[str] = []
    specs = plan_campaign(n, seed, config, prefix)
    injector.reset(vus=specs[0].base_vus if specs else None)
    try:
        for spec in specs:
            try:
                path, _ = run_experiment(spec, out_root, injector, capture_dir,
                                         sleep, now_ns)
                out.append(path)
            except Exception:
                log.exception("experiment %s failed", spec.experiment_id)
                failed.append(spec.experiment_id)
            finally:
                # Between experiments as well as after a failure: whatever the last one
                # left behind stops here rather than mislabelling the next.
                injector.reset(vus=spec.base_vus)
    finally:
        injector.reset()
    return out, failed


def campaign_wall_seconds(specs: list[ExperimentSpec]) -> float:
    return sum(s.settle_s + s.duration_s for s in specs)
