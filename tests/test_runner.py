"""Campaign planning and experiment orchestration (with a fake clock, no Docker)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rca.benchmark import runner
from rca.benchmark.injector import FaultInjector, allowed_targets
from rca.data import schema

FIXTURES = Path(__file__).parent / "fixtures" / "otlp"
START_NS = 1_700_000_000_000_000_000


class FakeClock:
    """Advances only when the code under test sleeps."""

    def __init__(self, start_ns: int = START_NS) -> None:
        self.now = start_ns

    def sleep(self, seconds: float) -> None:
        self.now += int(seconds * 1_000_000_000)

    def now_ns(self) -> int:
        return self.now


def test_plan_campaign_is_deterministic_and_valid():
    specs = runner.plan_campaign(40, seed=7)
    assert specs == runner.plan_campaign(40, seed=7)
    assert runner.plan_campaign(40, seed=8) != specs
    assert len({s.experiment_id for s in specs}) == 40

    kinds = {s.kind for s in specs}
    assert kinds == {"fault", "normal", "traffic-only"}
    for spec in specs:
        assert spec.base_vus in runner.CampaignConfig().vus_choices
        # Every kind records the same shape, so the fault-free experiments contribute
        # as much evaluation time as the faulty ones.
        assert spec.start_offset_s >= spec.warmup_s
        assert spec.hold_s > 0
        if spec.fault is not None:
            assert spec.fault.target in allowed_targets(spec.fault.fault_type)
            assert 0.15 <= spec.fault.intensity <= 1.0
        if spec.kind == "traffic-only":
            assert spec.spike_vus is not None and spec.spike_vus > spec.base_vus


def test_campaign_wall_seconds_matches_the_plan():
    specs = runner.plan_campaign(5, seed=1)
    expected = sum(s.settle_s + s.duration_s for s in specs)
    assert runner.campaign_wall_seconds(specs) == expected


def test_every_kind_records_the_same_amount_of_time():
    """A normal experiment used to be warm-up + cool-down and nothing else.

    That made the fault-free half of the evaluation a fraction of the faulty half: 8
    windows per normal experiment against ~20 per fault one, so the false-alarm rate was
    estimated from very little data.
    """
    config = runner.CampaignConfig(hold_s_range=(50.0, 50.0),
                                   extra_offset_s_range=(10.0, 10.0))
    specs = runner.plan_campaign(40, seed=5, config=config)
    assert {s.kind for s in specs} == {"fault", "normal", "traffic-only"}
    # Fixed ranges, so every experiment lands on exactly the same window length.
    assert {s.duration_s for s in specs} == {config.warmup_s + 10.0 + 50.0 + config.cooldown_s}
    for spec in specs:
        assert spec.hold_s == 50.0, spec.kind
        assert spec.start_offset_s == config.warmup_s + 10.0, spec.kind


def test_run_experiment_writes_a_canonical_experiment(tmp_path):
    clock = FakeClock()
    injector = FaultInjector(dry_run=True)
    # run_experiment records where each capture file ends when the window opens, so the
    # fixture has to arrive *after* that, the way the collector appends during a run.
    capture = tmp_path / "capture"
    capture.mkdir()
    for name in runner.CAPTURE_FILES:
        (capture / name).write_bytes(b"")
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 2:             # the settle is over; offsets have been taken
            for name in runner.CAPTURE_FILES:
                (capture / name).write_bytes((FIXTURES / name).read_bytes())
        clock.sleep(seconds)

    spec = runner.ExperimentSpec(
        experiment_id="exp-0",
        seed=3,
        kind="fault",
        base_vus=10,
        settle_s=0.0,
        warmup_s=60.0,
        cooldown_s=30.0,
        start_offset_s=60.0,
        hold_s=40.0,
        fault=runner.FaultSpec("network_latency", "cart", 0.5),
    )
    out, report = runner.run_experiment(spec, tmp_path, injector, capture,
                                        sleep=sleep, now_ns=clock.now_ns)

    manifest = schema.read_manifest(out)
    assert manifest.source == "otel-demo"
    # rca.eval.metrics.experiment_categories buckets on this key.
    assert manifest.extra["category"] == "fault"
    assert manifest.start_ns == START_NS
    assert manifest.end_ns == START_NS + int((60 + 40 + 30) * 1e9)
    assert manifest.warmup_ns == 60_000_000_000
    assert len(manifest.faults) == 1
    fault = manifest.faults[0]
    assert fault.start_ns == START_NS + 60_000_000_000
    assert fault.end_ns == START_NS + 100_000_000_000
    assert fault.params["delay_ms"] == 425
    # Traffic level is set before the window and the fault is injected then cleared.
    assert injector.commands[0] == "flag loadGeneratorTraffic=on"
    assert injector.commands[1] == "flag loadGeneratorVUs=10"
    assert "netem delay 425ms" in injector.commands[2]
    assert injector.commands[3].endswith("tc qdisc del dev eth0 root")
    # The fixture capture falls inside the window and after the recorded offsets, so
    # ingest picked it up.
    assert report.spans_kept == 5
    assert manifest.extra["capture_offsets"] == dict.fromkeys(runner.CAPTURE_FILES, 0)
    assert schema.read_experiment(out).manifest.traffic.base_rps > 0


def test_traffic_only_experiment_has_no_fault(tmp_path):
    clock = FakeClock()
    injector = FaultInjector(dry_run=True)
    spec = runner.ExperimentSpec(
        experiment_id="exp-1", seed=4, kind="traffic-only", base_vus=5,
        settle_s=0.0, warmup_s=60.0, cooldown_s=30.0,
        start_offset_s=60.0, hold_s=60.0, spike_vus=50,
    )
    out, _ = runner.run_experiment(spec, tmp_path, injector, FIXTURES,
                                   sleep=clock.sleep, now_ns=clock.now_ns)
    manifest = schema.read_manifest(out)
    assert manifest.faults == []
    assert manifest.extra["category"] == "traffic_spike"
    assert manifest.traffic.shape == "bursty"
    assert injector.commands[2] == "flag loadGeneratorVUs=50"
    assert injector.commands[3] == "flag loadGeneratorVUs=5"


class Boom(RuntimeError):
    pass


def _spec(experiment_id="exp-f", **kwargs):
    return runner.ExperimentSpec(
        experiment_id=experiment_id, seed=3, kind="fault", base_vus=10,
        settle_s=0.0, warmup_s=60.0, cooldown_s=30.0, start_offset_s=60.0, hold_s=40.0,
        fault=runner.FaultSpec("network_latency", "cart", 0.5), **kwargs)


def test_a_failure_during_the_hold_still_clears_the_fault(tmp_path):
    clock = FakeClock()
    injector = FaultInjector(dry_run=True)
    holds = []

    def sleep(seconds):
        # Blow up exactly once, during the fault hold.
        if seconds == 40.0 and not holds:
            holds.append(seconds)
            raise Boom("interrupted mid-hold")
        clock.sleep(seconds)

    with pytest.raises(Boom):
        runner.run_experiment(_spec(), tmp_path, injector, FIXTURES,
                              sleep=sleep, now_ns=clock.now_ns)
    # Without the try/finally the qdisc would still be on cart and every later
    # experiment would carry an unlabelled network fault.
    assert injector.commands[-1].endswith("tc qdisc del dev eth0 root")


def test_a_failure_during_a_traffic_spike_restores_the_base_level(tmp_path):
    clock = FakeClock()
    injector = FaultInjector(dry_run=True)
    spec = runner.ExperimentSpec(
        experiment_id="exp-s", seed=4, kind="traffic-only", base_vus=5,
        settle_s=0.0, warmup_s=60.0, cooldown_s=30.0,
        start_offset_s=60.0, hold_s=60.0, spike_vus=50)

    def sleep(seconds):
        if seconds == 60.0 and "flag loadGeneratorVUs=50" in injector.commands:
            raise Boom("interrupted mid-spike")
        clock.sleep(seconds)

    with pytest.raises(Boom):
        runner.run_experiment(spec, tmp_path, injector, FIXTURES,
                              sleep=sleep, now_ns=clock.now_ns)
    assert injector.commands[-1] == "flag loadGeneratorVUs=5"


def test_manifest_records_capture_offsets_and_clear_failures(tmp_path):
    clock = FakeClock()
    injector = FaultInjector(dry_run=True)
    out, _ = runner.run_experiment(_spec("exp-o"), tmp_path, injector, FIXTURES,
                                   sleep=clock.sleep, now_ns=clock.now_ns)
    extra = schema.read_manifest(out).extra
    # Where in each capture file this window starts, so ingest need not read from 0.
    assert set(extra["capture_offsets"]) == {"traces.jsonl", "metrics.jsonl", "logs.jsonl"}
    assert all(v >= 0 for v in extra["capture_offsets"].values())
    assert extra["clear_failures"] == []


def test_campaign_isolates_a_failing_experiment_and_resets_between_them(tmp_path):
    clock = FakeClock()
    injector = FaultInjector(dry_run=True)
    specs = runner.plan_campaign(3, seed=11, prefix="c")
    doomed = specs[1].experiment_id
    real_run = runner.run_experiment

    def run(spec, *args, **kwargs):
        if spec.experiment_id == doomed:
            raise Boom("ingest exploded")
        return real_run(spec, *args, **kwargs)

    runner.run_experiment = run
    try:
        paths, failed = runner.run_campaign(3, 11, tmp_path, injector, FIXTURES,
                                            prefix="c", sleep=clock.sleep,
                                            now_ns=clock.now_ns)
    finally:
        runner.run_experiment = real_run

    # One bad experiment must not cost the other two.
    assert failed == [doomed]
    assert len(paths) == 2
    assert doomed not in {p.name for p in paths}
    # A reset before the first experiment, after each one, and once at the end.
    assert injector.commands.count("docker unpause " + " ".join(
        sorted(set(inj_containers())))) == 5


def inj_containers():
    from rca.benchmark.injector import CONTAINERS
    return set(CONTAINERS.values())
