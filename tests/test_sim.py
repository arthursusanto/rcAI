"""Tests for the telemetry simulator."""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from rca.benchmark.injector import MIN_CPUS, allowed_targets
from rca.data import schema
from rca.data.splits import split_by_time
from rca.sim import FaultSpec, generate, generate_dataset, generate_experiment, sample_fault
from rca.sim.faults import (
    FAULT_TARGET_PAIRS,
    VALID_TARGETS,
    cpu_mechanism_of,
    valid_targets,
)
from rca.sim.topology import ALL_COMPONENTS

STEADY = schema.TrafficProfile(base_rps=20.0, shape="steady", params={})
# The calibrated request mix makes checkout (and so payment, email, quote) a few percent
# of traffic, which is the real regime but leaves too few spans to take a stable median
# of one dependency edge. Tests that measure such an edge run the system busier.
BUSY = schema.TrafficProfile(base_rps=30.0, shape="steady", params={})


def _mean_metric(exp, service, metric, a_s, b_s):
    m = exp.metrics
    t = (m["ts_ns"] - exp.manifest.start_ns) / 1e9
    sel = m[(m["service"] == service) & (m["metric"] == metric) & (t >= a_s) & (t < b_s)]
    return float(sel["value"].mean())


def _mean_latency_ms(exp, service, kind, a_s, b_s):
    s = exp.spans
    t = (s["start_ns"] - exp.manifest.start_ns) / 1e9
    sel = s[(s["service"] == service) & (s["kind"] == kind) & (t >= a_s) & (t < b_s)]
    return float(sel["duration_ns"].mean()) / 1e6


def test_determinism():
    spec = lambda: FaultSpec("cpu_saturation", "cart", 50.0, 25.0, 0.7)
    a = generate_experiment(11, 90.0, spec(), STEADY, warmup_s=30.0)
    b = generate_experiment(11, 90.0, spec(), STEADY, warmup_s=30.0)
    pd.testing.assert_frame_equal(a.spans, b.spans)
    pd.testing.assert_frame_equal(a.metrics, b.metrics)
    pd.testing.assert_frame_equal(a.logs, b.logs)
    assert a.manifest.to_json() == b.manifest.to_json()

    c = generate_experiment(12, 90.0, spec(), STEADY, warmup_s=30.0)
    assert len(c.spans) != len(a.spans) or not c.spans.equals(a.spans)


def test_schema_round_trip(tmp_path):
    exp = generate_experiment(5, 90.0, FaultSpec("network_latency", "payment", 45.0, 30.0, 0.8),
                              STEADY, warmup_s=30.0)
    d = schema.write_experiment(exp, tmp_path)
    assert d.name == "sim-000005"
    back = schema.read_experiment(d)

    for df, columns in ((back.metrics, schema.METRICS_COLUMNS),
                        (back.spans, schema.SPANS_COLUMNS),
                        (back.logs, schema.LOGS_COLUMNS)):
        assert list(df.columns) == list(columns)
        assert len(df) > 0

    assert set(back.spans["kind"]) <= {"server", "client", "producer", "consumer", "internal"}
    assert set(back.logs["severity"]) <= set(schema.SEVERITIES)
    assert set(back.metrics["metric"]) == set(schema.METRIC_NAMES)
    assert set(back.metrics["service"]) == set(ALL_COMPONENTS)

    m = back.manifest
    assert m.source == "sim"
    assert m.seed == 5
    assert m.start_ns < m.end_ns
    assert m.warmup_ns == int(30e9)
    assert m.extra["sim_version"]
    assert len(m.faults) == 1
    f = m.faults[0]
    assert f.fault_type == "network_latency" and f.target == "payment"
    assert m.start_ns < f.start_ns < f.end_ns <= m.end_ns
    # timestamps stay inside the experiment window
    for df in (back.metrics, back.logs):
        assert df["ts_ns"].between(m.start_ns, m.end_ns).all()
    assert back.spans["start_ns"].between(m.start_ns, m.end_ns).all()


def test_spans_form_valid_trees():
    exp = generate_experiment(21, 120.0, FaultSpec("packet_loss", "cart", 60.0, 40.0, 0.9),
                              STEADY)
    s = exp.spans
    assert s["span_id"].is_unique

    trace_of = dict(zip(s["span_id"], s["trace_id"]))
    child = s[s["parent_span_id"] != ""]
    assert child["parent_span_id"].isin(trace_of).all(), "dangling parent span"
    assert (child["parent_span_id"].map(trace_of).to_numpy()
            == child["trace_id"].to_numpy()).all(), "parent in a different trace"

    parents = s.set_index("span_id")[["start_ns", "duration_ns"]]
    j = child.join(parents, on="parent_span_id", rsuffix="_p")
    inside = ((j["start_ns"] >= j["start_ns_p"])
              & (j["start_ns"] + j["duration_ns"] <= j["start_ns_p"] + j["duration_ns_p"]))
    # kafka consumer spans are asynchronous: they are children of the producer span but
    # run after it, so only synchronous spans must be nested inside their parent.
    assert inside[j["kind"] != "consumer"].all()
    assert (j.loc[j["kind"] == "consumer", "start_ns"]
            >= j.loc[j["kind"] == "consumer", "start_ns_p"]).all()

    # exactly one root per trace
    assert (s["parent_span_id"] == "").sum() == s["trace_id"].nunique()
    assert (s["duration_ns"] > 0).all()


LIGHT = schema.TrafficProfile(base_rps=5.0, shape="steady", params={})
HOG_DUTY_AT_0_9 = 0.87                  # lerp(*HOG_DUTY_RANGE, 0.9)


def _p95_ms(exp, mask_fn, a_s, b_s):
    s = exp.spans
    t = (s["start_ns"] - exp.manifest.start_ns) / 1e9
    sel = s[mask_fn(s) & (t >= a_s) & (t < b_s)]
    assert len(sel) > 20, "not enough spans to compare"
    return float(np.percentile(sel["duration_ns"], 95)) / 1e6


CPU_TARGETS = ["cart", "image-provider"]        # a busy one and a nearly idle one
WARM_W, FAULT_W = (0.0, 60.0), (85.0, 145.0)


def _cpu_saturation(target, intensity, traffic=None, seed=31):
    spec = FaultSpec("cpu_saturation", target, 70.0, 80.0, intensity)
    exp = generate_experiment(seed, 200.0, spec, traffic or BUSY)
    return spec, exp


@pytest.mark.parametrize("target", CPU_TARGETS)
@pytest.mark.parametrize("intensity", [0.3, 0.75, 1.0])
def test_cpu_saturation_pins_cpu_util_at_every_intensity(target, intensity):
    """Intensity sets the quota; the hog is sized to fill whatever the service leaves.

    So the cgroup runs at its limit whatever the intensity -- what intensity changes is
    how long the CFS throttling gap is, not how full the cgroup is.
    """
    spec, exp = _cpu_saturation(target, intensity)
    assert spec.params["mechanism"] == "hog+quota"
    quota, duty = spec.params["cpus"], spec.params["hog_duty"]
    assert quota == pytest.approx(max(MIN_CPUS, 0.8 - 0.75 * intensity), abs=0.005)
    assert 0.0 <= duty <= quota
    assert quota - duty >= MIN_CPUS - 1e-9        # the service keeps at least that much

    cpu_warm = _mean_metric(exp, target, schema.METRIC_CPU_UTIL, *WARM_W)
    cpu_fault = _mean_metric(exp, target, schema.METRIC_CPU_UTIL, *FAULT_W)
    assert cpu_warm < 0.2
    assert 0.9 <= cpu_fault <= 1.6

    # an unrelated service keeps its own CPU
    assert (_mean_metric(exp, "email", schema.METRIC_CPU_UTIL, *FAULT_W)
            < _mean_metric(exp, "email", schema.METRIC_CPU_UTIL, *WARM_W) + 0.1)


@pytest.mark.parametrize("target", CPU_TARGETS)
def test_cpu_saturation_latency_grows_with_intensity(target):
    """A tighter quota is a longer stall, so tail latency rises monotonically."""
    server = (lambda s: (s["service"] == target) & (s["kind"] == "server"))
    p95 = []
    for intensity in (0.3, 0.75, 1.0):
        _, exp = _cpu_saturation(target, intensity)
        healthy = _p95_ms(exp, server, *WARM_W)
        p95.append(_p95_ms(exp, server, *FAULT_W))
        assert p95[-1] > 3.0 * healthy, f"{target} at {intensity}"
    assert p95[0] < p95[1] < p95[2], f"{target}: {p95}"


def test_cpu_saturation_stall_is_elapsed_time_not_work():
    """The cgroup is frozen, not busy: the stall must not show up as CPU demand."""
    _, mild = _cpu_saturation("image-provider", 0.3)
    _, hard = _cpu_saturation("image-provider", 1.0)
    server = (lambda s: (s["service"] == "image-provider") & (s["kind"] == "server"))
    assert _p95_ms(hard, server, *FAULT_W) > _p95_ms(mild, server, *FAULT_W)
    # far more elapsed time, but the work behind it is unchanged, so the reported
    # utilisation stays pinned rather than exploding
    for exp in (mild, hard):
        assert _mean_metric(exp, "image-provider", schema.METRIC_CPU_UTIL, *FAULT_W) <= 1.6


def test_cpu_saturation_propagates_to_callers():
    spec = FaultSpec("cpu_saturation", "cart", 70.0, 80.0, 0.9)
    exp = generate_experiment(31, 200.0, spec, BUSY)
    warm, fault = (0.0, 60.0), (85.0, 145.0)

    # only the requests that go through cart slow down, and cart is a small part of
    # their time, so the rise upstream is real but much smaller than at the target
    for service in ("frontend", "frontend-proxy"):
        via_cart = (lambda s, c=service: (s["service"] == c) & (s["kind"] == "server")
                    & (s["operation"] == "GET /api/cart"))
        assert _median_ms(exp, via_cart, *fault) > 1.15 * _median_ms(exp, via_cart, *warm), service

    # the product page never touches cart and must not move
    browse = (lambda s: (s["service"] == "frontend") & (s["kind"] == "server")
              & (s["operation"] == "GET /api/products"))
    assert _median_ms(exp, browse, *fault) < 1.1 * _median_ms(exp, browse, *warm)


def test_cpu_saturation_quota_mechanism_stays_available():
    """The quota-only squeeze is still selectable, and still reaches every container."""
    assert "product-catalog" not in VALID_TARGETS["cpu_saturation"]
    assert "product-catalog" in valid_targets("cpu_saturation", "quota-only")

    spec = FaultSpec("cpu_saturation", "product-catalog", 70.0, 80.0, 0.9,
                     {"mechanism": "quota-only"})
    exp = generate_experiment(31, 200.0, spec, BUSY)
    assert spec.params["cpus"] == pytest.approx(0.15, abs=0.01)
    assert "hog_duty" not in spec.params
    warm, fault = (0.0, 60.0), (85.0, 145.0)
    cpu_warm = _mean_metric(exp, "product-catalog", schema.METRIC_CPU_UTIL, *warm)
    cpu_fault = _mean_metric(exp, "product-catalog", schema.METRIC_CPU_UTIL, *fault)
    assert cpu_fault > 3.0 * cpu_warm

    with pytest.raises(ValueError):
        generate_experiment(1, 200.0, FaultSpec("cpu_saturation", "cart", 70.0, 60.0, 0.5,
                                                {"mechanism": "nice"}), BUSY)
    # a manifest written by the real injector resolves too
    assert cpu_mechanism_of({"mechanism": "exec-cpu-hog-quota"}) == "hog+quota"
    assert cpu_mechanism_of({"mechanism": "exec-cpu-hog"}) == "hog-only"
    assert cpu_mechanism_of({"mechanism": "docker-update-cpus"}) == "quota-only"


def test_queue_backlog_raises_consumer_queue_depth():
    spec = FaultSpec("queue_backlog", "accounting", 60.0, 60.0, 0.9)
    exp = generate_experiment(33, 150.0, spec, BUSY)
    warm = _mean_metric(exp, "accounting", schema.METRIC_QUEUE_DEPTH, 0.0, 30.0)
    fault = _mean_metric(exp, "accounting", schema.METRIC_QUEUE_DEPTH, 75.0, 115.0)
    assert warm < 1.0 and fault > 5.0

    # the other consumer is untouched and the synchronous path is unaffected
    other = _mean_metric(exp, "fraud-detection", schema.METRIC_QUEUE_DEPTH, 75.0, 115.0)
    assert other < 1.0
    assert (_mean_latency_ms(exp, "frontend", "server", 75.0, 115.0)
            < 1.3 * _mean_latency_ms(exp, "frontend", "server", 0.0, 30.0))


def test_normal_experiment_has_no_fault():
    exp = generate_experiment(41, 90.0, None, STEADY)
    assert exp.manifest.faults == []
    assert len(exp.spans) > 0
    error_rate = exp.spans["status_error"].mean()
    assert error_rate < 0.05


def test_valid_targets_come_from_the_injector():
    """VALID_TARGETS is the injector's capability narrowed by the sim's structure."""
    for fault_type, targets in VALID_TARGETS.items():
        assert targets and set(targets) <= set(schema.SERVICES)
        assert set(targets) <= set(allowed_targets(fault_type)), fault_type
        assert targets == [s for s in schema.SERVICES if s in targets], "SERVICES order"

    # families the injector limits
    assert VALID_TARGETS["error_rate"] == ["cart", "payment", "ad"]
    assert set(VALID_TARGETS["memory_leak"]) == set(allowed_targets("memory_leak"))
    assert "checkout" not in VALID_TARGETS["memory_leak"]

    # constraints the simulated call graph adds on top
    for fault_type in ("packet_loss", "dependency_failure"):
        assert not {"accounting", "fraud-detection"} & set(VALID_TARGETS[fault_type])
    assert not {"frontend", "frontend-proxy"} & set(VALID_TARGETS["dependency_failure"])
    assert VALID_TARGETS["queue_backlog"] == ["checkout", "accounting", "fraud-detection"]
    # accounting is excluded even though the injector can reach postgresql through it:
    # the netem goes on the shared astronomy-db and product-catalog is what suffers
    assert set(VALID_TARGETS["cache_slowdown"]) == {"cart", "product-catalog",
                                                    "recommendation"}


def test_fault_spec_rejects_invalid_target_and_early_start():
    with pytest.raises(ValueError):
        FaultSpec("queue_backlog", "payment", 40.0, 20.0, 0.5).validate()
    with pytest.raises(ValueError):
        FaultSpec("error_rate", "shipping", 40.0, 20.0, 0.5).validate()
    with pytest.raises(ValueError):
        generate_experiment(1, 90.0, FaultSpec("cpu_saturation", "cart", 10.0, 20.0, 0.5), STEADY)


def test_sample_fault_is_uniform_over_pairs():
    rng = np.random.default_rng(0)
    drawn = [(f.fault_type, f.target) for f in
             (sample_fault(rng, 180.0, 30.0) for _ in range(4000))]
    assert set(drawn) == set(FAULT_TARGET_PAIRS)

    # families are drawn in proportion to sqrt(number of targets): between uniform
    # over pairs and uniform over families
    by_type = pd.Series([f for f, _ in drawn]).value_counts(normalize=True)
    total = sum(np.sqrt(len(t)) for t in VALID_TARGETS.values())
    for fault_type, targets in VALID_TARGETS.items():
        expected = np.sqrt(len(targets)) / total
        assert abs(by_type[fault_type] - expected) < 0.35 * expected, fault_type

    # within a family the targets are drawn evenly, so no legal target is starved
    by_pair = pd.Series(Counter(drawn))
    for fault_type, targets in VALID_TARGETS.items():
        counts = pd.Series({t: by_pair.get((fault_type, t), 0) for t in targets})
        assert counts.min() > 0.7 * counts.mean(), fault_type

    # same seed, same draws
    a = [(f.fault_type, f.target) for f in
         (sample_fault(np.random.default_rng(7), 180.0, 30.0) for _ in range(5))]
    b = [(f.fault_type, f.target) for f in
         (sample_fault(np.random.default_rng(7), 180.0, 30.0) for _ in range(5))]
    assert a == b


def test_dependency_failure_makes_the_target_unreachable():
    """docker pause semantics: calls INTO the target fail and the target goes quiet."""
    spec = FaultSpec("dependency_failure", "payment", 60.0, 60.0, 0.9)
    exp = generate_experiment(37, 150.0, spec, BUSY)
    s = exp.spans
    t = (s["start_ns"] - exp.manifest.start_ns) / 1e9
    warm, fault = (t >= 0) & (t < 30), (t >= 75) & (t < 115)

    inbound = s["peer_service"] == "payment"
    err_warm = s.loc[inbound & warm, "status_error"].mean()
    err_fault = s.loc[inbound & fault, "status_error"].mean()
    assert err_warm < 0.05 and err_fault > 0.7

    # the target answers almost nothing, so its server spans and its CPU collapse
    served_warm = (inbound & warm).sum() and (s["service"] == "payment").loc[warm].sum()
    served_fault = (s["service"] == "payment").loc[fault].sum()
    assert served_fault < 0.5 * served_warm
    cpu_warm = _mean_metric(exp, "payment", schema.METRIC_CPU_UTIL, 0.0, 30.0)
    cpu_fault = _mean_metric(exp, "payment", schema.METRIC_CPU_UTIL, 75.0, 115.0)
    assert cpu_fault < 0.5 * cpu_warm

    # the caller sees the failure and propagates it up a required edge
    caller = (s["service"] == "checkout") & (s["kind"] == "server")
    assert s.loc[caller & fault, "status_error"].mean() > 0.7
    # payment is only reachable through checkout, so it is the checkout requests at the
    # frontend that fail -- a few percent of all of them, which is the real regime
    front = ((s["service"] == "frontend") & (s["kind"] == "server")
             & (s["operation"] == "POST /api/checkout"))
    assert s.loc[front & fault, "status_error"].mean() > 0.7
    assert s.loc[front & warm, "status_error"].mean() < 0.05

    assert "callee" not in exp.manifest.faults[0].params
    assert 0.0 < exp.manifest.faults[0].params["refused_fraction"] < 1.0


def test_dependency_failure_on_an_optional_edge_degrades_without_failing():
    spec = FaultSpec("dependency_failure", "ad", 60.0, 60.0, 1.0)
    exp = generate_experiment(38, 150.0, spec, STEADY)
    s = exp.spans
    t = (s["start_ns"] - exp.manifest.start_ns) / 1e9
    fault = (t >= 75) & (t < 115)
    inbound = s["peer_service"] == "ad"
    assert s.loc[inbound & fault, "status_error"].mean() > 0.9

    front = (s["service"] == "frontend") & (s["kind"] == "server")
    # the product page treats the ad as optional: it degrades but still answers
    browse = front & (s["operation"] == "GET /api/products")
    assert s.loc[browse & fault, "status_error"].mean() < 0.02
    # a request that exists only to fetch an ad has no fallback and fails
    ad_only = front & (s["operation"] == "GET /api/data")
    assert s.loc[ad_only & fault, "status_error"].mean() > 0.9


def test_all_fault_families_generate():
    rng = np.random.default_rng(0)
    for fault_type, targets in VALID_TARGETS.items():
        target = str(rng.choice(targets))
        spec = FaultSpec(fault_type, target, 40.0, 30.0, 0.8)
        exp = generate_experiment(int(rng.integers(1000)), 100.0, spec, STEADY, warmup_s=30.0)
        assert len(exp.spans) > 0
        assert exp.manifest.faults[0].fault_type == fault_type
        assert exp.manifest.faults[0].target == target


def test_generate_dataset(tmp_path):
    paths = generate_dataset(tmp_path, 8, seed=3, duration_range=(60.0, 80.0),
                             normal_fraction=0.25, traffic_only_spike_fraction=0.25)
    assert len(paths) == 8
    assert len(schema.list_experiments(tmp_path)) == 8

    categories = []
    for p in paths:
        m = schema.read_manifest(p)
        assert p.name == f"sim-{m.seed:06d}"
        categories.append(m.extra["category"])
        if m.extra["category"] == "fault":
            f = m.faults[0]
            assert m.start_ns + m.warmup_ns <= f.start_ns
            assert 0.2 <= f.intensity <= 1.0
            assert f.target in VALID_TARGETS[f.fault_type]
        else:
            assert m.faults == []
        if m.extra["category"] == "traffic_spike":
            assert "spike" in m.traffic.params
    assert set(categories) <= {"normal", "traffic_spike", "fault"}
    assert "fault" in categories


# --- logs ---------------------------------------------------------------------------

# The only bodies an ERROR/FATAL line may carry: one per failed span, plus restarts.
ERROR_PREFIXES = ("connection timeout to ", "upstream error from ", "call failed: ")
ERROR_BODIES = {"message processing failed", "request failed with internal error",
                "out of memory: container killed", *generate.RESTART_BODIES}

# Families where the fault itself produces no failed requests, so it must produce no
# ERROR lines either -- the old simulator gave every family an ERROR giveaway.
SILENT_FAMILIES = [("cpu_saturation", "recommendation"), ("network_latency", "payment"),
                   ("queue_backlog", "accounting"), ("cache_slowdown", "product-catalog")]


def _log_counts(exp, predicate, a_s, b_s):
    lg = exp.logs
    t = (lg["ts_ns"] - exp.manifest.start_ns) / 1e9
    return int((predicate(lg) & (t >= a_s) & (t < b_s)).sum())


def test_error_logs_only_come_from_failed_spans_and_restarts():
    """No fault family may invent ERROR lines; that is what leaked the label."""
    rng = np.random.default_rng(0)
    for fault_type, targets in VALID_TARGETS.items():
        target = str(rng.choice(targets))
        exp = generate_experiment(61, 150.0, FaultSpec(fault_type, target, 60.0, 60.0, 0.9),
                                  STEADY)
        lg = exp.logs
        bad = lg[lg["severity"].isin(["ERROR", "FATAL"])]["body"]
        unexpected = {b for b in set(bad)
                      if b not in ERROR_BODIES and not b.startswith(ERROR_PREFIXES)}
        assert not unexpected, f"{fault_type}@{target} invented ERROR bodies: {unexpected}"


def test_silent_faults_do_not_raise_errors_at_the_target():
    for fault_type, target in SILENT_FAMILIES:
        exp = generate_experiment(62, 150.0, FaultSpec(fault_type, target, 60.0, 60.0, 0.9),
                                  STEADY)
        def at_target(lg, target=target):
            return (lg["service"] == target) & (lg["severity"].isin(["ERROR", "FATAL"]))
        warm = _log_counts(exp, at_target, 0.0, 30.0) / 30.0
        fault = _log_counts(exp, at_target, 70.0, 120.0) / 50.0
        assert fault <= 3.0 * warm + 0.1, f"{fault_type}@{target} logs errors it should not"


def test_latency_faults_are_logged_by_callers_not_the_target():
    exp = generate_experiment(63, 150.0,
                              FaultSpec("network_latency", "payment", 60.0, 60.0, 0.9), STEADY)
    lg = exp.logs
    body = "upstream payment response time degraded"
    complaining = lg[lg["body"] == body]
    assert len(complaining) > 5
    # checkout is payment's only caller in the demo, and payment never blames itself
    assert set(complaining["service"]) == {"checkout"}
    assert set(complaining["severity"]) == {"WARN"}


def test_memory_leak_target_warns_about_gc_pressure():
    exp = generate_experiment(64, 180.0,
                              FaultSpec("memory_leak", "cart", 50.0, 100.0, 1.0,
                                        {"leak_full_s": 45.0}), STEADY, warmup_s=30.0)
    lg = exp.logs
    gc = lg[(lg["service"] == "cart") & lg["body"].isin(generate.GC_PRESSURE_BODIES)]
    t = (gc["ts_ns"] - exp.manifest.start_ns) / 1e9
    assert set(gc["severity"]) == {"WARN"}
    assert (t >= 50.0).mean() > 0.9          # they follow the leak, not the warm-up
    # the restart is the one event that logs FATAL
    fatal = lg[lg["severity"] == "FATAL"]
    assert len(fatal) == len(exp.manifest.extra["restarts"]) > 0
    assert set(fatal["service"]) == {"cart"}


def test_queue_backlog_logs_lag_on_the_consumer_only():
    exp = generate_experiment(65, 150.0,
                              FaultSpec("queue_backlog", "accounting", 60.0, 60.0, 0.9), STEADY)
    lg = exp.logs
    lag = lg[lg["body"].isin(generate.LAG_BODIES) & (lg["severity"] == "WARN")]
    t = (lag["ts_ns"] - exp.manifest.start_ns) / 1e9
    warm, during = lag[t < 30], lag[(t >= 70) & (t < 120)]

    # the stalled consumer becomes the dominant source, but the same phrasing keeps
    # appearing at a background rate elsewhere so the logs alone do not name it
    counts = during["service"].value_counts()
    assert counts.index[0] == "accounting"
    assert counts["accounting"] > 3 * counts.drop("accounting").max()
    assert counts["accounting"] / 50.0 > 5 * ((warm["service"] == "accounting").sum() / 30.0)
    assert (during["service"] != "accounting").sum() > 0
    # the other consumer is not implicated
    assert counts.get("fraud-detection", 0) < 0.2 * counts["accounting"]


def test_failed_calls_log_at_both_ends():
    """A failed call is logged by the caller and by the callee that answered it."""
    exp = generate_experiment(66, 150.0,
                              FaultSpec("error_rate", "cart", 60.0, 60.0, 0.9), STEADY)
    lg = exp.logs
    t = (lg["ts_ns"] - exp.manifest.start_ns) / 1e9
    err = lg[(lg["severity"] == "ERROR") & (t >= 70) & (t < 120)]
    assert (err["trace_id"] != "").all()

    callee = err[err["service"] == "cart"]
    caller = err[err["service"] == "frontend"]
    assert len(callee) > 20 and len(caller) > 20
    assert callee["body"].eq("request failed with internal error").all()
    assert caller["body"].str.endswith("cart").any()
    # the two ends are joinable through the trace they belong to
    assert len(set(callee["trace_id"]) & set(caller["trace_id"])) > 10


# --- where latency lands ------------------------------------------------------------

def _median_ms(exp, mask_fn, a_s, b_s):
    s = exp.spans
    t = (s["start_ns"] - exp.manifest.start_ns) / 1e9
    sel = s[mask_fn(s) & (t >= a_s) & (t < b_s)]
    assert len(sel) > 20, "not enough spans to compare"
    return float(np.median(sel["duration_ns"])) / 1e6


def test_network_latency_on_a_leaf_only_shows_at_its_callers():
    """tc netem is egress. A leaf sends only responses, so only its callers pay.

    Its own server span is untouched, which is what the live runs show for payment:
    unchanged self latency and an enormous inbound-vs-server ratio.
    """
    delay, intensity = 220.0, 0.9
    spec = FaultSpec("network_latency", "currency", 60.0, 60.0, intensity,
                     {"delay_ms": delay})
    exp = generate_experiment(71, 150.0, spec, BUSY)
    warm, fault = (0.0, 30.0), (80.0, 115.0)

    server = (lambda s: (s["service"] == "currency") & (s["kind"] == "server"))
    before, during = _median_ms(exp, server, *warm), _median_ms(exp, server, *fault)
    assert abs(during - before) < 0.5, "the target's own server span must not change"

    expected = delay * intensity
    inbound = (lambda s: (s["peer_service"] == "currency") & (s["kind"] == "client"))
    frontend_side = (lambda s: (s["service"] == "frontend")
                     & (s["peer_service"] == "currency"))
    for name, client in (("all callers", inbound), ("frontend", frontend_side)):
        added = _median_ms(exp, client, *fault) - _median_ms(exp, client, *warm)
        # exactly once: not 2x, not 0.5x
        assert 0.9 * expected < added < 1.1 * expected, f"{name} saw {added:.1f} ms"

    # The caller carries it upstream, but only on the requests that actually call
    # currency -- a few percent of them -- so the effect shows in the mean, not the
    # median of every frontend request.
    mean_warm = _mean_latency_ms(exp, "frontend", "server", *warm)
    mean_fault = _mean_latency_ms(exp, "frontend", "server", *fault)
    assert mean_fault > mean_warm + 0.05 * expected


def _window(exp, mask_fn, a_s, b_s):
    s = exp.spans
    t = (s["start_ns"] - exp.manifest.start_ns) / 1e9
    return s[mask_fn(s) & (t >= a_s) & (t < b_s)]


def test_network_latency_is_egress_in_both_directions():
    """A mid-graph target sends requests too, so its own outbound calls are delayed.

    Live: network_latency@cart put z 44 on cart's *own* server latency because every
    valkey call paid the delay, and z 42 on its callers -- who pay it twice, once for
    cart's slowed request and once for its slowed response.
    """
    delay, intensity = 588.0, 0.9
    spec = FaultSpec("network_latency", "cart", 70.0, 80.0, intensity, {"delay_ms": delay})
    exp = generate_experiment(31, 200.0, spec, BUSY)
    expected = delay * intensity
    warm, fault = (0.0, 60.0), (85.0, 145.0)

    outbound = (lambda s: (s["service"] == "cart") & (s["peer_service"] == "valkey"))
    server = (lambda s: (s["service"] == "cart") & (s["kind"] == "server"))
    inbound = (lambda s: (s["service"] == "frontend") & (s["peer_service"] == "cart"))

    # the target's own call out pays the delay once ...
    out_add = _median_ms(exp, outbound, *fault) - _median_ms(exp, outbound, *warm)
    assert 0.9 * expected < out_add < 1.1 * expected
    # ... which lands inside its server span ...
    srv_add = _median_ms(exp, server, *fault) - _median_ms(exp, server, *warm)
    assert 0.9 * expected < srv_add < 1.15 * expected
    # ... and the caller pays that plus the delayed response: twice, not once
    in_add = _median_ms(exp, inbound, *fault) - _median_ms(exp, inbound, *warm)
    assert 1.8 * expected < in_add < 2.2 * expected

    # the callee behind the target is innocent: valkey has no server span, so check a
    # service callee instead
    spec2 = FaultSpec("network_latency", "recommendation", 70.0, 80.0, intensity,
                      {"delay_ms": delay})
    exp2 = generate_experiment(31, 200.0, spec2, BUSY)
    callee = (lambda s: (s["service"] == "product-catalog") & (s["kind"] == "server"))
    assert _median_ms(exp2, callee, *fault) < 1.2 * _median_ms(exp2, callee, *warm)


def test_packet_loss_shows_up_as_latency_not_errors():
    """netem loss makes calls slow via TCP retransmits; it does not fail them.

    Matches the live demo at 24.8% loss on cart: client->cart mean 1.5 -> 341 ms,
    p95 974 ms, client error rate 0.000, and no application-level retries.
    """
    spec = FaultSpec("packet_loss", "cart", 70.0, 80.0, 0.6)
    exp = generate_experiment(72, 200.0, spec, STEADY)
    assert spec.params["loss_pct"] == pytest.approx(24.8, abs=0.1)
    warm, fault = (0.0, 60.0), (85.0, 145.0)

    inbound = (lambda s: (s["peer_service"] == "cart") & (s["kind"] == "client"))
    before, during = _window(exp, inbound, *warm), _window(exp, inbound, *fault)
    mean_before = before["duration_ns"].mean() / 1e6
    mean_during = during["duration_ns"].mean() / 1e6
    p95_before = np.percentile(before["duration_ns"], 95) / 1e6
    p95_during = np.percentile(during["duration_ns"], 95) / 1e6
    assert mean_during > 20 * mean_before and 200 < mean_during < 600
    assert p95_during > 10 * p95_before and 600 < p95_during < 1600

    # a few calls do exceed the caller's socket timeout, but errors stay near zero
    assert before["status_error"].mean() < 0.01     # background rate only
    assert during["status_error"].mean() < 0.05

    # and the application never retries: no second client span to the same peer for the
    # same operation under one parent (checkout calling GetCart and EmptyCart is not one)
    repeats = during.groupby(["trace_id", "parent_span_id", "operation"]).size()
    assert (repeats > 1).sum() == 0


def test_packet_loss_is_egress_so_the_target_pays_on_both_sides():
    """tc netem sits on the target's eth0: everything it *sends* is lossy.

    So a mid-graph target's own outbound calls retransmit too, which is what puts the
    latency into its own server spans -- measured live on checkout, whose calls to
    currency/cart/payment were the slow ones.
    """
    spec = FaultSpec("packet_loss", "cart", 70.0, 80.0, 0.9)
    exp = generate_experiment(31, 200.0, spec, BUSY)
    warm, fault = (0.0, 60.0), (85.0, 145.0)

    outbound = (lambda s: (s["service"] == "cart") & (s["peer_service"] == "valkey"))
    server = (lambda s: (s["service"] == "cart") & (s["kind"] == "server"))
    inbound = (lambda s: (s["service"] == "frontend") & (s["peer_service"] == "cart"))

    out_add = _median_ms(exp, outbound, *fault) - _median_ms(exp, outbound, *warm)
    srv_add = _median_ms(exp, server, *fault) - _median_ms(exp, server, *warm)
    in_add = _median_ms(exp, inbound, *fault) - _median_ms(exp, inbound, *warm)
    assert out_add > 50.0                       # the target's own calls retransmit
    assert srv_add > 0.8 * out_add              # ... which is inside its server span
    assert in_add > 1.5 * srv_add               # the caller pays that plus its own hop


def test_packet_loss_callers_grumble_less_than_for_a_failing_dependency():
    """Callers only notice slow calls, so they log far less than for hard failures."""
    lossy = generate_experiment(72, 150.0,
                                FaultSpec("packet_loss", "cart", 60.0, 60.0, 1.0), STEADY)
    slow = generate_experiment(72, 150.0,
                               FaultSpec("network_latency", "cart", 60.0, 60.0, 1.0), STEADY)

    def caller_warns(exp):
        lg = exp.logs
        t = (lg["ts_ns"] - exp.manifest.start_ns) / 1e9
        return int(((lg["severity"] == "WARN") & (t >= 70) & (t < 120)
                    & lg["service"].isin(["frontend", "checkout"])).sum())

    assert caller_warns(lossy) < caller_warns(slow)


def test_cache_slowdown_is_inside_the_target_server_span():
    """Unlike netem on the target, a slow backing store is a child of the target."""
    spec = FaultSpec("cache_slowdown", "product-catalog", 60.0, 60.0, 0.9)
    exp = generate_experiment(73, 150.0, spec, STEADY)
    warm, fault = (0.0, 30.0), (80.0, 115.0)

    db = (lambda s: (s["service"] == "product-catalog")
          & (s["peer_service"] == "postgresql"))
    server = (lambda s: (s["service"] == "product-catalog") & (s["kind"] == "server"))
    caller = (lambda s: (s["service"] == "frontend")
              & (s["peer_service"] == "product-catalog"))

    db_added = _median_ms(exp, db, *fault) - _median_ms(exp, db, *warm)
    server_added = _median_ms(exp, server, *fault) - _median_ms(exp, server, *warm)
    caller_added = _median_ms(exp, caller, *fault) - _median_ms(exp, caller, *warm)
    assert db_added > 20.0
    # the target's self-time genuinely grows here, and the caller inherits it
    assert server_added >= 0.9 * db_added
    assert caller_added >= 0.9 * server_added


# --- dataset timeline ---------------------------------------------------------------

def test_generate_dataset_is_chronological(tmp_path):
    paths = generate_dataset(tmp_path, 10, seed=4, duration_range=(60.0, 90.0), gap_s=45.0)
    manifests = [schema.read_manifest(p) for p in paths]
    assert [m.extra["run_index"] for m in manifests] == list(range(10))

    for prev, nxt in itertools.pairwise(manifests):
        assert prev.end_ns < nxt.start_ns, "experiments must not overlap"
        assert nxt.start_ns == prev.end_ns + int(45e9)
    assert manifests[0].start_ns == generate.EPOCH_NS

    # the chronological split now follows generation order instead of the random ids
    ordered = sorted(manifests, key=lambda m: (m.start_ns, m.experiment_id))
    assert [m.experiment_id for m in ordered] == [m.experiment_id for m in manifests]
    split = split_by_time(manifests, frac=0.2, val_frac=0.1)
    assert split["test"] == sorted(m.experiment_id for m in manifests[-2:])
    assert not set(split["train"]) & set(split["test"])


def test_generate_dataset_timeline_is_deterministic(tmp_path):
    a = [schema.read_manifest(p).start_ns
         for p in generate_dataset(tmp_path / "a", 6, seed=9, duration_range=(60.0, 90.0))]
    b = [schema.read_manifest(p).start_ns
         for p in generate_dataset(tmp_path / "b", 6, seed=9, duration_range=(60.0, 90.0))]
    assert a == b
    c = [schema.read_manifest(p).start_ns
         for p in generate_dataset(tmp_path / "c", 6, seed=10, duration_range=(60.0, 90.0))]
    assert c != a


def test_generation_is_fast():
    import time

    t0 = time.perf_counter()
    exp = generate_experiment(77, 180.0, None, STEADY)
    elapsed = time.perf_counter() - t0
    assert len(exp.spans) > 10_000
    assert elapsed < 10.0, f"180s experiment took {elapsed:.1f}s"
