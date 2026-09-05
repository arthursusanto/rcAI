"""Hand-built canonical experiments used by the feature tests.

Deliberately tiny and fully deterministic: three application services plus one infra
component (valkey, owned by cart), a slow-dependency fault on cart, and telemetry shaped
so that the interesting feature interactions are checkable by hand.

    frontend --> cart --> valkey   (cart's own cache: client spans only)
             \\-> product-catalog

The frontend does a substantial amount of its own work (30 ms) so the two services have
comparable *relative* baseline noise -- that is what makes "explained by downstream"
(callee latency z minus own latency z) meaningful rather than an artefact of one service
having a tighter baseline than another.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rca.data import schema

START_NS = 1_700_000_000_000_000_000
DURATION_S = 120
WARMUP_S = 60                # 6 windows: the campaign's warm-up, and >= MIN_BASELINE_WINDOWS
FAULT_START_S = 60
FAULT_END_S = 100
NO_PC_AFTER_S = 100        # product-catalog stops serving -> NaN trace features there
RPS = 5
WINDOW_NS = 10_000_000_000

APP_SERVICES = ["frontend", "cart", "product-catalog"]
EDGES = [("frontend", "cart"), ("frontend", "product-catalog"), ("cart", "valkey")]
BASE_LATENCY_NS = {
    "frontend_own": 30_000_000, "cart": 20_000_000, "product-catalog": 10_000_000,
    "valkey": 2_000_000,
}
FAULT_LATENCY_MULTIPLE = 15.0
FAULT_ERROR_RATE = 0.2
ANSWERED_WHILE_TIMING_OUT = 0.4   # share of calls that still reach the callee

# How the fault shows up in the spans:
#   callee_slow     -- the callee's own server spans get slow (a slow dependency)
#   caller_timeouts -- the callee stays healthy but most calls never reach it, and the
#                      ones that do time out on the caller's side (packet loss, a
#                      saturated link, a dependency that stopped answering)
FAULT_MODES = ("callee_slow", "caller_timeouts")


def make_experiment(
    experiment_id: str = "exp-0",
    seed: int = 7,
    with_fault: bool = True,
    traffic_shape: str = "steady",
    fault_mode: str = "callee_slow",
) -> schema.Experiment:
    """Build one synthetic canonical experiment."""
    rng = np.random.default_rng(seed)
    fault_window = (FAULT_START_S, FAULT_END_S) if with_fault else (DURATION_S, DURATION_S)
    faults = []
    if with_fault:
        faults.append(schema.Fault(
            fault_type="network_latency", target="cart",
            start_ns=START_NS + FAULT_START_S * 1_000_000_000,
            end_ns=START_NS + FAULT_END_S * 1_000_000_000,
            intensity=0.7,
        ))
    manifest = schema.Manifest(
        experiment_id=experiment_id,
        source="sim",
        seed=seed,
        start_ns=START_NS,
        end_ns=START_NS + DURATION_S * 1_000_000_000,
        traffic=schema.TrafficProfile(base_rps=float(RPS), shape=traffic_shape),
        faults=faults,
        services=list(APP_SERVICES),
        edges=list(EDGES),
        metric_interval_ns=1_000_000_000,
        warmup_ns=WARMUP_S * 1_000_000_000,
    )
    return schema.Experiment(
        manifest=manifest,
        metrics=_cast(_metrics(rng, fault_window), schema.METRICS_COLUMNS),
        spans=_cast(_spans(rng, fault_window, fault_mode), schema.SPANS_COLUMNS),
        logs=_cast(_logs(fault_window), schema.LOGS_COLUMNS),
    )


def make_manifest(
    experiment_id: str,
    fault_type: str | None,
    target: str = "cart",
    intensity: float = 0.5,
    shape: str = "steady",
    base_rps: float = 10.0,
    start_ns: int = START_NS,
    seed: int = 0,
) -> schema.Manifest:
    """A manifest with no telemetry attached, for the split tests."""
    faults = []
    if fault_type is not None:
        faults.append(schema.Fault(
            fault_type=fault_type, target=target, start_ns=start_ns + 60_000_000_000,
            end_ns=start_ns + 100_000_000_000, intensity=intensity,
        ))
    return schema.Manifest(
        experiment_id=experiment_id, source="sim", seed=seed, start_ns=start_ns,
        end_ns=start_ns + 120_000_000_000,
        traffic=schema.TrafficProfile(base_rps=base_rps, shape=shape),
        faults=faults, services=list(APP_SERVICES), edges=list(EDGES),
        metric_interval_ns=1_000_000_000, warmup_ns=40_000_000_000,
    )


@pytest.fixture
def experiment() -> schema.Experiment:
    return make_experiment()


# --- telemetry generation --------------------------------------------------------------
def _faulty(second: int, fault_window: tuple[int, int]) -> bool:
    return fault_window[0] <= second < fault_window[1]


def _metrics(rng: np.random.Generator, fault_window: tuple[int, int]) -> pd.DataFrame:
    rows = []
    cumulative = {s: 0.0 for s in [*APP_SERVICES, "valkey"]}
    for second in range(DURATION_S):
        ts = START_NS + second * 1_000_000_000
        faulty = _faulty(second, fault_window)
        for service in [*APP_SERVICES, "valkey"]:
            cpu = 0.20 + 0.02 * rng.standard_normal()
            queue = 1.0 + 0.2 * rng.random()
            mem = 4.0e8 + 1.0e6 * rng.standard_normal()
            if faulty and service == "cart":
                cpu += 0.5
                queue += 20.0
                mem += 5.0e6 * (second - fault_window[0])   # leak-like ramp
            if faulty and service == "valkey":
                queue += 40.0
            cumulative[service] += 1.0e6 * (1.0 + rng.random())
            for metric, value in [
                (schema.METRIC_CPU_UTIL, cpu),
                (schema.METRIC_MEM_BYTES, mem),
                (schema.METRIC_MEM_LIMIT, 1.0e9),
                (schema.METRIC_QUEUE_DEPTH, queue),
                (schema.METRIC_THREADS, 8.0 + rng.integers(0, 3)),
                (schema.METRIC_GC_PAUSE_MS, 0.5 + 0.1 * rng.random()),
                (schema.METRIC_NET_RX_BYTES, cumulative[service]),
                (schema.METRIC_NET_TX_BYTES, 0.5 * cumulative[service]),
            ]:
                rows.append((ts, service, metric, float(value)))
    return pd.DataFrame(rows, columns=["ts_ns", "service", "metric", "value"])


def _spans(
    rng: np.random.Generator, fault_window: tuple[int, int], fault_mode: str
) -> pd.DataFrame:
    assert fault_mode in FAULT_MODES
    rows: list[tuple] = []

    def span(trace, span_id, parent, service, operation, start, duration, error, peer, kind):
        rows.append((trace, span_id, parent, service, operation, int(start), int(duration),
                     bool(error), peer, kind))

    for request in range(DURATION_S * RPS):
        second = request // RPS
        start = START_NS + int(request * 1e9 / RPS)
        faulty = _faulty(second, fault_window)
        trace = f"trace-{request:06d}"

        cart_ns = BASE_LATENCY_NS["cart"] * rng.lognormal(0.0, 0.15)
        cart_server_ns = cart_ns * 0.98
        answered = True
        if faulty:
            cart_ns *= FAULT_LATENCY_MULTIPLE       # what the caller waits for
            if fault_mode == "callee_slow":
                cart_server_ns = cart_ns * 0.98     # ... because the callee is slow
            else:
                answered = rng.random() < ANSWERED_WHILE_TIMING_OUT
        valkey_ns = BASE_LATENCY_NS["valkey"] * rng.lognormal(0.0, 0.15)
        pc_ns = BASE_LATENCY_NS["product-catalog"] * rng.lognormal(0.0, 0.15)
        own_ns = BASE_LATENCY_NS["frontend_own"] * rng.lognormal(0.0, 0.25)
        has_pc = second < NO_PC_AFTER_S
        timing_out = faulty and fault_mode == "caller_timeouts"
        failed = timing_out or (faulty and rng.random() < FAULT_ERROR_RATE)
        total = own_ns + cart_ns + (pc_ns if has_pc else 0.0)

        root = f"sp-{request:06d}-root"
        span(trace, root, "", "frontend", "GET /checkout", start, total, failed, "", "server")

        call = f"sp-{request:06d}-c1"
        span(trace, call, root, "frontend", "cart.Get", start + 1_000_000, cart_ns, failed,
             "cart", "client")
        if answered:
            server = f"sp-{request:06d}-s1"
            span(trace, server, call, "cart", "cart.Get", start + 1_200_000, cart_server_ns,
                 failed and not timing_out, "", "server")
            # valkey emits nothing of its own: the cache call exists only as cart's client
            # span, exactly as a real database/cache/broker call is instrumented.
            span(trace, f"sp-{request:06d}-v1", server, "cart", "valkey.Get",
                 start + 1_400_000, valkey_ns, False, "valkey", "client")
        if failed:   # the frontend retries the failed call on the same parent span
            span(trace, f"sp-{request:06d}-c1r", root, "frontend", "cart.Get",
                 start + 2_000_000, cart_ns, timing_out, "cart", "client")
        if has_pc:
            pc_call = f"sp-{request:06d}-c2"
            span(trace, pc_call, root, "frontend", "pc.List", start + 2_000_000, pc_ns, False,
                 "product-catalog", "client")
            span(trace, f"sp-{request:06d}-s2", pc_call, "product-catalog", "pc.List",
                 start + 2_100_000, pc_ns * 0.95, False, "", "server")

    return pd.DataFrame(rows, columns=list(schema.SPANS_COLUMNS))


def _logs(fault_window: tuple[int, int]) -> pd.DataFrame:
    rows = []
    for second in range(DURATION_S):
        ts = START_NS + second * 1_000_000_000
        faulty = _faulty(second, fault_window)
        for service in APP_SERVICES:
            for _ in range(2):
                rows.append((ts, service, "INFO", "handled request", ""))
            if faulty and service == "cart":
                for index in range(3):
                    rows.append((ts, service, "ERROR", "upstream timeout",
                                 f"trace-{second * RPS + index:06d}"))
                rows.append((ts, service, "WARN", "retrying", ""))
    return pd.DataFrame(rows, columns=list(schema.LOGS_COLUMNS))


def _cast(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    return frame.astype(columns)


# --- a shared infra component, called by a service that does not own it -----------------
SHARED_INFRA_SERVICES = ["accounting", "product-catalog"]   # postgresql's owner is the latter
SHARED_INFRA_EDGES = [("accounting", "postgresql"), ("product-catalog", "postgresql")]
DB_LATENCY_NS = 5_000_000
DB_FAULT_MULTIPLE = 20.0


def make_shared_infra_experiment(seed: int = 3) -> schema.Experiment:
    """Two services querying the same database, one of which sees it go slow.

    postgresql is owned by product-catalog, but here it is *accounting* whose queries
    crawl. The evidence has to land on accounting -- the service that is actually
    waiting -- not on the component's nominal owner, which is querying it happily.
    """
    rng = np.random.default_rng(seed)
    manifest = schema.Manifest(
        experiment_id="exp-shared-infra", source="sim", seed=seed, start_ns=START_NS,
        end_ns=START_NS + DURATION_S * 1_000_000_000,
        traffic=schema.TrafficProfile(base_rps=float(RPS), shape="steady"),
        faults=[schema.Fault(
            fault_type="cache_slowdown", target="accounting",
            start_ns=START_NS + FAULT_START_S * 1_000_000_000,
            end_ns=START_NS + FAULT_END_S * 1_000_000_000, intensity=0.6,
        )],
        services=list(SHARED_INFRA_SERVICES), edges=list(SHARED_INFRA_EDGES),
        metric_interval_ns=1_000_000_000, warmup_ns=WARMUP_S * 1_000_000_000,
    )

    rows = []
    for request in range(DURATION_S * RPS):
        second = request // RPS
        start = START_NS + int(request * 1e9 / RPS)
        for service in SHARED_INFRA_SERVICES:
            query_ns = DB_LATENCY_NS * rng.lognormal(0.0, 0.15)
            if service == "accounting" and _faulty(second, (FAULT_START_S, FAULT_END_S)):
                query_ns *= DB_FAULT_MULTIPLE
            tag = f"{service[:4]}-{request:06d}"
            trace = f"trace-{tag}"
            server = f"sp-{tag}-s"
            rows.append((trace, server, "", service, "work", start, int(query_ns + 1e6),
                         False, "", "server"))
            rows.append((trace, f"sp-{tag}-q", server, service, "db.query",
                         start + 100_000, int(query_ns), False, "postgresql", "client"))
    spans = pd.DataFrame(rows, columns=list(schema.SPANS_COLUMNS))

    # only the component's own resource metrics exist, as in a real deployment
    metrics = pd.DataFrame(
        [(START_NS + second * 1_000_000_000, "postgresql", metric, value)
         for second in range(DURATION_S)
         for metric, value in ((schema.METRIC_CPU_UTIL, 0.3 + 0.02 * rng.standard_normal()),
                               (schema.METRIC_QUEUE_DEPTH, 2.0 + rng.random()))],
        columns=["ts_ns", "service", "metric", "value"],
    )
    logs = pd.DataFrame(columns=list(schema.LOGS_COLUMNS))
    return schema.Experiment(
        manifest=manifest,
        metrics=_cast(metrics, schema.METRICS_COLUMNS),
        spans=_cast(spans, schema.SPANS_COLUMNS),
        logs=_cast(logs, schema.LOGS_COLUMNS),
    )


# --- one caller, two calls to the same peer per request ---------------------------------
FANOUT_OPERATIONS = ("cart.GetCart", "cart.EmptyCart")


def make_retry_experiment(same_operation: bool, seed: int = 5) -> schema.Experiment:
    """checkout calls cart twice per request, both calls under the same parent span.

    With ``same_operation`` the two are the same call made twice -- a retry. Otherwise
    they are two different operations that merely share a parent, which is what the real
    demo does (fetch the basket, then empty it) and must not be read as a retry.
    """
    rng = np.random.default_rng(seed)
    manifest = schema.Manifest(
        experiment_id=f"exp-retry-{int(same_operation)}", source="sim", seed=seed,
        start_ns=START_NS, end_ns=START_NS + DURATION_S * 1_000_000_000,
        traffic=schema.TrafficProfile(base_rps=float(RPS), shape="steady"),
        faults=[], services=["checkout", "cart"], edges=[("checkout", "cart")],
        metric_interval_ns=1_000_000_000, warmup_ns=WARMUP_S * 1_000_000_000,
    )
    operations = (
        (FANOUT_OPERATIONS[0], FANOUT_OPERATIONS[0]) if same_operation else FANOUT_OPERATIONS
    )

    rows = []
    for request in range(DURATION_S * RPS):
        start = START_NS + int(request * 1e9 / RPS)
        trace, root = f"trace-{request:06d}", f"sp-{request:06d}-root"
        rows.append((trace, root, "", "checkout", "PlaceOrder", start, 40_000_000, False,
                     "", "server"))
        for index, operation in enumerate(operations):
            duration = int(BASE_LATENCY_NS["cart"] * rng.lognormal(0.0, 0.15))
            call = f"sp-{request:06d}-c{index}"
            offset = 1_000_000 * (index + 1)
            rows.append((trace, call, root, "checkout", operation, start + offset, duration,
                         False, "cart", "client"))
            rows.append((trace, f"sp-{request:06d}-s{index}", call, "cart", operation,
                         start + offset + 100_000, int(duration * 0.95), False, "", "server"))

    empty_metrics = pd.DataFrame(columns=["ts_ns", "service", "metric", "value"])
    return schema.Experiment(
        manifest=manifest,
        metrics=_cast(empty_metrics, schema.METRICS_COLUMNS),
        spans=_cast(pd.DataFrame(rows, columns=list(schema.SPANS_COLUMNS)),
                    schema.SPANS_COLUMNS),
        logs=_cast(pd.DataFrame(columns=list(schema.LOGS_COLUMNS)), schema.LOGS_COLUMNS),
    )
