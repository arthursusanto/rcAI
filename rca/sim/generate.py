"""Telemetry simulator: emits canonical :class:`rca.data.schema.Experiment` objects.

The model is an open queueing network over the OpenTelemetry Demo dependency graph.
Entry requests arrive as a Poisson process whose rate follows the traffic shape; each
request picks a request type whose static call tree is walked to build a trace. Every
node of the tree costs CPU demand at the component that serves it; the CPU utilisation
that demand implies drives an M/M/c waiting-time factor, so latency grows non-linearly
as a component approaches saturation. Faults change capacity, demand, network delay,
loss and error probabilities, and their effects therefore propagate along the graph
the same way they would on the real demo.

See :mod:`rca.sim.topology` for the graph and :mod:`rca.sim.faults` for the fault model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from rca.data.schema import (
    DEPENDENCY_EDGES,
    METRIC_CPU_UTIL,
    METRIC_GC_PAUSE_MS,
    METRIC_MEM_BYTES,
    METRIC_MEM_LIMIT,
    METRIC_NET_RX_BYTES,
    METRIC_NET_TX_BYTES,
    METRIC_QUEUE_DEPTH,
    METRIC_THREADS,
    SERVICES,
    Experiment,
    Fault,
    Manifest,
    TrafficProfile,
    write_experiment,
)

from . import faults as fault_mod
from .faults import Effects, FaultSpec, build_effects, sample_fault
from .topology import (
    ALL_COMPONENTS,
    CPU_CAPACITY_MS,
    DEFAULT_MIX,
    FLAT_TYPES,
    GC_RUNTIMES,
    INFRA_CALL_MS,
    KAFKA_CONSUMERS,
    NET_RX_BYTES,
    NET_TX_BYTES,
    PROFILES,
    TYPE_NAMES,
    reachability,
)

SIM_VERSION = "1.0"
METRIC_INTERVAL_S = 1.0
# Fault-free prefix usable as a per-experiment baseline. Six default-width
# feature windows: rca.features.baseline needs at least five to fit one.
WARMUP_S = 60.0
EPOCH_NS = 1_700_000_000_000_000_000

MAX_RHO = 0.97                  # utilisation is never allowed to reach 1 exactly
WAIT_FACTOR_CAP = 25.0
LOG_SIGMA = 0.35                # per-call lognormal noise on service time
OUTLIER_PROB = 0.004            # benign GC pause / cold cache on a server span
BASE_ERROR_PROB = 0.0008        # background error rate with no fault
RTO_MS = (200.0, 300.0)         # one retransmission timeout per lost segment
SEGMENTS_PER_CALL = 2.5         # segments one direction of a call puts on the wire
CONSUMER_BASE_LAG_MS = 25.0


# --- traffic ------------------------------------------------------------------------

def _rate_series(traffic: TrafficProfile, n_sec: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_sec, dtype=float)
    p = traffic.params
    base = float(traffic.base_rps)
    if traffic.shape == "steady":
        rate = np.full(n_sec, base)
    elif traffic.shape == "ramp":
        end_mult = float(p.get("end_multiplier", 2.0))
        rate = base * (0.6 + (end_mult - 0.6) * t / max(1, n_sec - 1))
    elif traffic.shape == "diurnal":
        period = float(p.get("period_s", 120.0))
        amp = float(p.get("amplitude", 0.35))
        phase = float(p.get("phase", 0.0))
        rate = base * (1.0 + amp * np.sin(2 * math.pi * t / period + phase))
    elif traffic.shape == "bursty":
        rate = np.full(n_sec, base)
        every = float(p.get("burst_every_s", 45.0))
        length = int(p.get("burst_len_s", 8))
        mult = float(p.get("burst_multiplier", 2.5))
        n_bursts = max(1, int(n_sec / every))
        for s in rng.integers(0, max(1, n_sec - length), n_bursts):
            rate[s:s + length] *= mult
    else:
        raise ValueError(f"unknown traffic shape {traffic.shape!r}")

    spike = p.get("spike")
    if spike:
        s = int(spike["start_s"])
        rate[s:s + int(spike["len_s"])] *= float(spike["multiplier"])
    return np.maximum(rate, 0.5)


def sample_traffic(rng: np.random.Generator, duration_s: float, warmup_s: float,
                   with_spike: bool = False) -> TrafficProfile:
    base = float(rng.uniform(1.5, 15.0))
    shape = str(rng.choice(["steady", "ramp", "diurnal", "bursty"], p=[0.4, 0.2, 0.2, 0.2]))
    params: dict = {}
    if shape == "ramp":
        params["end_multiplier"] = float(rng.uniform(1.5, 2.5))
    elif shape == "diurnal":
        params["period_s"] = float(rng.uniform(80.0, 220.0))
        params["amplitude"] = float(rng.uniform(0.2, 0.5))
        params["phase"] = float(rng.uniform(0.0, 2 * math.pi))
    elif shape == "bursty":
        params["burst_every_s"] = float(rng.uniform(30.0, 70.0))
        params["burst_len_s"] = int(rng.integers(5, 15))
        params["burst_multiplier"] = float(rng.uniform(1.8, 3.2))
    if with_spike:
        length = float(rng.uniform(20.0, 45.0))
        start = float(rng.uniform(warmup_s + 5.0, max(warmup_s + 6.0, duration_s - length - 10.0)))
        params["spike"] = {"start_s": start, "len_s": length,
                           "multiplier": float(rng.uniform(2.5, 4.0))}
    return TrafficProfile(base_rps=base, shape=shape, params=params)


# --- queueing -----------------------------------------------------------------------

def _wait_factor(rho: np.ndarray, cores: float) -> np.ndarray:
    """Sakasegawa approximation of the M/M/c sojourn time as a multiple of service time."""
    rho = np.clip(rho, 0.0, MAX_RHO)
    exponent = math.sqrt(2.0 * (cores + 1.0)) - 1.0
    return np.minimum(1.0 + rho ** exponent / (cores * (1.0 - rho)), WAIT_FACTOR_CAP)


# Static CPU demand in ms contributed to each component by one request of each type.
# Weighted by cumulative reachability, not each node's own probability: a server node
# always has prob 1.0 and is reached only as often as the client node above it.
_TYPE_DEMAND = {
    name: {c: sum(r * n.base_ms
                  for n, r in zip(flat, reachability(flat), strict=True)
                  if n.cpu_service == c)
           for c in ALL_COMPONENTS}
    for name, flat in FLAT_TYPES.items()
}


def baseline_cores(counts_by_type: np.ndarray, warmup_s: float) -> dict[str, float]:
    """Mean cores each component draws over the fault-free warm-up.

    This is what the injector measures at inject time to size the quota squeeze, so the
    simulated squeeze has to be sized from the same thing.
    """
    n = max(1, min(len(counts_by_type), int(warmup_s)))
    out = {}
    for c in ALL_COMPONENTS:
        weights = np.array([_TYPE_DEMAND[name][c] for name in TYPE_NAMES])
        demand = float((counts_by_type[:n] @ weights).mean())
        out[c] = PROFILES[c].idle_util + demand / CPU_CAPACITY_MS[c]
    return out


def _utilisation(counts_by_type: np.ndarray, eff: Effects) -> tuple[dict, dict]:
    """Expected per-second CPU utilisation and the waiting factor it implies."""
    rho, wait = {}, {}
    for c in ALL_COMPONENTS:
        weights = np.array([_TYPE_DEMAND[name][c] for name in TYPE_NAMES])
        demand = counts_by_type @ weights * eff.demand_factor[c] * eff.reachable_frac[c]
        prof = PROFILES[c]
        cores = prof.idle_util + demand / CPU_CAPACITY_MS[c]
        util = cores / np.maximum(eff.capacity_factor[c], 1e-3)
        rho[c] = np.clip(util, 0.0, MAX_RHO)
        wait[c] = _wait_factor(rho[c], prof.cores)
    wait[""] = np.ones(eff.n_sec)
    return rho, wait


# --- span engine --------------------------------------------------------------------

class _SpanSink:
    """Collects span column fragments and the per-second accumulators the metrics need."""

    def __init__(self, n_sec: int):
        self.cols: dict[str, list[np.ndarray]] = {
            k: [] for k in ("trace_no", "span_no", "parent_no", "service", "operation",
                            "kind", "peer", "start_ns", "duration_ns", "status_error")}
        self.n_spans = 0
        self.cpu_ms = {c: np.zeros(n_sec) for c in ALL_COMPONENTS}
        self.busy_s = {c: np.zeros(n_sec) for c in ALL_COMPONENTS}
        self.calls = {c: np.zeros(n_sec) for c in ALL_COMPONENTS}
        self.n_sec = n_sec

    def add(self, trace_no, span_no, parent_no, service, operation, kind, peer,
            start_ns, duration_ns, status_error) -> None:
        m = len(trace_no)
        if m == 0:
            return
        c = self.cols
        c["trace_no"].append(trace_no)
        c["span_no"].append(span_no)
        c["parent_no"].append(parent_no)
        c["service"].append(np.full(m, service, dtype=object))
        c["operation"].append(np.full(m, operation, dtype=object))
        c["kind"].append(np.full(m, kind, dtype=object))
        c["peer"].append(np.full(m, peer, dtype=object))
        c["start_ns"].append(start_ns)
        c["duration_ns"].append(duration_ns)
        c["status_error"].append(status_error)


def _lognormal(rng: np.random.Generator, n: int, sigma: float = LOG_SIGMA) -> np.ndarray:
    return rng.lognormal(-0.5 * sigma * sigma, sigma, n)


def _run_type(name: str, trace_no: np.ndarray, sec: np.ndarray, t0_ns: np.ndarray,
              eff: Effects, wait: dict, rng: np.random.Generator, sink: _SpanSink,
              end_ns: int, next_span_no: int) -> int:
    """Build every span of every trace of one request type, vectorised over traces."""
    flat = FLAT_TYPES[name]
    n = len(trace_no)
    if n == 0:
        return next_span_no
    k = len(flat)
    zeros = lambda: np.zeros(n)
    demand = [zeros() for _ in range(k)]     # CPU ms charged to cpu_service
    own = [zeros() for _ in range(k)]        # elapsed own work, excluding children
    net = [zeros() for _ in range(k)]
    killed = [np.zeros(n, bool) for _ in range(k)]
    kill_ms = [zeros() for _ in range(k)]
    err = [np.zeros(n, bool) for _ in range(k)]
    alive = [np.zeros(n, bool) for _ in range(k)]
    dur = [zeros() for _ in range(k)]
    start_off = [zeros() for _ in range(k)]

    # --- per-node draws
    for i, node in enumerate(flat):
        d = node.base_ms * eff.demand_factor[node.cpu_service][sec] * _lognormal(rng, n)
        demand[i] = d
        e = d * wait[node.queue_service][sec] * eff.elapsed_factor[node.service][sec]
        if node.cache:
            e = e * eff.cache_factor[node.service][sec]
        if node.kind in ("server", "consumer"):
            if node.kind == "consumer":
                # netem on the target delays its kafka fetch, and a consumer has no
                # caller-side client span for that delay to land in
                e = e + eff.net_delay_ms[node.service][sec]
            # a request that lands in a CFS throttling gap waits out the rest of the
            # period; this is elapsed time only, the cgroup is frozen, not busy
            stalled = rng.random(n) < eff.throttle_prob[node.service][sec]
            e = e + stalled * rng.random(n) * eff.throttle_ms[node.service][sec]
            outlier = rng.random(n) < OUTLIER_PROB
            e = e + outlier * rng.gamma(2.0, 30.0, n)
            err[i] = rng.random(n) < (eff.error_prob[node.service][sec] + BASE_ERROR_PROB)
        own[i] = e
        if node.kind in ("client", "producer"):
            # per-hop overhead is a property of the callee: for the cheap gRPC services
            # it dominates their own service time, which is what real traces show.
            # netem sits on the target's egress, so it delays everything the target
            # *sends*: this hop pays once if the target is the callee (its response) and
            # once if the target is the caller (its request). A mid-graph target is hit
            # on both sides; a leaf only on the response.
            net[i] = (PROFILES[node.peer].hop_ms * _lognormal(rng, n, 0.45)
                      + eff.net_delay_ms[node.peer][sec]
                      + eff.net_delay_ms[node.service][sec])
            p_unreachable = eff.unreachable[node.peer]
            if p_unreachable.any():
                # The callee is down: the call never reaches it, so no server span is
                # emitted. Some attempts are refused immediately, the rest hang.
                killed[i] = rng.random(n) < p_unreachable[sec]
                refused = rng.random(n) < eff.refused_frac
                kill_ms[i] = np.where(refused, eff.refused_ms, eff.unreach_timeout_ms) \
                    * _lognormal(rng, n, 0.3)
            # same egress logic for loss: the target's inbound *and* outbound calls
            # both cross the lossy interface
            p_loss = eff.loss_prob[node.peer] + eff.loss_prob[node.service]
            if node.kind == "client" and p_loss.any():
                # netem drops individual segments and a call puts several on the wire, so
                # the number of retransmissions per call is Poisson in the loss rate. Each
                # one costs an RTO: TCP recovers, so the call is slow, not failed, and the
                # application never issues a retry of its own.
                n_lost = rng.poisson(p_loss[sec] * SEGMENTS_PER_CALL)
                net[i] = net[i] + n_lost * rng.uniform(*RTO_MS, n)
                # a few calls do back off past the caller's socket timeout and fail
                p_give = eff.give_up_prob[node.peer] + eff.give_up_prob[node.service]
                gave_up = rng.random(n) < p_give[sec]
                killed[i] = killed[i] | gave_up
                kill_ms[i] = np.where(gave_up, eff.timeout_ms * _lognormal(rng, n, 0.15),
                                      kill_ms[i])

    # --- reachability, top-down (a parent always has a lower index)
    alive[0] = np.ones(n, bool)
    for i, node in enumerate(flat):
        if i == 0:
            continue
        a = alive[node.parent] & ~killed[node.parent]
        if node.prob < 1.0:
            a = a & (rng.random(n) < node.prob)
        alive[i] = a

    # --- durations and error propagation, bottom-up
    contrib = [zeros() for _ in range(k)]
    for i in range(k - 1, -1, -1):
        node = flat[i]
        kids = zeros()
        for block in node.blocks:
            m = contrib[block[0]]
            for c in block[1:]:
                m = np.maximum(m, contrib[c])
            kids = kids + m
            for c in block:
                if not flat[c].optional:
                    err[i] = err[i] | (err[c] & alive[c])
        err[i] = err[i] | killed[i]
        d = np.where(killed[i], kill_ms[i], own[i] + net[i] + kids)
        dur[i] = np.where(alive[i], d, 0.0)
        contrib[i] = np.where(alive[i], dur[i], 0.0)
        err[i] = err[i] & alive[i]

    # --- start offsets, top-down
    for i, node in enumerate(flat):
        if node.kind in ("server", "consumer", "internal"):
            cursor = start_off[i] + 0.4 * own[i]
        else:
            cursor = start_off[i] + own[i] + 0.5 * net[i]
        for block in node.blocks:
            m = zeros()
            for c in block:
                start_off[c] = cursor
                m = np.maximum(m, contrib[c])
            cursor = cursor + m
        for c in node.async_children:
            svc = flat[c].service
            lag = CONSUMER_BASE_LAG_MS + eff.consumer_wait_ms[svc][sec]
            start_off[c] = start_off[i] + dur[i] + lag

    # --- emit
    span_no = [np.zeros(n, np.int64) for _ in range(k)]
    for i, node in enumerate(flat):
        mask = alive[i].copy()
        starts = t0_ns + (start_off[i] * 1e6).astype(np.int64)
        mask &= starts < end_ns
        idx = np.flatnonzero(mask)
        m = len(idx)
        span_no[i][idx] = np.arange(next_span_no, next_span_no + m, dtype=np.int64)
        next_span_no += m
        parent = span_no[node.parent][idx] if node.parent >= 0 else np.full(m, -1, np.int64)
        sink.add(trace_no[idx], span_no[i][idx], parent, node.service, node.operation,
                 node.kind, node.peer, starts[idx],
                 np.maximum((dur[i][idx] * 1e6).astype(np.int64), 1000),
                 err[i][idx])

        # accumulators
        live = alive[i] & ~killed[i]
        comp = node.cpu_service
        sink.cpu_ms[comp] += np.bincount(sec[live], weights=demand[i][live],
                                         minlength=sink.n_sec)
        inbound = node.kind in ("server", "consumer") or node.parent < 0 or (
            node.kind in ("client", "producer") and node.peer in INFRA_CALL_MS)
        if inbound:
            sink.calls[comp] += np.bincount(sec[live], minlength=sink.n_sec).astype(float)
        if node.kind in ("server", "consumer"):
            sink.busy_s[node.service] += np.bincount(
                sec[alive[i]], weights=dur[i][alive[i]] / 1000.0, minlength=sink.n_sec)
    return next_span_no


# --- metrics ------------------------------------------------------------------------

def _metrics(sink: _SpanSink, eff: Effects, rng: np.random.Generator,
             start_ns: int, n_sec: int) -> pd.DataFrame:
    ts = start_ns + (np.arange(n_sec) * int(1e9 * METRIC_INTERVAL_S))
    frames = []
    for c in ALL_COMPONENTS:
        prof = PROFILES[c]
        calls = sink.calls[c]
        # everything running inside the cgroup, over the (possibly squeezed) allotment
        cores = prof.idle_util + sink.cpu_ms[c] / CPU_CAPACITY_MS[c]
        cpu = cores / np.maximum(eff.cpu_alloc_factor[c], 1e-3) + eff.cpu_extra[c]
        cpu = np.clip(cpu + rng.normal(0.0, 0.004, n_sec), 0.0, 1.6)

        base_mem = prof.mem_base_mb * 1e6
        drift = np.cumsum(rng.normal(0.0, base_mem * 6e-4, n_sec))
        if prof.runtime in GC_RUNTIMES:
            band = 0.28 * base_mem
            per_call = band / max(1.0, calls.mean() * 15.0)
            heap = np.mod(np.cumsum(calls * per_call * rng.uniform(0.85, 1.15, n_sec)), band)
            gc_event = np.diff(heap, prepend=heap[0]) < 0
        else:
            heap = np.zeros(n_sec)
            gc_event = np.zeros(n_sec, bool)
        mem = np.maximum(base_mem + heap + drift + eff.mem_extra[c], 0.3 * base_mem)

        gc = gc_event * rng.gamma(2.0, 6.0, n_sec) + rng.gamma(1.0, 0.3, n_sec)
        gc = gc + eff.gc_extra_ms[c]
        for svc, i in eff.restarts:
            if svc == c:
                gc[i] += 180.0

        if c in KAFKA_CONSUMERS:
            queue = eff.kafka_lag[c]                       # consumer lag in messages
        elif c == "kafka":
            queue = sum(eff.kafka_lag[k] for k in KAFKA_CONSUMERS)
        else:
            queue = sink.busy_s[c] + rng.gamma(1.0, 0.2, n_sec)   # in-flight requests
        threads = prof.threads_base + sink.busy_s[c] * 2.0 + rng.gamma(1.0, 0.8, n_sec)
        rx = np.cumsum(calls * NET_RX_BYTES[c] * rng.uniform(0.9, 1.1, n_sec))
        tx = np.cumsum(calls * NET_TX_BYTES[c] * rng.uniform(0.9, 1.1, n_sec))

        for metric, values in (
            (METRIC_CPU_UTIL, cpu),
            (METRIC_MEM_BYTES, mem),
            (METRIC_MEM_LIMIT, np.full(n_sec, prof.mem_limit_mb * 1e6)),
            (METRIC_QUEUE_DEPTH, np.maximum(queue, 0.0)),
            (METRIC_NET_RX_BYTES, rx),
            (METRIC_NET_TX_BYTES, tx),
            (METRIC_THREADS, np.round(threads)),
            (METRIC_GC_PAUSE_MS, np.maximum(gc, 0.0)),
        ):
            frames.append(pd.DataFrame({"ts_ns": ts, "service": c, "metric": metric,
                                        "value": values}))
    return pd.concat(frames, ignore_index=True)


# --- logs ---------------------------------------------------------------------------

INFO_BODIES = [
    "request handled", "response sent status=200", "processing request",
    "cache hit", "query completed", "flag evaluated", "span exported",
]
WARN_BODIES = [
    "slow response detected", "retrying upstream call", "connection pool nearly exhausted",
    "high memory usage", "consumer lag growing", "gc pause exceeded threshold",
    "request queue depth elevated", "cache miss ratio high", "socket read timeout",
    "connection reset by peer", "cpu throttling detected", "slow query detected",
]
# Fault-specific WARN phrasings. Which service emits them is decided per family by
# _fault_warn_sources; ERROR never comes from this block.
GC_PRESSURE_BODIES = ["heap usage above 90%", "OOM pressure",
                      "gc pause exceeded threshold", "allocation failure, retrying"]
QUEUE_DEPTH_BODIES = ["request queue depth elevated", "event loop lag high"]
LAG_BODIES = ["consumer lag growing", "commit offset behind head", "rebalance in progress"]
CACHE_BODIES = ["slow query detected", "cache miss latency high", "cache latency degraded"]
SLOW_CALL_BODIES = ["slow response detected", "socket read timeout",
                    "upstream {target} response time degraded"]
LOSSY_CALL_BODIES = ["slow response detected", "upstream {target} p99 latency high",
                     "request exceeded its latency budget"]
UNREACHABLE_BODIES = ["connection timeout to {target}", "{target} unreachable",
                      "circuit breaker open for {target}"]
RESTART_BODIES = ["reconnecting to dependencies after restart", "cache cold after restart",
                  "dropping in-flight requests"]


def _emit(rows: list, sec: np.ndarray, service, severity, bodies, trace_ids,
          start_ns: int, rng: np.random.Generator) -> None:
    if len(sec) == 0:
        return
    ts = start_ns + sec.astype(np.int64) * 1_000_000_000 + rng.integers(0, 1_000_000_000, len(sec))
    rows.append(pd.DataFrame({"ts_ns": ts, "service": service, "severity": severity,
                              "body": bodies, "trace_id": trace_ids}))


def _caller_shares(spans: pd.DataFrame, target: str, start_ns: int, n_sec: int,
                   errored_only: bool) -> dict[str, np.ndarray]:
    """Each caller's share of the calls into ``target``, per second."""
    sel = spans[(spans["peer_service"] == target) & (spans["kind"] == "client")]
    if errored_only:
        sel = sel[sel["status_error"]]
    if sel.empty:
        return {}
    secs = np.clip((sel["start_ns"].to_numpy() - start_ns) // 1_000_000_000, 0, n_sec - 1)
    callers = sel["service"].to_numpy().astype(str)
    total = np.maximum(np.bincount(secs, minlength=n_sec), 1)
    return {svc: np.bincount(secs[callers == svc], minlength=n_sec) / total
            for svc in np.unique(callers)}


def _fault_warn_sources(spec: FaultSpec, eff: Effects, spans: pd.DataFrame,
                        rng: np.random.Generator, start_ns: int,
                        n_sec: int) -> list[tuple[str, list[str], np.ndarray]]:
    """(service, bodies, per-second rate) for the fault-specific WARN lines.

    Who logs is family-specific and mirrors what a real service can actually notice: a
    throttled or lossy service has no idea it is degraded, its callers do. Families that
    only show up as failed requests contribute nothing here -- their log signal is the
    span-correlated ERROR lines instead.
    """
    t = np.arange(n_sec, dtype=float)
    env = fault_mod.envelope(t, spec.start_s, spec.start_s + spec.duration_s,
                             float(spec.params.get("ramp_s", 5.0)),
                             float(spec.params.get("recover_s", 5.0)))
    base = 0.8 * spec.intensity * env
    ft, target = spec.fault_type, spec.target

    if ft == "cpu_saturation":
        # a CFS-throttled process rarely says so; only some runtimes report backlog,
        # and only in a minority of experiments
        if rng.random() > 0.3:
            return []
        return [(target, QUEUE_DEPTH_BODIES, 0.4 * base)]
    if ft == "memory_leak":
        prof = PROFILES[target]
        headroom = 0.92 * prof.mem_limit_mb * 1e6 - prof.mem_base_mb * 1e6
        frac = np.clip(eff.mem_extra[target] / headroom, 0.0, 1.0)
        return [(target, GC_PRESSURE_BODIES, base * frac)]
    if ft == "error_rate":
        return []
    if ft == "queue_backlog":
        return [(c, LAG_BODIES, base) for c in spec.params.get("consumers", [target])]
    if ft == "cache_slowdown":
        return [(target, CACHE_BODIES, base)]

    # network_latency / packet_loss / dependency_failure: the target logs nothing and
    # its callers complain in proportion to the slow or failed calls they make into it.
    # packet loss only makes calls slower, so its callers grumble far less often than
    # callers of a service that is timing out or refusing connections outright.
    templates, errored_only, scale = {
        "network_latency": (SLOW_CALL_BODIES, False, 1.0),
        "packet_loss": (LOSSY_CALL_BODIES, False, 0.25),
        "dependency_failure": (UNREACHABLE_BODIES, True, 1.0),
    }[ft]
    bodies = [b.format(target=target) for b in templates]
    shares = _caller_shares(spans, target, start_ns, n_sec, errored_only)
    return [(svc, bodies, scale * base * share) for svc, share in shares.items()]


def _logs(sink: _SpanSink, spans: pd.DataFrame, eff: Effects, spec: FaultSpec | None,
          rng: np.random.Generator, start_ns: int, n_sec: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for c in ALL_COMPONENTS:
        calls = sink.calls[c]
        n_info = rng.poisson(0.30 * calls)
        sec = np.repeat(np.arange(n_sec), n_info)
        _emit(rows, sec, c, "INFO", rng.choice(INFO_BODIES, len(sec)), "", start_ns, rng)

        # background WARNs, including the "suspicious" phrasings, so logs alone do not
        # identify the culprit
        n_warn = rng.poisson(0.02 * calls + 0.015)
        sec = np.repeat(np.arange(n_sec), n_warn)
        _emit(rows, sec, c, "WARN", rng.choice(WARN_BODIES, len(sec)), "", start_ns, rng)

    if spec is not None:
        for svc, bodies, rate in _fault_warn_sources(spec, eff, spans, rng, start_ns, n_sec):
            n_f = rng.poisson(rate)
            sec = np.repeat(np.arange(n_sec), n_f)
            _emit(rows, sec, svc, "WARN", rng.choice(bodies, len(sec)), "", start_ns, rng)

    # a restart is the one thing that logs FATAL, plus a short burst of start-up errors
    for svc, i in eff.restarts:
        _emit(rows, np.array([i]), svc, "FATAL", ["out of memory: container killed"],
              "", start_ns, rng)
        n = int(rng.integers(2, 6))
        _emit(rows, np.full(n, min(i + 1, n_sec - 1)), svc, "ERROR",
              rng.choice(RESTART_BODIES, n), "", start_ns, rng)

    # ERROR lines correlated with failed spans. Both sides of a failed call log: the
    # caller for its failed client span and, when it answered at all, the callee for its
    # failed server span.
    bad = spans[spans["status_error"]]
    if len(bad):
        keep = bad.iloc[np.flatnonzero(rng.random(len(bad)) < 0.6)]
        peer = keep["peer_service"].to_numpy().astype(str)
        kind = keep["kind"].to_numpy().astype(str)
        prefix = rng.choice(["connection timeout to ", "upstream error from ",
                             "call failed: "], len(keep))
        body = np.where(peer != "", np.char.add(prefix, peer),
                        np.where(kind == "consumer", "message processing failed",
                                 "request failed with internal error"))
        ts = keep["start_ns"].to_numpy() + keep["duration_ns"].to_numpy()
        rows.append(pd.DataFrame({
            "ts_ns": np.minimum(ts, start_ns + n_sec * 1_000_000_000 - 1),
            "service": keep["service"].to_numpy(), "severity": "ERROR",
            "body": body, "trace_id": keep["trace_id"].to_numpy()}))

    if not rows:
        return pd.DataFrame({"ts_ns": [], "service": [], "severity": [], "body": [],
                             "trace_id": []})
    return pd.concat(rows, ignore_index=True)


# --- experiment ---------------------------------------------------------------------

def generate_experiment(seed: int, duration_s: float = 180.0,
                        fault_spec: FaultSpec | None = None,
                        traffic: TrafficProfile | None = None,
                        warmup_s: float = WARMUP_S, source: str = "sim",
                        start_ns: int = EPOCH_NS,
                        extra: dict | None = None) -> Experiment:
    """Simulate one experiment. The same ``seed`` and arguments give identical output."""
    rng = np.random.default_rng(seed)
    n_sec = math.ceil(duration_s)
    if traffic is None:
        traffic = TrafficProfile(base_rps=20.0, shape="steady", params={})
    if fault_spec is not None:
        fault_spec.validate()
        if fault_spec.start_s < warmup_s:
            raise ValueError("fault must start after the warm-up window")
    end_ns = start_ns + n_sec * 1_000_000_000

    # arrivals
    rate = _rate_series(traffic, n_sec, rng)
    counts = rng.poisson(rate)
    n_traces = int(counts.sum())
    sec_idx = np.repeat(np.arange(n_sec), counts)
    arrival_ns = (start_ns + sec_idx.astype(np.int64) * 1_000_000_000
                  + rng.integers(0, 1_000_000_000, n_traces))

    mix_vec = rng.dirichlet(np.array([DEFAULT_MIX[t] for t in TYPE_NAMES]) * 250.0)
    types = rng.choice(len(TYPE_NAMES), size=n_traces, p=mix_vec)
    counts_by_type = np.stack(
        [np.bincount(sec_idx[types == k], minlength=n_sec) for k in range(len(TYPE_NAMES))],
        axis=1).astype(float)

    produced = counts_by_type[:, TYPE_NAMES.index("checkout")]
    eff = build_effects(fault_spec, n_sec, rng, produced,
                        baseline_cores(counts_by_type, warmup_s))
    _, wait = _utilisation(counts_by_type, eff)

    sink = _SpanSink(n_sec)
    next_span_no = 0
    for k, name in enumerate(TYPE_NAMES):
        sel = np.flatnonzero(types == k)
        next_span_no = _run_type(name, sel, sec_idx[sel], arrival_ns[sel], eff, wait,
                                 rng, sink, end_ns, next_span_no)

    spans = _assemble_spans(sink, n_traces, next_span_no, rng)
    metrics = _metrics(sink, eff, rng, start_ns, n_sec)
    logs = _logs(sink, spans, eff, fault_spec, rng, start_ns, n_sec)

    manifest = Manifest(
        experiment_id=f"{source}-{seed:06d}",
        source=source,
        seed=int(seed),
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        traffic=traffic,
        faults=[] if fault_spec is None else [Fault(
            fault_type=fault_spec.fault_type,
            target=fault_spec.target,
            start_ns=int(start_ns + fault_spec.start_s * 1e9),
            end_ns=int(start_ns + (fault_spec.start_s + fault_spec.duration_s) * 1e9),
            intensity=float(fault_spec.intensity),
            params=dict(fault_spec.params),
        )],
        services=list(SERVICES),
        edges=list(DEPENDENCY_EDGES),
        metric_interval_ns=int(1e9 * METRIC_INTERVAL_S),
        warmup_ns=int(warmup_s * 1e9),
        extra={
            "sim_version": SIM_VERSION,
            "request_mix": {name: float(p) for name, p in zip(TYPE_NAMES, mix_vec)},
            "ref_rps": fault_mod.REF_RPS,
            "consumer_capacity_msgs_s": fault_mod.CONSUMER_CAPACITY,
            "n_traces": n_traces,
            "restarts": [[s, int(i)] for s, i in eff.restarts],
            **(extra or {}),
        },
    )
    return Experiment(manifest=manifest, metrics=metrics, spans=spans, logs=logs)


def _assemble_spans(sink: _SpanSink, n_traces: int, n_spans: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    cols = {k: (np.concatenate(v) if v else np.array([])) for k, v in sink.cols.items()}
    trace_nonce = int(rng.integers(1, 2 ** 60))
    span_nonce = int(rng.integers(1, 2 ** 30))
    trace_ids = np.array([f"{trace_nonce:016x}{i:016x}" for i in range(n_traces)], dtype=object)
    span_ids = np.array([f"{span_nonce:08x}{i:08x}" for i in range(n_spans)], dtype=object)
    parent_no = cols["parent_no"].astype(np.int64)
    return pd.DataFrame({
        "trace_id": trace_ids[cols["trace_no"].astype(np.int64)],
        "span_id": span_ids[cols["span_no"].astype(np.int64)],
        "parent_span_id": np.where(parent_no >= 0, span_ids[np.maximum(parent_no, 0)], ""),
        "service": cols["service"],
        "operation": cols["operation"],
        "start_ns": cols["start_ns"].astype(np.int64),
        "duration_ns": cols["duration_ns"].astype(np.int64),
        "status_error": cols["status_error"].astype(bool),
        "peer_service": cols["peer"],
        "kind": cols["kind"],
    })


# --- dataset ------------------------------------------------------------------------

def generate_dataset(out_root: Path, n_experiments: int, seed: int = 0,
                     duration_range: tuple[float, float] = (120.0, 240.0),
                     normal_fraction: float = 0.20,
                     traffic_only_spike_fraction: float = 0.10,
                     warmup_s: float = WARMUP_S, source: str = "sim",
                     start_ns: int = EPOCH_NS, gap_s: float = 60.0,
                     progress: bool = False) -> list[Path]:
    """Generate ``n_experiments`` randomised experiments under ``out_root``.

    Experiments are laid out back to back on a single timeline in generation order,
    separated by ``gap_s`` of teardown/setup, so a chronological split of the dataset
    is well defined. The layout is a deterministic function of ``seed``.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.choice(1_000_000, size=n_experiments, replace=False)
    gap_ns = int(gap_s * 1e9)
    cursor = int(start_ns)
    out: list[Path] = []
    for i, exp_seed in enumerate(seeds):
        duration_s = float(rng.uniform(*duration_range))
        roll = rng.random()
        is_normal = roll < normal_fraction
        is_spike = normal_fraction <= roll < normal_fraction + traffic_only_spike_fraction
        traffic = sample_traffic(rng, duration_s, warmup_s, with_spike=is_spike)
        spec = None if (is_normal or is_spike) else sample_fault(rng, duration_s, warmup_s)
        exp = generate_experiment(int(exp_seed), duration_s, spec, traffic,
                                  warmup_s=warmup_s, source=source, start_ns=cursor,
                                  extra={"category": "normal" if is_normal else
                                         "traffic_spike" if is_spike else "fault",
                                         "run_index": i})
        cursor = exp.manifest.end_ns + gap_ns
        out.append(write_experiment(exp, Path(out_root)))
        if progress:
            print(f"[{i + 1}/{n_experiments}] {exp.manifest.experiment_id} "
                  f"{'none' if spec is None else spec.fault_type + '@' + spec.target}")
    return out
