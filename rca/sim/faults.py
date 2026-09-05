"""Fault model: which faults are valid where, and what they do to the system.

A fault is turned into per-second effect arrays (:class:`Effects`) that the span,
metric and log generators read. Effects ramp in over a few seconds after the start and
decay over a few seconds after the end, so nothing is a step function.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rca.benchmark.injector import (
    CPU_MECHANISMS,
    DEFAULT_CPU_MECHANISM,
    HOG_DUTY_RANGE,
    MIN_CPUS,
    allowed_targets,
)
from rca.data.schema import FAULT_TYPES, SERVICES

from .topology import (
    ALL_COMPONENTS,
    CACHE_PEER,
    CALL_EDGES,
    KAFKA_CONSUMERS,
    PROFILES,
    REF_RPS,
)
from .topology import DEFAULT_MIX as _MIX

# Services reachable by a synchronous inbound RPC. packet_loss needs a caller to retry
# and dependency_failure needs a caller to see the service go unreachable; the two
# kafka-only consumers have neither (that scenario is queue_backlog).
SYNC_TARGETS = sorted({c for _, c in CALL_EDGES if c in SERVICES})

# Per-consumer kafka throughput in messages/s. Sized off the top of the generated
# traffic range rather than its mean, so a normal system never lags at any traffic level
# the generator produces but a large spike still briefly can.
CONSUMER_CAPACITY = 6.0 * REF_RPS * _MIX["checkout"]

# The CFS accounting period: a cgroup that exhausts its quota is frozen for the
# remainder of the current period.
CFS_PERIOD_MS = 100.0

# What the simulated topology can express, per family. Absent = no extra constraint.
_STRUCTURAL: dict[str, set[str]] = {
    "packet_loss": set(SYNC_TARGETS),
    "dependency_failure": set(SYNC_TARGETS),
    "queue_backlog": {"checkout", *KAFKA_CONSUMERS},
    "cache_slowdown": set(CACHE_PEER),
}

# Single source of truth: what the real injector can do to the demo, narrowed to what
# the simulated call graph can actually express. Do not hand-edit these lists.
def valid_targets(fault_type: str,
                  cpu_mechanism: str = DEFAULT_CPU_MECHANISM) -> list[str]:
    """What the real injector can do, narrowed to what the simulated graph expresses."""
    allowed = allowed_targets(fault_type, cpu_mechanism)
    return [s for s in allowed if s in _STRUCTURAL.get(fault_type, set(SERVICES))]


VALID_TARGETS: dict[str, list[str]] = {ft: valid_targets(ft) for ft in FAULT_TYPES}


def cpu_mechanism_of(params: dict) -> str:
    """Resolve params["mechanism"] to one of injector.CPU_MECHANISMS.

    The injector writes a descriptive value into the same key on the way out
    ("exec-cpu-hog-quota", "exec-cpu-hog", "docker-update-cpus"), so a manifest written
    by a real run has to resolve as well as a selector written by a caller.
    """
    raw = str(params.get("mechanism", DEFAULT_CPU_MECHANISM))
    hog = "hog" in raw
    quota = "quota" in raw or "cpus" in raw
    if hog and quota:
        return "hog+quota"
    if hog:
        return "hog-only"
    if quota:
        return "quota-only"
    raise ValueError(f"cpu_saturation mechanism must be one of {CPU_MECHANISMS}, "
                     f"got {raw!r}")

# Every valid (fault_type, target) combination, so a fault can be drawn uniformly over
# pairs rather than over families with wildly different numbers of legal targets.
FAULT_TARGET_PAIRS: list[tuple[str, str]] = [
    (ft, t) for ft, targets in VALID_TARGETS.items() for t in targets]


@dataclass
class FaultSpec:
    """A fault relative to experiment time zero (seconds)."""
    fault_type: str
    target: str
    start_s: float
    duration_s: float
    intensity: float
    params: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.fault_type not in FAULT_TYPES:
            raise ValueError(f"unknown fault type {self.fault_type!r}")
        if self.fault_type == "cpu_saturation":
            # the hog needs a shell in the image; the quota squeeze reaches every container
            allowed = valid_targets("cpu_saturation", cpu_mechanism_of(self.params))
        else:
            allowed = VALID_TARGETS[self.fault_type]
        if self.target not in allowed:
            raise ValueError(
                f"{self.fault_type} is not valid on {self.target!r}; allowed: {allowed}")


class Effects:
    """Per-second effect arrays. Neutral values are 1.0 (factors) and 0.0 (additive)."""

    def __init__(self, n_sec: int):
        one = lambda: np.ones(n_sec)
        zero = lambda: np.zeros(n_sec)
        self.n_sec = n_sec
        self.capacity_factor = {c: one() for c in ALL_COMPONENTS}   # share of the allotment
        self.cpu_alloc_factor = {c: one() for c in ALL_COMPONENTS}  # size of the allotment
        self.throttle_prob = {c: zero() for c in ALL_COMPONENTS}    # requests hit a CFS gap
        self.throttle_ms = {c: zero() for c in ALL_COMPONENTS}      # longest such gap
        self.demand_factor = {c: one() for c in ALL_COMPONENTS}     # CPU work per call
        self.elapsed_factor = {c: one() for c in ALL_COMPONENTS}    # elapsed only, no CPU cost
        self.cache_factor = {c: one() for c in ALL_COMPONENTS}      # elapsed on cache/db spans
        self.net_delay_ms = {c: zero() for c in ALL_COMPONENTS}     # on the wire INTO c
        self.loss_prob = {c: zero() for c in ALL_COMPONENTS}        # segments lost INTO c
        self.give_up_prob = {c: zero() for c in ALL_COMPONENTS}     # callers of c that time out
        self.unreachable = {c: zero() for c in ALL_COMPONENTS}      # calls INTO c never land
        self.reachable_frac = {c: one() for c in ALL_COMPONENTS}    # share of load c still sees
        self.error_prob = {c: zero() for c in ALL_COMPONENTS}       # server/consumer errors
        self.cpu_extra = {c: zero() for c in ALL_COMPONENTS}        # externally burnt cpu
        self.mem_extra = {c: zero() for c in ALL_COMPONENTS}        # leaked bytes
        self.gc_extra_ms = {c: zero() for c in ALL_COMPONENTS}
        self.consumer_wait_ms = {c: np.zeros(n_sec) for c in KAFKA_CONSUMERS}
        self.kafka_lag = {c: np.zeros(n_sec) for c in KAFKA_CONSUMERS}
        self.restarts: list[tuple[str, int]] = []
        self.timeout_ms = 2000.0            # caller socket timeout
        self.refused_ms = 6.0               # connection refused, fails fast
        self.unreach_timeout_ms = 3000.0    # connection hangs until the caller gives up
        self.refused_frac = 0.5             # share of unreachable calls that fail fast


def envelope(t: np.ndarray, start: float, end: float, ramp: float, recover: float) -> np.ndarray:
    """0 before start, ramps to 1, holds, decays to 0 after end."""
    rise = np.clip((t - start) / max(ramp, 1e-6), 0.0, 1.0)
    fall = np.clip(1.0 - (t - end) / max(recover, 1e-6), 0.0, 1.0)
    return rise * fall


def _kafka_lag(produced: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    lag = np.zeros(len(produced))
    backlog = 0.0
    for i in range(len(produced)):
        backlog = max(0.0, backlog + produced[i] - capacity[i])
        lag[i] = backlog
    return lag


def _memory_leak(eff: Effects, spec: FaultSpec, t: np.ndarray, env: np.ndarray,
                 rng: np.random.Generator) -> None:
    target = spec.target
    prof = PROFILES[target]
    limit = prof.mem_limit_mb * 1e6
    base = prof.mem_base_mb * 1e6
    headroom = 0.92 * limit - base
    full_s = spec.params.setdefault("leak_full_s", float(rng.uniform(40.0, 140.0)))
    rate = spec.intensity * headroom / full_s          # bytes per second at full effect
    restart_s = spec.params.setdefault("restart_s", float(rng.uniform(3.0, 7.0)))

    leaked = 0.0
    unavailable_until = -1.0
    for i in range(eff.n_sec):
        leaked += rate * env[i]
        if leaked >= headroom:
            eff.restarts.append((target, i))
            leaked = 0.0
            unavailable_until = t[i] + restart_s
        eff.mem_extra[target][i] = leaked
        frac = leaked / headroom
        eff.gc_extra_ms[target][i] = 60.0 * frac ** 3 * env[i]
        eff.demand_factor[target][i] = 1.0 + 0.7 * spec.intensity * frac * env[i]
        if t[i] < unavailable_until:
            eff.error_prob[target][i] = 0.95
            eff.demand_factor[target][i] = 1.0
            eff.mem_extra[target][i] = -0.35 * base       # cold process, smaller heap


def build_effects(spec: FaultSpec | None, n_sec: int, rng: np.random.Generator,
                  produced_per_s: np.ndarray,
                  baseline_cores: dict[str, float] | None = None) -> Effects:
    """Turn a fault spec (or None) into per-second effect arrays."""
    eff = Effects(n_sec)
    baseline_cores = baseline_cores or {}
    t = np.arange(n_sec, dtype=float)
    capacity = np.full(n_sec, CONSUMER_CAPACITY)

    if spec is not None:
        spec.validate()
        i = float(spec.intensity)
        start, end = spec.start_s, spec.start_s + spec.duration_s
        ramp = spec.params.setdefault("ramp_s", float(rng.uniform(3.0, 12.0)))
        recover = spec.params.setdefault("recover_s", float(rng.uniform(3.0, 15.0)))
        env = envelope(t, start, end, ramp, recover)
        target = spec.target
        kind = spec.fault_type

        if kind == "cpu_saturation":
            spec.params.setdefault("mechanism", DEFAULT_CPU_MECHANISM)
            baseline = max(baseline_cores.get(target, 0.02), 1e-4)
            mechanism = cpu_mechanism_of(spec.params)
            if mechanism != "quota-only":
                # A busy loop inside the target's cgroup, plus a quota squeeze sized
                # from what the service was actually using. The hog alone does not slow
                # an idle service -- CFS schedules it promptly -- so the squeeze is what
                # creates the pressure: the allotment leaves the service about half the
                # cores it was drawing, its utilisation goes to ~2, and it queues.
                if mechanism == "hog+quota":
                    # Intensity drives the quota; the hog is then sized to fill whatever
                    # the service does not need, so the service is always left about half
                    # its draw and cpu_util reads ~1 at every intensity. What grows with
                    # intensity is the CFS stall: the cgroup exhausts a quota of Q cores
                    # partway into each 100 ms period and everything in it freezes for
                    # the rest of it.
                    quota = spec.params.setdefault(
                        "cpus", round(max(MIN_CPUS, 0.8 - 0.75 * i), 4))
                    headroom = max(MIN_CPUS, 0.5 * baseline)
                    duty = spec.params.setdefault(
                        "hog_duty", round(max(0.0, quota - headroom), 4))
                    eff.throttle_prob[target] = (1.0 - quota) * env
                    eff.throttle_ms[target] = (1.0 - quota) * CFS_PERIOD_MS * env
                else:
                    # hog-only: no quota, so no throttling gap -- measured live, this
                    # raises cpu_util and leaves latency alone.
                    quota = 1.0
                    duty = spec.params.setdefault(
                        "hog_duty", round(HOG_DUTY_RANGE[0]
                                          + (HOG_DUTY_RANGE[1] - HOG_DUTY_RANGE[0]) * i, 3))
                hog_t = duty * env                       # cores the hog burns
                alloc_t = 1.0 + (quota - 1.0) * env      # the allotment, ramping down
                eff.capacity_factor[target] = np.maximum(alloc_t - hog_t, 1e-3)
                eff.cpu_alloc_factor[target] = alloc_t
                eff.cpu_extra[target] = hog_t / alloc_t
            else:
                # CFS quota squeeze: the allotment itself shrinks and nothing burns extra
                # cycles. Leaves an idle service with no symptom at all, which is why it
                # is no longer the default.
                cpus = spec.params.setdefault("cpus", round(max(1.0 - 0.95 * i, 0.05), 2))
                eff.capacity_factor[target] = 1.0 - (1.0 - cpus) * env
                eff.cpu_alloc_factor[target] = eff.capacity_factor[target]
        elif kind == "memory_leak":
            _memory_leak(eff, spec, t, env, rng)
        elif kind == "network_latency":
            # netem on the target egress: the delay is paid once, on the wire, so it
            # lands in the callers client spans and never in the targets server spans
            delay = spec.params.setdefault("delay_ms", float(rng.uniform(25.0, 320.0)))
            eff.net_delay_ms[target] = delay * i * env
        elif kind == "packet_loss":
            # netem loss on the target. TCP retransmits the lost segment, so the call is
            # slow, not failed: no application error and no application-level retry. Only
            # a call whose backoff runs past the caller's socket timeout actually fails.
            loss_pct = spec.params.setdefault("loss_pct", round(2.0 + 38.0 * i, 1))
            eff.loss_prob[target] = (loss_pct / 100.0) * env
            eff.give_up_prob[target] = 0.02 * i * env
            eff.cpu_extra[target] = 0.06 * i * env
            eff.timeout_ms = spec.params.setdefault(
                "client_timeout_ms", float(rng.uniform(1000.0, 3000.0)))
        elif kind == "dependency_failure":
            # The target itself stops answering (docker pause / *Unreachable flag), so
            # every call INTO it fails. intensity 0.2 -> ~30% of calls, 1.0 -> all of them.
            fail = np.clip(0.125 + 0.875 * i, 0.0, 1.0)
            eff.unreachable[target] = fail * env
            eff.reachable_frac[target] = 1.0 - fail * env
            eff.refused_frac = spec.params.setdefault(
                "refused_fraction", float(rng.uniform(0.2, 0.9)))
            eff.refused_ms = spec.params.setdefault(
                "refused_ms", float(rng.uniform(2.0, 20.0)))
            eff.unreach_timeout_ms = spec.params.setdefault(
                "timeout_ms", float(rng.uniform(1500.0, 5000.0)))
            spec.params.setdefault("failed_call_fraction", float(fail))
        elif kind == "error_rate":
            eff.error_prob[target] = 0.7 * i * env
        elif kind == "queue_backlog":
            slowed = list(KAFKA_CONSUMERS) if target == "checkout" else [target]
            spec.params.setdefault("consumers", slowed)
            capacity = capacity * (1.0 - 0.92 * i * env)
            # a stalled consumer is blocked, not busy: latency grows, CPU does not
            for c in slowed:
                eff.elapsed_factor[c] = 1.0 + 3.0 * i * env
        elif kind == "cache_slowdown":
            eff.cache_factor[target] = 1.0 + 30.0 * i * env
            eff.demand_factor[target] = 1.0 + 0.4 * i * env
            peer = CACHE_PEER[target]
            if peer:
                eff.capacity_factor[peer] = 1.0 - 0.3 * i * env

    # kafka lag applies to every consumer; only the slowed ones have reduced capacity.
    for c in KAFKA_CONSUMERS:
        cap = capacity if _consumer_is_slowed(spec, c) else np.full(n_sec, CONSUMER_CAPACITY)
        lag = _kafka_lag(produced_per_s, cap)
        eff.kafka_lag[c] = lag
        eff.consumer_wait_ms[c] = 1000.0 * lag / np.maximum(cap, 0.05)
    return eff


def _consumer_is_slowed(spec: FaultSpec | None, consumer: str) -> bool:
    if spec is None or spec.fault_type != "queue_backlog":
        return False
    return spec.target == "checkout" or spec.target == consumer


def sample_fault(rng: np.random.Generator, duration_s: float, warmup_s: float,
                 fault_type: str | None = None, target: str | None = None) -> FaultSpec:
    """Draw a random valid fault that fits inside the experiment after the warm-up.

    Pairs are weighted 1/sqrt(n_targets of the family): a compromise between uniform
    over pairs (which starves families with few legal targets, e.g. error_rate with 3)
    and uniform over families (which over-represents the few targets of those families).
    """
    pairs = [(f, t) for f, t in FAULT_TARGET_PAIRS
             if (fault_type is None or f == fault_type) and (target is None or t == target)]
    if not pairs:
        raise ValueError(f"no valid fault for type={fault_type!r} target={target!r}")
    w = np.array([1.0 / np.sqrt(len(VALID_TARGETS[f])) for f, _ in pairs])
    fault_type, target = pairs[int(rng.choice(len(pairs), p=w / w.sum()))]
    tail = 10.0                                   # leave room to observe the recovery
    earliest = warmup_s + 5.0
    latest = max(earliest + 1.0, duration_s - tail - 30.0)
    start = float(rng.uniform(earliest, latest))
    max_dur = max(20.0, duration_s - tail - start)
    dur = float(rng.uniform(min(30.0, max_dur), max_dur))
    return FaultSpec(fault_type, target, start, dur, float(rng.uniform(0.2, 1.0)))
