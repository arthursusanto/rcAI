"""Canonical experiment data contract.

Every telemetry source (real OpenTelemetry Demo ingest, simulator) writes exactly this
layout; every downstream stage (features, models, serving) reads only this.

experiments/<experiment_id>/
    manifest.json     -> Manifest
    metrics.parquet   -> METRICS_COLUMNS
    spans.parquet     -> SPANS_COLUMNS
    logs.parquet      -> LOGS_COLUMNS

Timestamps are int64 nanoseconds since the Unix epoch. Experiment time zero is
manifest.start_ns; faults are expressed in absolute ns too.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

# --- label spaces -------------------------------------------------------------------

SERVICES: list[str] = [
    "frontend",
    "frontend-proxy",
    "cart",
    "checkout",
    "currency",
    "email",
    "payment",
    "product-catalog",
    "recommendation",
    "shipping",
    "ad",
    "quote",
    "accounting",
    "fraud-detection",
    "image-provider",
]

# Infra components carry telemetry but are attributed to an owning application service.
INFRA_OWNER: dict[str, str] = {
    "kafka": "checkout",
    "valkey": "cart",
    "postgresql": "product-catalog",
    "flagd": "frontend",
    "load-generator": "frontend-proxy",
}

FAULT_TYPES: list[str] = [
    "cpu_saturation",
    "memory_leak",
    "network_latency",
    "packet_loss",
    "dependency_failure",
    "error_rate",
    "queue_backlog",
    "cache_slowdown",
]

NO_FAULT = "none"

# Caller -> callee edges of the OpenTelemetry Demo (application services plus the
# infra components they talk to). Source of truth for graph features.
DEPENDENCY_EDGES: list[tuple[str, str]] = [
    ("load-generator", "frontend-proxy"),
    ("frontend-proxy", "frontend"),
    ("frontend-proxy", "image-provider"),
    ("frontend-proxy", "flagd"),
    ("frontend", "ad"),
    ("frontend", "cart"),
    ("frontend", "checkout"),
    ("frontend", "currency"),
    ("frontend", "product-catalog"),
    ("frontend", "recommendation"),
    ("frontend", "shipping"),
    ("frontend", "flagd"),
    ("cart", "valkey"),
    ("cart", "flagd"),
    ("checkout", "cart"),
    ("checkout", "currency"),
    ("checkout", "email"),
    ("checkout", "payment"),
    ("checkout", "product-catalog"),
    ("checkout", "shipping"),
    ("checkout", "kafka"),
    ("checkout", "flagd"),
    ("recommendation", "product-catalog"),
    ("recommendation", "flagd"),
    ("shipping", "quote"),
    ("product-catalog", "postgresql"),
    ("product-catalog", "flagd"),
    ("payment", "flagd"),
    ("ad", "flagd"),
    ("kafka", "accounting"),
    ("kafka", "fraud-detection"),
    ("accounting", "postgresql"),
    ("fraud-detection", "flagd"),
]

# --- metric names -------------------------------------------------------------------
# Resource/infra metrics are sampled per service at a fixed interval (default 1 s).
METRIC_CPU_UTIL = "cpu_util"            # fraction of allotted CPU, 0..1+ (bursts can exceed 1)
METRIC_MEM_BYTES = "mem_bytes"          # resident memory
METRIC_MEM_LIMIT = "mem_limit_bytes"    # container limit
METRIC_QUEUE_DEPTH = "queue_depth"      # pending items (kafka lag for consumers, pool wait otherwise)
METRIC_NET_RX_BYTES = "net_rx_bytes"    # cumulative
METRIC_NET_TX_BYTES = "net_tx_bytes"    # cumulative
METRIC_THREADS = "threads"              # active workers / goroutines
METRIC_GC_PAUSE_MS = "gc_pause_ms"      # gc pause time within the interval

METRIC_NAMES = [
    METRIC_CPU_UTIL, METRIC_MEM_BYTES, METRIC_MEM_LIMIT, METRIC_QUEUE_DEPTH,
    METRIC_NET_RX_BYTES, METRIC_NET_TX_BYTES, METRIC_THREADS, METRIC_GC_PAUSE_MS,
]

# --- tables -------------------------------------------------------------------------
METRICS_COLUMNS = {
    "ts_ns": "int64",
    "service": "string",
    "metric": "string",
    "value": "float64",
}

SPANS_COLUMNS = {
    "trace_id": "string",
    "span_id": "string",
    "parent_span_id": "string",     # "" for root spans
    "service": "string",            # emitting service
    "operation": "string",
    "start_ns": "int64",
    "duration_ns": "int64",
    "status_error": "bool",
    "peer_service": "string",       # callee for client spans, "" otherwise
    "kind": "string",               # server | client | producer | consumer | internal
}

LOGS_COLUMNS = {
    "ts_ns": "int64",
    "service": "string",
    "severity": "string",           # DEBUG | INFO | WARN | ERROR | FATAL
    "body": "string",
    "trace_id": "string",           # "" if not correlated
}

SEVERITIES = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]


@dataclass
class Fault:
    fault_type: str                 # one of FAULT_TYPES
    target: str                     # one of SERVICES (the root-cause label)
    start_ns: int
    end_ns: int
    intensity: float                # 0..1, family-specific meaning (documented by the injector)
    params: dict = field(default_factory=dict)   # raw injector parameters for reproducibility


@dataclass
class TrafficProfile:
    base_rps: float                 # mean request rate at the entry point
    shape: str                      # steady | ramp | diurnal | bursty
    params: dict = field(default_factory=dict)


@dataclass
class Manifest:
    experiment_id: str
    source: str                     # "otel-demo" | "sim"
    seed: int
    start_ns: int
    end_ns: int
    traffic: TrafficProfile
    faults: list[Fault]             # empty for pure-normal / traffic-only experiments
    services: list[str]
    edges: list[tuple[str, str]]
    metric_interval_ns: int
    warmup_ns: int                  # initial fault-free period usable as a per-experiment baseline
    version: str = "1"
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        d = json.loads(text)
        d["traffic"] = TrafficProfile(**d["traffic"])
        d["faults"] = [Fault(**f) for f in d["faults"]]
        d["edges"] = [tuple(e) for e in d["edges"]]
        return cls(**d)


@dataclass
class Experiment:
    manifest: Manifest
    metrics: pd.DataFrame
    spans: pd.DataFrame
    logs: pd.DataFrame


def _coerce(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    out = df[list(columns)].copy()
    for col, dtype in columns.items():
        out[col] = out[col].astype(dtype)
    return out


EXPERIMENT_FILES = ("metrics.parquet", "spans.parquet", "logs.parquet", "manifest.json")


def write_experiment(exp: Experiment, root: Path) -> Path:
    """Write one experiment directory. manifest.json is written last, deliberately.

    It is the marker ``list_experiments`` keys on, so a run killed part-way through
    leaves a directory that is skipped rather than one that reads back as an experiment
    with a missing or truncated parquet.
    """
    d = Path(root) / exp.manifest.experiment_id
    d.mkdir(parents=True, exist_ok=True)
    _coerce(exp.metrics, METRICS_COLUMNS).sort_values("ts_ns").to_parquet(
        d / "metrics.parquet", index=False)
    _coerce(exp.spans, SPANS_COLUMNS).sort_values("start_ns").to_parquet(
        d / "spans.parquet", index=False)
    _coerce(exp.logs, LOGS_COLUMNS).sort_values("ts_ns").to_parquet(
        d / "logs.parquet", index=False)
    (d / "manifest.json").write_text(exp.manifest.to_json(), newline="\n")
    return d


def read_manifest(exp_dir: Path) -> Manifest:
    return Manifest.from_json((Path(exp_dir) / "manifest.json").read_text())


def read_experiment(exp_dir: Path) -> Experiment:
    exp_dir = Path(exp_dir)
    return Experiment(
        manifest=read_manifest(exp_dir),
        metrics=pd.read_parquet(exp_dir / "metrics.parquet"),
        spans=pd.read_parquet(exp_dir / "spans.parquet"),
        logs=pd.read_parquet(exp_dir / "logs.parquet"),
    )


def list_experiments(root: Path) -> list[Path]:
    """Complete experiment directories, in id order.

    A directory missing any of EXPERIMENT_FILES is an interrupted write and is skipped:
    reading it would either raise or, worse, yield an experiment with empty telemetry.
    """
    return sorted(p for p in Path(root).iterdir()
                  if all((p / name).exists() for name in EXPERIMENT_FILES))
