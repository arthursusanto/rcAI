"""Fixed-width window table construction.

``build_windows`` turns one canonical experiment (``rca.data.schema.Experiment``) into
the window table described by ``rca.features.schema``: one row per (window, service) for
every application service in the manifest, with metric / trace / log / graph features and
the supervision labels.

Structure (the graph features force two passes):

1. ``raw_window_features``  -- vectorized per-(window, service) aggregates of metrics,
   spans and logs, plus per-(window, service, peer) outbound aggregates.
2. ``fit_baseline`` on the warm-up windows, then ``apply_baseline`` -- robust z-scores
   and the per-dependency maxima that need the baseline.
3. ``graph_features`` -- aggregates of the *neighbours'* pass-1/2 features over the
   dependency graph, which can only be computed once every service has its own row.
4. ``temporal_features`` -- trailing-window context per (experiment, service), which can
   only be computed once the rows exist. It is deliberately a standalone function over
   the *finished* window table, taking nothing a streaming detector would not have: only
   the current window and the ones before it, addressed by ``window_idx`` so that an
   overlapping stride is handled without special cases. Serving calls it on its buffer of
   past window rows and gets exactly the values this build produced.

Infra components (kafka, valkey, ...) are not rows of their own, and the two halves of
their telemetry are attributed differently. Their *metrics* are the component's own, so
they fold onto the owning application service (``f_metrics_infra_*``) -- the only
attribution available, and lossy for a shared component. Their *spans* do not exist: a
database, cache or broker call is a single client span on whoever made the call, so the
trace-side infra features (``f_traces_infra_*``) sit on the calling service's row,
owner or not. Dependency edges touching infra are dropped rather than rewritten onto an
owner, so no fabricated service-to-service edge ever enters the graph features; the one
place the unfolded graph is still used is ``f_graph_depth_from_entry``, where infra
counts as a hop on the path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rca.data import schema
from rca.features import schema as fschema
from rca.features.baseline import Baseline, fit_baseline, ratio_column, z_columns

# --- knobs --------------------------------------------------------------------------
DEFAULT_WINDOW_NS = 10_000_000_000

METRIC_QUANTITIES = [
    "cpu_util", "mem_frac", "queue_depth", "threads", "gc_pause_ms",
    "net_rx_rate", "net_tx_rate",
]
SLOPE_QUANTITIES = ["mem_frac", "queue_depth"]
# Byte rates climb with traffic, so a z-score of one flags a load ramp as an anomaly.
# What is stable under load is bytes *per request*; that is what gets the baseline
# treatment, while the raw rates stay as plain features.
PER_REQUEST_QUANTITIES = {"net_rx_rate": "net_rx_per_req", "net_tx_rate": "net_tx_per_req"}
LEVEL_Z_QUANTITIES = [q for q in METRIC_QUANTITIES if q not in PER_REQUEST_QUANTITIES]
INFRA_QUANTITIES = ["cpu_util", "queue_depth"]

# A consumer span is inbound work just like a server span (kafka consumers have no
# server spans at all); a producer span is an outbound call just like a client span.
INFRA_COMPONENTS = frozenset(schema.INFRA_OWNER)

INBOUND_KINDS = ["server", "consumer"]
OUTBOUND_KINDS = ["client", "producer"]

ANOMALY_LATENCY_Z = 3.0     # neighbour counts as latency-anomalous above this z
ANOMALY_ERROR_RATE = 0.05   # ... or above this error rate
TIMEOUT_MULTIPLE = 3.0      # failed call longer than this * warm-up p95 looks like a timeout
CHILD_DOMINANT_FRAC = 0.5   # child longer than this fraction of the parent dominates it
MAX_ANCESTOR_HOPS = 6       # internal spans between a server span and the calls it issues

# Raw window aggregate -> robust z feature, per service.
Z_MAP = {f"f_metrics_{q}_mean": f"f_metrics_{q}_z" for q in LEVEL_Z_QUANTITIES}
Z_MAP.update({f"_metrics_{n}": f"f_metrics_{n}_z" for n in PER_REQUEST_QUANTITIES.values()})
Z_MAP["f_metrics_infra_queue_depth_mean"] = "f_metrics_infra_queue_depth_z"
# A slow leak is invisible in the level but obvious in the slope, so the slope gets a
# baseline of its own (warm-up slopes are noise around zero).
Z_MAP["f_metrics_mem_frac_slope"] = "f_metrics_mem_frac_slope_z"
Z_MAP["_lat_log_med"] = "f_traces_latency_z"
Z_MAP["f_logs_error_rate"] = "f_logs_error_z"
# ... and per (service, peer), for the per-dependency client-side view.
PEER_Z_MAP = {"_client_lat_log_med": "_client_lat_z"}

WINDOW_FEATURE_COLUMNS = (
    [f"f_metrics_{q}_mean" for q in METRIC_QUANTITIES]
    + [f"f_metrics_{q}_z" for q in LEVEL_Z_QUANTITIES]
    + [f"f_metrics_{n}_z" for n in PER_REQUEST_QUANTITIES.values()]
    + [f"f_metrics_{q}_slope" for q in SLOPE_QUANTITIES]
    + ["f_metrics_mem_frac_slope_z"]
    + ["f_metrics_infra_cpu_util_mean", "f_metrics_infra_queue_depth_mean",
       "f_metrics_infra_queue_depth_z"]
    + ["f_traces_request_rate", "f_traces_error_rate", "f_traces_latency_p50_ms",
       "f_traces_latency_p95_ms", "f_traces_latency_max_ms", "f_traces_latency_z",
       "f_traces_self_time_ratio", "f_traces_self_time_ms",
       "f_traces_child_dominated_frac", "f_traces_client_rate",
       "f_traces_client_error_rate_max", "f_traces_client_latency_z_max",
       "f_traces_retry_frac", "f_traces_timeout_frac",
       "f_traces_infra_error_rate", "f_traces_infra_p95_ms",
       "f_traces_infra_latency_z", "f_traces_latency_ratio"]
    + ["f_logs_rate", "f_logs_error_rate", "f_logs_error_z", "f_logs_warn_rate",
       "f_logs_error_trace_frac"]
    + ["f_graph_callee_latency_z_max", "f_graph_callee_latency_z_mean",
       "f_graph_callee_error_rate_max", "f_graph_callee_error_rate_mean",
       "f_graph_caller_latency_z_max", "f_graph_caller_latency_z_mean",
       "f_graph_caller_error_rate_max", "f_graph_caller_error_rate_mean",
       "f_graph_downstream_explained", "f_graph_n_callees_anomalous",
       "f_graph_n_callers_anomalous", "f_graph_depth_from_entry",
       "f_graph_outbound_path_gap_max", "f_graph_outbound_path_gap_mean"]
    + ["f_graph_inbound_client_latency_z_max", "f_graph_inbound_client_error_rate",
       "f_graph_inbound_timeout_frac", "f_graph_inbound_retry_frac",
       "f_graph_inbound_client_rate", "f_graph_inbound_vs_server_latency",
       "f_graph_inbound_unanswered_frac"]
)

# --- temporal context -------------------------------------------------------------------
# One window is a thin sample: at the traffic levels the calibrated benchmark runs at,
# most windows hold a handful of spans and a single-window feature is mostly sampling
# noise. These summarise the trailing K windows of the same (experiment, service), which
# is both a stronger signal and computable online from a buffer of past window rows.
TEMPORAL_K = 3
TEMPORAL_SOURCES = [
    "f_traces_latency_z", "f_traces_error_rate", "f_traces_latency_ratio",
    "f_graph_inbound_client_latency_z_max", "f_graph_inbound_unanswered_frac",
    "f_graph_inbound_timeout_frac", "f_traces_client_latency_z_max",
    "f_metrics_cpu_util_z", "f_metrics_mem_frac_z", "f_metrics_mem_frac_slope_z",
    "f_metrics_queue_depth_z", "f_logs_error_z", "f_graph_outbound_path_gap_max",
]
# Change since the previous window: a fault that is ramping looks different from one
# that has been steady for a minute, and the level alone cannot tell them apart.
TEMPORAL_DELTA_SOURCES = [
    "f_traces_latency_z", "f_metrics_cpu_util_z", "f_metrics_mem_frac_z",
    "f_metrics_queue_depth_z", "f_traces_error_rate",
]
# Pooled over the trailing windows rather than averaged over them: a rate or an error
# fraction from three sparse windows is far better estimated from the combined counts
# than from the mean of three noisy per-window estimates.
POOLED_RATE = "f_traces_request_rate"
POOLED_ERROR_RATE = "f_traces_error_rate"
POOLED_LATENCY_Z = "f_traces_latency_z"


def _temporal(source: str, suffix: str) -> str:
    return f"f_temporal_{source.removeprefix('f_')}_{suffix}"


TEMPORAL_FEATURE_COLUMNS = (
    [_temporal(c, "mean3") for c in TEMPORAL_SOURCES]
    + [_temporal(c, "max3") for c in TEMPORAL_SOURCES]
    + [_temporal(c, "delta") for c in TEMPORAL_DELTA_SOURCES]
    + ["f_temporal_traces_request_rate3", "f_temporal_traces_error_rate3",
       "f_temporal_traces_latency_z3"]
)

FEATURE_COLUMNS = WINDOW_FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS

# Columns produced by pass 1 (everything that needs no baseline and no neighbours).
BASELINE_COLUMNS = [
    "f_traces_client_latency_z_max", "f_traces_client_error_rate_max",
    "f_traces_timeout_frac", "f_traces_infra_latency_z", "f_traces_latency_ratio",
    *Z_MAP.values(),
]
PASS1_COLUMNS = [
    c for c in WINDOW_FEATURE_COLUMNS
    if c not in BASELINE_COLUMNS and not c.startswith("f_graph_")
] + ["_lat_log_med", "_lat_med_ns"] + [
    f"_metrics_{n}" for n in PER_REQUEST_QUANTITIES.values()
]

COLUMNS = (
    fschema.KEY_COLUMNS
    + list(fschema.LABEL_COLUMNS)
    + list(fschema.GROUP_COLUMNS)
    + FEATURE_COLUMNS
)


# --- public entry point ---------------------------------------------------------------
def build_windows(
    exp: schema.Experiment, window_ns: float = DEFAULT_WINDOW_NS, stride_ns: float | None = None
) -> pd.DataFrame:
    """Build the window table of one experiment (see ``rca.features.schema``)."""
    manifest = exp.manifest
    window_ns = int(window_ns)
    stride_ns = int(stride_ns) if stride_ns else window_ns
    services = list(manifest.services)
    starts = _window_starts(manifest.start_ns, manifest.end_ns, window_ns, stride_ns)
    if len(starts) == 0 or not services:
        return empty_table()

    grid = pd.MultiIndex.from_product(
        [range(len(starts)), services], names=["window_idx", "service"]
    )
    raw, peer_raw, outbound = raw_window_features(
        exp, grid, manifest.start_ns, window_ns, stride_ns, len(starts)
    )
    warmup_windows = _warmup_windows(manifest.warmup_ns, window_ns, stride_ns)
    baseline = fit_baseline(raw, peer_raw, outbound, warmup_windows)
    feat = pd.concat(
        [raw, apply_baseline(raw, peer_raw, outbound, baseline, window_ns / 1e9)], axis=1
    )
    feat = pd.concat([feat, graph_features(feat, manifest.edges, services)], axis=1)

    missing = set(WINDOW_FEATURE_COLUMNS) - set(feat.columns)
    if missing:
        raise AssertionError(f"feature columns not produced: {sorted(missing)}")

    out = _labels(manifest, starts, window_ns, services, grid)
    for column in WINDOW_FEATURE_COLUMNS:
        out[column] = feat[column].to_numpy(dtype="float64")
    out = out.reset_index(drop=True)
    return pd.concat([out, temporal_features(out)], axis=1)[COLUMNS]


# --- windowing ------------------------------------------------------------------------
def _window_starts(start_ns: int, end_ns: int, window_ns: int, stride_ns: int) -> np.ndarray:
    """Window starts aligned to ``start_ns``; a partial final window is dropped."""
    count = (end_ns - start_ns - window_ns) // stride_ns + 1
    return start_ns + np.arange(max(count, 0), dtype="int64") * stride_ns


def _warmup_windows(warmup_ns: int, window_ns: int, stride_ns: int) -> int:
    """How many leading windows lie *entirely* inside the warm-up period.

    Counting windows by their start instead would, at a stride shorter than the window,
    let the last baseline windows reach past the warm-up and into the fault -- fitting
    the "this is what healthy looks like" baseline on the anomaly itself.
    """
    return int(max((warmup_ns - window_ns) // stride_ns + 1, 0))


def _assign(
    ts: np.ndarray, start_ns: int, window_ns: int, stride_ns: int, n_windows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row positions and window indices of every (row, containing window) pair."""
    rel = ts - start_ns
    hi = np.minimum(np.floor_divide(rel, stride_ns), n_windows - 1)
    lo = np.maximum(-np.floor_divide(window_ns - 1 - rel, stride_ns), 0)
    counts = np.maximum(hi - lo + 1, 0)
    rows = np.repeat(np.arange(len(ts)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return rows, np.repeat(lo, counts) + offsets


def _expand(
    frame: pd.DataFrame, ts_column: str, start_ns: int, window_ns: int, stride_ns: int,
    n_windows: int,
) -> pd.DataFrame:
    """Attach ``window_idx``, duplicating rows that fall into overlapping windows."""
    if frame.empty:
        return frame.assign(window_idx=np.zeros(0, dtype="int64"))
    rows, windows = _assign(
        frame[ts_column].to_numpy(dtype="int64"), start_ns, window_ns, stride_ns, n_windows
    )
    out = frame.iloc[rows].copy()
    out["window_idx"] = windows
    return out


# --- pass 1: per-service raw aggregates -----------------------------------------------
def raw_window_features(
    exp: schema.Experiment, grid: pd.MultiIndex, start_ns: int, window_ns: int,
    stride_ns: int, n_windows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw (un-normalized) window aggregates.

    Returns ``(raw, peer_raw, outbound)``: the per-(window, service) table on the full
    grid, the per-(window, service, peer) outbound table, and the window-assigned
    outbound spans (needed for timeout detection against the warm-up baseline).
    """
    services = grid.levels[1]
    window_seconds = window_ns / 1e9

    def expand(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        return _expand(frame, column, start_ns, window_ns, stride_ns, n_windows)

    metrics = _prepare_metrics(exp.metrics, start_ns)
    own_metrics = expand(metrics[metrics["service"].isin(services)], "ts_ns")
    # Resource metrics of an infra component are the component's own, and ownership is
    # the only attribution available for them. Note the limitation: a *shared* component
    # (postgresql serves both product-catalog and accounting) puts its cpu, memory and
    # queue depth on the single owning service only. The caller-side f_traces_infra_*
    # features are what give every other client of that component its own evidence.
    infra = metrics[metrics["service"].isin(INFRA_COMPONENTS)].copy()
    infra["service"] = _owner(infra["service"])
    infra_metrics = expand(infra[infra["service"].isin(services)], "ts_ns")

    inbound, outbound = _prepare_spans(exp.spans, services)
    inbound = expand(inbound, "start_ns")
    outbound = expand(outbound, "start_ns")

    logs = exp.logs.copy()
    logs["service"] = _owner(logs["service"])
    logs = expand(logs[logs["service"].isin(services)], "ts_ns")

    parts = [
        _metric_features(own_metrics, "f_metrics_", METRIC_QUANTITIES, SLOPE_QUANTITIES),
        _metric_features(infra_metrics, "f_metrics_infra_", INFRA_QUANTITIES, []),
        _inbound_features(inbound, window_seconds),
        _infra_outbound_features(outbound),
        _outbound_features(outbound, window_seconds),
        _log_features(logs, window_seconds),
    ]
    raw = pd.DataFrame(index=grid)
    for part in parts:
        if part is not None:
            raw = raw.join(part)
    unknown = set(raw.columns) - set(PASS1_COLUMNS)
    if unknown:
        raise AssertionError(f"unregistered pass-1 columns: {sorted(unknown)}")
    raw = raw.reindex(columns=PASS1_COLUMNS).astype("float64")

    # Bytes per request. Both sides are per-second rates over the same window, so the
    # window length cancels and this is exactly (bytes in the window / requests in it).
    requests = raw["f_traces_request_rate"].to_numpy(dtype="float64")
    for quantity, name in PER_REQUEST_QUANTITIES.items():
        rate = raw[f"f_metrics_{quantity}_mean"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            raw[f"_metrics_{name}"] = np.where(requests > 0, rate / requests, np.nan)
    return raw.reset_index(), _peer_features(outbound), outbound


def _owner(services: pd.Series) -> pd.Series:
    """Map infra components onto the application service that owns them.

    The result is plain ``object`` dtype so that every service key in this module
    compares and joins the same way, whatever dtype the source table used.
    """
    plain = services.astype(object)
    return plain.map(schema.INFRA_OWNER).fillna(plain)


def _prepare_metrics(metrics: pd.DataFrame, origin_ns: int) -> pd.DataFrame:
    """Long metric table -> wide table of the derived quantities we window over."""
    columns = ["service", "ts_ns", "_t_s", *METRIC_QUANTITIES]
    if metrics.empty:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
        return empty.astype({"service": object, "ts_ns": "int64"})
    wide = metrics.pivot_table(
        index=["service", "ts_ns"], columns="metric", values="value", aggfunc="mean"
    )
    wide = wide.reset_index().sort_values(["service", "ts_ns"], ignore_index=True)
    wide["service"] = wide["service"].astype(object)
    for name in schema.METRIC_NAMES:
        if name not in wide.columns:
            wide[name] = np.nan
    wide["mem_frac"] = wide[schema.METRIC_MEM_BYTES] / wide[schema.METRIC_MEM_LIMIT]
    grouped = wide.groupby("service", sort=False)
    seconds = grouped["ts_ns"].diff() / 1e9
    wide["net_rx_rate"] = grouped[schema.METRIC_NET_RX_BYTES].diff() / seconds
    wide["net_tx_rate"] = grouped[schema.METRIC_NET_TX_BYTES].diff() / seconds
    # Seconds since the experiment start: keeps the least-squares slope well conditioned.
    wide["_t_s"] = (wide["ts_ns"] - origin_ns) / 1e9
    return wide[columns]


def _metric_features(
    metrics: pd.DataFrame, prefix: str, quantities: list[str], slopes: list[str]
) -> pd.DataFrame | None:
    if metrics.empty:
        return None
    grouped = metrics.groupby(["window_idx", "service"], sort=False)
    out = grouped[quantities].mean()
    out.columns = [f"{prefix}{q}_mean" for q in quantities]
    for quantity in slopes:
        out[f"{prefix}{quantity}_slope"] = _slope(metrics, quantity)
    return out


def _slope(metrics: pd.DataFrame, column: str) -> pd.Series:
    """Least-squares slope per (window, service), in units per second."""
    frame = metrics[["window_idx", "service", "_t_s", column]].dropna()
    seconds = frame["_t_s"].to_numpy(dtype="float64")
    values = frame[column].to_numpy(dtype="float64")
    frame = frame.assign(_t=seconds, _tt=seconds * seconds, _ty=seconds * values)
    grouped = frame.groupby(["window_idx", "service"], sort=False)
    n = grouped.size()
    sum_t, sum_y = grouped["_t"].sum(), grouped[column].sum()
    denominator = n * grouped["_tt"].sum() - sum_t * sum_t
    return (n * grouped["_ty"].sum() - sum_t * sum_y) / denominator.where(denominator > 0)


def _prepare_spans(
    spans: pd.DataFrame, services: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split spans into the application services' inbound and outbound work."""
    spans = spans.copy()
    spans["service"] = spans["service"].astype(object)
    outbound_all = spans[spans["kind"].isin(OUTBOUND_KINDS)].copy()
    outbound_all["_ancestor"] = _inbound_ancestor(spans, outbound_all)
    child_sum = outbound_all.groupby("_ancestor")["duration_ns"].sum()
    child_max = outbound_all.groupby("_ancestor")["duration_ns"].max()

    inbound = spans[spans["kind"].isin(INBOUND_KINDS)].copy()
    duration = inbound["duration_ns"].to_numpy(dtype="float64")
    children = inbound["span_id"].map(child_sum).fillna(0.0).to_numpy(dtype="float64")
    slowest = inbound["span_id"].map(child_max).fillna(0.0).to_numpy(dtype="float64")
    self_ns = np.maximum(duration - children, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        inbound["_self_ratio"] = np.where(duration > 0, self_ns / duration, np.nan)
    inbound["_self_ms"] = self_ns / 1e6
    inbound["_child_dominated"] = (slowest > CHILD_DOMINANT_FRAC * duration).astype("float64")
    inbound["_log_duration"] = np.log1p(duration)

    app_inbound = inbound[inbound["service"].isin(services)]

    outbound = outbound_all[outbound_all["service"].isin(services)].copy()
    outbound["_log_duration"] = np.log1p(outbound["duration_ns"].to_numpy(dtype="float64"))
    # Peers are *not* folded onto owners here: "cart -> valkey" is exactly the
    # per-dependency view that makes a cache slowdown visible from the caller side.
    outbound["peer"] = outbound["peer_service"].astype(object)
    # Identity of the request a call belongs to: the inbound span it hangs under, not
    # whatever internal span happens to be its immediate parent.
    outbound["_call"] = outbound["_ancestor"].fillna(outbound["parent_span_id"])
    return app_inbound, outbound[outbound["peer"].ne("")]


def _inbound_ancestor(spans: pd.DataFrame, outbound: pd.DataFrame) -> np.ndarray:
    """Span id of the inbound span each outbound call ultimately sits under.

    Real instrumentation puts internal spans between a service's server span and the
    calls it issues (server -> internal -> client), so counting only *direct* children
    misses every outbound call and each service looks as if it does all the work itself
    (self-time ratio pinned at 1.00). Walk up ``parent_span_id`` while staying inside the
    same service until an inbound span turns up, with a hop cap so that neither a deep
    nesting nor a malformed trace can run away. ``None`` where no such ancestor exists.
    """
    unique = spans.drop_duplicates("span_id").set_index("span_id")
    parent_of, kind_of, service_of = unique["parent_span_id"], unique["kind"], unique["service"]

    service = outbound["service"].to_numpy(dtype=object)
    current = outbound["parent_span_id"].to_numpy(dtype=object).copy()
    ancestor = np.full(len(outbound), None, dtype=object)
    pending = np.flatnonzero(pd.notna(current) & (current != ""))

    for _ in range(MAX_ANCESTOR_HOPS):
        if len(pending) == 0:
            break
        keys = pd.Index(current[pending])
        same = service_of.reindex(keys).fillna("").to_numpy(dtype=object) == service[pending]
        is_inbound = np.isin(
            kind_of.reindex(keys).fillna("").to_numpy(dtype=object), INBOUND_KINDS)
        parents = parent_of.reindex(keys).fillna("").to_numpy(dtype=object)

        found = same & is_inbound
        ancestor[pending[found]] = current[pending[found]]
        walk = same & ~is_inbound & (parents != "")
        current[pending[walk]] = parents[walk]
        pending = pending[walk]
    return ancestor


def _inbound_features(inbound: pd.DataFrame, window_seconds: float) -> pd.DataFrame | None:
    if inbound.empty:
        return None
    grouped = inbound.groupby(["window_idx", "service"], sort=False)
    duration = grouped["duration_ns"]
    return pd.DataFrame({
        "f_traces_request_rate": grouped.size() / window_seconds,
        "f_traces_error_rate": grouped["status_error"].mean(),
        "f_traces_latency_p50_ms": duration.quantile(0.5) / 1e6,
        "f_traces_latency_p95_ms": duration.quantile(0.95) / 1e6,
        "f_traces_latency_max_ms": duration.max() / 1e6,
        "f_traces_self_time_ratio": grouped["_self_ratio"].mean(),
        "f_traces_self_time_ms": grouped["_self_ms"].mean(),
        "f_traces_child_dominated_frac": grouped["_child_dominated"].mean(),
        "_lat_log_med": grouped["_log_duration"].median(),
        "_lat_med_ns": duration.median(),
    })


def _infra_outbound_features(outbound: pd.DataFrame) -> pd.DataFrame | None:
    """Calls a service makes into *any* infra component (a database, cache or broker).

    Infra is not separately instrumented: both the simulator and the real OTel Demo
    represent such a call as a single client (or producer) span on the calling service,
    with ``peer_service`` naming the component. Whoever makes the call is who sees it,
    so these land on the caller's row regardless of which service nominally owns the
    component -- a slow postgresql is evidence about accounting when accounting is the
    one waiting on it.
    """
    infra = outbound[_calls_infra(outbound)]
    if infra.empty:
        return None
    grouped = infra.groupby(["window_idx", "service"], sort=False)
    return pd.DataFrame({
        "f_traces_infra_error_rate": grouped["status_error"].mean(),
        "f_traces_infra_p95_ms": grouped["duration_ns"].quantile(0.95) / 1e6,
    })


def _calls_infra(frame: pd.DataFrame) -> np.ndarray:
    """Rows whose ``peer`` is an infra component, whichever service is calling it."""
    return frame["peer"].isin(INFRA_COMPONENTS).to_numpy(dtype=bool)


def _outbound_features(outbound: pd.DataFrame, window_seconds: float) -> pd.DataFrame | None:
    if outbound.empty:
        return None
    grouped = outbound.groupby(["window_idx", "service"], sort=False)
    # A retry is the *same* call made twice: the operation is part of the identity, or
    # a service that legitimately calls one peer twice per request (checkout asking cart
    # for the basket and then emptying it) reads as retrying every single time.
    calls = outbound.groupby(
        ["window_idx", "service", "_call", "peer", "operation"], sort=False
    ).size()
    retried = (calls > 1).astype("float64").groupby(level=["window_idx", "service"]).mean()
    return pd.DataFrame({
        "f_traces_client_rate": grouped.size() / window_seconds,
        "f_traces_retry_frac": retried,
    })


def _peer_features(outbound: pd.DataFrame) -> pd.DataFrame:
    """Per-dependency (window, service, peer) outbound aggregates."""
    columns = ["window_idx", "service", "peer", "_client_error_rate", "_client_lat_log_med"]
    if outbound.empty:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})
        return empty.astype({"window_idx": "int64", "service": object, "peer": object})
    grouped = outbound.groupby(["window_idx", "service", "peer"], sort=False)
    return pd.DataFrame({
        "_client_error_rate": grouped["status_error"].mean(),
        "_client_lat_log_med": grouped["_log_duration"].median(),
    }).reset_index()


def _log_features(logs: pd.DataFrame, window_seconds: float) -> pd.DataFrame | None:
    if logs.empty:
        return None
    logs = logs.assign(
        _error=logs["severity"].isin(["ERROR", "FATAL"]).astype("float64"),
        _warn=(logs["severity"] == "WARN").astype("float64"),
    )
    logs["_error_traced"] = logs["_error"] * (logs["trace_id"] != "").astype("float64")
    grouped = logs.groupby(["window_idx", "service"], sort=False)
    errors = grouped["_error"].sum()
    return pd.DataFrame({
        "f_logs_rate": grouped.size() / window_seconds,
        "f_logs_error_rate": errors / window_seconds,
        "f_logs_warn_rate": grouped["_warn"].sum() / window_seconds,
        "f_logs_error_trace_frac": (grouped["_error_traced"].sum() / errors).where(errors > 0),
    })


# --- pass 2a: warm-up normalization ----------------------------------------------------
def apply_baseline(
    raw: pd.DataFrame, peer_raw: pd.DataFrame, outbound: pd.DataFrame, baseline: Baseline,
    window_seconds: float,
) -> pd.DataFrame:
    """Robust z-scores plus the per-dependency features that need the warm-up baseline."""
    out = z_columns(raw, ["service"], baseline.stats, Z_MAP, window_seconds)
    out["f_traces_latency_ratio"] = ratio_column(
        raw, ["service"], baseline.stats, "_lat_med_ns")
    peer = peer_raw.join(z_columns(peer_raw, ["service", "peer"], baseline.peer_stats,
                                   PEER_Z_MAP, window_seconds))
    index = pd.MultiIndex.from_frame(raw[["window_idx", "service"]])
    worst = peer.groupby(["window_idx", "service"], sort=False).agg(
        f_traces_client_latency_z_max=("_client_lat_z", "max"),
        f_traces_client_error_rate_max=("_client_error_rate", "max"),
    )
    for column in worst.columns:
        out[column] = _align(worst[column], index)
    # Infra dependencies get their own z, on the same per-(service, peer) baseline.
    infra = peer[_calls_infra(peer)].groupby(["window_idx", "service"], sort=False)
    out["f_traces_infra_latency_z"] = _align(infra["_client_lat_z"].max(), index)
    outbound = outbound.assign(_timeout=_timeout_flag(outbound, baseline))
    out["f_traces_timeout_frac"] = _align(
        _mean_by(outbound, "service", "_timeout"), index)
    for column, values in _outbound_path_gap(peer, raw, out, index).items():
        out[column] = values
    for column, values in _caller_view(raw, peer, outbound, window_seconds, index).items():
        out[column] = values
    return out


def _outbound_path_gap(
    peer: pd.DataFrame, raw: pd.DataFrame, out: pd.DataFrame, index: pd.MultiIndex
) -> dict[str, np.ndarray]:
    """How much slower a call looks to the caller than it does to the callee.

    Egress shaping -- a netem qdisc on a service's own interface, a saturated uplink, a
    failing NIC -- slows everything that service *sends* while the services it calls stay
    perfectly healthy and say so in their own server spans. Every other feature then
    points at the callees: their inbound view lights up and the real culprit's own server
    latency looks fine. The gap between the two sides of the same call is what names the
    caller. Near zero when a dependency is genuinely slow (both sides see the same
    thing), large when the time is being lost on the path out.
    """
    server_z = pd.Series(
        out["f_traces_latency_z"].to_numpy(dtype="float64"),
        index=pd.MultiIndex.from_frame(raw[["window_idx", "service"]]),
    )
    callee_z = server_z.reindex(
        pd.MultiIndex.from_frame(peer[["window_idx", "peer"]])).to_numpy(dtype="float64")
    gaps = peer.assign(
        _gap=peer["_client_lat_z"].to_numpy(dtype="float64") - callee_z
    ).groupby(["window_idx", "service"], sort=False)["_gap"]
    return {
        "f_graph_outbound_path_gap_max": _align(gaps.max(), index),
        "f_graph_outbound_path_gap_mean": _align(gaps.mean(), index),
    }


def _align(values: pd.Series, index: pd.MultiIndex) -> np.ndarray:
    if values.empty:
        return np.full(len(index), np.nan)
    return values.reindex(index).to_numpy(dtype="float64")


def _timeout_flag(outbound: pd.DataFrame, baseline: Baseline) -> np.ndarray:
    """Per call: did it fail *and* run far past the warm-up p95 for that dependency?"""
    if outbound.empty:
        return np.zeros(0, dtype="float64")
    peer_index = pd.MultiIndex.from_frame(outbound[["service", "peer"]])
    limit = baseline.client_p95_ns.reindex(peer_index).to_numpy(dtype="float64")
    duration = outbound["duration_ns"].to_numpy(dtype="float64")
    return np.where(
        np.isnan(limit), np.nan,
        (outbound["status_error"].to_numpy() & (duration > TIMEOUT_MULTIPLE * limit)).astype(
            "float64"
        ),
    )


def _mean_by(frame: pd.DataFrame, key: str, column: str) -> pd.Series:
    return frame.groupby(["window_idx", key], sort=False)[column].mean()


def _calls_into_services(frame: pd.DataFrame) -> pd.DataFrame:
    """Caller-side rows whose ``peer`` is an application service -- the callee's row.

    Calls into infra are dropped rather than folded onto the component's owner. Infra is
    never a label row, ownership is only a guess about which service a *shared* component
    belongs to, and folding actively lies: accounting's postgresql calls are not traffic
    into product-catalog. That evidence already lives on the caller's own row as
    ``f_traces_infra_*``.
    """
    keep = ~_calls_infra(frame) & frame["peer"].ne(frame["service"]).to_numpy(dtype=bool)
    return frame[keep]


def _caller_view(
    raw: pd.DataFrame, peer: pd.DataFrame, outbound: pd.DataFrame, window_seconds: float,
    index: pd.MultiIndex,
) -> dict[str, np.ndarray]:
    """How a service's callers see it, attributed to the *callee's* row.

    For a dropped packet, a saturated link or a dependency that stops answering, the
    callee's own server spans look healthy -- or never happen at all -- and the entire
    symptom lives in the callers' client spans. Pooling those onto the callee's row is
    what makes such a service localizable.
    """
    columns = [
        "f_graph_inbound_client_latency_z_max", "f_graph_inbound_client_error_rate",
        "f_graph_inbound_timeout_frac", "f_graph_inbound_retry_frac",
        "f_graph_inbound_client_rate", "f_graph_inbound_vs_server_latency",
        "f_graph_inbound_unanswered_frac",
    ]
    calls = _calls_into_services(outbound)
    if calls.empty:
        return {column: np.full(len(index), np.nan) for column in columns}

    grouped = calls.groupby(["window_idx", "peer"], sort=False)
    count = grouped.size()
    retries = calls.groupby(
        ["window_idx", "peer", "_call", "operation"], sort=False).size()
    out = {
        "f_graph_inbound_client_latency_z_max": _align(
            _calls_into_services(peer).groupby(["window_idx", "peer"], sort=False)[
                "_client_lat_z"].max(), index),
        "f_graph_inbound_client_error_rate": _align(grouped["status_error"].mean(), index),
        "f_graph_inbound_timeout_frac": _align(grouped["_timeout"].mean(), index),
        "f_graph_inbound_retry_frac": _align(
            (retries > 1).astype("float64").groupby(level=["window_idx", "peer"]).mean(),
            index),
        "f_graph_inbound_client_rate": _align(count / window_seconds, index),
    }

    calls_in = _align(count.astype("float64"), index)
    client_ns = _align(grouped["duration_ns"].median(), index)
    server_ns = raw["_lat_med_ns"].to_numpy(dtype="float64")
    served = np.nan_to_num(raw["f_traces_request_rate"].to_numpy(dtype="float64")) * (
        window_seconds)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["f_graph_inbound_vs_server_latency"] = client_ns / np.where(
            server_ns > 0, server_ns, np.nan)
        out["f_graph_inbound_unanswered_frac"] = np.clip(
            1.0 - served / np.where(calls_in > 0, calls_in, np.nan), 0.0, 1.0)
    return out


# --- pass 2b: graph aggregates ---------------------------------------------------------
def graph_features(
    feat: pd.DataFrame, edges: list[tuple[str, str]], services: list[str]
) -> pd.DataFrame:
    """Aggregate each service's neighbours' own features over the dependency graph."""
    service_edges = _service_edges(edges, services)
    neighbours = feat[["window_idx", "service", "f_traces_latency_z", "f_traces_error_rate"]]
    neighbours = neighbours.rename(columns={
        "service": "neighbour", "f_traces_latency_z": "lz", "f_traces_error_rate": "er",
    })
    index = pd.MultiIndex.from_frame(feat[["window_idx", "service"]])
    callees = pd.DataFrame(service_edges, columns=["service", "neighbour"])
    callers = callees.rename(columns={"service": "neighbour", "neighbour": "service"})

    out = pd.concat(
        [_neighbour_agg(callees, neighbours, "callee", index),
         _neighbour_agg(callers, neighbours, "caller", index)], axis=1
    )
    out.index = feat.index
    out["f_graph_downstream_explained"] = (
        out["f_graph_callee_latency_z_max"] - feat["f_traces_latency_z"].to_numpy()
    )
    depth = _depth_from_entry(edges, services)
    out["f_graph_depth_from_entry"] = feat["service"].map(depth).to_numpy(dtype="float64")
    return out


def _service_edges(edges: list[tuple[str, str]], services: list[str]) -> list[tuple[str, str]]:
    """Edges between application services; every edge touching infra is dropped.

    Not folded onto owners: rewriting ``cart -> flagd`` into ``cart -> frontend`` invents
    a dependency that does not exist, and the callee/caller aggregates would then read a
    stranger's health as cart's downstream. What a service sees of the infra it calls is
    already on its own row as ``f_traces_infra_*``.
    """
    known = set(services) - INFRA_COMPONENTS
    return sorted({
        (source, target) for source, target in edges
        if source != target and source in known and target in known
    })


def _neighbour_agg(
    pairs: pd.DataFrame, neighbours: pd.DataFrame, role: str, index: pd.MultiIndex
) -> pd.DataFrame:
    columns = [
        f"f_graph_{role}_latency_z_max", f"f_graph_{role}_latency_z_mean",
        f"f_graph_{role}_error_rate_max", f"f_graph_{role}_error_rate_mean",
        f"f_graph_n_{role}s_anomalous",
    ]
    if pairs.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    joined = pairs.merge(neighbours, on="neighbour", how="inner")
    joined["_anomalous"] = (
        (joined["lz"] > ANOMALY_LATENCY_Z) | (joined["er"] > ANOMALY_ERROR_RATE)
    ).astype("float64")
    grouped = joined.groupby(["window_idx", "service"], sort=False)
    out = pd.DataFrame({
        columns[0]: grouped["lz"].max(),
        columns[1]: grouped["lz"].mean(),
        columns[2]: grouped["er"].max(),
        columns[3]: grouped["er"].mean(),
        columns[4]: grouped["_anomalous"].sum(),
    })
    return out.reindex(index)


def _depth_from_entry(edges: list[tuple[str, str]], services: list[str]) -> dict[str, float]:
    """Hops from the entry point, counted over the *unfolded* graph.

    Infra components are not rows, but they are still hops on the path. The only route to
    a broker consumer runs ``checkout -> kafka -> accounting``, so dropping infra from the
    graph here would leave the consumers unreachable and their depth undefined; counting
    kafka as a hop puts them two below checkout. Depths are rebased so the shallowest
    application service is 0 -- that is the entry point of the system proper, the load
    generator sitting in front of it not being part of it.
    """
    callees: dict[str, list[str]] = {}
    called = set()
    for source, target in edges:
        callees.setdefault(source, []).append(target)
        called.add(target)

    frontier = [node for node in callees if node not in called]
    depth = {node: 0.0 for node in frontier}
    while frontier:
        nxt = []
        for node in frontier:
            for callee in callees.get(node, ()):
                if callee not in depth:
                    depth[callee] = depth[node] + 1.0
                    nxt.append(callee)
        frontier = nxt

    reachable = [depth[s] for s in services if s in depth]
    if not reachable:
        return {}
    entry = min(reachable)
    return {s: depth[s] - entry for s in services if s in depth}


# --- pass 3: temporal context -----------------------------------------------------------
def temporal_features(windows: pd.DataFrame, k: int = TEMPORAL_K) -> pd.DataFrame:
    """Trailing-window context per (experiment, service), from the finished window table.

    Every value comes from the current window and the ``k - 1`` before it -- never from a
    later one -- so the online path can call this on a buffer of past window rows and get
    exactly the values the offline build produced. It takes the finished table rather than
    the intermediate frames for the same reason: a streaming detector has window rows, not
    spans. The result is indexed like ``windows``.

    Trailing means trailing *windows*, addressed by ``window_idx``, so an overlapping
    stride is honoured automatically (three windows at a half stride span two window
    widths of wall clock, not three). A row whose predecessor is absent -- the start of an
    experiment, or a gap in a serving buffer -- contributes nothing rather than silently
    borrowing a stranger's row.

    ``*_mean3`` / ``*_max3`` skip NaN, so a service that only shows up in one of the three
    windows still gets a value; ``*_delta`` does not, since "changed by" is meaningless
    without both ends. The pooled statistics weight each window by its request count,
    which is what makes them worth having over the plain means: three windows of two
    spans each estimate an error rate far better combined than averaged.
    """
    if windows.empty:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in TEMPORAL_FEATURE_COLUMNS},
            index=windows.index,
        )

    keys = ["experiment_id", "service"]
    sources = list(dict.fromkeys(
        [*TEMPORAL_SOURCES, *TEMPORAL_DELTA_SOURCES,
         POOLED_RATE, POOLED_ERROR_RATE, POOLED_LATENCY_Z]
    ))
    at = {name: position for position, name in enumerate(sources)}

    ordered = windows.sort_values([*keys, "window_idx"])
    grouped = ordered.groupby(keys, sort=False)
    window_idx = ordered["window_idx"].to_numpy()
    seconds = (
        ordered["window_end_ns"].to_numpy(dtype="float64")
        - ordered["window_start_ns"].to_numpy(dtype="float64")
    ) / 1e9

    values, present = [], []
    for lag in range(k):
        # copy=True: under pandas copy-on-write (default from 3.0) a homogeneous
        # float frame hands back a read-only view, and the NaN fill below writes.
        lagged = grouped[sources].shift(lag).to_numpy(dtype="float64", copy=True)
        exists = grouped["window_idx"].shift(lag).to_numpy(dtype="float64") == window_idx - lag
        lagged[~exists, :] = np.nan
        values.append(lagged)
        present.append(exists)
    stack = np.stack(values)                      # (k, rows, sources)
    exists = np.stack(present)                    # (k, rows)

    observed = np.sum(~np.isnan(stack), axis=0)
    out = {}
    for source in TEMPORAL_SOURCES:
        column = stack[:, :, at[source]]
        out[_temporal(source, "mean3")] = _nan_mean(column)
        out[_temporal(source, "max3")] = _nan_max(column)
    for source in TEMPORAL_DELTA_SOURCES:
        previous = stack[1][:, at[source]] if k > 1 else np.nan
        out[_temporal(source, "delta")] = stack[0][:, at[source]] - previous

    # Pooled: reconstruct per-window request counts, treating an unobserved window as
    # zero requests but an absent one as no window at all.
    rate = stack[:, :, at[POOLED_RATE]]
    counts = np.where(np.isnan(rate), 0.0, rate) * seconds
    span = np.sum(exists, axis=0) * seconds
    seen = observed[:, at[POOLED_RATE]] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        out["f_temporal_traces_request_rate3"] = np.where(
            seen & (span > 0), np.sum(counts, axis=0) / np.where(span > 0, span, np.nan), np.nan
        )
    out["f_temporal_traces_error_rate3"] = _pooled(
        stack[:, :, at[POOLED_ERROR_RATE]], counts)
    out["f_temporal_traces_latency_z3"] = _pooled(
        stack[:, :, at[POOLED_LATENCY_Z]], counts)

    return pd.DataFrame(out, index=ordered.index).reindex(windows.index)[
        TEMPORAL_FEATURE_COLUMNS]


def _nan_mean(column: np.ndarray) -> np.ndarray:
    seen = np.sum(~np.isnan(column), axis=0)
    total = np.nansum(column, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(seen > 0, total / np.maximum(seen, 1), np.nan)


def _nan_max(column: np.ndarray) -> np.ndarray:
    seen = np.sum(~np.isnan(column), axis=0)
    largest = np.max(np.where(np.isnan(column), -np.inf, column), axis=0)
    return np.where(seen > 0, largest, np.nan)


def _pooled(column: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Count-weighted mean over the trailing windows; NaN when nothing was observed."""
    weights = np.where(np.isnan(column), 0.0, counts)
    total = weights.sum(axis=0)
    weighted = (np.where(np.isnan(column), 0.0, column) * weights).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, weighted / np.where(total > 0, total, np.nan), np.nan)


# --- labels and grouping ---------------------------------------------------------------
def _labels(
    manifest: schema.Manifest, starts: np.ndarray, window_ns: int, services: list[str],
    grid: pd.MultiIndex,
) -> pd.DataFrame:
    ends = starts + window_ns
    faults = sorted(manifest.faults, key=lambda f: f.start_ns)
    n = len(starts)
    active = np.full(n, -1, dtype="int64")
    for position, fault in enumerate(faults):
        overlapping = (fault.start_ns < ends) & (fault.end_ns > starts)
        active = np.where((active < 0) & overlapping, position, active)

    has_fault = active >= 0
    target = np.array(
        [schema.INFRA_OWNER.get(f.target, f.target) for f in faults] + [""], dtype=object
    )
    fault_type = np.array([f.fault_type for f in faults] + [schema.NO_FAULT], dtype=object)
    intensity = np.array([f.intensity for f in faults] + [0.0], dtype="float64")
    fault_start = np.array([f.start_ns for f in faults] + [0], dtype="int64")
    pick = np.where(has_fault, active, len(faults))

    per_window = pd.DataFrame({
        "window_idx": np.arange(n, dtype="int64"),
        "window_start_ns": starts,
        "window_end_ns": ends,
        "is_fault_window": has_fault,
        "root_service": target[pick],
        "fault_type": fault_type[pick],
        "fault_intensity": intensity[pick],
        "since_fault_start_ns": np.where(has_fault, starts - fault_start[pick], -1),
    })
    out = pd.DataFrame(index=grid).reset_index()
    out = out.merge(per_window, on="window_idx", how="left")
    out["experiment_id"] = manifest.experiment_id
    out["is_root"] = out["service"].to_numpy() == out["root_service"].to_numpy()
    out["source"] = manifest.source
    out["seed"] = np.int64(manifest.seed)
    out["traffic_shape"] = manifest.traffic.shape
    out["base_rps"] = float(manifest.traffic.base_rps)
    return _cast(out)


def _cast(out: pd.DataFrame) -> pd.DataFrame:
    for column, dtype in {**fschema.LABEL_COLUMNS, **fschema.GROUP_COLUMNS}.items():
        out[column] = out[column].astype(dtype)
    out["experiment_id"] = out["experiment_id"].astype("string")
    out["service"] = out["service"].astype("string")
    for column in ["window_idx", "window_start_ns", "window_end_ns"]:
        out[column] = out[column].astype("int64")
    return out


def empty_table() -> pd.DataFrame:
    out = pd.DataFrame({
        "experiment_id": pd.Series(dtype="string"),
        "window_idx": pd.Series(dtype="int64"),
        "window_start_ns": pd.Series(dtype="int64"),
        "window_end_ns": pd.Series(dtype="int64"),
        "service": pd.Series(dtype="string"),
    })
    for column, dtype in {**fschema.LABEL_COLUMNS, **fschema.GROUP_COLUMNS}.items():
        out[column] = pd.Series(dtype=dtype)
    for column in FEATURE_COLUMNS:
        out[column] = pd.Series(dtype="float64")
    return out[COLUMNS]
