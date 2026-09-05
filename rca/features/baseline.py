"""Warm-up baselines: robust per-service statistics used to normalize window features.

The per-deployment normalization the feature set relies on is a robust z-score against
the experiment's *own* warm-up period: every raw window aggregate (metric means, median
log latency, log error rate, ...) is compared to the median / MAD of the same aggregate
over the fault-free warm-up windows of that experiment. That makes a "1.0 CPU" feature
mean the same thing for a service whose normal CPU is 0.1 and for one whose normal is
0.9, and makes features comparable across experiments and deployments.

The baseline is fitted from the raw window aggregate table, not from the raw telemetry,
so the serving path can fit exactly the same object from a live warm-up period: aggregate
the warm-up windows with ``rca.features.windows.raw_window_features``, call
``fit_baseline``, keep the returned object, then normalize each incoming window with
``z_columns``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826       # scale factor making the MAD a consistent estimator of sigma
MEANAD_TO_SIGMA = 1.2533    # ... and the mean absolute deviation about the median
Z_CLIP = 50.0               # with sensible floors, saturating here should be rare
MIN_BASELINE_WINDOWS = 5    # fewer observations than this -> baseline is not trustworthy
SMALL_SAMPLE_N = 10         # below this the MAD is too noisy; use the mean deviation

# Why two scale estimators: over a warm-up of six windows the MAD is the median of six
# deviations, i.e. essentially the average of the third and fourth -- it throws away most
# of the little information there is and its own sampling spread then inflates every
# z-score downstream. The mean absolute deviation about the median uses all of them and
# is far less variable at these sample sizes, at the cost of some robustness we can
# afford because the warm-up period is fault-free by construction. Above SMALL_SAMPLE_N
# observations the MAD is well enough determined and its robustness is worth having.

# --- sigma floors ---------------------------------------------------------------------
# A robust z divides by the warm-up MAD, and plenty of warm-up baselines are legitimately
# flat: a healthy service logs no errors, keeps an empty queue and pauses for no GC, so
# both its median and its MAD are 0. Dividing by (almost) zero turns a single stray event
# into an enormous z, which is why the floor cannot be a fraction of the median -- it has
# to be an absolute quantity, and the honest one is the measurement's own resolution:
# the smallest deviation that means anything at all for that signal.
#
# Keyed by feature base name: the source column minus its ``f_`` prefix and ``_mean``
# suffix, with the ``infra_`` infix dropped so an owned component shares the floor of the
# same quantity on its owner. Latency baselines (plain and per-dependency) live in log
# space and share the ``log_latency`` key.
SIGMA_FLOORS: dict[str, float] = {
    "log_latency": 0.05,            # ~5% change in latency; below that is timing jitter
    "metrics_cpu_util": 0.02,       # 2% of one core
    "metrics_mem_frac": 0.01,       # 1% of the container's memory limit
    "metrics_queue_depth": 1.0,     # one queued item
    "metrics_threads": 1.0,         # one worker
    "metrics_gc_pause_ms": 1.0,     # one millisecond of pause
}

# Event rates (per second): one extra event anywhere in the window is the resolution.
RATE_FEATURES = frozenset({
    "logs_rate", "logs_error_rate", "logs_warn_rate",
    "traces_request_rate", "traces_client_rate",
})

# Bytes per request: proportional to the payload a request already moves, with an
# absolute knee so a service exchanging almost nothing does not turn a header into news.
PER_REQUEST_FEATURES = frozenset({"metrics_net_rx_per_req", "metrics_net_tx_per_req"})
PER_REQUEST_FLOOR_FRACTION = 0.05
PER_REQUEST_FLOOR_BYTES = 100.0

# Slopes (units per second): the smallest slope that moves the level by its own floor
# over the course of one window. Keyed to the level feature it is a slope of.
SLOPE_FEATURES: dict[str, str] = {"metrics_mem_frac_slope": "metrics_mem_frac"}

# Anything not covered above: guards division by zero and nothing else. A new z feature
# should get its own entry in SIGMA_FLOORS rather than fall back to this.
DEFAULT_SIGMA_FLOOR = 1e-6


def sigma_floor(source: str, median: np.ndarray, window_seconds: float) -> np.ndarray:
    """Smallest deviation of ``source`` that is worth calling a deviation."""
    base = _base_name(source)
    if base in RATE_FEATURES:
        return np.full(np.shape(median), 1.0 / window_seconds)
    if base in PER_REQUEST_FEATURES:
        return np.maximum(
            PER_REQUEST_FLOOR_FRACTION * np.abs(median), PER_REQUEST_FLOOR_BYTES)
    if base in SLOPE_FEATURES:
        return np.full(np.shape(median), SIGMA_FLOORS[SLOPE_FEATURES[base]] / window_seconds)
    return np.full(np.shape(median), SIGMA_FLOORS.get(base, DEFAULT_SIGMA_FLOOR))


def _base_name(source: str) -> str:
    if source.endswith("_lat_log_med"):
        return "log_latency"
    return (source.removeprefix("f_").removeprefix("_").removesuffix("_mean")
            .replace("_infra_", "_"))


def robust_stats(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    """Median / MAD / count of ``columns``, grouped by ``keys``.

    Returns one row per key with columns ``<col>__med``, ``<col>__mad``,
    ``<col>__meanad`` (mean absolute deviation about the median) and ``<col>__n``.
    """
    med = frame.groupby(keys, sort=True)[columns].median()
    count = frame.groupby(keys, sort=True)[columns].count()
    index = _index_of(frame, keys)
    dev = pd.DataFrame(
        np.abs(frame[columns].to_numpy(dtype="float64") - med.reindex(index).to_numpy()),
        columns=columns,
    )
    for key in keys:
        dev[key] = frame[key].to_numpy()
    grouped_dev = dev.groupby(keys, sort=True)[columns]
    mad = grouped_dev.median()
    meanad = grouped_dev.mean()

    out = pd.concat({"med": med, "mad": mad, "meanad": meanad, "n": count}, axis=1)
    out.columns = [f"{col}__{stat}" for stat, col in out.columns]
    return out


def z_columns(
    frame: pd.DataFrame, keys: list[str], stats: pd.DataFrame, mapping: dict[str, str],
    window_seconds: float,
) -> pd.DataFrame:
    """Robust z-score of ``mapping`` source columns against ``stats``.

    ``frame`` must carry ``keys`` as ordinary columns. The result is indexed like
    ``frame`` and holds the mapping's destination columns.
    """
    index = _index_of(frame, keys)
    out = {}
    for src, dst in mapping.items():
        med = _lookup(stats, f"{src}__med", index)
        mad = _lookup(stats, f"{src}__mad", index)
        meanad = _lookup(stats, f"{src}__meanad", index)
        n = _lookup(stats, f"{src}__n", index)
        spread = np.where(
            n < SMALL_SAMPLE_N, MEANAD_TO_SIGMA * meanad, MAD_TO_SIGMA * mad)
        scale = np.maximum(spread, sigma_floor(src, med, window_seconds))
        z = (frame[src].to_numpy(dtype="float64") - med) / scale
        out[dst] = np.where(n >= MIN_BASELINE_WINDOWS, np.clip(z, -Z_CLIP, Z_CLIP), np.nan)
    return pd.DataFrame(out, index=frame.index)


def ratio_column(
    frame: pd.DataFrame, keys: list[str], stats: pd.DataFrame, source: str
) -> np.ndarray:
    """``source`` divided by its warm-up median -- the plain "2x warm-up" rule's input.

    Unitless and directly interpretable, unlike a z-score, which is what the rule
    baselines want; NaN where the warm-up is missing, too short, or zero.
    """
    index = _index_of(frame, keys)
    med = _lookup(stats, f"{source}__med", index)
    n = _lookup(stats, f"{source}__n", index)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = frame[source].to_numpy(dtype="float64") / np.where(med > 0, med, np.nan)
    return np.where(n >= MIN_BASELINE_WINDOWS, ratio, np.nan)


@dataclass
class Baseline:
    """Warm-up statistics of one experiment (or of one live warm-up period)."""

    stats: pd.DataFrame         # per service, over raw window aggregates
    peer_stats: pd.DataFrame    # per (service, peer), over raw per-dependency aggregates
    client_p95_ns: pd.Series    # per (service, peer), p95 of outbound call duration


def fit_baseline(
    raw: pd.DataFrame,
    peer_raw: pd.DataFrame,
    outbound: pd.DataFrame,
    warmup_windows: int,
) -> Baseline:
    """Fit warm-up statistics from the raw window aggregates of the warm-up period.

    ``raw`` / ``peer_raw`` are raw (un-normalized) window aggregate tables and
    ``outbound`` the window-assigned outbound spans; only rows with
    ``window_idx < warmup_windows`` are used.
    """
    raw_w = raw[raw["window_idx"] < warmup_windows]
    peer_w = peer_raw[peer_raw["window_idx"] < warmup_windows]
    out_w = outbound[outbound["window_idx"] < warmup_windows]
    return Baseline(
        stats=robust_stats(raw_w, ["service"], _numeric(raw_w, ["window_idx"])),
        peer_stats=robust_stats(peer_w, ["service", "peer"], _numeric(peer_w, ["window_idx"])),
        client_p95_ns=out_w.groupby(["service", "peer"], sort=True)["duration_ns"].quantile(0.95),
    )


def _index_of(frame: pd.DataFrame, keys: list[str]) -> pd.Index:
    if len(keys) == 1:
        return pd.Index(frame[keys[0]])
    return pd.MultiIndex.from_frame(frame[keys])


def _lookup(stats: pd.DataFrame, column: str, index: pd.Index) -> np.ndarray:
    if column not in stats.columns:
        return np.full(len(index), np.nan)
    return stats[column].reindex(index).to_numpy(dtype="float64")


def _numeric(frame: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [
        c
        for c in frame.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]
