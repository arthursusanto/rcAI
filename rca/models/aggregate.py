"""System-level window vectors.

Stage 1 must answer "is *the system* faulty" without knowing which service failed, so
the per-service rows of a window are collapsed into one row per (experiment, window):
for a fixed list of key features, the max and mean over services plus the number of
services above an anomaly threshold.

NaN means "not observable in this window". ``max``/``mean`` skip NaN and stay NaN when a
feature is missing for every service; the count column is NaN in that case too, so a
missing modality never looks like "zero services are anomalous".
"""
from __future__ import annotations

import pandas as pd

from rca.features.schema import FEATURE_PREFIX

AGGREGATE_PREFIX = "a_"

KEYS = ["experiment_id", "window_idx"]

# feature column -> "this service looks anomalous" threshold for the count statistic.
KEY_FEATURES: dict[str, float] = {
    "f_traces_latency_z": 3.0,
    "f_traces_error_rate": 0.05,
    "f_metrics_cpu_util_z": 3.0,
    "f_metrics_mem_frac_z": 3.0,
    "f_metrics_queue_depth_z": 3.0,
    "f_logs_error_z": 3.0,
    "f_traces_client_error_rate_max": 0.05,
    "f_traces_timeout_frac": 0.02,
    "f_traces_retry_frac": 0.02,
    "f_graph_downstream_explained": 1.0,
    # caller-side view: the only place a wire delay / unreachable target is visible
    "f_traces_client_latency_z_max": 3.0,
    "f_graph_inbound_client_latency_z_max": 3.0,
    "f_graph_inbound_vs_server_latency": 2.0,
    "f_graph_inbound_unanswered_frac": 0.1,
    # trailing-window context (pooled / max over the last 3 windows) so the detector
    # can integrate evidence on sparse-traffic services
    "f_temporal_traces_latency_z3": 3.0,
    "f_temporal_traces_error_rate3": 0.05,
    "f_temporal_graph_inbound_client_latency_z_max_max3": 3.0,
    "f_temporal_graph_inbound_unanswered_frac_max3": 0.1,
    "f_temporal_graph_inbound_timeout_frac_max3": 0.02,
    "f_temporal_metrics_cpu_util_z_max3": 3.0,
    "f_temporal_metrics_mem_frac_slope_z_mean3": 3.0,
    "f_temporal_metrics_queue_depth_z_mean3": 3.0,
    "f_temporal_logs_error_z_max3": 3.0,
    # egress faults: my calls out are slow while the callees say they are fine
    "f_graph_outbound_path_gap_max": 3.0,
    "f_temporal_graph_outbound_path_gap_max_max3": 3.0,
}

# Window-level columns carried through so the aggregate is usable on its own.
CARRIED = [
    "window_start_ns", "window_end_ns", "is_fault_window", "root_service", "fault_type",
    "fault_intensity", "since_fault_start_ns",
]


def aggregate_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment_id, window_idx), sorted by those keys.

    Key features absent from ``windows`` (ablations, older feature builds) are skipped.
    """
    grouped = windows.groupby(KEYS, sort=True)
    out = pd.DataFrame(index=grouped.size().index)

    for name, threshold in KEY_FEATURES.items():
        if name not in windows.columns:
            continue
        short = AGGREGATE_PREFIX + name[len(FEATURE_PREFIX):]
        column = windows[name]
        observed = column.notna().groupby([windows[key] for key in KEYS], sort=True).any()
        over = (column > threshold).groupby([windows[key] for key in KEYS], sort=True).sum()
        out[f"{short}_max"] = grouped[name].max()
        out[f"{short}_mean"] = grouped[name].mean()
        out[f"{short}_n_over"] = over.astype("float64").where(observed)

    for name in CARRIED:
        if name in windows.columns:
            out[name] = grouped[name].first()
    return out.reset_index()


def aggregate_feature_columns(aggregate: pd.DataFrame) -> list[str]:
    return [c for c in aggregate.columns if c.startswith(AGGREGATE_PREFIX)]
