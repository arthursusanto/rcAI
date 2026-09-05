"""Window table contract (output of rca.features, input to rca.models / rca.eval).

One row per (experiment_id, window_idx, service). Feature columns are prefixed ``f_``
and are float64 with NaN meaning "not observable in this window" (missing modality,
no traffic). Everything else is a key, label, or grouping column and must be present.
"""
from __future__ import annotations

KEY_COLUMNS = ["experiment_id", "window_idx", "window_start_ns", "window_end_ns", "service"]

LABEL_COLUMNS = {
    "is_fault_window": "bool",      # a fault is active anywhere in the system during this window
    "root_service": "string",       # root-cause service for the window, "" if none
    "fault_type": "string",         # one of FAULT_TYPES or "none"
    "is_root": "bool",              # this row's service is the root service
    "fault_intensity": "float64",   # 0 when no fault
    "since_fault_start_ns": "int64",  # window_start - fault_start, -1 when no fault
}

GROUP_COLUMNS = {
    "source": "string",             # otel-demo | sim
    "seed": "int64",
    "traffic_shape": "string",
    "base_rps": "float64",
}

FEATURE_PREFIX = "f_"

# Modality tag for each feature family; used by ablations / missing-telemetry robustness.
# "temporal" is not a telemetry source but a derived family: trailing-window context over
# the others, tagged so it can be ablated on its own.
# A feature column name starts with one of these after the prefix, e.g. f_metrics_cpu_z.
MODALITIES = ["metrics", "traces", "logs", "graph", "temporal"]


def feature_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith(FEATURE_PREFIX)]


def modality_of(column: str) -> str:
    body = column[len(FEATURE_PREFIX):]
    for m in MODALITIES:
        if body.startswith(m + "_"):
            return m
    raise ValueError(f"feature column {column!r} has no modality tag")
