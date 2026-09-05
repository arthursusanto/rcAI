"""A hand-built window table for the model and evaluation tests.

Four services, twelve 10-second windows per experiment, and a deliberately obvious fault
signature on the root service, so every model can learn it and every metric can be
checked by hand. Shape and dtypes follow ``rca.features.schema``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from rca.data import schema

SERVICES = ["frontend", "cart", "payment", "shipping"]
DEPTH = {"frontend": 0.0, "cart": 1.0, "payment": 1.0, "shipping": 1.0}
START_NS = 1_700_000_000_000_000_000
WINDOW_NS = 10_000_000_000
N_WINDOWS = 12
ENTRY = "frontend"

FEATURES = [
    "f_metrics_cpu_util_mean", "f_metrics_cpu_util_z", "f_metrics_mem_frac_mean",
    "f_metrics_mem_frac_z", "f_metrics_queue_depth_z",
    "f_traces_latency_z", "f_traces_latency_ratio", "f_traces_error_rate",
    "f_traces_request_rate", "f_traces_client_error_rate_max",
    "f_traces_timeout_frac", "f_traces_retry_frac", "f_traces_infra_error_rate",
    "f_logs_error_z",
    "f_graph_downstream_explained", "f_graph_depth_from_entry",
    "f_graph_inbound_client_error_rate", "f_graph_inbound_unanswered_frac",
    "f_graph_inbound_client_latency_z_max",
    "f_temporal_traces_latency_z_mean3",
]
# Feature values that identify a fault family on the root service's row.
SIGNATURE = {
    "cpu_saturation": {"f_metrics_cpu_util_z": 12.0, "f_metrics_cpu_util_mean": 0.95},
    "error_rate": {"f_traces_error_rate": 0.4, "f_logs_error_z": 12.0,
                   "f_traces_retry_frac": 0.3},
}


class Spec(NamedTuple):
    experiment_id: str
    fault_type: str | None = None
    root_service: str = "cart"
    fault_start_s: float = 45.0
    fault_end_s: float = 85.0
    traffic_shape: str = "steady"
    category: str = ""
    # Requests per second reaching every service; drives the target-exposure strata.
    request_rate: float = 5.0


DEFAULT_SPECS = [
    Spec("exp-train-normal", None),
    Spec("exp-train-cpu", "cpu_saturation", "cart"),
    Spec("exp-train-err", "error_rate", "payment"),
    Spec("exp-train-cpu2", "cpu_saturation", "shipping", 35.0, 75.0),
    Spec("exp-val-err", "error_rate", "cart", 55.0, 95.0),
    Spec("exp-val-normal", None, traffic_shape="bursty", category="traffic_spike"),
    Spec("exp-test-cpu", "cpu_saturation", "payment"),
    Spec("exp-test-normal", None),
    # 0.05 rps x 10 s x 5 fault windows = 2.5 requests to the root: barely exercised.
    Spec("exp-test-quiet", "error_rate", "shipping", request_rate=0.05),
]


def make_windows(specs=DEFAULT_SPECS, seed: int = 0) -> pd.DataFrame:
    """The window table for ``specs``; one row per (experiment, window, service)."""
    rng = np.random.default_rng(seed)
    rows = []
    for spec in specs:
        start_ns = START_NS + int(spec.fault_start_s * 1e9)
        end_ns = START_NS + int(spec.fault_end_s * 1e9)
        for index in range(N_WINDOWS):
            window_start = START_NS + index * WINDOW_NS
            window_end = window_start + WINDOW_NS
            faulty = bool(
                spec.fault_type and start_ns < window_end and end_ns > window_start
            )
            for service in SERVICES:
                rows.append(_row(rng, spec, index, window_start, window_end, service,
                                 faulty, start_ns))
    frame = pd.DataFrame(rows)
    frame["f_traces_infra_error_rate"] = np.nan     # a modality column nothing observes
    return frame.astype({
        "window_idx": "int64", "window_start_ns": "int64", "window_end_ns": "int64",
        "since_fault_start_ns": "int64", "seed": "int64",
        "is_fault_window": "bool", "is_root": "bool",
        **{name: "float64" for name in FEATURES},
    })


def make_manifests(specs=DEFAULT_SPECS) -> list[schema.Manifest]:
    """Matching manifests, so the evaluation can bucket fault-free time."""
    manifests = []
    for spec in specs:
        faults = []
        if spec.fault_type:
            faults.append(schema.Fault(
                fault_type=spec.fault_type, target=spec.root_service,
                start_ns=START_NS + int(spec.fault_start_s * 1e9),
                end_ns=START_NS + int(spec.fault_end_s * 1e9), intensity=0.8,
            ))
        category = spec.category or ("fault" if spec.fault_type else "normal")
        manifests.append(schema.Manifest(
            experiment_id=spec.experiment_id, source="sim", seed=0, start_ns=START_NS,
            end_ns=START_NS + N_WINDOWS * WINDOW_NS,
            traffic=schema.TrafficProfile(base_rps=10.0, shape=spec.traffic_shape),
            faults=faults, services=list(SERVICES),
            edges=[(ENTRY, s) for s in SERVICES if s != ENTRY],
            metric_interval_ns=1_000_000_000, warmup_ns=30_000_000_000,
            extra={"category": category},
        ))
    return manifests


def _row(rng, spec, index, window_start, window_end, service, faulty, fault_start_ns):
    is_root = faulty and service == spec.root_service
    values = {name: float(0.2 * rng.standard_normal()) for name in FEATURES}
    values["f_metrics_cpu_util_mean"] = 0.2 + 0.02 * rng.random()
    values["f_metrics_mem_frac_mean"] = 0.3 + 0.02 * rng.random()
    values["f_traces_error_rate"] = 0.002 * rng.random()
    values["f_traces_latency_ratio"] = 1.0 + 0.02 * rng.random()
    values["f_traces_request_rate"] = spec.request_rate
    values["f_traces_timeout_frac"] = 0.0
    values["f_traces_retry_frac"] = 0.0
    values["f_graph_downstream_explained"] = 0.0
    values["f_graph_depth_from_entry"] = DEPTH[service]
    values["f_graph_inbound_client_error_rate"] = 0.001 * rng.random()
    values["f_graph_inbound_unanswered_frac"] = 0.0
    values["f_graph_inbound_client_latency_z_max"] = 0.2 * rng.random()
    if service == ENTRY:
        values["f_traces_client_error_rate_max"] = 0.001 * rng.random()
    else:                     # leaves make no outbound calls: nothing to observe
        values["f_traces_client_error_rate_max"] = np.nan

    if is_root:
        values["f_traces_latency_z"] = 9.0
        values["f_traces_latency_ratio"] = 8.0
        values["f_metrics_queue_depth_z"] = 4.0
        values["f_graph_downstream_explained"] = -3.0
        values["f_graph_inbound_client_error_rate"] = 0.4
        values["f_graph_inbound_unanswered_frac"] = 0.5
        values["f_graph_inbound_client_latency_z_max"] = 9.0
        values.update(SIGNATURE[spec.fault_type])
    elif faulty and service == ENTRY:
        # The entry point sees the fault through its dependency, not in itself.
        values["f_traces_latency_z"] = 6.0
        values["f_traces_latency_ratio"] = 4.0
        values["f_graph_downstream_explained"] = 5.0
        values["f_traces_client_error_rate_max"] = 0.3

    # The trailing-window view simply mirrors the instantaneous one here.
    values["f_temporal_traces_latency_z_mean3"] = values["f_traces_latency_z"]

    return {
        "experiment_id": spec.experiment_id,
        "window_idx": index,
        "window_start_ns": window_start,
        "window_end_ns": window_end,
        "service": service,
        "is_fault_window": faulty,
        "root_service": spec.root_service if faulty else "",
        "fault_type": spec.fault_type if faulty else "none",
        "is_root": is_root,
        "fault_intensity": 0.8 if faulty else 0.0,
        "since_fault_start_ns": window_start - fault_start_ns if faulty else -1,
        "source": "sim",
        "seed": 0,
        "traffic_shape": spec.traffic_shape,
        "base_rps": 10.0,
        **values,
    }
