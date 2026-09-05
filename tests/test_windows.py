"""Tests for the window table construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import (
    FAULT_START_S,
    NO_PC_AFTER_S,
    RPS,
    START_NS,
    WINDOW_NS,
    make_experiment,
    make_retry_experiment,
    make_shared_infra_experiment,
)

from rca.data import schema
from rca.features import schema as fschema
from rca.features.windows import (
    FEATURE_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
    TEMPORAL_K,
    TEMPORAL_SOURCES,
    WINDOW_FEATURE_COLUMNS,
    _depth_from_entry,
    _service_edges,
    _warmup_windows,
    _window_starts,
    build_windows,
    temporal_features,
)

FAULT_WINDOWS = [6, 7, 8, 9]     # 10 s windows overlapping the 60 s .. 100 s fault
INBOUND_COLUMNS = [c for c in FEATURE_COLUMNS if c.startswith("f_graph_inbound_")]


def rows(windows: pd.DataFrame, service: str) -> pd.DataFrame:
    return windows[windows["service"] == service].set_index("window_idx")


def test_table_matches_contract(experiment):
    windows = build_windows(experiment)

    assert list(windows.columns[:5]) == fschema.KEY_COLUMNS
    assert fschema.feature_columns(windows.columns) == FEATURE_COLUMNS
    # the per-window family stays focused; the temporal family is derived from it
    assert 40 <= len(WINDOW_FEATURE_COLUMNS) <= 70
    assert FEATURE_COLUMNS == WINDOW_FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS
    assert all(fschema.modality_of(c) == "temporal" for c in TEMPORAL_FEATURE_COLUMNS)
    for column in FEATURE_COLUMNS:
        assert windows[column].dtype == np.float64
        fschema.modality_of(column)   # every feature carries a valid modality tag
    for column, dtype in {**fschema.LABEL_COLUMNS, **fschema.GROUP_COLUMNS}.items():
        assert windows[column].dtype == dtype

    # one row per (window, service) for the manifest services only, aligned to start_ns
    assert len(windows) == 12 * 3
    assert set(windows["service"]) == set(experiment.manifest.services)
    assert windows["window_start_ns"].min() == START_NS
    assert (windows["window_end_ns"] - windows["window_start_ns"] == WINDOW_NS).all()
    assert windows["window_end_ns"].max() == experiment.manifest.end_ns
    assert windows["seed"].iloc[0] == experiment.manifest.seed
    assert windows["traffic_shape"].iloc[0] == "steady"


def test_labels_align_with_fault_interval(experiment):
    windows = build_windows(experiment)
    fault = experiment.manifest.faults[0]

    flagged = sorted(set(windows.loc[windows["is_fault_window"], "window_idx"]))
    assert flagged == FAULT_WINDOWS
    # the label is exactly "the window overlaps the fault interval"
    overlaps = (windows["window_start_ns"] < fault.end_ns) & (
        windows["window_end_ns"] > fault.start_ns)
    assert (windows["is_fault_window"] == overlaps).all()

    faulty = windows[windows["is_fault_window"]]
    assert (faulty["root_service"] == "cart").all()
    assert (faulty["fault_type"] == "network_latency").all()
    assert faulty["fault_intensity"].eq(0.7).all()
    assert (faulty["is_root"] == (faulty["service"] == "cart")).all()
    assert rows(faulty, "cart").loc[FAULT_START_S // 10, "since_fault_start_ns"] == 0
    assert rows(faulty, "cart").loc[FAULT_START_S // 10 + 1, "since_fault_start_ns"] == WINDOW_NS

    clean = windows[~windows["is_fault_window"]]
    assert (clean["root_service"] == "").all()
    assert (clean["fault_type"] == "none").all()
    assert (~clean["is_root"]).all()
    assert clean["fault_intensity"].eq(0.0).all()
    assert clean["since_fault_start_ns"].eq(-1).all()


def test_no_fault_experiment_is_all_negative():
    windows = build_windows(make_experiment(with_fault=False))
    assert not windows["is_fault_window"].any()
    assert (windows["fault_type"] == "none").all()
    assert not windows["is_root"].any()


def test_slow_dependency_localizes_to_the_callee(experiment):
    """B is slow and A calls B: B looks self-slow, A looks explained by downstream."""
    windows = build_windows(experiment)
    fault = windows["window_idx"].isin(FAULT_WINDOWS)
    cart = rows(windows[fault], "cart")
    frontend = rows(windows[fault], "frontend")

    # the callee spends the extra time inside itself
    assert (cart["f_traces_self_time_ratio"] > 0.8).all()
    assert (cart["f_traces_latency_z"] > 3).all()

    # the caller is just as slow end to end, but almost none of it is its own work
    assert (frontend["f_traces_self_time_ratio"] < 0.3).all()
    assert (frontend["f_traces_child_dominated_frac"] > 0.9).all()
    assert (frontend["f_traces_latency_z"] > 3).all()

    # ... which is exactly what "explained by downstream" measures: positive for the
    # victim, and its callee latency z is the larger of the two
    assert (frontend["f_graph_downstream_explained"] > 0).all()
    assert (frontend["f_graph_callee_latency_z_max"] > frontend["f_traces_latency_z"]).all()
    assert (frontend["f_graph_n_callees_anomalous"] >= 1).all()
    assert (cart["f_graph_caller_latency_z_max"] > 3).all()

    # the caller also sees it from the client side, per dependency
    assert (frontend["f_traces_client_latency_z_max"] > 3).all()
    assert (frontend["f_traces_client_error_rate_max"] > 0.05).all()
    assert (frontend["f_traces_retry_frac"] > 0).all()
    assert (frontend["f_traces_timeout_frac"] > 0).all()
    assert (cart["f_logs_error_z"] > 3).all()

    # when the callee really is slow, its callers wait exactly as long as it takes,
    # and every call is answered -- the opposite of a caller-visible timeout
    assert cart["f_graph_inbound_vs_server_latency"].between(0.9, 1.5).all()
    assert cart["f_graph_inbound_unanswered_frac"].lt(0.3).all()

    # both sides of the call see the same slowdown, so there is no path gap to explain:
    # the time is being spent in cart, not on the way to it
    assert frontend["f_graph_outbound_path_gap_max"].abs().lt(3).all()
    assert frontend["f_graph_outbound_path_gap_mean"].abs().lt(3).all()

    # depth is measured from the entry point, which is the frontend here
    assert (frontend["f_graph_depth_from_entry"] == 0).all()
    assert (cart["f_graph_depth_from_entry"] == 1).all()

    # a quiet window before the fault shows none of this
    quiet = rows(windows[windows["window_idx"] == 2], "cart")
    assert quiet["f_traces_latency_z"].abs().lt(3).all()


def test_infra_folds_into_its_owner(experiment):
    """valkey is not a row of its own; its telemetry shows up on cart."""
    windows = build_windows(experiment)
    assert "valkey" not in set(windows["service"])

    cart = rows(windows, "cart")
    assert cart["f_metrics_infra_queue_depth_mean"].notna().all()
    # valkey emits no spans of its own: these come from cart's client spans to it
    assert cart["f_traces_infra_p95_ms"].notna().all()
    assert cart["f_traces_infra_error_rate"].notna().all()
    assert cart["f_traces_infra_latency_z"].notna().all()
    assert cart["f_traces_infra_p95_ms"].between(1.0, 5.0).all()
    assert cart["f_traces_infra_latency_z"].abs().lt(3).all()   # valkey itself is healthy
    # the infra queue backs up during the fault, and is separate from cart's own queue
    assert cart.loc[FAULT_WINDOWS, "f_metrics_infra_queue_depth_mean"].min() > 30
    assert cart.loc[2, "f_metrics_infra_queue_depth_mean"] < 3
    assert cart.loc[FAULT_WINDOWS, "f_metrics_infra_queue_depth_z"].min() > 3
    assert not np.isclose(
        cart.loc[2, "f_metrics_infra_queue_depth_mean"], cart.loc[2, "f_metrics_queue_depth_mean"]
    )
    # frontend owns flagd and product-catalog owns postgresql, neither of which is
    # called here, so their infra features stay NaN rather than 0
    frontend = rows(windows, "frontend")
    assert frontend["f_metrics_infra_queue_depth_mean"].isna().all()
    assert frontend["f_traces_infra_p95_ms"].isna().all()
    assert rows(windows, "product-catalog")["f_traces_infra_latency_z"].isna().all()


def test_infra_features_accept_producer_spans(experiment):
    """A broker call is a producer span, and must land in the same infra group."""
    spans = experiment.spans.copy()
    is_cache = spans["peer_service"] == "valkey"
    spans.loc[is_cache, "kind"] = "producer"
    experiment.spans = spans

    cart = rows(build_windows(experiment), "cart")
    assert cart["f_traces_infra_p95_ms"].notna().all()
    assert cart["f_traces_infra_latency_z"].notna().all()


def test_missing_spans_give_nan_not_zero(experiment):
    """product-catalog stops serving at 100 s: trace features go NaN, metrics do not."""
    windows = build_windows(experiment)
    catalog = rows(windows, "product-catalog")
    silent = catalog.loc[NO_PC_AFTER_S // 10:]
    served = catalog.loc[: NO_PC_AFTER_S // 10 - 1]

    trace_columns = [c for c in FEATURE_COLUMNS if fschema.modality_of(c) == "traces"
                     and not c.startswith("f_traces_infra")]
    assert silent[trace_columns].isna().all().all()
    assert served["f_traces_request_rate"].gt(0).all()
    assert silent["f_metrics_cpu_util_mean"].notna().all()
    assert silent["f_logs_rate"].notna().all()

    # bytes per request needs a request count, so it goes with the traffic, not with
    # the metrics: no inbound spans, no denominator
    assert silent["f_metrics_net_rx_per_req_z"].isna().all()
    assert silent["f_metrics_net_tx_per_req_z"].isna().all()
    assert served["f_metrics_net_rx_per_req_z"].notna().all()
    assert served["f_metrics_net_rx_rate_mean"].notna().all()


def test_deterministic(experiment):
    pd.testing.assert_frame_equal(build_windows(experiment), build_windows(make_experiment()))


def test_overlapping_stride_shares_telemetry(experiment):
    """A half-window stride yields overlapping windows, not a re-partition."""
    windows = build_windows(experiment, stride_ns=WINDOW_NS // 2)
    starts = sorted(set(windows["window_start_ns"]))
    assert len(starts) == 23                                  # 120 s, 10 s wide, 5 s stride
    assert starts[1] - starts[0] == WINDOW_NS // 2
    assert windows["window_end_ns"].max() == experiment.manifest.end_ns
    # the overlapping window straddling the fault start is labelled faulty
    fault_start_ns = experiment.manifest.faults[0].start_ns
    straddling = windows[windows["window_start_ns"] == fault_start_ns - WINDOW_NS // 2]
    assert straddling["is_fault_window"].all()
    assert straddling["since_fault_start_ns"].eq(-WINDOW_NS // 2).all()


def test_partial_final_window_is_dropped(experiment):
    experiment.manifest.end_ns -= 3_000_000_000
    windows = build_windows(experiment)
    assert windows["window_idx"].max() == 10
    assert windows["window_end_ns"].max() <= experiment.manifest.end_ns


@pytest.mark.parametrize("modality", ["metrics", "traces", "logs"])
def test_missing_modality_leaves_its_features_nan(experiment, modality):
    """Dropping a whole telemetry modality must not break the build."""
    table = {"metrics": "metrics", "traces": "spans", "logs": "logs"}[modality]
    setattr(experiment, table, getattr(experiment, table).iloc[:0])
    windows = build_windows(experiment)

    dropped = [c for c in FEATURE_COLUMNS if fschema.modality_of(c) == modality]
    assert windows[dropped].isna().all().all()
    assert len(windows) == 12 * 3


def test_caller_visible_timeouts_light_up_the_callee():
    """B stays healthy but calls into B time out: the symptom must land on B's row.

    This is the packet-loss / unreachable-dependency shape, where B's own server spans
    are fast and clean (or missing entirely) and every trace of the fault lives in A's
    client spans.
    """
    windows = build_windows(make_experiment(fault_mode="caller_timeouts"))
    fault = windows["window_idx"].isin(FAULT_WINDOWS)
    cart = rows(windows[fault], "cart")
    frontend = rows(windows[fault], "frontend")
    catalog = rows(windows[fault], "product-catalog")

    # cart's own telemetry looks entirely healthy -- this is why it is hard to localize
    assert cart["f_traces_latency_z"].abs().lt(3).all()
    assert cart["f_traces_error_rate"].eq(0).all()
    assert cart["f_traces_self_time_ratio"].gt(0.8).all()

    # ... but its callers cannot reach it, and that is now on cart's row
    assert (cart["f_graph_inbound_client_latency_z_max"] > 3).all()
    assert (cart["f_graph_inbound_client_error_rate"] > 0.5).all()
    assert (cart["f_graph_inbound_timeout_frac"] > 0.5).all()
    assert (cart["f_graph_inbound_retry_frac"] > 0).all()
    assert (cart["f_graph_inbound_client_rate"] > cart["f_traces_request_rate"]).all()
    assert (cart["f_graph_inbound_vs_server_latency"] > 5).all()
    assert (cart["f_graph_inbound_unanswered_frac"] > 0.5).all()

    # the caller's own row stays clean of the inbound family: nothing calls it here
    assert frontend[INBOUND_COLUMNS].isna().all().all()

    # ... but the path gap does land on the caller, which is the point: cart reports
    # healthy server spans while frontend's calls into it crawl
    assert (frontend["f_graph_outbound_path_gap_max"] > 10).all()
    assert (frontend["f_graph_outbound_path_gap_mean"] > 3).all()
    assert (frontend["f_temporal_graph_outbound_path_gap_max_max3"] > 10).all()
    # the gap is a caller-side feature: a service nobody sees calling has none
    assert cart["f_graph_outbound_path_gap_max"].isna().all()

    # and a healthy sibling callee of the same caller stays quiet
    assert catalog["f_graph_inbound_client_error_rate"].eq(0).all()
    assert catalog["f_graph_inbound_timeout_frac"].eq(0).all()
    assert catalog["f_graph_inbound_unanswered_frac"].eq(0).all()
    assert catalog["f_graph_inbound_vs_server_latency"].between(0.9, 1.5).all()


def test_memory_ramp_shows_in_the_slope_z(experiment):
    """A slow leak barely moves the level, so the slope carries the signal."""
    windows = build_windows(experiment)
    cart = rows(windows, "cart")
    assert (cart.loc[FAULT_WINDOWS, "f_metrics_mem_frac_slope_z"] > 3).all()
    assert abs(cart.loc[2, "f_metrics_mem_frac_slope_z"]) < 3
    quiet = rows(windows, "frontend").loc[FAULT_WINDOWS, "f_metrics_mem_frac_slope_z"]
    assert quiet.abs().lt(3).all()


def test_shared_infra_is_attributed_to_the_calling_service():
    """A slow database belongs to whoever is waiting on it, not to its nominal owner."""
    windows = build_windows(make_shared_infra_experiment())
    fault = windows["window_idx"].isin(FAULT_WINDOWS)
    accounting = rows(windows[fault], "accounting")
    catalog = rows(windows[fault], "product-catalog")

    # accounting does not own postgresql, but it is the one whose queries crawl
    assert (accounting["f_traces_infra_latency_z"] > 3).all()
    assert (accounting["f_traces_infra_p95_ms"] > 50).all()

    # the owner, querying the same database happily, stays clean
    assert catalog["f_traces_infra_latency_z"].abs().lt(3).all()
    assert (catalog["f_traces_infra_p95_ms"] < 20).all()

    # and accounting's database traffic is never re-attributed to the owner as if it
    # were traffic into product-catalog: infra is not a callee
    inbound = windows[INBOUND_COLUMNS]
    assert inbound.isna().all().all()

    # resource metrics are the component's own, so they still fold onto the owner --
    # and onto the owner *only*, which is the documented limitation of that half
    assert catalog["f_metrics_infra_cpu_util_mean"].notna().all()
    assert accounting["f_metrics_infra_cpu_util_mean"].isna().all()


def test_graph_edges_are_never_fabricated_from_infra():
    """Every edge the graph features use must appear verbatim in the contract."""
    edges = _service_edges(schema.DEPENDENCY_EDGES, schema.SERVICES)
    assert set(edges) <= set(schema.DEPENDENCY_EDGES)
    assert not any(set(edge) & set(schema.INFRA_OWNER) for edge in edges)

    # flagd is owned by frontend and half the system calls it; folding those edges onto
    # the owner used to invent a dependency on frontend for every one of them
    assert [edge for edge in edges if edge[1] == "frontend"] == [
        ("frontend-proxy", "frontend")]
    for caller in ("cart", "checkout", "payment", "ad", "recommendation"):
        assert (caller, "frontend") not in edges


def test_depth_from_entry_is_finite_for_every_service():
    """Infra is not a row, but it is still a hop: consumers are reachable through it."""
    depth = _depth_from_entry(schema.DEPENDENCY_EDGES, schema.SERVICES)
    assert set(depth) == set(schema.SERVICES)
    assert all(np.isfinite(value) for value in depth.values())

    assert depth["frontend-proxy"] == 0        # the load generator is not part of the system
    assert depth["frontend"] == 1
    # the only route to the broker consumers is checkout -> kafka -> consumer
    for consumer in ("accounting", "fraud-detection"):
        assert depth[consumer] == depth["checkout"] + 2


@pytest.mark.parametrize(("same_operation", "expected"), [(False, 0.0), (True, 1.0)])
def test_retry_needs_the_same_operation(same_operation, expected):
    """Two calls to one peer under one parent are a retry only if they are the same call."""
    windows = build_windows(make_retry_experiment(same_operation=same_operation))
    checkout = rows(windows, "checkout")
    cart = rows(windows, "cart")

    assert checkout["f_traces_retry_frac"].eq(expected).all()
    assert cart["f_graph_inbound_retry_frac"].eq(expected).all()
    # either way both calls are counted as traffic
    assert checkout["f_traces_client_rate"].eq(2 * RPS).all()
    assert cart["f_graph_inbound_client_rate"].eq(2 * RPS).all()


def test_baseline_windows_never_reach_past_the_warm_up(experiment):
    """At a stride shorter than the window, a baseline window must still fit inside it."""
    manifest = experiment.manifest
    stride_ns = WINDOW_NS // 2
    starts = _window_starts(manifest.start_ns, manifest.end_ns, WINDOW_NS, stride_ns)
    count = _warmup_windows(manifest.warmup_ns, WINDOW_NS, stride_ns)
    warmup_end = manifest.start_ns + manifest.warmup_ns

    assert count > 0
    assert (starts[:count] + WINDOW_NS <= warmup_end).all()
    assert starts[count] + WINDOW_NS > warmup_end
    # counting windows by their start would have taken one more, whose tail sits in the
    # fault -- fitting the healthy baseline on the anomaly
    assert manifest.warmup_ns // stride_ns > count


def trailing(values: list[float], index: int, k: int = TEMPORAL_K) -> list[float]:
    """The window at ``index`` and the k-1 before it, dropping NaN."""
    window = values[max(index - k + 1, 0): index + 1]
    return [v for v in window if not np.isnan(v)]


@pytest.mark.parametrize(
    "source", ["f_traces_latency_z", "f_metrics_cpu_util_z", "f_traces_error_rate"])
def test_temporal_stats_equal_hand_computed_rolling_values(experiment, source):
    windows = build_windows(experiment)
    for service in experiment.manifest.services:
        row = rows(windows, service)
        values = row[source].tolist()
        for index in range(len(values)):
            seen = trailing(values, index)
            mean3 = row[f"f_temporal_{source[2:]}_mean3"].iloc[index]
            max3 = row[f"f_temporal_{source[2:]}_max3"].iloc[index]
            if not seen:
                assert np.isnan(mean3) and np.isnan(max3)
            else:
                assert mean3 == pytest.approx(sum(seen) / len(seen), nan_ok=False)
                assert max3 == pytest.approx(max(seen))


def test_temporal_deltas_need_a_previous_window(experiment):
    windows = build_windows(experiment)
    for service in experiment.manifest.services:
        row = rows(windows, service)
        for source in ["f_traces_latency_z", "f_metrics_cpu_util_z"]:
            delta = row[f"f_temporal_{source[2:]}_delta"]
            assert np.isnan(delta.loc[0])       # nothing precedes the first window
            values = row[source]
            for index in range(1, len(values)):
                expected = values.iloc[index] - values.iloc[index - 1]
                if np.isnan(expected):
                    assert np.isnan(delta.iloc[index])
                else:
                    assert delta.iloc[index] == pytest.approx(expected)


def test_temporal_features_never_look_ahead(experiment):
    """Changing a window must not move any row that precedes it."""
    windows = build_windows(experiment)
    last = windows["window_idx"].max()
    tampered = windows.copy()
    for source in TEMPORAL_SOURCES:
        tampered.loc[tampered["window_idx"] == last, source] = 999.0

    before = temporal_features(windows)
    after = temporal_features(tampered)
    earlier = windows["window_idx"] < last
    pd.testing.assert_frame_equal(before[earlier], after[earlier])
    # ... and the tampering really did change something, or the test proves nothing
    assert not before[~earlier].equals(after[~earlier])


def test_temporal_features_match_an_online_trailing_buffer(experiment):
    """What serving computes from a buffer of past rows must equal the offline build."""
    windows = build_windows(experiment)
    offline = temporal_features(windows)
    for index in sorted(windows["window_idx"].unique()):
        buffered = windows[windows["window_idx"] <= index]
        online = temporal_features(buffered)
        current = buffered["window_idx"] == index
        pd.testing.assert_frame_equal(online[current], offline[windows["window_idx"] == index])


def test_pooled_stats_weight_windows_by_their_request_count():
    """Three sparse windows are better combined than averaged."""
    counts, errors = [1.0, 1.0, 8.0], [0.0, 0.0, 0.5]     # requests/s and error fraction
    table = pd.DataFrame({
        "experiment_id": "e", "service": "cart", "window_idx": [0, 1, 2],
        "window_start_ns": [0, 10**10, 2 * 10**10],
        "window_end_ns": [10**10, 2 * 10**10, 3 * 10**10],
        **{source: np.nan for source in TEMPORAL_SOURCES},
    })
    table["f_traces_request_rate"] = counts
    table["f_traces_error_rate"] = errors

    out = temporal_features(table)
    pooled = out["f_temporal_traces_error_rate3"].iloc[2]
    assert pooled == pytest.approx(sum(c * e for c, e in zip(counts, errors, strict=True))
                                   / sum(counts))
    assert pooled == pytest.approx(0.4)
    assert out["f_temporal_traces_error_rate_mean3"].iloc[2] == pytest.approx(0.5 / 3)
    assert out["f_temporal_traces_request_rate3"].iloc[2] == pytest.approx(10.0 / 3)
