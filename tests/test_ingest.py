"""OTLP-JSON -> canonical frames, against hand-built fixtures in the real OTLP shape."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rca.benchmark import ingest
from rca.data import schema

FIXTURES = Path(__file__).parent / "fixtures" / "otlp"
START_NS = 1_700_000_000_000_000_000
END_NS = START_NS + 120_000_000_000
TRACE_ID = "5b8aa5a2d2c872e8321cf37308d69df2"


@pytest.fixture
def manifest():
    return schema.Manifest(
        experiment_id="fixture",
        source="otel-demo",
        seed=0,
        start_ns=START_NS,
        end_ns=END_NS,
        traffic=schema.TrafficProfile(base_rps=1.0, shape="steady"),
        faults=[],
        services=list(schema.SERVICES),
        edges=list(schema.DEPENDENCY_EDGES),
        metric_interval_ns=5_000_000_000,
        warmup_ns=60_000_000_000,
    )


@pytest.fixture
def parsed(manifest):
    return ingest.build_experiment(FIXTURES, manifest)


# --- spans ---------------------------------------------------------------------------
def test_spans_columns_and_dtypes(parsed):
    exp, _ = parsed
    assert list(exp.spans.columns) == list(schema.SPANS_COLUMNS)
    for col, dtype in schema.SPANS_COLUMNS.items():
        assert exp.spans[col].dtype == dtype


def test_spans_form_one_trace_across_three_services(parsed):
    exp, _ = parsed
    spans = exp.spans
    assert len(spans) == 5
    assert set(spans["trace_id"]) == {TRACE_ID}
    assert set(spans["service"]) == {"frontend-proxy", "frontend", "product-catalog"}
    root = spans[spans["parent_span_id"] == ""]
    assert len(root) == 1
    assert root.iloc[0]["service"] == "frontend-proxy"


def test_span_kind_status_and_peer(parsed):
    exp, _ = parsed
    by_id = exp.spans.set_index("span_id")
    # Hex-encoded, integer span kind (the collector's pdata encoding).
    assert by_id.loc["aaaaaaaaaaaaaaa1", "kind"] == "server"
    assert by_id.loc["aaaaaaaaaaaaaaa2", "kind"] == "client"
    # peer.service wins where present; server.address is the fallback.
    assert by_id.loc["bbbbbbbbbbbbbbb2", "peer_service"] == "product-catalog"
    assert by_id.loc["aaaaaaaaaaaaaaa2", "peer_service"] == "frontend"
    # Server spans never carry a peer.
    assert by_id.loc["bbbbbbbbbbbbbbb1", "peer_service"] == ""
    # Base64 ids and spelled-out enums (the protojson encoding) decode the same way.
    catalog = by_id.loc["ccccccccccccccc1"]
    assert catalog["kind"] == "server"
    assert bool(catalog["status_error"]) is True
    assert catalog["parent_span_id"] == "bbbbbbbbbbbbbbb2"
    assert catalog["duration_ns"] == 26_000_000
    assert not exp.spans[exp.spans["span_id"] != "ccccccccccccccc1"]["status_error"].any()


def test_spans_clipped_and_unknown_services_dropped(parsed):
    _, report = parsed
    assert report.spans_in == 7
    assert report.spans_kept == 5
    assert report.out_of_window["spans"] == 1
    assert report.dropped_services["otel-collector"] == 1


# --- metrics -------------------------------------------------------------------------
def _value(df, service, metric, ts):
    row = df[(df["service"] == service) & (df["metric"] == metric) & (df["ts_ns"] == ts)]
    assert len(row) == 1, f"expected exactly one {service}/{metric} at {ts}, got {len(row)}"
    return row.iloc[0]["value"]


def test_metrics_columns_and_mapping(parsed):
    exp, _ = parsed
    metrics = exp.metrics
    assert list(metrics.columns) == list(schema.METRICS_COLUMNS)
    assert set(metrics["metric"]) <= set(schema.METRIC_NAMES)
    m0 = START_NS + 60_000_000_000
    assert _value(metrics, "frontend", schema.METRIC_MEM_BYTES, m0) == 104857600
    assert _value(metrics, "frontend", schema.METRIC_MEM_LIMIT, m0) == 262144000
    assert _value(metrics, "frontend", schema.METRIC_THREADS, m0) == 37
    # Two interfaces at the same timestamp are summed into one canonical row.
    assert _value(metrics, "frontend", schema.METRIC_NET_RX_BYTES, m0) == 1250


def test_cpu_util_is_a_fraction_of_the_container_allotment(parsed):
    exp, report = parsed
    metrics = exp.metrics
    m0, m1 = START_NS + 60_000_000_000, START_NS + 65_000_000_000
    # cpu_util comes from container.cpu.usage.total, a cumulative CPU-nanosecond counter
    # differenced against elapsed time: frontend burns 1.25e9 ns of CPU in the 5 s to m0,
    # which is 0.25 cores, then 3.0e9 ns -> 0.60 cores. It has no CPU limit, so the
    # denominator is the core count it can see.
    assert _value(metrics, "frontend", schema.METRIC_CPU_UTIL, m0) == pytest.approx(0.25 / 8)
    assert _value(metrics, "frontend", schema.METRIC_CPU_UTIL, m1) == pytest.approx(0.60 / 8)
    # valkey-cart uses a steady 0.04 cores across a limit change: 1.0 CPU at m0, then
    # squeezed to 0.05 before m1, so identical usage reads as 4% and then 80% of its
    # allotment. (Its container.name is also normalised to the schema's infra name.)
    assert _value(metrics, "valkey", schema.METRIC_CPU_UTIL, m0) == pytest.approx(0.04)
    assert _value(metrics, "valkey", schema.METRIC_CPU_UTIL, m1) == pytest.approx(0.8)
    # Neither denominator present: the value stays as cores used.
    assert _value(metrics, "image-provider", schema.METRIC_CPU_UTIL, m0) == pytest.approx(0.12)
    # The counter's first point per series has no predecessor, so it yields no rate.
    assert (metrics["ts_ns"] == START_NS + 55_000_000_000).sum() == 0
    assert report.cpu_denominator == {"cpu_logical_count": 2, "cpu_limit": 2, "none": 1}
    # Every point that did not have a container.cpu.limit is attributed to its service,
    # so a container missing the compose baseline CPU limit is visible in the report.
    assert report.cpu_limit_missing == {"frontend": 2, "image-provider": 1}
    # quote's capture carries container.cpu.utilization and no usage counter, so it has
    # no cpu_util at all -- named in the report rather than silently absent.
    assert report.cpu_utilization_only == {"quote": 1}
    assert metrics[metrics["service"] == "quote"].empty
    # The denominators are consumed, not emitted as canonical metrics or dropped.
    assert "container.cpu.limit" not in report.dropped_metric_names
    assert "container.cpu.logical.count" not in report.dropped_metric_names
    assert "container.cpu.utilization" not in report.dropped_metric_names


def test_cumulative_cpu_counter_becomes_cores_and_survives_chunking():
    # A counter climbing 0.4, 0.6 then 0.3 core-seconds per 5 s interval.
    points = [(0, 0.0), (5, 0.4), (10, 1.0), (15, 1.3)]
    rows = [{"ts_ns": START_NS + t * 1_000_000_000, "service": "cart",
             "metric": schema.METRIC_CPU_UTIL, "value": v} for t, v in points]
    whole = ingest._differenced(pd.DataFrame(rows))
    assert [round(v, 6) for v in whole["value"]] == [0.08, 0.12, 0.06]
    assert list(whole["ts_ns"]) == [START_NS + t * 1_000_000_000 for t in (5, 10, 15)]

    # One point per call, the way the live tail feeds it: without the carried state every
    # point is the first of its series and the whole metric silently disappears.
    state: dict = {}
    chunks = [ingest._differenced(pd.DataFrame([row]), state) for row in rows]
    assert [len(chunk) for chunk in chunks] == [0, 1, 1, 1]
    assert [round(chunk["value"].iloc[0], 6) for chunk in chunks[1:]] == [0.08, 0.12, 0.06]


IP_SPAN = {"resourceSpans": [{
    "resource": {"attributes": [
        {"key": "service.name", "value": {"stringValue": "checkout"}}]},
    "scopeSpans": [{"spans": [{
        "traceId": TRACE_ID, "spanId": "aaaaaaaaaaaaaaa9", "kind": 3,
        "name": "oteldemo.PaymentService/Charge",
        "startTimeUnixNano": str(START_NS + 1), "endTimeUnixNano": str(START_NS + 2),
        "attributes": [
            {"key": "server.address", "value": {"stringValue": "172.18.0.10"}},
            {"key": "server.port", "value": {"intValue": "50051"}}],
    }]}],
}]}


def test_container_ips_resolve_through_the_manifest_ip_map(tmp_path, manifest):
    ip_map = {"172.18.0.10": "payment"}
    assert ingest.normalize_service("172.18.0.10", ip_map) == "payment"
    assert ingest.normalize_service("172.18.0.10:50051", ip_map) == "payment"
    assert ingest.normalize_service("172.18.0.99", ip_map) == ""
    # A map entry naming something outside the label space resolves to nothing.
    assert ingest.normalize_service("10.0.0.1", {"10.0.0.1": "telemetry-docs"}) == ""
    # Hostnames still win and need no map at all.
    assert ingest.normalize_service("cart", ip_map) == "cart"

    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(IP_SPAN) + "\n", newline="\n")

    report = ingest.IngestReport()
    spans = ingest.parse_spans(path, START_NS, END_NS, report)
    # Go's gRPC instrumentation writes the resolved address, so without the map the
    # dependency edge is lost -- and the report names exactly what went unresolved.
    assert spans.iloc[0]["peer_service"] == ""
    assert report.peer_unresolved == {"172.18.0.10": 1}

    manifest.extra["ip_map"] = ip_map
    experiment, report = ingest.build_experiment(tmp_path, manifest)
    assert experiment.spans.iloc[0]["peer_service"] == "payment"
    assert report.peer_unresolved == {}


def test_report_dict_caps_its_open_ended_counters():
    report = ingest.IngestReport()
    for i in range(50):
        report.dropped_metric_names[f"m{i:02d}"] = i
    dumped = report.to_dict()
    assert len(dumped["dropped_metric_names"]) == ingest.MAX_REPORTED_NAMES
    assert dumped["dropped_metric_names_distinct"] == 50
    assert "m49" in dumped["dropped_metric_names"]      # the biggest counts are kept
    assert "m00" not in dumped["dropped_metric_names"]


def test_kafka_consumer_lag_becomes_queue_depth(parsed):
    exp, _ = parsed
    m0, m1 = START_NS + 60_000_000_000, START_NS + 65_000_000_000
    assert _value(exp.metrics, "accounting", schema.METRIC_QUEUE_DEPTH, m0) == 12
    assert _value(exp.metrics, "accounting", schema.METRIC_QUEUE_DEPTH, m1) == 4210
    assert _value(exp.metrics, "fraud-detection", schema.METRIC_QUEUE_DEPTH, m0) == 3


def test_cumulative_gc_histogram_is_differenced(parsed):
    exp, _ = parsed
    gc = exp.metrics[exp.metrics["metric"] == schema.METRIC_GC_PAUSE_MS]
    # The first cumulative point has no predecessor, so only the interval survives.
    assert len(gc) == 1
    assert gc.iloc[0]["service"] == "ad"
    assert gc.iloc[0]["value"] == pytest.approx(45.0)


def test_unmapped_metrics_are_dropped_with_a_count(parsed):
    _, report = parsed
    assert report.dropped_metric_names == {"kafka.brokers": 1}
    assert report.metric_points_in == 20
    assert report.metric_points_kept == 14


# --- logs ----------------------------------------------------------------------------
def test_logs(parsed):
    exp, report = parsed
    logs = exp.logs
    assert list(logs.columns) == list(schema.LOGS_COLUMNS)
    assert len(logs) == 3
    by_service = logs.set_index("service")
    assert by_service.loc["frontend", "severity"] == "INFO"
    assert by_service.loc["product-catalog", "severity"] == "ERROR"
    # severityText only, no severityNumber.
    assert by_service.loc["cart", "severity"] == "WARN"
    assert by_service.loc["cart", "trace_id"] == ""
    # Base64 trace id on the log record decodes to the same trace as the spans.
    assert by_service.loc["product-catalog", "trace_id"] == TRACE_ID
    assert report.logs_in == 4
    assert report.out_of_window["logs"] == 1
    # The last line of the fixture is a half-flushed record.
    assert report.bad_lines == 1


# --- naming --------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("valkey-cart", "valkey"),
    ("astronomy-db", "postgresql"),
    ("frontend-web", "frontend"),
    ("product-catalog:3550", "product-catalog"),
    ("http://email:6060", "email"),
    ("/cart", "cart"),
    ("jaeger", ""),
    (None, ""),
])
def test_normalize_service(raw, expected):
    assert ingest.normalize_service(raw) == expected


def test_ingest_writes_a_readable_experiment(tmp_path, parsed):
    exp, _ = parsed
    out, report = ingest.ingest(FIXTURES, exp.manifest, tmp_path)
    round_tripped = schema.read_experiment(out)
    assert len(round_tripped.spans) == report.spans_kept
    assert round_tripped.manifest.source == "otel-demo"
    assert round_tripped.manifest.extra["ingest_report"]["logs_kept"] == 3


def test_reader_streams_from_an_offset_and_survives_junk_lines(tmp_path):
    path = tmp_path / "traces.jsonl"
    first = json.dumps({"resourceSpans": [{"resource": {}, "scopeSpans": []}]})
    second = json.dumps(IP_SPAN["resourceSpans"][0])
    lines = [first, "not json at all", "12345", json.dumps(IP_SPAN)]
    path.write_text("\n".join(lines) + "\n", newline="\n")

    report = ingest.IngestReport()
    entries = ingest._read_lines(path, "resourceSpans", report)
    # A generator, not a list: the capture files are gigabytes and must not be
    # materialised to pull a two-minute window out of them.
    assert iter(entries) is entries
    assert len(list(entries)) == 2
    # Unparseable text and valid-but-not-an-object lines are both bad lines.
    assert report.bad_lines == 2

    # Reading from the offset of the last line skips everything before it.
    offset = len("\n".join(lines[:-1]).encode()) + 1
    report = ingest.IngestReport()
    assert len(list(ingest._read_lines(path, "resourceSpans", report, offset))) == 1
    assert report.bad_lines == 0
    assert second     # (silences the unused-name lint if the fixture changes shape)


def test_capture_offsets_from_the_manifest_skip_earlier_windows(tmp_path, manifest):
    path = tmp_path / "traces.jsonl"
    earlier = json.loads(json.dumps(IP_SPAN))
    earlier["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["spanId"] = "aaaaaaaaaaaaaaa8"
    head = json.dumps(earlier) + "\n"
    path.write_text(head + json.dumps(IP_SPAN) + "\n", newline="\n")

    manifest.extra["capture_offsets"] = {"traces.jsonl": len(head.encode())}
    experiment, _ = ingest.build_experiment(tmp_path, manifest)
    assert list(experiment.spans["span_id"]) == ["aaaaaaaaaaaaaaa9"]
    # No offsets recorded (an older capture, or a hand-written manifest): read it all.
    manifest.extra.pop("capture_offsets")
    experiment, _ = ingest.build_experiment(tmp_path, manifest)
    assert len(experiment.spans) == 2


def test_a_counter_reset_is_missing_data_not_zero():
    # A container restart takes the cumulative counter back to zero. Reporting the
    # interval as 0 cores would read as an idle service in the middle of a fault.
    rows = [{"ts_ns": START_NS + t * 1_000_000_000, "service": "cart",
             "metric": schema.METRIC_CPU_UTIL, "value": v}
            for t, v in [(0, 1.0), (5, 1.4), (10, 0.1), (15, 0.5)]]
    report = ingest.IngestReport()
    out = ingest._differenced(pd.DataFrame(rows), None, report).sort_values("ts_ns")
    assert [round(v, 6) for v in out["value"]] == [0.08, 0.08]
    assert list(out["ts_ns"]) == [START_NS + t * 1_000_000_000 for t in (5, 15)]
    assert report.counter_resets == 1


@pytest.mark.parametrize("raw, expected", [
    ("cart:7070", "cart"),
    ("[2001:db8::10]:50051", ""),                 # an IPv6 with no map entry
    ("2001:db8::10", ""),                         # ... and bare, not split at the colon
    ("http://[2001:db8::10]:8080/x", ""),
])
def test_ipv6_peer_addresses_are_parsed_whole(raw, expected):
    assert ingest.normalize_service(raw) == expected


def test_ipv6_peer_addresses_resolve_through_the_map():
    ip_map = {"2001:db8::10": "payment"}
    assert ingest.normalize_service("[2001:db8::10]:50051", ip_map) == "payment"
    assert ingest.normalize_service("2001:db8::10", ip_map) == "payment"
    assert ingest.normalize_service("http://[2001:db8::10]:8080/charge", ip_map) == "payment"
