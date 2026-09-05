"""Canonical experiment -> OTLP-JSON lines, for the serving tests.

The inverse of ``rca.benchmark.ingest``, only as faithful as the tests need: enough of
the real collector's shape (resource attributes, scope wrappers, hex ids, the metric
names of ``METRIC_MAP``) that the parser reads it back, and nothing more. It lives in
``tests`` because it is test scaffolding, not part of the pipeline.

The round trip is deliberately lossy where the parser is: ``gc_pause_ms`` is emitted the
way a real SDK does -- a *cumulative* total in seconds -- and ``cpu_util`` the way
docker_stats does -- a cumulative CPU-time counter, which ingest has to difference
against elapsed time -- so both lose their first point per series. That is exactly the
path the streaming tail has to get right.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rca.data import schema

# canonical metric -> (OTLP name, multiplier applied on the way out, mode), where mode is
#   "gauge"      the value as it stands
#   "cumulative" a running total of the per-interval values (what an SDK reports)
#   "rate"       a running total of value x elapsed time (what a CPU-time counter is)
EMIT_AS: dict[str, tuple[str, float, str]] = {
    schema.METRIC_CPU_UTIL: ("container.cpu.usage.total", 1.0, "rate"),
    schema.METRIC_MEM_BYTES: ("container.memory.usage.total", 1.0, "gauge"),
    schema.METRIC_MEM_LIMIT: ("container.memory.usage.limit", 1.0, "gauge"),
    schema.METRIC_NET_RX_BYTES: ("container.network.io.usage.rx_bytes", 1.0, "gauge"),
    schema.METRIC_NET_TX_BYTES: ("container.network.io.usage.tx_bytes", 1.0, "gauge"),
    schema.METRIC_THREADS: ("container.pids.count", 1.0, "gauge"),
    schema.METRIC_QUEUE_DEPTH: ("kafka.consumer_group.lag_sum", 1.0, "gauge"),
    schema.METRIC_GC_PAUSE_MS: ("jvm.gc.duration", 0.001, "cumulative"),
}
CPU_NAME = EMIT_AS[schema.METRIC_CPU_UTIL][0]
SPAN_KINDS = {
    "internal": "SPAN_KIND_INTERNAL", "server": "SPAN_KIND_SERVER",
    "client": "SPAN_KIND_CLIENT", "producer": "SPAN_KIND_PRODUCER",
    "consumer": "SPAN_KIND_CONSUMER",
}
FILES = {"spans": "traces.jsonl", "metrics": "metrics.jsonl", "logs": "logs.jsonl"}


def experiment_lines(exp: schema.Experiment) -> dict[str, list[tuple[int, str]]]:
    """One OTLP-JSON line per (timestamp, service) of each signal, in time order.

    Returns ``{filename: [(ts_ns, line), ...]}`` so a test can append the lines whose
    timestamp falls in a given tick and reproduce a collector's incremental writes.
    """
    return {
        FILES["metrics"]: _metric_lines(exp.metrics),
        FILES["spans"]: _span_lines(exp.spans),
        FILES["logs"]: _log_lines(exp.logs),
    }


def write_all(exp: schema.Experiment, directory: Path) -> Path:
    """Write the whole experiment out at once (the offline capture)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, lines in experiment_lines(exp).items():
        with open(directory / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(line + "\n" for _, line in lines)
    return directory


# --- per signal ------------------------------------------------------------------------
def _resource(service: str, key: str = "service.name") -> dict:
    return {"attributes": [{"key": key, "value": {"stringValue": service}}]}


def _metric_lines(metrics: pd.DataFrame) -> list[tuple[int, str]]:
    frame = metrics.sort_values(["ts_ns", "service"], kind="mergesort")
    cumulative: dict[tuple[str, str], float] = {}
    last_ts: dict[tuple[str, str], int] = {}
    lines = []
    for (ts_ns, service), rows in frame.groupby(["ts_ns", "service"], sort=True):
        emitted = []
        for row in rows.itertuples():
            spec = EMIT_AS.get(str(row.metric))
            if spec is None:
                continue
            name, scale, mode = spec
            value = float(row.value) * scale
            key = (str(service), name)
            if mode == "cumulative":
                value = cumulative[key] = cumulative.get(key, 0.0) + value
            elif mode == "rate":
                # CPU-nanoseconds: the first point of a series is the counter's start.
                elapsed = int(ts_ns) - last_ts.get(key, int(ts_ns))
                value = cumulative[key] = cumulative.get(key, 0.0) + value * elapsed
                last_ts[key] = int(ts_ns)
            emitted.append({
                "name": name,
                "gauge": {"dataPoints": [
                    {"timeUnixNano": str(int(ts_ns)), "asDouble": value}
                ]},
            })
        if any(metric["name"] == CPU_NAME for metric in emitted):
            # cpu_util is a fraction of the allotment, so declaring a 1.0 CPU limit makes
            # the counter above read back as the same fraction.
            emitted.append({"name": "container.cpu.limit", "gauge": {"dataPoints": [
                {"timeUnixNano": str(int(ts_ns)), "asDouble": 1.0}]}})
        if emitted:
            # docker_stats identifies the container, not a service.name.
            lines.append((int(ts_ns), json.dumps({"resourceMetrics": [{
                "resource": _resource(str(service), "container.name"),
                "scopeMetrics": [{"metrics": emitted}],
            }]})))
    return lines


def _span_lines(spans: pd.DataFrame) -> list[tuple[int, str]]:
    frame = spans.sort_values(["start_ns", "span_id"], kind="mergesort")
    lines = []
    for (start_ns, service), rows in frame.groupby(["start_ns", "service"], sort=True):
        encoded = []
        for row in rows.itertuples():
            kind = str(row.kind)
            span = {
                "traceId": str(row.trace_id),
                "spanId": str(row.span_id),
                "name": str(row.operation),
                "kind": SPAN_KINDS[kind],
                "startTimeUnixNano": str(int(row.start_ns)),
                "endTimeUnixNano": str(int(row.start_ns) + int(row.duration_ns)),
                "status": {"code": 2 if bool(row.status_error) else 0},
            }
            if str(row.parent_span_id):
                span["parentSpanId"] = str(row.parent_span_id)
            # The parser reads a peer only off client/producer spans, as the demo emits.
            if kind in ("client", "producer") and str(row.peer_service):
                span["attributes"] = [
                    {"key": "peer.service", "value": {"stringValue": str(row.peer_service)}}
                ]
            encoded.append(span)
        lines.append((int(start_ns), json.dumps({"resourceSpans": [{
            "resource": _resource(str(service)),
            "scopeSpans": [{"spans": encoded}],
        }]})))
    return lines


def _log_lines(logs: pd.DataFrame) -> list[tuple[int, str]]:
    frame = logs.sort_values(["ts_ns", "service"], kind="mergesort")
    lines = []
    for (ts_ns, service), rows in frame.groupby(["ts_ns", "service"], sort=True):
        records = []
        for row in rows.itertuples():
            record = {
                "timeUnixNano": str(int(row.ts_ns)),
                "severityText": str(row.severity),
                "body": {"stringValue": str(row.body)},
            }
            if str(row.trace_id):
                record["traceId"] = str(row.trace_id)
            records.append(record)
        lines.append((int(ts_ns), json.dumps({"resourceLogs": [{
            "resource": _resource(str(service)),
            "scopeLogs": [{"logRecords": records}],
        }]})))
    return lines
