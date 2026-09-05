"""OTLP-JSON (collector ``file`` exporter output) -> canonical experiment frames.

The demo's collector is configured by ``deploy/compose/otelcol-config-extras.yml`` to
write three files, one JSON object per line, in OTLP-JSON encoding:

    <capture>/traces.jsonl    ExportTraceServiceRequest   {"resourceSpans": [...]}
    <capture>/metrics.jsonl   ExportMetricsServiceRequest {"resourceMetrics": [...]}
    <capture>/logs.jsonl      ExportLogsServiceRequest    {"resourceLogs": [...]}

Everything here is pure parsing over those files plus a ``[start_ns, end_ns]`` clip,
so it is exercised by fixtures in ``tests/fixtures/otlp/`` without Docker.

Field encodings handled: ids as hex or base64, enums as ints or names, 64-bit ints as
JSON strings or numbers -- protojson and the collector's pdata marshaller disagree on
several of these, so both spellings are accepted.
"""
from __future__ import annotations

import base64
import bisect
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from rca.data import schema

# --- service naming -----------------------------------------------------------------
# Container name -> canonical name, plus the browser SDK's separate service.name.
SERVICE_ALIASES: dict[str, str] = {
    "valkey-cart": "valkey",
    "astronomy-db": "postgresql",
    "frontend-web": "frontend",
}
KNOWN_SERVICES: frozenset[str] = frozenset(schema.SERVICES) | frozenset(schema.INFRA_OWNER)


def normalize_service(raw: str | None, ip_map: dict[str, str] | None = None) -> str:
    """Canonical service name, or "" when the name is not part of the label space.

    Accepts bare names, ``host:port`` and URLs, since peer attributes carry all three.

    ``ip_map`` is ``{container ip: service}``, captured per experiment by the runner into
    ``manifest.extra["ip_map"]``. Go's gRPC instrumentation writes the *resolved address*
    into ``server.address`` (``172.18.0.10``) where the JS SDK writes the hostname, so
    without the map every checkout -> payment / cart / currency / product-catalog client
    span is ingested with no peer and those dependency edges disappear. Its keys are IP
    addresses, so a hostname can never collide with one.
    """
    if not raw:
        return ""
    name = str(raw)
    if "://" in name:
        name = name.split("://", 1)[1]
    name = name.lstrip("/").split("/", 1)[0].strip()
    if name.startswith("["):
        name = name[1:].split("]", 1)[0]        # [2001:db8::1]:8080 -> 2001:db8::1
    elif name.count(":") == 1:
        name = name.split(":", 1)[0]            # host:port; a bare IPv6 has more colons
    name = SERVICE_ALIASES.get(name, name)
    if name in KNOWN_SERVICES:
        return name
    mapped = (ip_map or {}).get(name, "")
    return mapped if mapped in KNOWN_SERVICES else ""


# --- metric mapping -------------------------------------------------------------------
# (canonical name, multiplier). Sources: docker_stats and host_metrics are receivers in
# the demo's base collector config; kafkametrics is added by compose.full.yaml; the JVM
# metrics come from the Java agent on ad / fraud-detection / kafka.
METRIC_MAP: dict[str, tuple[str, float]] = {
    # docker_stats: cumulative CPU-nanoseconds. x1e-9 makes it core-seconds, which
    # _differenced turns into cores used (core-seconds per second) and _normalize_cpu
    # then divides by the container's CPU allotment -- the fraction METRIC_CPU_UTIL is
    # defined as. `container.cpu.utilization` looks like the obvious source and is NOT
    # used: on dockerstatsreceiver v0.159.0 it is an average over the container's whole
    # lifetime rather than a per-interval rate, so it does not move during a fault.
    # Measured on the live demo: a container squeezed to 0.02 CPU and visibly throttled
    # (`docker stats` climbing to 2.02%, i.e. saturating its new quota) kept reporting
    # its lifetime 9.2% throughout, while this counter tracked the squeeze exactly.
    "container.cpu.usage.total": (schema.METRIC_CPU_UTIL, 1e-9),
    "container.memory.usage.total": (schema.METRIC_MEM_BYTES, 1.0),
    "container.memory.usage.limit": (schema.METRIC_MEM_LIMIT, 1.0),
    "container.network.io.usage.rx_bytes": (schema.METRIC_NET_RX_BYTES, 1.0),
    "container.network.io.usage.tx_bytes": (schema.METRIC_NET_TX_BYTES, 1.0),
    "container.pids.count": (schema.METRIC_THREADS, 1.0),
    # kafkametrics consumers scraper; the demo's consumer groups are literally named
    # "accounting" and "fraud-detection", so the group attribute is the service.
    "kafka.consumer_group.lag_sum": (schema.METRIC_QUEUE_DEPTH, 1.0),
    # Histogram, seconds, cumulative -> per-interval milliseconds (differenced below).
    "jvm.gc.duration": (schema.METRIC_GC_PAUSE_MS, 1000.0),
}

# Canonical metrics that arrive as a cumulative total but are defined per-interval.
_DIFFERENCED = frozenset({schema.METRIC_GC_PAUSE_MS, schema.METRIC_CPU_UTIL})
# ... and of those, the ones defined as a per-second *rate* rather than an interval
# total, so the difference is divided by the elapsed time as well.
_RATE_DIFFERENCED = frozenset({schema.METRIC_CPU_UTIL})

_NS_PER_S = 1_000_000_000

# Superseded by container.cpu.usage.total (see METRIC_MAP), so it is read only to be
# counted: IngestReport.cpu_utilization_only names the services whose capture carries
# this metric and no usage counter, which is the one case that yields no cpu_util at all.
CPU_UTILIZATION_METRIC = "container.cpu.utilization"

# Not canonical metrics themselves: they are the denominator for cpu_util. docker_stats
# derives container.cpu.limit from HostConfig (NanoCPUs, else CpusetCpus, else
# CPUQuota/CPUPeriod) and only emits it when one of those is actually set.
# deploy/compose/docker-compose.override.yml gives all 15 application services a constant
# 1.0 CPU limit and the injector's cpu_saturation clear restores it, so on a correctly
# deployed stack container.cpu.limit is always present and constant outside the fault.
# The logical-core-count fallback below is only a safety net for a stack without those
# limits; every point that uses it is counted in IngestReport.cpu_limit_missing.
CPU_LIMIT_METRIC = "container.cpu.limit"
CPU_LOGICAL_METRIC = "container.cpu.logical.count"
_CPU_DENOMINATORS = (CPU_LIMIT_METRIC, CPU_LOGICAL_METRIC)

_SPAN_KINDS = {
    0: "internal", 1: "internal", 2: "server", 3: "client", 4: "producer", 5: "consumer",
    "SPAN_KIND_UNSPECIFIED": "internal", "SPAN_KIND_INTERNAL": "internal",
    "SPAN_KIND_SERVER": "server", "SPAN_KIND_CLIENT": "client",
    "SPAN_KIND_PRODUCER": "producer", "SPAN_KIND_CONSUMER": "consumer",
}

_PEER_ATTRS = ("peer.service", "net.peer.name", "server.address", "net.sock.peer.name")

_SEVERITY_TEXT = {
    "TRACE": "DEBUG", "DEBUG": "DEBUG", "FINE": "DEBUG", "VERBOSE": "DEBUG",
    "INFO": "INFO", "INFORMATION": "INFO", "NOTICE": "INFO",
    "WARN": "WARN", "WARNING": "WARN",
    "ERROR": "ERROR", "ERR": "ERROR", "SEVERE": "ERROR",
    "FATAL": "FATAL", "CRITICAL": "FATAL", "PANIC": "FATAL", "EMERGENCY": "FATAL",
}


def _severity(number, text) -> str:
    """OTLP severity -> schema.SEVERITIES.

    OTLP numbers TRACE 1-4, DEBUG 5-8, INFO 9-12, WARN 13-16, ERROR 17-20, FATAL 21-24;
    the schema has no TRACE level, so TRACE folds into DEBUG.
    """
    n = _int(number, 0)
    if n:
        return schema.SEVERITIES[min(max((n - 1) // 4 - 1, 0), len(schema.SEVERITIES) - 1)]
    return _SEVERITY_TEXT.get(str(text or "").strip().upper(), "INFO")


# --- OTLP-JSON primitives -------------------------------------------------------------
def _int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _hex_id(value) -> str:
    """Trace/span id as lowercase hex. Accepts hex (pdata) or base64 (protojson)."""
    if not value:
        return ""
    s = str(value)
    if len(s) in (16, 32) and all(c in "0123456789abcdefABCDEF" for c in s):
        return s.lower()
    try:
        return base64.b64decode(s + "=" * (-len(s) % 4)).hex()
    except ValueError:      # binascii.Error subclasses ValueError
        return s.lower()


def _any_value(v):
    if not isinstance(v, dict):
        return v
    for key in ("stringValue", "boolValue", "doubleValue"):
        if key in v:
            return v[key]
    if "intValue" in v:
        return _int(v["intValue"])
    if "arrayValue" in v:
        return [_any_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return _attributes(v["kvlistValue"].get("values", []))
    if "bytesValue" in v:
        return v["bytesValue"]
    return ""


def _attributes(items) -> dict:
    return {kv.get("key", ""): _any_value(kv.get("value")) for kv in (items or [])}


def _read_lines(path: Path, root_key: str, report: IngestReport, offset: int = 0):
    """Yield the resource entries of every well-formed line of an OTLP-JSON file.

    A generator, not a list. The capture files are append-only for a whole campaign and
    reach gigabytes; holding one as parsed Python objects costs several times its size
    on the heap, and every experiment used to pay that for the whole file to keep a
    two-minute window out of it.

    ``offset`` is a byte position recorded when the experiment started (the runner puts
    it in ``manifest.extra["capture_offsets"]``), so a window parses only the bytes
    appended during it. It is always a line boundary in practice -- the exporter appends
    whole lines -- and if it is not, the partial first line is counted as a bad line.
    """
    path = Path(path)
    if not path.exists():
        return
    # Binary: a text-mode seek only accepts an opaque cookie from tell(), and json.loads
    # takes utf-8 bytes directly.
    with open(path, "rb") as fh:
        if offset:
            fh.seek(offset)
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                document = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # A line the exporter was still flushing when we read the file.
                report.bad_lines += 1
                continue
            if not isinstance(document, dict):
                # Valid JSON, but not an OTLP export request (a bare number or list).
                report.bad_lines += 1
                continue
            yield from document.get(root_key) or []


# --- report ---------------------------------------------------------------------------
# Entries kept per open-ended counter when the report is serialised into a manifest.
MAX_REPORTED_NAMES = 30


@dataclass
class IngestReport:
    spans_in: int = 0
    spans_kept: int = 0
    metric_points_in: int = 0
    metric_points_kept: int = 0
    logs_in: int = 0
    logs_kept: int = 0
    bad_lines: int = 0
    dropped_metric_names: Counter = field(default_factory=Counter)
    dropped_services: Counter = field(default_factory=Counter)
    out_of_window: Counter = field(default_factory=Counter)
    cpu_denominator: Counter = field(default_factory=Counter)
    # service -> cpu_util points that had no container.cpu.limit to divide by. Non-empty
    # means the compose baseline CPU limit is missing for that container, which makes its
    # cpu_util incomparable with the rest (and, during cpu_saturation, a step artefact).
    cpu_limit_missing: Counter = field(default_factory=Counter)
    # service -> container.cpu.utilization points seen for a service that has no
    # container.cpu.usage.total. Non-empty means the capture predates the switch to the
    # usage counter and has no usable cpu_util for those services.
    cpu_utilization_only: Counter = field(default_factory=Counter)
    # Intervals dropped because a cumulative counter went backwards (container restart).
    counter_resets: int = 0
    # Peer attribute value -> client/producer spans whose callee could not be resolved.
    # Container IPs showing up here mean manifest.extra["ip_map"] is missing or stale.
    peer_unresolved: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "spans_in": self.spans_in, "spans_kept": self.spans_kept,
            "metric_points_in": self.metric_points_in,
            "metric_points_kept": self.metric_points_kept,
            "logs_in": self.logs_in, "logs_kept": self.logs_kept,
            "bad_lines": self.bad_lines,
            # The demo emits ~380 distinct metric names and 9 of them map to the
            # canonical schema, so the untruncated counter is mostly noise and bloats
            # every manifest it is stored in. The distinct count keeps it honest.
            "dropped_metric_names": dict(
                self.dropped_metric_names.most_common(MAX_REPORTED_NAMES)),
            "dropped_metric_names_distinct": len(self.dropped_metric_names),
            "dropped_services": dict(self.dropped_services),
            "out_of_window": dict(self.out_of_window),
            "cpu_denominator": dict(self.cpu_denominator),
            "cpu_limit_missing": dict(self.cpu_limit_missing),
            "cpu_utilization_only": dict(self.cpu_utilization_only),
            "counter_resets": self.counter_resets,
            "peer_unresolved": dict(self.peer_unresolved.most_common(MAX_REPORTED_NAMES)),
        }


def _frame(rows, columns: dict[str, str]) -> pd.DataFrame:
    """Rows (list of dicts, or a DataFrame) -> the exact columns and dtypes wanted."""
    df = pd.DataFrame(rows, columns=list(columns))
    for col, dtype in columns.items():
        df[col] = df[col].astype(dtype)
    return df


# --- per-signal parsing ---------------------------------------------------------------
def _peer_service(attrs: dict, ip_map: dict | None, report: IngestReport) -> str:
    """Callee of a client/producer span, from whichever peer attribute carries it."""
    present = [str(attrs[key]) for key in _PEER_ATTRS if attrs.get(key)]
    for value in present:
        peer = normalize_service(value, ip_map)
        if peer:
            return peer
    if present:
        report.peer_unresolved[present[0]] += 1
    return ""


def parse_spans(path, start_ns: int, end_ns: int, report: IngestReport,
                ip_map: dict[str, str] | None = None, offset: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    for rs in _read_lines(Path(path), "resourceSpans", report, offset):
        res = _attributes(rs.get("resource", {}).get("attributes"))
        service = normalize_service(res.get("service.name"))
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                report.spans_in += 1
                if not service:
                    report.dropped_services[str(res.get("service.name", ""))] += 1
                    continue
                ts = _int(span.get("startTimeUnixNano"))
                if not (start_ns <= ts <= end_ns):
                    report.out_of_window["spans"] += 1
                    continue
                attrs = _attributes(span.get("attributes"))
                kind = _SPAN_KINDS.get(span.get("kind", 1), "internal")
                peer = (_peer_service(attrs, ip_map, report)
                        if kind in ("client", "producer") else "")
                status = span.get("status") or {}
                code = status.get("code", 0)
                rows.append({
                    "trace_id": _hex_id(span.get("traceId")),
                    "span_id": _hex_id(span.get("spanId")),
                    "parent_span_id": _hex_id(span.get("parentSpanId")),
                    "service": service,
                    "operation": span.get("name", ""),
                    "start_ns": ts,
                    "duration_ns": max(0, _int(span.get("endTimeUnixNano")) - ts),
                    "status_error": code in (2, "STATUS_CODE_ERROR"),
                    "peer_service": peer,
                    "kind": kind,
                })
    report.spans_kept = len(rows)
    return _frame(rows, schema.SPANS_COLUMNS)


def _data_points(metric: dict) -> list[dict]:
    for key in ("gauge", "sum", "histogram", "exponentialHistogram", "summary"):
        if key in metric:
            return metric[key].get("dataPoints", []) or []
    return []


def _point_value(point: dict) -> float | None:
    if "asDouble" in point:
        return float(point["asDouble"])
    if "asInt" in point:
        return float(_int(point["asInt"]))
    if "sum" in point:            # histogram / summary total
        return float(point["sum"])
    return None


def _latest_at(series: tuple[list[int], list[float]], ts: int) -> float | None:
    """Value of a step series at ts: the last point at or before ts, else the first."""
    timestamps, values = series
    if not timestamps:
        return None
    index = bisect.bisect_right(timestamps, ts)
    return values[index - 1] if index else values[0]


def _normalize_cpu(df: pd.DataFrame, denominators: dict, report: IngestReport
                   ) -> pd.DataFrame:
    """Divide cpu_util (by now, cores used) by the container's CPU allotment.

    Runs *after* ``_differenced``: the raw signal is a cumulative CPU-time counter and
    only becomes "cores used" once it has been differenced against elapsed time.

    The allotment is container.cpu.limit, which the compose override makes a constant
    1.0 CPU outside a fault and which docker_stats tracks through
    `docker update --cpus` (the receiver re-inspects the container on Docker `update`
    events). The core-count fallback exists only so a stack deployed without those
    limits still yields a number; it is counted per service in
    ``report.cpu_limit_missing`` because mixing the two denominators inside one
    experiment turns the start of a cpu_saturation fault into a step artefact.
    """
    mask = df["metric"] == schema.METRIC_CPU_UTIL
    if not mask.any():
        return df
    sorted_series = {
        (service, metric): (
            [ts for ts, _ in sorted(points)], [value for _, value in sorted(points)])
        for (service, metric), points in denominators.items()
    }
    normalized = []
    for ts_ns, service, value in zip(df.loc[mask, "ts_ns"], df.loc[mask, "service"],
                                     df.loc[mask, "value"]):
        for metric, label in ((CPU_LIMIT_METRIC, "cpu_limit"),
                              (CPU_LOGICAL_METRIC, "cpu_logical_count")):
            series = sorted_series.get((service, metric))
            allotment = _latest_at(series, ts_ns) if series else None
            if allotment and allotment > 0:
                value /= allotment
                report.cpu_denominator[label] += 1
                if metric != CPU_LIMIT_METRIC:
                    report.cpu_limit_missing[service] += 1
                break
        else:
            report.cpu_denominator["none"] += 1
            report.cpu_limit_missing[service] += 1
        normalized.append(value)
    df.loc[mask, "value"] = normalized
    return df


def parse_metrics(path, start_ns: int, end_ns: int, report: IngestReport,
                  state: dict | None = None, offset: int = 0) -> pd.DataFrame:
    """Parse metric points. ``state`` carries cumulative totals across calls.

    A streaming caller that hands over one chunk of newly appended lines at a time
    passes the same mutable ``state`` dict every call, so a cumulative series keeps
    its predecessor across chunk boundaries (see ``_differenced``). Offline callers
    pass nothing and the behaviour is unchanged.
    """
    rows: list[dict] = []
    denominators: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    utilization: Counter = Counter()
    for rm in _read_lines(Path(path), "resourceMetrics", report, offset):
        res = _attributes(rm.get("resource", {}).get("attributes"))
        # service.name for SDK metrics, container.name for docker_stats.
        res_service = normalize_service(res.get("service.name") or res.get("container.name"))
        for sm in rm.get("scopeMetrics", []):
            for metric in sm.get("metrics", []):
                name = metric.get("name", "")
                mapped = METRIC_MAP.get(name)
                points = _data_points(metric)
                if name == CPU_UTILIZATION_METRIC:
                    if res_service:
                        utilization[res_service] += len(points)
                    continue
                if name in _CPU_DENOMINATORS:
                    # Consumed as the cpu_util denominator, never emitted on its own, so
                    # it counts as neither an input point nor a dropped one. Denominator
                    # points are deliberately not window-clipped: one from just before
                    # the window still describes the container's allotment inside it.
                    for point in points:
                        value = _point_value(point)
                        if res_service and value is not None:
                            denominators[(res_service, name)].append(
                                (_int(point.get("timeUnixNano")), value))
                    continue
                report.metric_points_in += len(points)
                if mapped is None:
                    report.dropped_metric_names[name] += len(points)
                    continue
                canonical, scale = mapped
                for point in points:
                    attrs = _attributes(point.get("attributes"))
                    # kafkametrics carries no service resource; its consumer group is
                    # the consuming service's name.
                    service = res_service or normalize_service(attrs.get("group"))
                    if not service:
                        raw = res.get("service.name") or res.get("container.name") or ""
                        report.dropped_services[str(raw or attrs.get("group", ""))] += 1
                        continue
                    ts = _int(point.get("timeUnixNano")) or _int(point.get("startTimeUnixNano"))
                    if not (start_ns <= ts <= end_ns):
                        report.out_of_window["metrics"] += 1
                        continue
                    value = _point_value(point)
                    if value is None:
                        report.dropped_metric_names[name] += 1
                        continue
                    rows.append({"ts_ns": ts, "service": service,
                                 "metric": canonical, "value": value * scale})

    with_usage = {row["service"] for row in rows
                  if row["metric"] == schema.METRIC_CPU_UTIL}
    for service, count in utilization.items():
        if service not in with_usage:
            report.cpu_utilization_only[service] += count

    df = _frame(rows, schema.METRICS_COLUMNS)
    if not df.empty:
        # Several points can share (ts, service, metric) -- e.g. network bytes per
        # interface, or kafka lag per topic. Summing is the right aggregate for all
        # mapped metrics, and is a no-op where there is only one point.
        df = df.groupby(["ts_ns", "service", "metric"], as_index=False)["value"].sum()
        df = _normalize_cpu(_differenced(df, state, report), denominators, report)
        df = _frame(df, schema.METRICS_COLUMNS)
    report.metric_points_kept = len(df)
    return df


def _differenced(df: pd.DataFrame, state: dict | None = None,
                 report: IngestReport | None = None) -> pd.DataFrame:
    """Turn cumulative totals into per-interval values for _DIFFERENCED metrics.

    Metrics in ``_RATE_DIFFERENCED`` become a per-*second* rate -- the difference divided
    by the elapsed time -- which is what turns a cumulative CPU-seconds counter into
    cores used.

    ``state`` maps (service, metric) to the last ``(ts_ns, value)`` seen, and is updated
    in place. Passing one across successive calls makes chunk-by-chunk parsing produce
    exactly what a single call over the concatenated chunks would: without it the first
    point of every chunk has no predecessor and is dropped, which silently deletes the
    whole series when the chunks are one point long (a live collector tail).
    """
    mask = df["metric"].isin(_DIFFERENCED)
    if not mask.any():
        return df
    cumulative = df[mask].sort_values(["service", "metric", "ts_ns"]).copy()
    grouped = cumulative.groupby(["service", "metric"], sort=False)[["ts_ns", "value"]]
    previous = grouped.shift()
    if state is not None:
        nothing = (float("nan"), float("nan"))
        carried = pd.DataFrame(
            [state.get(key, nothing)
             for key in zip(cumulative["service"], cumulative["metric"])],
            columns=["ts_ns", "value"], index=cumulative.index, dtype="float64")
        previous = previous.fillna(carried)
        state.update({key: (int(row.ts_ns), float(row.value))
                      for key, row in grouped.last().iterrows()})
    # A counter reset (a container restart) makes the difference negative. That
    # interval has no measurable value at all, so it must be dropped, never reported as
    # a zero -- a fabricated zero in the middle of a fault reads as an idle service.
    raw = cumulative["value"] - previous["value"]
    delta = raw.where(raw >= 0.0)
    if report is not None:
        resets = int(((raw < 0.0) & raw.notna()).sum())
        if resets:
            report.counter_resets += resets
    seconds = (cumulative["ts_ns"] - previous["ts_ns"]) / _NS_PER_S
    is_rate = cumulative["metric"].isin(_RATE_DIFFERENCED)
    # A rate needs a positive time base; duplicate timestamps yield no rate at all.
    cumulative["value"] = delta.where(~is_rate, delta / seconds.where(seconds > 0))
    # The first point of a series has no predecessor and no interval value.
    cumulative = cumulative[cumulative["value"].notna()]
    return pd.concat([df[~mask], cumulative], ignore_index=True)


def parse_logs(path, start_ns: int, end_ns: int, report: IngestReport,
               offset: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    for rl in _read_lines(Path(path), "resourceLogs", report, offset):
        res = _attributes(rl.get("resource", {}).get("attributes"))
        service = normalize_service(res.get("service.name"))
        for sl in rl.get("scopeLogs", []):
            for rec in sl.get("logRecords", []):
                report.logs_in += 1
                if not service:
                    report.dropped_services[str(res.get("service.name", ""))] += 1
                    continue
                ts = _int(rec.get("timeUnixNano")) or _int(rec.get("observedTimeUnixNano"))
                if not (start_ns <= ts <= end_ns):
                    report.out_of_window["logs"] += 1
                    continue
                body = _any_value(rec.get("body"))
                rows.append({
                    "ts_ns": ts,
                    "service": service,
                    "severity": _severity(rec.get("severityNumber"), rec.get("severityText")),
                    "body": body if isinstance(body, str) else json.dumps(body),
                    "trace_id": _hex_id(rec.get("traceId")),
                })
    report.logs_kept = len(rows)
    return _frame(rows, schema.LOGS_COLUMNS)


# --- top level ------------------------------------------------------------------------
def build_experiment(
    capture_dir: Path | str,
    manifest: schema.Manifest,
    traces_name: str = "traces.jsonl",
    metrics_name: str = "metrics.jsonl",
    logs_name: str = "logs.jsonl",
) -> tuple[schema.Experiment, IngestReport]:
    """Parse a capture directory into one canonical Experiment.

    The clip window is ``manifest.start_ns .. manifest.end_ns``. The capture files are
    append-only across a whole campaign, so ``manifest.extra["capture_offsets"]`` -- the
    size of each file when the experiment started -- says where this window begins and
    keeps the cost of a run proportional to the window rather than to the campaign.
    Absent (a hand-written manifest, or a capture from before offsets were recorded), the
    whole file is read and clipped as before.
    """
    capture = Path(capture_dir)
    report = IngestReport()
    ip_map = manifest.extra.get("ip_map") or {}
    offsets = manifest.extra.get("capture_offsets") or {}
    spans = parse_spans(capture / traces_name, manifest.start_ns, manifest.end_ns, report,
                        ip_map, offsets.get(traces_name, 0))
    metrics = parse_metrics(capture / metrics_name, manifest.start_ns, manifest.end_ns,
                            report, None, offsets.get(metrics_name, 0))
    logs = parse_logs(capture / logs_name, manifest.start_ns, manifest.end_ns, report,
                      offsets.get(logs_name, 0))
    return schema.Experiment(manifest=manifest, metrics=metrics, spans=spans, logs=logs), report


def ingest(capture_dir: Path | str, manifest: schema.Manifest,
           out_root: Path | str) -> tuple[Path, IngestReport]:
    """Parse a capture window and write the canonical experiment directory."""
    experiment, report = build_experiment(capture_dir, manifest)
    experiment.manifest.extra["ingest_report"] = report.to_dict()
    return schema.write_experiment(experiment, Path(out_root)), report
