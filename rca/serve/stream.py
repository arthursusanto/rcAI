"""Near-real-time telemetry sources and the streaming detector.

Two pieces:

``TelemetrySource``
    Yields :class:`Frame` chunks in time order. ``ReplaySource`` reads a canonical
    experiment directory and pushes it out tick by tick (optionally in wall-clock time);
    ``OtlpFileSource`` tails the collector's file-exporter JSONL files and pushes
    whatever arrived since the last poll. Both expose a ``manifest``, so the detector
    gets its window origin, service list and dependency graph the same way either way.

``StreamingDetector``
    Holds a rolling buffer of raw telemetry, fits the feature baseline once the warm-up
    period has passed, then scores one window at a time and drives an incident state
    machine.

Feature parity with the offline path
------------------------------------
The detector does **not** re-implement any feature. It calls
``rca.features.windows.raw_window_features`` / ``apply_baseline`` / ``graph_features`` /
``temporal_features`` on a ``schema.Experiment`` view of its buffer, with the same window
origin, width and stride the offline build uses and a grid restricted to the single
window being scored. The third pass reads finished window rows rather than telemetry, so
it is fed a rolling buffer of the last ``k - 1`` scored rows -- seeded from the tail of
the warm-up, so even the first scored window has the trailing context offline gave it.
Every pass-1 aggregate, z-score and graph aggregate is therefore computed by exactly the
same code on exactly the same telemetry, and matches ``build_windows`` value for value.

Three things make that hold:

* **Lag.** A window is scored only once the stream clock has passed its end by
  ``lag_windows`` windows, so late-completing spans (a slow parent's children, a call
  that timed out) are buffered before the window is aggregated -- offline they always
  are.
* **History.** The buffer keeps ``history_windows`` of past telemetry, so the metric
  rate columns (per-service differences) see the sample preceding the window, as they
  do offline.
* **No clipping.** ``_assign`` folds telemetry past the last window into it; the
  detector sizes ``n_windows`` from the buffer so nothing is folded. The one window
  where streaming and offline can differ is the experiment's *final* one, where the
  offline build deliberately folds the dropped partial tail in.
"""
from __future__ import annotations

import itertools
import logging
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from rca.benchmark import ingest
from rca.data import schema
from rca.features import windows as W
from rca.features.baseline import Baseline, fit_baseline

DEFAULT_WINDOW_NS = W.DEFAULT_WINDOW_NS
DEFAULT_TICK_NS = 1_000_000_000
LAG_WINDOWS = 1             # windows of grace for late-arriving telemetry
HISTORY_WINDOWS = 3         # windows of raw telemetry kept behind the scored window
EVIDENCE_SERVICES = 3       # evidence is assembled for the top-k ranked services
EVIDENCE_ITEMS = 5          # ... and at most this many spans / logs each
DEFAULT_REFRESH_WINDOWS = 60    # refit the baseline every N windows on a live stream
STUCK_POSITIVE_MIN = 6          # ... or sooner, if the detector is stuck positive
MAX_LATENCY_SAMPLES = 1000      # latency stats are over a rolling sample, not all time
MAX_INCIDENT_WINDOWS = 500      # an incident keeps its most recent windows only
MAX_EVIDENCE_WINDOWS = 5        # ... and the bulky evidence for fewer still
MAX_INCIDENTS = 500             # the detector keeps its most recent incidents

# Evidence metric -> (raw value column, z-score column) of the window table.
EVIDENCE_METRICS = {
    "cpu_util": ("f_metrics_cpu_util_mean", "f_metrics_cpu_util_z"),
    "mem_frac": ("f_metrics_mem_frac_mean", "f_metrics_mem_frac_z"),
    "queue_depth": ("f_metrics_queue_depth_mean", "f_metrics_queue_depth_z"),
    "latency_p95_ms": ("f_traces_latency_p95_ms", "f_traces_latency_z"),
    "error_rate": ("f_traces_error_rate", None),
    "log_error_rate": ("f_logs_error_rate", "f_logs_error_z"),
}
NOTABLE_SEVERITIES = ["FATAL", "ERROR", "WARN"]

LOGGER = logging.getLogger(__name__)

# Absolute thresholds that decide whether a window is fit to fit a baseline on. They have
# to be absolute: this runs *before* there is a baseline, and asking "is this window
# normal for this window" would call a uniformly broken period perfectly normal.
# Failure evidence -- these block a fit. Calibrated against the fault-free window
# distribution of the 60-experiment development dataset (rows with enough requests to
# count): error rate p99 0.032, p99.9 0.062, max 0.190; unanswered p99.9 0.008, max 0.143.
# A gate at the "textbook" 5 % error rate fires on ~1 % of healthy rows, and a warm-up
# needs every service clean in every window at once, so it would defer the fit for ever
# on a healthy system. 0.25 sits clear of the healthy maximum and still catches 11.3 % of
# true-root fault rows outright (against 14.3 % at 0.05) -- the loss is small, and the
# families it misses are the ones the *other* two rules exist for.
HEALTH_ERROR_RATE = 0.25        # inbound requests answered with an error
HEALTH_UNANSWERED_FRAC = 0.2    # calls into the service that never reached its handler
HEALTH_TIMEOUT_FRAC = 0.05      # outbound calls that failed only after hanging
# Saturation -- reported, never blocking (see warmup_health).
HEALTH_CPU_UTIL = 0.9           # fraction of the service's allotted CPU
HEALTH_QUEUE_DEPTH = 50.0       # pending items
HEALTH_TIMEOUT_NS = 1_000_000_000   # a failed call slower than this looks like a timeout
# A rate needs a denominator before it means anything: one failure out of four requests
# is 25 %, and a quiet service sees four requests in a window on a good day.
HEALTH_MIN_SAMPLES = 20


# --- frames and sources ---------------------------------------------------------------
@dataclass
class Frame:
    """Telemetry that arrived between the previous tick and ``now_ns``."""

    now_ns: int
    metrics: pd.DataFrame
    spans: pd.DataFrame
    logs: pd.DataFrame


class TelemetrySource:
    """Yields telemetry in time order. Subclasses set ``manifest`` and ``kind``."""

    kind = "source"
    manifest: schema.Manifest

    def frames(self):
        raise NotImplementedError

    def stop(self) -> None:
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return getattr(self, "_stopped", False)


class ReplaySource(TelemetrySource):
    """Replay a canonical experiment directory tick by tick.

    ``speed`` is a wall-clock multiplier: 1.0 replays in real time, 10.0 ten times
    faster, 0.0 as fast as possible (tests and batch replay). Spans are released at
    their *completion* time, which is when a real collector would export them.
    """

    kind = "replay"

    def __init__(self, exp_dir: Path | str, speed: float = 1.0,
                 tick_ns: int = DEFAULT_TICK_NS, start_offset_ns: int = 0):
        self.exp_dir = Path(exp_dir)
        self.experiment = schema.read_experiment(self.exp_dir)
        self.speed = float(speed)
        self.tick_ns = int(tick_ns)
        self.start_offset_ns = int(start_offset_ns)
        self._stopped = False
        manifest = self.experiment.manifest
        self.origin_ns = int(manifest.start_ns) + self.start_offset_ns
        # Starting part way in is what a detector attached to an already-running system
        # sees, so the stream -- and therefore the window grid and the warm-up -- begins
        # there. The faults keep their absolute times, so the ground truth stays honest.
        self.manifest = (manifest if not self.start_offset_ns
                         else replace(manifest, start_ns=self.origin_ns))

    def frames(self):
        exp = self.experiment
        metrics = exp.metrics.sort_values("ts_ns", ignore_index=True)
        logs = exp.logs.sort_values("ts_ns", ignore_index=True)
        spans = exp.spans.assign(
            _done_ns=exp.spans["start_ns"] + exp.spans["duration_ns"]
        ).sort_values("_done_ns", ignore_index=True)

        start, end = self.origin_ns, self.experiment.manifest.end_ns
        previous = start
        wall = time.perf_counter()
        for now_ns in range(start + self.tick_ns, end + self.tick_ns + 1, self.tick_ns):
            if self.stopped:
                return
            yield Frame(
                now_ns=now_ns,
                metrics=_slice(metrics, "ts_ns", previous, now_ns),
                spans=_slice(spans, "_done_ns", previous, now_ns).drop(columns="_done_ns"),
                logs=_slice(logs, "ts_ns", previous, now_ns),
            )
            previous = now_ns
            if self.speed > 0:
                wall += self.tick_ns / 1e9 / self.speed
                delay = wall - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)


class OtlpFileSource(TelemetrySource):
    """Tail the OTel collector's file-exporter JSONL files.

    Each poll reads the bytes appended since the last one, keeps any trailing partial
    line for next time, and parses the complete lines with ``rca.benchmark.ingest`` --
    the same parser the offline capture path uses.
    """

    kind = "live"

    def __init__(self, capture_dir: Path | str, poll_s: float = 1.0,
                 start_ns: int | None = None, warmup_ns: int = 120_000_000_000,
                 experiment_id: str = "live", traces_name: str = "traces.jsonl",
                 metrics_name: str = "metrics.jsonl", logs_name: str = "logs.jsonl",
                 clock=time.time_ns):
        self.capture_dir = Path(capture_dir)
        self.poll_s = float(poll_s)
        # The stream clock. Wall time for a live collector; overridden when driving a
        # capture whose timestamps are not "now".
        self.clock = clock
        self._stopped = False
        start = int(start_ns if start_ns is not None else self.clock())
        self.manifest = live_manifest(experiment_id, start, warmup_ns)
        # Cumulative metric series (gc pause) are differenced against their predecessor.
        # Each poll is only a slice of the series, so the parser has to carry the last
        # cumulative value across polls or every slice loses its first point -- which,
        # for a series delivered one point per poll, is the whole series.
        self.metric_state: dict = {}
        self._tails = [
            ("metrics", _Tail(self.capture_dir / metrics_name,
                              partial(ingest.parse_metrics, state=self.metric_state))),
            ("spans", _Tail(self.capture_dir / traces_name, ingest.parse_spans)),
            ("logs", _Tail(self.capture_dir / logs_name, ingest.parse_logs)),
        ]
        # Whatever is already in the files predates this run; start from the end.
        for _, tail in self._tails:
            tail.seek_end()

    def frames(self):
        start = self.manifest.start_ns
        while not self.stopped:
            time.sleep(self.poll_s)
            now_ns = int(self.clock())
            read = {name: tail.read(start, now_ns) for name, tail in self._tails}
            yield Frame(now_ns=now_ns, **read)


def live_manifest(experiment_id: str, start_ns: int, warmup_ns: int) -> schema.Manifest:
    """Manifest describing a live stream: the full label space and no ground truth."""
    return schema.Manifest(
        experiment_id=experiment_id, source="otel-demo", seed=0, start_ns=int(start_ns),
        end_ns=int(start_ns) + 10 ** 15,
        traffic=schema.TrafficProfile(base_rps=0.0, shape="unknown"),
        faults=[], services=list(schema.SERVICES),
        edges=[tuple(e) for e in schema.DEPENDENCY_EDGES],
        metric_interval_ns=1_000_000_000, warmup_ns=int(warmup_ns),
    )


class _Tail:
    """Byte-offset tail of one append-only OTLP-JSON file."""

    def __init__(self, path: Path, parser):
        self.path = Path(path)
        self.parser = parser
        self.offset = 0
        self.partial = b""

    def seek_end(self) -> None:
        self.offset = self.path.stat().st_size if self.path.exists() else 0

    def read(self, start_ns: int, end_ns: int) -> pd.DataFrame:
        """Parse the lines appended since the last call, clipped to the run window."""
        report = ingest.IngestReport()
        chunk = self._new_bytes()
        if not chunk:
            # The parsers return a correctly typed empty frame for a missing file.
            return self.parser(self.path.parent / "__none__", start_ns, end_ns, report)
        # The ingest parsers read a path, so hand them the complete new lines as one.
        with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as fh:
            fh.write(chunk)
            scratch = Path(fh.name)
        try:
            return self.parser(scratch, start_ns, end_ns, report)
        finally:
            scratch.unlink(missing_ok=True)

    def _new_bytes(self) -> bytes:
        if not self.path.exists():
            return b""
        size = self.path.stat().st_size
        if size < self.offset:          # rotated or truncated underneath us
            self.offset, self.partial = 0, b""
        if size == self.offset:
            return b""
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            data = self.partial + fh.read()
            # read() goes to the real end of file, which a writer may have moved past
            # ``size`` since the stat above. Trusting the stat re-delivers those bytes
            # on the next poll, duplicating spans, logs and metric points.
            self.offset = fh.tell()
        cut = data.rfind(b"\n")
        if cut < 0:                     # no complete line yet
            self.partial = data
            return b""
        self.partial = data[cut + 1:]
        return data[:cut + 1]


def _slice(frame: pd.DataFrame, column: str, low: int, high: int) -> pd.DataFrame:
    """Rows of a table already sorted on ``column`` with ``low <= value < high``."""
    values = frame[column].to_numpy(dtype="int64")
    lo = int(np.searchsorted(values, low, "left"))
    hi = int(np.searchsorted(values, high, "left"))
    return frame.iloc[lo:hi]


# --- rolling buffer ---------------------------------------------------------------------
def _empty(columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=d) for c, d in columns.items()})


class _Buffer:
    """Rolling buffer of one canonical telemetry table."""

    def __init__(self, columns: dict[str, str], ts_column: str):
        self.columns = columns
        self.ts_column = ts_column
        self._parts: list[pd.DataFrame] = [_empty(columns)]

    def add(self, frame: pd.DataFrame | None) -> None:
        if frame is not None and len(frame):
            self._parts.append(frame[list(self.columns)])

    def frame(self) -> pd.DataFrame:
        if len(self._parts) > 1:
            filled = [part for part in self._parts if len(part)]
            self._parts = [pd.concat(filled, ignore_index=True) if filled
                           else _empty(self.columns)]
        return self._parts[0]

    def trim(self, min_ts: int) -> None:
        frame = self.frame()
        self._parts = [frame[frame[self.ts_column] >= min_ts]]

    def max_ts(self) -> int | None:
        frame = self.frame()
        return int(frame[self.ts_column].max()) if len(frame) else None


# --- incidents ---------------------------------------------------------------------------
@dataclass
class Incident:
    """One detected incident and the per-window history that produced it."""

    incident_id: str
    experiment_id: str
    opened_ns: int
    closed_ns: int | None = None
    windows: list[dict] = field(default_factory=list)
    n_windows_total: int = 0
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_windows_total = self.n_windows_total or len(self.windows)
        for record in self.windows:
            self._remember(record)
        self._forget_stale_evidence()

    @property
    def is_open(self) -> bool:
        return self.closed_ns is None

    def add(self, record: dict) -> None:
        """Append a window, keeping only the most recent ``MAX_INCIDENT_WINDOWS``.

        A long-running incident is otherwise unbounded; the count stays exact because
        ``n_windows_total`` is not what gets trimmed.
        """
        self.windows.append(record)
        self.n_windows_total += 1
        self._remember(record)
        del self.windows[:-MAX_INCIDENT_WINDOWS]
        self._forget_stale_evidence()

    def _remember(self, record: dict) -> None:
        """The verdict on show is the newest positive window's, evidence included."""
        if record["detect"] and record.get("evidence"):
            self.evidence = record["evidence"]

    def _forget_stale_evidence(self) -> None:
        """Evidence is by far the biggest part of a record; keep only the newest few.

        The entries are replaced rather than mutated: the same dicts are handed to
        callers of ``push`` and to the store, and must not change underneath them.
        """
        for position in range(max(len(self.windows) - MAX_EVIDENCE_WINDOWS, 0)):
            record = self.windows[position]
            if record.get("evidence"):
                self.windows[position] = {**record, "evidence": {}}

    def current(self) -> dict:
        """The most recent positive window -- the verdict the incident is showing."""
        for record in reversed(self.windows):
            if record["detect"]:
                return record
        return self.windows[-1]

    def summary(self) -> dict:
        window = self.current()
        ranked = list(zip(window["ranked_services"], window["root_scores"]))
        return {
            "incident_id": self.incident_id,
            "experiment_id": self.experiment_id,
            "opened_ns": self.opened_ns,
            "closed_ns": self.closed_ns,
            "is_open": self.is_open,
            "n_windows": self.n_windows_total,
            "current_window_idx": window["window_idx"],
            "last_window_idx": self.windows[-1]["window_idx"],
            "last_window_end_ns": self.windows[-1]["window_end_ns"],
            "detect_prob": window["detect_prob"],
            "ranking": [{"service": s, "score": v} for s, v in ranked[:EVIDENCE_SERVICES]],
            "root_top1": window["root_top1"],
            "root_margin": window["root_margin"],
            "fault_type_pred": window["fault_type_pred"],
            "fault_type_probs": window["fault_type_probs"],
            "fault_type_conf": window["fault_type_conf"],
        }

    def full(self) -> dict:
        """Summary plus the per-window history and the newest positive window's evidence."""
        return {
            **self.summary(),
            "evidence": self.evidence,
            "windows": [{k: v for k, v in record.items() if k != "evidence"}
                        for record in self.windows],
        }


# --- the detector -------------------------------------------------------------------------
class StreamingDetector:
    """Buffer telemetry, score one window at a time, maintain the incident state."""

    def __init__(self, model, manifest: schema.Manifest, *,
                 window_ns: int = DEFAULT_WINDOW_NS, warmup_ns: int | None = None,
                 debounce_open: int = 1, debounce_close: int = 3,
                 lag_windows: int = LAG_WINDOWS, history_windows: int = HISTORY_WINDOWS,
                 store=None, evidence_services: int = EVIDENCE_SERVICES,
                 evidence_items: int = EVIDENCE_ITEMS, refresh_windows: int = 0):
        self.model = model
        self.manifest = manifest
        self.experiment_id = manifest.experiment_id
        self.services = list(manifest.services)
        self.edges = [tuple(edge) for edge in manifest.edges]
        self.start_ns = int(manifest.start_ns)
        self.window_ns = int(window_ns)
        self.warmup_ns = int(manifest.warmup_ns if warmup_ns is None else warmup_ns)
        # Windows *fully inside* the warm-up, matching rca.features.windows: a window
        # that straddles the end of the warm-up already contains post-warm-up telemetry.
        self.warmup_windows = max(
            (self.warmup_ns - self.window_ns) // self.window_ns + 1, 0)
        if self.warmup_windows < 1:
            raise ValueError(
                f"a {self.warmup_ns / 1e9:g}s warm-up holds no whole "
                f"{self.window_ns / 1e9:g}s window")
        self.debounce_open = max(int(debounce_open), 1)
        self.debounce_close = max(int(debounce_close), 1)
        self.lag_windows = int(lag_windows)
        self.history_windows = max(int(history_windows), 1)
        self.store = store
        self.evidence_services = int(evidence_services)
        self.evidence_items = int(evidence_items)
        # 0 disables the periodic refit; the baseline then describes the warm-up for the
        # process lifetime, which is right for a bounded replay and wrong for a live
        # stream that runs for weeks through deploys and traffic-pattern changes.
        self.refresh_windows = int(refresh_windows)

        self.metrics = _Buffer(schema.METRICS_COLUMNS, "ts_ns")
        self.spans = _Buffer(schema.SPANS_COLUMNS, "start_ns")
        self.logs = _Buffer(schema.LOGS_COLUMNS, "ts_ns")

        self.now_ns = self.start_ns
        self.baseline: Baseline | None = None
        self.baseline_state = "warming-up"
        self.next_window = 0
        self.n_windows_scored = 0
        self.n_baseline_refreshes = 0
        self.n_stuck_refits = 0
        # No real incident lasts this long; past it, suspect the baseline, not the system.
        self.stuck_positive_limit = max(STUCK_POSITIVE_MIN, 3 * self.warmup_windows)
        self._positive_streak = 0
        # Candidate warm-up windows and their health, sliding until a clean run passes.
        self._warmup_pool: deque[tuple] = deque(maxlen=self.warmup_windows)
        self._warmup_cursor = 0
        # Aggregates of the most recent scored windows, whatever the verdict was.
        self._recent_aggregates: deque[tuple] = deque(maxlen=self.warmup_windows)
        # Finished per-service window rows (post graph pass, pre temporal) of the last
        # k - 1 windows: the third pass summarises trailing windows, so serving needs
        # the rows themselves, not the telemetry they came from.
        self.temporal_k = int(W.TEMPORAL_K)
        self._temporal_buffer: deque[pd.DataFrame] = deque(maxlen=max(self.temporal_k - 1, 0))
        self.latencies: deque[dict] = deque(maxlen=MAX_LATENCY_SAMPLES)
        # Raw aggregates of the most recent windows eligible to re-fit the baseline.
        self._baseline_pool: deque[tuple] = deque(maxlen=self.warmup_windows)
        self._since_refresh = 0
        self.last_window: dict | None = None
        self.n_incidents_total = 0
        self.incidents: deque[Incident] = deque(maxlen=MAX_INCIDENTS)
        self.open_incident: Incident | None = None
        self._pending: list[dict] = []
        self._negatives = 0
        self._ids = itertools.count(1)

    # --- ingestion --------------------------------------------------------------------
    def push(self, frame: Frame) -> list[dict]:
        """Buffer one frame and score every window it makes available."""
        self.metrics.add(frame.metrics)
        self.spans.add(frame.spans)
        self.logs.add(frame.logs)
        self.now_ns = max(self.now_ns, int(frame.now_ns))
        return self._advance()

    def run(self, source: TelemetrySource) -> None:
        """Consume a source to exhaustion."""
        for frame in source.frames():
            self.push(frame)

    @property
    def ready(self) -> bool:
        return self.baseline is not None

    def warmup_remaining_ns(self) -> int:
        """Time to the earliest possible fit, assuming every window from now on is clean."""
        if self.baseline is not None:
            return 0
        clean = 0
        for _, reasons in reversed(self._warmup_pool):
            if reasons:
                break
            clean += 1
        outstanding = self.warmup_windows - clean
        return max(self._due(self._warmup_cursor + outstanding - 1) - self.now_ns, 0)

    def _advance(self) -> list[dict]:
        if self.baseline is None:
            self._advance_warmup()
            if self.baseline is None:
                return []
        results = []
        while self.now_ns >= self._due(self.next_window):
            results.append(self._score(self.next_window))
            self.next_window += 1
            self._trim(self.next_window)
        return results

    def _advance_warmup(self) -> None:
        """Aggregate warm-up windows one at a time; fit once a clean run of them passes.

        The warm-up slides. Fitting the first ``warmup_windows`` unconditionally means a
        detector started during an incident bakes that incident into its own definition
        of normal and stays blind to it for the rest of the run.
        """
        while self.baseline is None and self.now_ns >= self._due(self._warmup_cursor):
            window_idx = self._warmup_cursor
            aggregates = self._aggregates(window_idx)
            reasons = warmup_health(aggregates[0], aggregates[2], self.services,
                                    self.window_ns / 1e9)
            self._warmup_pool.append((aggregates, reasons))
            self._warmup_cursor += 1
            self._trim(self._warmup_cursor)
            if reasons:
                LOGGER.info(
                    "warm-up window %d is not baseline material: %s", window_idx,
                    "; ".join(warmup_health(aggregates[0], aggregates[2], self.services,
                                            self.window_ns / 1e9, include_advisory=True)[:4]))
            if any(entry for _, entry in self._warmup_pool):
                self.baseline_state = "untrusted"
                continue
            if len(self._warmup_pool) < self.warmup_windows:
                self.baseline_state = "warming-up"
                continue
            self._fit_from([entry for entry, _ in self._warmup_pool])
            self.next_window = self._warmup_cursor
            self.baseline_state = "ready"
            self._seed_temporal_buffer()

    def _seed_temporal_buffer(self) -> None:
        """Build the last k - 1 warm-up windows' rows so window one has its trailing context.

        Offline, the first scored window's temporal features look back into the warm-up.
        Those rows only become computable once the baseline exists, so they are built here,
        from the same buffered telemetry and the baseline just fitted.
        """
        first = max(self.next_window - self._temporal_buffer.maxlen, 0)
        for window_idx in range(first, self.next_window):
            self._temporal_buffer.append(self._window_rows(window_idx)[0])

    def _due(self, window_idx: int) -> int:
        """Stream clock at which ``window_idx`` has had its full grace period."""
        return self.start_ns + (window_idx + 1 + self.lag_windows) * self.window_ns

    def _trim(self, next_window: int) -> None:
        keep_from = self.start_ns + (next_window - self.history_windows) * self.window_ns
        if keep_from <= self.start_ns:
            return
        for buffer in (self.metrics, self.spans, self.logs):
            buffer.trim(keep_from)

    # --- features ---------------------------------------------------------------------
    def _raw(self, low: int, high: int):
        """Pass-1 aggregates for windows ``[low, high]`` from the buffered telemetry."""
        view = schema.Experiment(
            manifest=self.manifest, metrics=self.metrics.frame(),
            spans=self.spans.frame(), logs=self.logs.frame(),
        )
        grid = pd.MultiIndex.from_product(
            [range(low, high + 1), self.services], names=["window_idx", "service"]
        )
        return W.raw_window_features(
            view, grid, self.start_ns, self.window_ns, self.window_ns,
            self._buffer_windows(high),
        )

    def _buffer_windows(self, at_least: int) -> int:
        """Window count large enough that no buffered telemetry is clipped into it."""
        stamps = [buffer.max_ts() for buffer in (self.metrics, self.spans, self.logs)]
        newest = max([s for s in stamps if s is not None], default=self.start_ns)
        return max(int((newest - self.start_ns) // self.window_ns) + 1, at_least + 1)

    def _aggregates(self, window_idx: int) -> tuple:
        """Pass-1 aggregates of one window, restricted to it and to what a fit needs."""
        return _restrict(*self._raw(window_idx, window_idx), window_idx=window_idx)

    def _fit_from(self, entries: list[tuple]) -> None:
        """Fit the baseline from per-window aggregates, renumbered so all count as warm-up."""
        raws, peers, outs = [], [], []
        for position, (raw, peer_raw, outbound) in enumerate(entries):
            raws.append(raw.assign(window_idx=position))
            peers.append(peer_raw.assign(window_idx=position))
            outs.append(outbound.assign(window_idx=position))
        self.baseline = fit_baseline(
            _concat(raws), _concat(peers), _concat(outs), len(entries)
        )

    def features(self, window_idx: int) -> pd.DataFrame:
        """The window table of one window: exactly the offline columns and values."""
        return self._features(window_idx)[0]

    def _features(self, window_idx: int) -> tuple[pd.DataFrame, tuple, pd.DataFrame]:
        """Full window table, the aggregates a re-fit needs, and the pre-temporal rows."""
        rows, aggregates = self._window_rows(window_idx)
        return self._with_temporal(rows), aggregates, rows

    def _with_temporal(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Attach the trailing-window features, mirroring the offline third pass.

        ``temporal_features`` reads only the current row and the ``k - 1`` before it, so
        running it over the buffer plus this window and keeping this window's rows gives
        exactly what the offline call over the whole table gives.
        """
        parts = [*self._temporal_buffer, rows]
        frame = pd.concat(parts, ignore_index=True) if len(parts) > 1 else rows
        temporal = W.temporal_features(frame, self.temporal_k)
        current = frame["window_idx"].to_numpy() == rows["window_idx"].iloc[0]
        return pd.concat(
            [rows.reset_index(drop=True), temporal[current].reset_index(drop=True)], axis=1
        )

    def _window_rows(self, window_idx: int) -> tuple[pd.DataFrame, tuple]:
        """One window's per-service rows before the temporal pass, plus its aggregates."""
        raw, peer_raw, outbound = self._raw(window_idx, window_idx)
        aggregates = _restrict(raw, peer_raw, outbound, window_idx=window_idx)
        applied = W.apply_baseline(
            raw, peer_raw, outbound, self.baseline, self.window_ns / 1e9
        )
        feat = pd.concat([raw, applied], axis=1)
        feat = pd.concat([feat, W.graph_features(feat, self.edges, self.services)], axis=1)
        start = self.start_ns + window_idx * self.window_ns
        out = pd.DataFrame({
            "experiment_id": self.experiment_id,
            "window_idx": np.int64(window_idx),
            "window_start_ns": np.int64(start),
            "window_end_ns": np.int64(start + self.window_ns),
            "service": feat["service"].astype(str),
        })
        for column in W.WINDOW_FEATURE_COLUMNS:
            out[column] = feat[column].to_numpy(dtype="float64")
        return out.reset_index(drop=True), aggregates

    # --- scoring ----------------------------------------------------------------------
    def _score(self, window_idx: int) -> dict:
        began = time.perf_counter()
        table, aggregates, rows = self._features(window_idx)
        built = time.perf_counter()
        prediction = self.model.predict(table).iloc[0]
        done = time.perf_counter()

        latency = {
            "feature_ms": (built - began) * 1e3,
            "predict_ms": (done - built) * 1e3,
            "total_ms": (done - began) * 1e3,
        }
        self.latencies.append(latency)
        ranked = [str(name) for name in prediction["ranked_services"]]
        record = {
            "experiment_id": self.experiment_id,
            "window_idx": int(window_idx),
            "window_start_ns": int(prediction["window_start_ns"]),
            "window_end_ns": int(prediction["window_end_ns"]),
            "detect_prob": float(prediction["detect_prob"]),
            "detect": bool(prediction["detect"]),
            "ranked_services": ranked,
            "root_scores": [float(v) for v in prediction["root_scores"]],
            "root_top1": str(prediction["root_top1"]),
            "root_margin": float(prediction["root_margin"]),
            "fault_type_pred": str(prediction["fault_type_pred"]),
            "fault_type_probs": {str(k): float(v)
                                 for k, v in prediction["fault_type_probs"].items()},
            "fault_type_conf": float(prediction["fault_type_conf"]),
            "latency_ms": latency,
        }
        record["evidence"] = self._evidence(
            table, record["window_start_ns"], record["window_end_ns"],
            ranked[:self.evidence_services],
        )
        self._temporal_buffer.append(rows)
        self.n_windows_scored += 1
        self.last_window = record
        if self.store is not None:
            self.store.add_window({k: v for k, v in record.items() if k != "evidence"})
        was_open = self.open_incident is not None
        self._update_incident(record)
        self._positive_streak = self._positive_streak + 1 if record["detect"] else 0
        self._recent_aggregates.append(aggregates)
        if self.refresh_windows:
            self._refresh_step(record, aggregates, was_open)
            self._escape_stuck_positive()
        return record

    # --- evidence ---------------------------------------------------------------------
    def _evidence(self, table: pd.DataFrame, start_ns: int, end_ns: int,
                  services: list[str]) -> dict:
        """Supporting telemetry from the buffer, per service, for this window only."""
        spans = self.spans.frame()
        spans = spans[(spans["start_ns"] >= start_ns) & (spans["start_ns"] < end_ns)]
        spans = spans.assign(owner=_owner(spans["service"]))
        logs = self.logs.frame()
        logs = logs[(logs["ts_ns"] >= start_ns) & (logs["ts_ns"] < end_ns)]
        logs = logs.assign(owner=_owner(logs["service"]))
        rows = table.set_index("service")
        return {
            service: {
                "service": service,
                "spans": self._span_evidence(spans[spans["owner"] == service]),
                "logs": self._log_evidence(logs[logs["owner"] == service]),
                "metrics": _metric_evidence(rows.loc[service]),
            }
            for service in services if service in rows.index
        }

    def _span_evidence(self, spans: pd.DataFrame) -> dict:
        def top(frame: pd.DataFrame) -> list[dict]:
            return [{
                "trace_id": str(row.trace_id), "span_id": str(row.span_id),
                "operation": str(row.operation), "service": str(row.service),
                "duration_ms": float(row.duration_ns) / 1e6, "peer": str(row.peer_service),
                "kind": str(row.kind), "status_error": bool(row.status_error),
                "start_ns": int(row.start_ns),
            } for row in frame.nlargest(self.evidence_items, "duration_ns").itertuples()]

        return {
            "error": top(spans[spans["status_error"]]),
            "slow": top(spans),
            "n_spans": len(spans),
            "n_errors": int(spans["status_error"].sum()) if len(spans) else 0,
        }

    def _log_evidence(self, logs: pd.DataFrame) -> list[dict]:
        notable = logs[logs["severity"].isin(NOTABLE_SEVERITIES)]
        if notable.empty:
            return []
        order = {severity: i for i, severity in enumerate(NOTABLE_SEVERITIES)}
        notable = notable.assign(_rank=notable["severity"].map(order))
        notable = notable.sort_values(["_rank", "ts_ns"]).head(self.evidence_items)
        return [{
            "ts_ns": int(row.ts_ns), "severity": str(row.severity), "body": str(row.body),
            "trace_id": str(row.trace_id), "service": str(row.service),
        } for row in notable.itertuples()]

    # --- incident state machine ---------------------------------------------------------
    def _update_incident(self, record: dict) -> None:
        if self.open_incident is None:
            self._maybe_open(record)
            return
        incident = self.open_incident
        incident.add(record)
        if record["detect"]:
            self._negatives = 0
        else:
            self._negatives += 1
            if self._negatives >= self.debounce_close:
                incident.closed_ns = incident.current()["window_end_ns"]
                self.open_incident = None
                self._negatives = 0
        self._persist(incident)

    def _maybe_open(self, record: dict) -> None:
        if not record["detect"]:
            self._pending = []
            return
        self._pending.append(record)
        if len(self._pending) < self.debounce_open:
            return
        incident = Incident(
            incident_id=f"inc-{next(self._ids):04d}",
            experiment_id=self.experiment_id,
            opened_ns=self._pending[0]["window_start_ns"],
            windows=list(self._pending),
        )
        self._pending = []
        self._negatives = 0
        self.open_incident = incident
        self.n_incidents_total += 1
        self.incidents.append(incident)
        self._persist(incident)

    def _persist(self, incident: Incident) -> None:
        if self.store is not None:
            self.store.save_incident(incident)

    # --- periodic baseline refresh ------------------------------------------------------
    def _refresh_step(self, record: dict, aggregates: tuple, was_open: bool) -> None:
        """Collect a healthy window, and every ``refresh_windows`` re-fit the baseline.

        Only windows the detector called *negative* and that sat outside an open incident
        are eligible, so an ongoing outage can never be normalised into "this is fine".
        """
        inside_incident = was_open or self.open_incident is not None
        if not record["detect"] and not inside_incident:
            self._baseline_pool.append(aggregates)
        self._since_refresh += 1
        if self._since_refresh < self.refresh_windows:
            return
        # The counter resets whether or not the re-fit happens, so a system that stays
        # unhealthy retries on the next cycle instead of on every window.
        self._since_refresh = 0
        if len(self._baseline_pool) < self.warmup_windows:
            return
        self._refit_baseline()

    def _refit_baseline(self) -> None:
        """Re-fit from the pooled healthy windows."""
        self._fit_from(list(self._baseline_pool))
        self.n_baseline_refreshes += 1

    def _escape_stuck_positive(self) -> None:
        """Re-fit when the detector is permanently positive on healthy-looking telemetry.

        A baseline fitted on a degraded period makes normal traffic look anomalous for
        ever. If the alarm has been on longer than any real incident and the raw
        telemetry passes the same absolute rules the warm-up has to pass, the baseline is
        what is wrong, so it is replaced regardless of the detector's own verdict.
        """
        if self._positive_streak <= self.stuck_positive_limit:
            return
        if len(self._recent_aggregates) < self.warmup_windows:
            return
        entries = list(self._recent_aggregates)
        seconds = self.window_ns / 1e9
        for raw, _, outbound in entries:
            if warmup_health(raw, outbound, self.services, seconds):
                return
        self._fit_from(entries)
        self.n_stuck_refits += 1
        self._positive_streak = 0
        LOGGER.warning(
            "baseline re-fitted at window %s: %d consecutive positive windows on "
            "telemetry that passes the warm-up health rules",
            self.last_window["window_idx"], self.stuck_positive_limit)

    # --- reporting ------------------------------------------------------------------------
    def latency_stats(self) -> dict:
        """Per-window inference latency over the last ``MAX_LATENCY_SAMPLES`` windows."""
        out: dict = {"n": len(self.latencies)}
        if not self.latencies:
            return out
        for key in ("feature_ms", "predict_ms", "total_ms"):
            values = np.array([entry[key] for entry in self.latencies], dtype="float64")
            out[key] = {
                "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)), "max": float(values.max()),
            }
        return out

    def state(self) -> dict:
        threshold = getattr(self.model, "detect_threshold_", None)
        return {
            "experiment_id": self.experiment_id,
            "ready": self.ready,
            "status": "ready" if self.ready else "warming-up",
            "baseline_state": self.baseline_state,
            "now_ns": self.now_ns,
            "start_ns": self.start_ns,
            "window_ns": self.window_ns,
            "warmup_ns": self.warmup_ns,
            "warmup_remaining_s": self.warmup_remaining_ns() / 1e9,
            "windows_scored": self.n_windows_scored,
            "last_window_idx": None if self.last_window is None
            else self.last_window["window_idx"],
            "last_window_end_ns": None if self.last_window is None
            else self.last_window["window_end_ns"],
            "detect_threshold": None if threshold is None else float(threshold),
            "open_incident": None if self.open_incident is None
            else self.open_incident.incident_id,
            "n_incidents": self.n_incidents_total,
            "refresh_windows": self.refresh_windows,
            "baseline_refreshes": self.n_baseline_refreshes,
            "stuck_refits": self.n_stuck_refits,
            "positive_streak": self._positive_streak,
            "latency": self.latency_stats(),
        }


def warmup_health(raw: pd.DataFrame, outbound: pd.DataFrame, services: list[str],
                  window_seconds: float, include_advisory: bool = False) -> list[str]:
    """Why this window is unfit to fit a baseline on. Empty means it looks healthy.

    Reads only pass-1 aggregates and the raw outbound spans, so it needs no baseline --
    which is the point, since it gates the baseline's own initial fit. NaN means "not
    observable in this window" and is never taken as evidence of trouble.

    Only *failure evidence* blocks a fit: requests answered with errors, calls that never
    reached a handler, calls that failed after hanging. Resource saturation is reported
    when ``include_advisory`` is set but never blocks, because a hot CPU or a deep queue
    is load, not failure -- and the benchmark deliberately contains traffic-only spikes
    as hard negatives. Blocking on them would leave a busy-but-healthy deployment unable
    to ever fit a baseline, which is worse than the stale baseline this guards against.
    """
    served = np.nan_to_num(
        raw["f_traces_request_rate"].to_numpy(dtype="float64")) * window_seconds
    calls_in = _calls_into(raw, outbound, services)
    unanswered, hung = _unanswered_fraction(served, calls_in), _hung_fraction(raw, outbound)
    reasons: list[str] = []
    # (values, denominator, threshold, label). A rate is only consulted once its
    # denominator is large enough to mean anything.
    rules = [
        (raw["f_traces_error_rate"].to_numpy(dtype="float64"), served,
         HEALTH_ERROR_RATE, "error_rate"),
        (unanswered, calls_in, HEALTH_UNANSWERED_FRAC, "unanswered_frac"),
        (hung, _outbound_calls(raw, outbound), HEALTH_TIMEOUT_FRAC, "timeout_frac"),
    ]
    if include_advisory:
        rules += [
            (raw["f_metrics_cpu_util_mean"].to_numpy(dtype="float64"), None,
             HEALTH_CPU_UTIL, "cpu_util"),
            (raw["f_metrics_queue_depth_mean"].to_numpy(dtype="float64"), None,
             HEALTH_QUEUE_DEPTH, "queue_depth"),
        ]
    for values, denominator, limit, label in rules:
        # NaN means "not observable in this window", which is not evidence of trouble.
        over = np.nan_to_num(values, nan=0.0) > limit
        if denominator is not None:
            over &= np.nan_to_num(denominator, nan=0.0) >= HEALTH_MIN_SAMPLES
        for position in np.flatnonzero(over):
            reasons.append(f"{raw['service'].iloc[position]} {label}="
                           f"{values[position]:.3g} > {limit:g}")
    return reasons


def _calls_into(raw: pd.DataFrame, outbound: pd.DataFrame,
                services: list[str]) -> np.ndarray:
    """How many calls the rest of the system made into each service this window."""
    calls = outbound[outbound["peer"].isin(services)
                     & outbound["peer"].ne(outbound["service"])]
    if calls.empty:
        return np.full(len(raw), np.nan)
    count = calls.groupby(["window_idx", "peer"], sort=False).size()
    count.index = count.index.set_names(["window_idx", "service"])
    return count.reindex(
        pd.MultiIndex.from_frame(raw[["window_idx", "service"]])).to_numpy(dtype="float64")


def _outbound_calls(raw: pd.DataFrame, outbound: pd.DataFrame) -> np.ndarray:
    if outbound.empty:
        return np.full(len(raw), np.nan)
    count = outbound.groupby(["window_idx", "service"], sort=False).size()
    return count.reindex(
        pd.MultiIndex.from_frame(raw[["window_idx", "service"]])).to_numpy(dtype="float64")


def _unanswered_fraction(served: np.ndarray, calls_in: np.ndarray) -> np.ndarray:
    """Share of the calls made *into* a service that its own handler never served.

    Callers count what they sent; the callee's server spans count what it answered. A
    service that has stopped answering shows the gap without any notion of "normal".
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(1.0 - served / np.where(calls_in > 0, calls_in, np.nan), 0.0, 1.0)


def _hung_fraction(raw: pd.DataFrame, outbound: pd.DataFrame) -> np.ndarray:
    """Share of a service's outbound calls that failed only after hanging.

    The feature build calls this a timeout relative to the warm-up p95; with no baseline
    to compare against, an absolute wall-clock threshold is the honest substitute -- a
    call that failed after a whole second did not fail fast.
    """
    if outbound.empty:
        return np.full(len(raw), np.nan)
    hung = (outbound["status_error"].to_numpy(dtype=bool)
            & (outbound["duration_ns"].to_numpy(dtype="float64") > HEALTH_TIMEOUT_NS))
    fraction = outbound.assign(_hung=hung.astype("float64")).groupby(
        ["window_idx", "service"], sort=False)["_hung"].mean()
    return fraction.reindex(pd.MultiIndex.from_frame(
        raw[["window_idx", "service"]])).to_numpy(dtype="float64")



def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, ignoring empty frames (pandas deprecates mixing them in)."""
    filled = [frame for frame in frames if len(frame)]
    return pd.concat(filled, ignore_index=True) if filled else frames[0]


def _restrict(raw: pd.DataFrame, peer_raw: pd.DataFrame, outbound: pd.DataFrame,
              window_idx: int) -> tuple:
    """One window's aggregates, and only the outbound columns a fit or a health check reads.

    ``raw_window_features`` returns ``peer_raw`` and ``outbound`` for the whole buffer;
    keeping the other windows' rows would duplicate them across pooled entries and skew
    every statistic fitted from the pool.
    """
    return (
        raw,
        peer_raw[peer_raw["window_idx"] == window_idx],
        outbound.loc[outbound["window_idx"] == window_idx,
                     ["window_idx", "service", "peer", "duration_ns", "status_error"]],
    )


def _owner(services: pd.Series) -> pd.Series:
    """Infra component -> the application service owning it (the features' rule)."""
    plain = services.astype(object)
    return plain.map(schema.INFRA_OWNER).fillna(plain)


def _metric_evidence(row: pd.Series) -> dict:
    return {
        name: {
            "value": _finite(row.get(value_column)),
            "z": None if z_column is None else _finite(row.get(z_column)),
        }
        for name, (value_column, z_column) in EVIDENCE_METRICS.items()
    }


def _finite(value) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def ground_truth(manifest: schema.Manifest) -> dict:
    """The manifest's faults, for the UI to overlay. Ground truth, never a prediction."""
    return {
        "ground_truth": True,
        "experiment_id": manifest.experiment_id,
        "source": manifest.source,
        "start_ns": manifest.start_ns,
        "end_ns": manifest.end_ns,
        "warmup_ns": manifest.warmup_ns,
        "faults": [{
            "fault_type": fault.fault_type,
            "target": schema.INFRA_OWNER.get(fault.target, fault.target),
            "raw_target": fault.target,
            "start_ns": fault.start_ns,
            "end_ns": fault.end_ns,
            "intensity": fault.intensity,
        } for fault in manifest.faults],
    }
