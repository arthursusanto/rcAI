"""FastAPI incident API and the engine that drives the streaming detector.

The engine owns one background thread pumping a :class:`~rca.serve.stream.TelemetrySource`
into a :class:`~rca.serve.stream.StreamingDetector`. Every endpoint reads the detector's
state under the same lock the thread writes it under, so a request always sees a
consistent window.

Evidence links point at the demo's own observability stack; the base URLs come from the
environment (``JAEGER_URL``, ``GRAFANA_URL``, ...) so a Kubernetes deployment can point
them elsewhere without a code change. In replay mode the evidence itself is still shown
inline, straight from the experiment files, because those external systems hold live
data only.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from rca.data import schema
from rca.serve.store import IncidentStore
from rca.serve.stream import (
    DEFAULT_REFRESH_WINDOWS,
    DEFAULT_WINDOW_NS,
    Incident,
    OtlpFileSource,
    ReplaySource,
    StreamingDetector,
    TelemetrySource,
    ground_truth,
)

LOGGER = logging.getLogger(__name__)

UI_FILE = Path(__file__).parent / "ui" / "index.html"

# One bad frame is a bad frame; a run of them is a broken pump, and the process should
# stop advertising itself as healthy so an orchestrator restarts it.
MAX_CONSECUTIVE_FRAME_ERRORS = 5

DEFAULT_JAEGER_URL = "http://localhost:8080/jaeger/ui"
DEFAULT_GRAFANA_URL = "http://localhost:8080/grafana"
DEFAULT_METRICS_QUERY = (
    'histogram_quantile(0.95, sum by (le) '
    '(rate(duration_milliseconds_bucket{{service_name="{service}"}}[1m])))'
)
DEFAULT_LOGS_QUERY = '{{service_name="{service}"}} |= "error"'


# --- evidence links -------------------------------------------------------------------
@dataclass
class Links:
    """URL templates for the observability stack the demo ships with."""

    jaeger_url: str = DEFAULT_JAEGER_URL
    grafana_url: str = DEFAULT_GRAFANA_URL
    metrics_datasource: str = "webstore-metrics"
    logs_datasource: str = "webstore-logs"
    metrics_query: str = DEFAULT_METRICS_QUERY
    logs_query: str = DEFAULT_LOGS_QUERY

    @classmethod
    def from_env(cls) -> Links:
        env = os.environ
        return cls(
            jaeger_url=env.get("JAEGER_URL", DEFAULT_JAEGER_URL).rstrip("/"),
            grafana_url=env.get("GRAFANA_URL", DEFAULT_GRAFANA_URL).rstrip("/"),
            metrics_datasource=env.get("GRAFANA_METRICS_DATASOURCE", "webstore-metrics"),
            logs_datasource=env.get("GRAFANA_LOGS_DATASOURCE", "webstore-logs"),
            metrics_query=env.get("METRICS_QUERY_TEMPLATE", DEFAULT_METRICS_QUERY),
            logs_query=env.get("LOGS_QUERY_TEMPLATE", DEFAULT_LOGS_QUERY),
        )

    def trace_url(self, trace_id: str) -> str:
        return f"{self.jaeger_url}/trace/{quote(str(trace_id))}"

    def service_traces_url(self, service: str) -> str:
        return f"{self.jaeger_url}/search?service={quote(service)}"

    def explore_url(self, datasource: str, query: str) -> str:
        pane = {"datasource": datasource,
                "queries": [{"refId": "A", "datasource": datasource, "expr": query}],
                "range": {"from": "now-1h", "to": "now"}}
        return (f"{self.grafana_url}/explore?schemaVersion=1&orgId=1"
                f"&panes={quote(json.dumps({'rca': pane}))}")

    def for_service(self, service: str, trace_ids: list[str]) -> dict:
        return {
            "service": service,
            "jaeger_service": self.service_traces_url(service),
            "traces": [{"trace_id": t, "url": self.trace_url(t)} for t in trace_ids],
            "grafana_metrics": self.explore_url(
                self.metrics_datasource, self.metrics_query.format(service=service)),
            "grafana_logs": self.explore_url(
                self.logs_datasource, self.logs_query.format(service=service)),
            "templates": {
                "trace": f"{self.jaeger_url}/trace/{{trace_id}}",
                "logs_query": self.logs_query,
                "metrics_query": self.metrics_query,
            },
        }


def _incident_links(incident: dict, links: Links, evidence: dict | None) -> dict:
    """Links for the incident's top-ranked service, seeded with its evidence traces."""
    service = incident.get("root_top1") or ""
    trace_ids: list[str] = []
    for span in (evidence or {}).get(service, {}).get("spans", {}).get("error", []):
        trace_ids.append(span["trace_id"])
    for span in (evidence or {}).get(service, {}).get("spans", {}).get("slow", []):
        trace_ids.append(span["trace_id"])
    unique = list(dict.fromkeys(trace_ids))[:5]
    return links.for_service(service, unique)


# --- engine ----------------------------------------------------------------------------
class Engine:
    """Runs one telemetry source into one detector on a background thread."""

    def __init__(self, model, *, window_ns: int = DEFAULT_WINDOW_NS,
                 warmup_ns: int | None = None, out_dir: Path | str | None = None,
                 links: Links | None = None, debounce_open: int = 1,
                 debounce_close: int = 3, refresh_windows: int | None = None):
        self.model = model
        self.window_ns = int(window_ns)
        self.warmup_ns = warmup_ns
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.links = links or Links.from_env()
        self.debounce_open = debounce_open
        self.debounce_close = debounce_close
        # None means "per source": a live stream drifts away from its warm-up and needs
        # the periodic re-fit, a bounded replay does not and must stay reproducible.
        self.refresh_windows = refresh_windows

        self.lock = threading.Lock()
        self.store = IncidentStore(self.out_dir)
        self.detector: StreamingDetector | None = None
        self.source: TelemetrySource | None = None
        self.mode = "idle"
        self.finished = False
        self.error: str | None = None
        self.frame_errors = 0
        self._consecutive_errors = 0
        self._thread: threading.Thread | None = None

    def start(self, source: TelemetrySource) -> None:
        self.stop()
        with self.lock:
            self.store = IncidentStore(self.out_dir)
            self.detector = StreamingDetector(
                self.model, source.manifest, window_ns=self.window_ns,
                warmup_ns=self.warmup_ns, store=self.store,
                debounce_open=self.debounce_open, debounce_close=self.debounce_close,
                refresh_windows=self._refresh_windows(source),
            )
            self.source = source
            self.mode = source.kind
            self.finished = False
            self.error = None
            self.frame_errors = 0
            self._consecutive_errors = 0
        self._thread = threading.Thread(target=self._pump, args=(source,), daemon=True)
        self._thread.start()

    def _refresh_windows(self, source: TelemetrySource) -> int:
        if self.refresh_windows is not None:
            return int(self.refresh_windows)
        return DEFAULT_REFRESH_WINDOWS if source.kind == "live" else 0

    def stop(self) -> None:
        source, thread = self.source, self._thread
        if source is not None:
            source.stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        self._thread = None

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _pump(self, source: TelemetrySource) -> None:
        """Feed the detector. One frame's failure must not end the run.

        A malformed export, a partial write, a transient parse error: none of those are
        a reason to stop detecting for the lifetime of the process, which is what a
        single try/round the whole loop used to do. Only a sustained run of failures is
        treated as fatal.
        """
        try:
            for frame in source.frames():
                with self.lock:
                    if self.source is not source:
                        return
                    if not self._push(frame):
                        return
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("telemetry source failed")
        finally:
            # A source that was already replaced must not mark the new run finished.
            if self.source is source:
                self.finished = True

    def _push(self, frame) -> bool:
        """Push one frame; False means the pump has failed too often to continue."""
        try:
            self.detector.push(frame)
        except Exception as exc:
            self.frame_errors += 1
            self._consecutive_errors += 1
            LOGGER.warning("frame at %s failed (%d in a row): %s",
                           frame.now_ns, self._consecutive_errors, exc, exc_info=True)
            if self._consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                self.error = (f"{self._consecutive_errors} consecutive frame failures; "
                              f"last was {type(exc).__name__}: {exc}")
                return False
            return True
        self._consecutive_errors = 0
        return True

    @property
    def alive(self) -> bool:
        """False once the pump has died or given up, so liveness can fail."""
        if self.error is not None:
            return False
        if self.mode == "idle" or self.finished or self._thread is None:
            return True         # nothing running is not the same as something broken
        return self._thread.is_alive()

    # --- views ------------------------------------------------------------------------
    def state(self) -> dict:
        with self.lock:
            detector = self.detector
            base = {"mode": self.mode, "finished": self.finished, "error": self.error,
                    "alive": self.alive, "frame_errors": self.frame_errors,
                    "model": type(self.model).__name__,
                    "model_name": getattr(self.model, "name", "unknown")}
            if detector is None:
                return {**base, "ready": False, "status": "idle"}
            return {**base, **detector.state()}

    def services(self) -> dict:
        with self.lock:
            detector = self.detector
            if detector is None:
                return {"services": [], "edges": [], "scores": {}}
            window = detector.last_window
            scores = {} if window is None else dict(
                zip(window["ranked_services"], window["root_scores"])
            )
            return {
                "services": detector.services,
                "edges": [list(edge) for edge in
                          _folded_edges(detector.edges, detector.services)],
                "scores": scores,
                "window_idx": None if window is None else window["window_idx"],
            }

    def ground_truth(self) -> dict:
        with self.lock:
            if self.detector is None:
                return {"ground_truth": True, "faults": [], "experiment_id": None}
            return ground_truth(self.detector.manifest)

    def recent_windows(self, n: int) -> list[dict]:
        with self.lock:
            return self.store.recent_windows(n)

    def incidents(self) -> list[dict]:
        with self.lock:
            return [self._serialize(incident, incident.summary())
                    for incident in self.store.incidents()]

    def incident(self, incident_id: str) -> dict | None:
        with self.lock:
            found = self.store.incident(incident_id)
            return None if found is None else self._serialize(found, found.full())

    def _serialize(self, incident: Incident, payload: dict) -> dict:
        evidence = incident.evidence
        return {**payload, "links": _incident_links(payload, self.links, evidence)}


def _folded_edges(edges, services) -> list[tuple[str, str]]:
    """Dependency edges rewritten onto owning services (the graph the model sees)."""
    known = set(services)
    folded = set()
    for source, target in edges:
        source = schema.INFRA_OWNER.get(source, source)
        target = schema.INFRA_OWNER.get(target, target)
        if source != target and source in known and target in known:
            folded.add((source, target))
    return sorted(folded)


# --- app ---------------------------------------------------------------------------------
class ReplayRequest(BaseModel):
    experiment_dir: str
    speed: float = 0.0


def create_app(model, *, replay: Path | str | None = None, speed: float = 1.0,
               capture: Path | str | None = None, out_dir: Path | str | None = None,
               window_ns: int = DEFAULT_WINDOW_NS, warmup_ns: int | None = None,
               links: Links | None = None, debounce_open: int = 1,
               debounce_close: int = 3, poll_s: float = 1.0,
               refresh_windows: int | None = None) -> FastAPI:
    """Build the incident API. Starts a replay or a live capture if one is given."""
    engine = Engine(model, window_ns=window_ns, warmup_ns=warmup_ns, out_dir=out_dir,
                    links=links, debounce_open=debounce_open,
                    debounce_close=debounce_close, refresh_windows=refresh_windows)
    app = FastAPI(title="rca incident API")
    app.state.engine = engine

    if replay is not None:
        engine.start(ReplaySource(replay, speed=speed))
    elif capture is not None:
        engine.start(OtlpFileSource(
            capture, poll_s=poll_s,
            warmup_ns=warmup_ns if warmup_ns is not None else 120_000_000_000,
        ))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(UI_FILE, media_type="text/html")

    @app.get("/health")
    def health():
        """Liveness: is the ingest pump still running? 503 restarts the pod."""
        alive = engine.alive
        return JSONResponse(
            status_code=200 if alive else 503,
            content={"status": "ok" if alive else "unhealthy", "mode": engine.mode,
                     "error": engine.error, "frame_errors": engine.frame_errors},
        )

    @app.get("/ready")
    def ready():
        """Readiness: is a baseline fitted, so the verdicts mean anything yet?"""
        state = engine.state()
        ok = bool(state.get("ready")) and engine.alive
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"ready": ok, "status": state.get("status", "idle"),
                     "baseline_state": state.get("baseline_state"),
                     "mode": engine.mode, "error": engine.error},
        )

    @app.get("/state")
    def state():
        return engine.state()

    @app.get("/incidents")
    def incidents():
        return {"incidents": engine.incidents()}

    @app.get("/incidents/{incident_id}")
    def incident(incident_id: str):
        found = engine.incident(incident_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no incident {incident_id!r}")
        return found

    @app.get("/windows/recent")
    def recent(n: int = 100):
        return {"windows": engine.recent_windows(n)}

    @app.post("/replay/start")
    def replay_start(request: ReplayRequest):
        directory = Path(request.experiment_dir)
        if not (directory / "manifest.json").exists():
            raise HTTPException(status_code=400,
                                detail=f"{directory} is not an experiment directory")
        engine.start(ReplaySource(directory, speed=request.speed))
        return {"started": True, "experiment_dir": str(directory), "speed": request.speed}

    @app.get("/replay/ground-truth")
    def replay_ground_truth():
        return engine.ground_truth()

    @app.get("/services")
    def services():
        return engine.services()

    return app
