"""Streaming inference, incident state, replay and the incident API.

The load-bearing test is ``test_streaming_features_match_offline``: the online path must
compute the *same* feature values as ``rca.features.windows.build_windows``, or every
number the offline evaluation reports is a number about a different system.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import otlp_writer
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rca.benchmark import ingest
from rca.cli.serve import format_replay, replay_experiment
from rca.data import schema
from rca.features.schema import FEATURE_PREFIX
from rca.features.windows import (
    DEFAULT_WINDOW_NS,
    FEATURE_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
    WINDOW_FEATURE_COLUMNS,
    build_windows,
)
from rca.models.aggregate import AGGREGATE_PREFIX, KEYS, aggregate_windows
from rca.models.base import TwoStageModel
from rca.serve.api import (
    MAX_CONSECUTIVE_FRAME_ERRORS,
    Engine,
    Links,
    create_app,
)
from rca.serve.store import IncidentStore, read_jsonl
from rca.serve.stream import (
    DEFAULT_REFRESH_WINDOWS,
    MAX_EVIDENCE_WINDOWS,
    MAX_INCIDENT_WINDOWS,
    MAX_INCIDENTS,
    MAX_LATENCY_SAMPLES,
    Incident,
    OtlpFileSource,
    ReplaySource,
    StreamingDetector,
    ground_truth,
    warmup_health,
)
from rca.sim import FaultSpec, generate_experiment

SEED = 11
# The feature baseline needs MIN_BASELINE_WINDOWS whole windows of warm-up, and the fault
# has to leave clean windows on both sides of it for the incident to open and close.
DURATION_S = 220.0
WARMUP_S = 60.0
FAULT_START_S = 100.0
FAULT_DURATION_S = 40.0
FAULT_TARGET = "cart"
FAULT_TYPE = "network_latency"

# A tc-netem style delay lands on the *callers* client spans, not on the target's own
# server spans, so the target's own latency z stays flat and the signal shows up in the
# graph features that aggregate what a service's callers observed of it. Which of them
# carries it is not this test's business -- the stub picks the first that both separates
# the fixture and points at the fault target, so feature work can move the signal around
# without silently turning these tests into no-ops.
CANDIDATE_FEATURES = [
    "f_graph_inbound_client_latency_z_max",
    "f_graph_inbound_vs_server_latency",
    "f_traces_client_latency_z_max",
    "f_traces_latency_z",
]


# --- fixtures ---------------------------------------------------------------------------
def _experiment(seed: int = SEED, target: str = FAULT_TARGET) -> schema.Experiment:
    return generate_experiment(
        seed, DURATION_S,
        FaultSpec(FAULT_TYPE, target, FAULT_START_S, FAULT_DURATION_S, 0.8),
        warmup_s=WARMUP_S,
    )


@pytest.fixture(scope="module")
def exp_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    return schema.write_experiment(_experiment(), root)


@pytest.fixture(scope="module")
def offline(exp_dir) -> pd.DataFrame:
    return build_windows(schema.read_experiment(exp_dir))


def _pick_feature(windows: pd.DataFrame) -> str:
    """First candidate that separates the fixture's fault and ranks the target first."""
    per_window = windows.groupby("window_idx", sort=True)
    fault = per_window["is_fault_window"].first().to_numpy(bool)
    onset = int(np.argmax(fault))
    for name in CANDIDATE_FEATURES:
        if name not in windows.columns:
            continue
        score = per_window[name].max().to_numpy(dtype="float64")
        quiet, during = score[:onset], score[fault]
        if onset == 0 or np.isnan(quiet).all() or np.isnan(during).all():
            continue
        if not np.nanmax(quiet) < np.nanmedian(during):
            continue
        top = {windows[windows["window_idx"] == k].nlargest(1, name)["service"].iloc[0]
               for k in np.flatnonzero(fault)}
        if top == {FAULT_TARGET}:
            return name
    raise AssertionError(
        f"no candidate feature separates {FAULT_TYPE} on {FAULT_TARGET}: "
        f"{CANDIDATE_FEATURES}")


class _Squash:
    """Calibrator stand-in: a robust z becomes a probability, crossing 0.5 at ``at``."""

    def __init__(self, at: float):
        self.at = float(at)

    def transform(self, scores):
        z = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
        return 1.0 / (1.0 + np.exp(-(z - self.at) / 2.0))


class StubModel(TwoStageModel):
    """A TwoStageModel whose stages are fixed functions of the CPU z-score.

    Enough signal to detect and localize a CPU fault, with no training cost -- these
    tests are about the serving plumbing, not the model. ``fit`` only picks the
    detection crossing point, from the offline table, so the stub self-calibrates
    instead of hard-coding a magnitude that feature work keeps moving.
    """

    name = "stub"

    def fit(self, windows, val=None):
        self.features_ = list(windows.columns)
        self.root_feature_ = _pick_feature(windows)
        self.detect_feature_ = AGGREGATE_PREFIX + self.root_feature_[len(FEATURE_PREFIX):]
        aggregate = aggregate_windows(windows)
        score = self.score_detect(aggregate, windows)
        fault = aggregate["is_fault_window"].to_numpy(bool)
        quiet = float(score[:int(np.argmax(fault))].max())
        during = float(np.median(score[fault]))
        assert quiet < during, f"fixture is not separable: {quiet} vs {during}"
        crossing = 0.5 * (quiet + during)
        self.detect_calibrator_ = _Squash(crossing)
        self.detect_threshold_ = 0.5
        self.detect_threshold_raw_ = crossing
        self.root_calibrator_ = _Squash(crossing)
        self.classes_ = list(schema.FAULT_TYPES)
        self.temperature_ = 1.0
        return self

    def score_detect(self, aggregate, windows):
        """Max over services of the localization feature, one value per window.

        Read straight off the aggregate when the feature is one of the model layer's
        key features, and reduced from the per-service rows when it is not -- the rule
        baseline reaches into ``windows`` for the same reason.
        """
        if self.detect_feature_ in aggregate.columns:
            return aggregate[self.detect_feature_].fillna(0.0).to_numpy(dtype="float64")
        per_window = windows.groupby(KEYS, sort=True)[self.root_feature_].max()
        keys = pd.MultiIndex.from_arrays([aggregate[key] for key in KEYS])
        return per_window.reindex(keys).fillna(0.0).to_numpy(dtype="float64")

    def score_root(self, rows):
        return rows[self.root_feature_].fillna(0.0).to_numpy(dtype="float64")

    def score_fault(self, rows):
        return np.full((len(rows), len(self.classes_)), 1.0 / len(self.classes_))


class RecordingStub(StubModel):
    """Keeps every window table the detector handed to ``predict``."""

    def __init__(self):
        super().__init__()
        self.seen: dict[int, pd.DataFrame] = {}

    def predict(self, windows):
        self.seen[int(windows["window_idx"].iloc[0])] = windows.copy()
        return super().predict(windows)


@pytest.fixture(scope="module")
def model(offline) -> TwoStageModel:
    return StubModel().fit(offline)


@pytest.fixture(scope="module")
def replayed(offline, exp_dir):
    """One full replay of the fixture experiment, with the scored tables kept."""
    recorder = RecordingStub().fit(offline)
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(recorder, source.manifest, store=IncidentStore())
    records = []
    for frame in source.frames():
        records.extend(detector.push(frame))
    return SimpleNamespace(detector=detector, records=records, seen=recorder.seen,
                           model=recorder)


# --- the consistency test ------------------------------------------------------------------
def test_streaming_features_match_offline(replayed, offline):
    """Every streamed window's features equal the offline build's, value for value.

    The experiment's final window is excluded: offline, ``_assign`` folds the dropped
    partial tail into it, and a stream has no tail to fold.
    """
    records, seen = replayed.records, replayed.seen
    last_offline = int(offline["window_idx"].max())
    streamed = [seen[r["window_idx"]] for r in records
                if r["window_idx"] < last_offline]
    assert streamed, "no windows were scored"
    stream_table = pd.concat(streamed, ignore_index=True)

    keys = ["window_idx", "service"]
    columns = [c for c in offline.columns if c.startswith("f_")]
    _assert_covers_every_pass(columns)
    expected = offline.set_index(keys)[columns]
    actual = stream_table.set_index(keys)[columns]
    expected = expected.reindex(actual.index)

    for column in columns:
        left = expected[column].to_numpy(dtype="float64")
        right = actual[column].to_numpy(dtype="float64")
        assert np.allclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True), column


def _assert_covers_every_pass(columns: list[str]) -> None:
    """The comparison must span all three passes, not just the ones that still exist."""
    assert set(columns) == set(WINDOW_FEATURE_COLUMNS) | set(TEMPORAL_FEATURE_COLUMNS)
    assert len(columns) == len(FEATURE_COLUMNS)
    # The third pass is the one a streaming detector is most likely to get wrong: it
    # reads finished rows of *earlier* windows, not the telemetry of this one.
    assert sum(c.startswith("f_temporal_") for c in columns) == len(TEMPORAL_FEATURE_COLUMNS)


def test_streaming_features_match_offline_named_columns(replayed, offline):
    """Spot check with an explicit tolerance on the columns the verdict hinges on."""
    records, seen = replayed.records, replayed.seen
    window_idx = records[len(records) // 2]["window_idx"]
    streamed = seen[window_idx].set_index("service")
    expected = offline[offline["window_idx"] == window_idx].set_index("service")
    for column in ["f_traces_latency_z", "f_traces_error_rate", "f_metrics_cpu_util_mean",
                   "f_logs_error_rate", "f_graph_callee_latency_z_max",
                   "f_temporal_traces_latency_z_mean3", "f_temporal_traces_error_rate3",
                   "f_temporal_metrics_cpu_util_z_delta"]:
        assert np.allclose(streamed[column].to_numpy(dtype="float64"),
                           expected[column].to_numpy(dtype="float64"),
                           rtol=1e-6, atol=1e-9, equal_nan=True), column


# --- detector behaviour ------------------------------------------------------------------
def test_warmup_then_ready(model, exp_dir):
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest)
    frames = source.frames()
    window_s = int(DEFAULT_WINDOW_NS / 1e9)
    # The last warm-up window closes at WARMUP_S and is scored a lag window later.
    for _ in range(int(WARMUP_S)):
        assert detector.push(next(frames)) == []
    assert not detector.ready
    for _ in range(window_s * detector.lag_windows):
        detector.push(next(frames))
    assert detector.ready
    assert detector.warmup_remaining_ns() == 0


def test_incident_opens_after_the_fault_starts(replayed, exp_dir):
    detector, records = replayed.detector, replayed.records
    fault = schema.read_manifest(exp_dir).faults[0]
    assert detector.incidents, "the fault was never detected"
    incident = detector.incidents[0]
    assert incident.opened_ns >= fault.start_ns
    assert incident.opened_ns < fault.end_ns
    assert incident.current()["root_top1"] == FAULT_TARGET
    # Windows before the fault are all negative.
    early = [r for r in records if r["window_end_ns"] <= fault.start_ns]
    assert early and not any(r["detect"] for r in early)


def test_incident_closes_after_the_fault_ends(replayed, exp_dir):
    detector = replayed.detector
    fault = schema.read_manifest(exp_dir).faults[0]
    incident = detector.incidents[0]
    if incident.closed_ns is not None:
        assert incident.closed_ns >= fault.end_ns - DEFAULT_WINDOW_NS


def test_evidence_covers_the_top_service(replayed):
    records = replayed.records
    positive = next(r for r in records if r["detect"])
    top = positive["ranked_services"][0]
    evidence = positive["evidence"][top]
    assert evidence["service"] == top
    assert evidence["spans"]["n_spans"] > 0
    assert evidence["spans"]["slow"], "no spans in the evidence"
    owners = {schema.INFRA_OWNER.get(s["service"], s["service"])
              for s in evidence["spans"]["slow"]}
    assert owners == {top}
    for span in evidence["spans"]["slow"]:
        assert span["trace_id"] and span["operation"]
    assert set(evidence["metrics"]) >= {"cpu_util", "latency_p95_ms", "queue_depth"}
    assert all({"value", "z"} == set(entry) for entry in evidence["metrics"].values())
    # Which particular metric or z is observable in a window is the feature layer's
    # business; that the evidence block carries real numbers through is this test's.
    assert any(entry["value"] is not None for entry in evidence["metrics"].values())
    assert len(positive["evidence"]) == 3


def test_debounce_needs_consecutive_positives(model, exp_dir):
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest, debounce_open=3, debounce_close=2)
    detector.run(source)
    assert detector.incidents
    incident = detector.incidents[0]
    # The incident is backdated to the first of the three positives that opened it.
    assert incident.windows[0]["detect"]
    assert incident.opened_ns == incident.windows[0]["window_start_ns"]


def test_latency_is_measured(replayed):
    detector, records = replayed.detector, replayed.records
    stats = detector.latency_stats()
    assert stats["n"] == len(records)
    for key in ("feature_ms", "predict_ms", "total_ms"):
        assert stats[key]["p50"] > 0
    assert all(r["latency_ms"]["total_ms"] > 0 for r in records)


# --- store ---------------------------------------------------------------------------------
def test_store_persists_windows_and_incidents(model, exp_dir, tmp_path):
    out = tmp_path / "run"
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest, store=IncidentStore(out))
    detector.run(source)
    windows = read_jsonl(out / "windows.jsonl")
    incidents = read_jsonl(out / "incidents.jsonl")
    assert len(windows) == detector.n_windows_scored
    assert "evidence" not in windows[0]
    assert {"detect_prob", "ranked_services", "window_idx"} <= set(windows[0])
    assert incidents and incidents[-1]["incident_id"].startswith("inc-")


# --- replay tool -----------------------------------------------------------------------------
def test_replay_experiment_reports_detection(model, exp_dir, tmp_path):
    summary = replay_experiment(model, exp_dir, out_dir=tmp_path / "replay")
    assert summary["ground_truth"]["target"] == FAULT_TARGET
    assert summary["detection"]["detected"]
    assert summary["detection"]["detection_delay_s"] >= 0
    assert summary["detection"]["alert_delay_s"] > summary["detection"]["detection_delay_s"]
    assert summary["detection"]["root_at_detection"] == FAULT_TARGET
    assert summary["latency"]["n"] == summary["windows_scored"]
    text = format_replay(summary)
    assert "ranking at detection" in text and "final ranking" in text


def replay_text(fault_type, target, predicted_root, predicted_fault):
    """``format_replay`` output for one hand-built summary; ``target=None`` = no fault."""
    truth = None if target is None else {
        "fault_type": fault_type, "target": target,
        "start_ns": 0, "end_ns": 10, "duration_s": 60.0, "intensity": 0.8,
    }
    return format_replay({
        "experiment_id": "exp-0",
        "ground_truth": truth,
        "windows_scored": 6,
        "positive_windows": 3,
        "detection": {
            "detected": True,
            # replay_summary only fills the detection-time keys when there is a fault to
            # measure against, so a fault-free summary must not carry them either.
            **({} if truth is None else {
                "detection_delay_s": 0.0,
                "alert_delay_s": 12.0,
                "detect_prob_at_detection": 0.9,
                "ranking_at_detection": [(predicted_root, 0.7), ("other", 0.2)],
                "fault_type_at_detection": predicted_fault,
                "correct_at_detection": predicted_root == target,
            }),
            "final_root": predicted_root,
            "final_ranking": [(predicted_root, 0.8), ("other", 0.1)],
            "final_fault_type": predicted_fault,
            "final_margin": 0.7,
        },
        "latency": {"n": 0},
        "incidents": [],
    })


def graded_lines(text):
    """The two verdict-bearing lines of a replay report, keyed by which one they are."""
    lines = text.splitlines()
    return {
        "detection": next(x for x in lines if x.startswith("ranking at detection")),
        "final": next(x for x in lines if x.startswith("final ranking")),
    }


@pytest.mark.parametrize(
    "truth_fault, truth_target, root, fault, expected",
    [
        # The bug this pins: one marker sat after the fault class but graded the root, so
        # a wrong service with the right family rendered as "-> fault X  [correct]".
        ("queue_backlog", "checkout", "cart", "queue_backlog",
         ("[root WRONG]", "[fault correct]")),
        ("dependency_failure", "ad", "ad", "cpu_saturation",
         ("[root correct]", "[fault WRONG]")),
        ("packet_loss", "currency", "currency", "packet_loss",
         ("[root correct]", "[fault correct]")),
        ("memory_leak", "email", "quote", "network_latency",
         ("[root WRONG]", "[fault WRONG]")),
    ],
)
def test_replay_grades_root_and_fault_separately(truth_fault, truth_target, root, fault,
                                                 expected):
    """Root and fault class are graded independently, on *both* reported rankings.

    Asserted per line, not over the whole report: a correct final line would otherwise
    mask a regression on the detection line, which is where the original defect was.
    """
    root_marker, fault_marker = expected
    lines = graded_lines(replay_text(truth_fault, truth_target, root, fault))
    for which, line in lines.items():
        assert root_marker in line, which
        assert fault_marker in line, which
        # The verdict for one stage must never be readable as the verdict for the other.
        opposite_root = "[root correct]" if root_marker == "[root WRONG]" else "[root WRONG]"
        opposite_fault = ("[fault correct]" if fault_marker == "[fault WRONG]"
                          else "[fault WRONG]")
        assert opposite_root not in line, which
        assert opposite_fault not in line, which
        # An unqualified marker is the old ambiguous format and must not come back.
        assert "[correct]" not in line and "[WRONG]" not in line, which


def test_replay_without_ground_truth_grades_nothing():
    """A fault-free experiment has nothing to grade against, so no verdict is claimed."""
    text = replay_text(None, None, "cart", "cpu_saturation")
    assert "no fault" in text
    for marker in ("[root correct]", "[root WRONG]", "[fault correct]", "[fault WRONG]"):
        assert marker not in text


# --- API ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(model, exp_dir):
    app = create_app(model, replay=exp_dir, speed=0.0, links=Links())
    app.state.engine.join(timeout=120)
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_state(client):
    assert client.get("/health").json()["status"] == "ok"
    state = client.get("/state").json()
    assert state["mode"] == "replay"
    assert state["ready"] is True
    assert state["error"] is None
    assert state["windows_scored"] > 0
    assert state["latency"]["total_ms"]["p95"] > 0
    assert state["last_window_end_ns"] > state["start_ns"]


def test_windows_recent(client):
    windows = client.get("/windows/recent?n=5").json()["windows"]
    assert 0 < len(windows) <= 5
    assert {"window_idx", "detect_prob", "detect", "root_top1"} <= set(windows[0])


def test_incidents_and_evidence(client):
    incidents = client.get("/incidents").json()["incidents"]
    assert incidents
    summary = incidents[0]
    assert {"incident_id", "ranking", "fault_type_probs", "links"} <= set(summary)
    assert len(summary["ranking"]) == 3

    full = client.get(f"/incidents/{summary['incident_id']}").json()
    assert full["windows"] and "evidence" not in full["windows"][0]
    top = full["root_top1"]
    assert top in full["evidence"]
    assert full["evidence"][top]["spans"]["slow"]
    links = full["links"]
    assert links["service"] == top
    assert links["jaeger_service"].startswith("http://localhost:8080/jaeger/ui")
    assert links["traces"] and links["traces"][0]["url"].startswith(
        "http://localhost:8080/jaeger/ui/trace/")
    assert "explore" in links["grafana_metrics"] and "explore" in links["grafana_logs"]
    assert links["templates"]["trace"].endswith("/trace/{trace_id}")

    assert client.get("/incidents/inc-9999").status_code == 404


def test_ground_truth_is_labelled(client, exp_dir):
    truth = client.get("/replay/ground-truth").json()
    assert truth["ground_truth"] is True
    assert len(truth["faults"]) == 1
    fault = truth["faults"][0]
    assert fault["fault_type"] == FAULT_TYPE
    assert fault["target"] == FAULT_TARGET
    assert fault["end_ns"] - fault["start_ns"] == int(FAULT_DURATION_S * 1e9)


def test_services_graph(client):
    payload = client.get("/services").json()
    assert set(payload["services"]) == set(schema.SERVICES)
    assert payload["edges"] and all(len(edge) == 2 for edge in payload["edges"])
    known = set(payload["services"])
    assert all(a in known and b in known for a, b in payload["edges"])
    assert set(payload["scores"]) == known
    assert all(0.0 <= v <= 1.0 for v in payload["scores"].values())


def test_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "rca incident console" in response.text
    assert "http" not in response.text.split("<script>")[0].replace("http-equiv", "")


def test_replay_start_endpoint(model, exp_dir):
    app = create_app(model, links=Links())
    with TestClient(app) as test_client:
        assert test_client.get("/state").json()["status"] == "idle"
        bad = test_client.post("/replay/start", json={"experiment_dir": str(exp_dir.parent)})
        assert bad.status_code == 400
        started = test_client.post(
            "/replay/start", json={"experiment_dir": str(exp_dir), "speed": 0.0}
        )
        assert started.json()["started"] is True
        app.state.engine.join(timeout=120)
        assert test_client.get("/state").json()["windows_scored"] > 0
    app.state.engine.stop()


def test_ground_truth_helper(exp_dir):
    manifest = schema.read_manifest(exp_dir)
    truth = ground_truth(manifest)
    assert truth["ground_truth"] is True
    assert truth["experiment_id"] == manifest.experiment_id


# --- the whole prediction path ------------------------------------------------------------
def test_streaming_predictions_match_offline(replayed, offline):
    """Streaming verdicts equal the offline ones, window for window.

    Feature parity is necessary but not sufficient: this runs the *same* model over the
    offline table and checks the detection probability, the ranking and the fault class
    come out identical, so the online path reproduces the offline evaluation end to end.
    """
    records, model = replayed.records, replayed.model
    last_offline = int(offline["window_idx"].max())
    expected = model.predict(offline).set_index("window_idx")
    compared = 0
    for record in records:
        if record["window_idx"] == last_offline:
            continue                      # offline folds the dropped tail into this one
        row = expected.loc[record["window_idx"]]
        assert record["detect_prob"] == pytest.approx(float(row["detect_prob"]), abs=1e-12)
        assert record["detect"] == bool(row["detect"])
        assert record["ranked_services"] == [str(s) for s in row["ranked_services"]]
        assert record["root_scores"] == pytest.approx([float(v) for v in row["root_scores"]],
                                                      abs=1e-12)
        assert record["fault_type_pred"] == str(row["fault_type_pred"])
        compared += 1
    assert compared >= 5


# --- the live source ----------------------------------------------------------------------
OTLP_FIXTURES = Path(__file__).parent / "fixtures" / "otlp"
OTLP_START_NS = 1_700_000_000_000_000_000


def test_otlp_file_source_tails_appended_lines(tmp_path):
    """New lines only, complete lines only -- a half-written line waits for its newline."""
    capture = tmp_path / "capture"
    capture.mkdir()
    names = ["traces.jsonl", "metrics.jsonl", "logs.jsonl"]
    for name in names:
        (capture / name).touch()

    source = OtlpFileSource(capture, poll_s=0.0, start_ns=OTLP_START_NS)
    frames = source.frames()
    first = next(frames)
    assert first.spans.empty and first.metrics.empty and first.logs.empty

    traces = (OTLP_FIXTURES / "traces.jsonl").read_text(encoding="utf-8")
    cut = len(traces) // 2
    _append(capture / "traces.jsonl", traces[:cut])
    for name in names[1:]:
        _append(capture / name, (OTLP_FIXTURES / name).read_text(encoding="utf-8"))
    second = next(frames)
    assert not second.metrics.empty and not second.logs.empty

    _append(capture / "traces.jsonl", traces[cut:])
    third = next(frames)
    source.stop()

    report = ingest.IngestReport()
    expected = ingest.parse_spans(OTLP_FIXTURES / "traces.jsonl", OTLP_START_NS,
                                  OTLP_START_NS + 10 ** 15, report)
    streamed = pd.concat([second.spans, third.spans], ignore_index=True)
    assert len(streamed) == len(expected)
    assert set(streamed["span_id"]) == set(expected["span_id"])
    assert list(streamed.columns) == list(schema.SPANS_COLUMNS)


def _append(path, text: str) -> None:
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write(text)


# --- the live tail, end to end ---------------------------------------------------------------
def _drive_capture(source, detector, lines, capture, manifest, clock, tick_ns=1_000_000_000):
    """Append each tick's OTLP lines, advance the clock, poll, push. One tick per poll."""
    frames = source.frames()
    cursor = {name: 0 for name in lines}
    for now_ns in range(manifest.start_ns + tick_ns, manifest.end_ns + tick_ns + 1, tick_ns):
        for name, entries in lines.items():
            with open(capture / name, "a", encoding="utf-8", newline="\n") as fh:
                while cursor[name] < len(entries) and entries[cursor[name]][0] < now_ns:
                    fh.write(entries[cursor[name]][1] + "\n")
                    cursor[name] += 1
        clock[0] = now_ns
        detector.push(next(frames))
    source.stop()


def test_otlp_stream_matches_offline_build(tmp_path):
    """The live path, over the real collector file format, equals the offline build.

    The ReplaySource tests feed the detector canonical frames; this one goes through
    OTLP-JSON files written a tick at a time, so it also covers OtlpFileSource, the
    byte-offset tail, and the cumulative-metric differencing that only a chunked reader
    can get wrong: gc_pause_ms arrives as a cumulative total, one point per poll, and
    without state carried across polls every point is dropped as "first of its series".
    """
    exp = generate_experiment(
        21, 120.0, FaultSpec(FAULT_TYPE, FAULT_TARGET, 60.0, 30.0, 0.8),
        traffic=schema.TrafficProfile(base_rps=4.0, shape="steady"), warmup_s=WARMUP_S,
    )
    offline_dir = otlp_writer.write_all(exp, tmp_path / "offline")
    parsed, report = ingest.build_experiment(offline_dir, exp.manifest)
    assert report.metric_points_kept > 0
    assert (parsed.metrics["metric"] == schema.METRIC_GC_PAUSE_MS).any(), (
        "the fixture must exercise a differenced cumulative metric")
    offline = build_windows(parsed)

    lines = otlp_writer.experiment_lines(exp)
    capture = tmp_path / "capture"
    capture.mkdir()
    for name in lines:
        (capture / name).touch()
    clock = [exp.manifest.start_ns]
    source = OtlpFileSource(capture, poll_s=0.0, start_ns=exp.manifest.start_ns,
                            warmup_ns=exp.manifest.warmup_ns, clock=lambda: clock[0])
    recorder = RecordingStub().fit(offline)
    detector = StreamingDetector(recorder, source.manifest)
    _drive_capture(source, detector, lines, capture, exp.manifest, clock)

    last = int(offline["window_idx"].max())
    scored = [k for k in sorted(recorder.seen) if k < last]
    assert len(scored) >= 3, f"only scored {sorted(recorder.seen)}"
    columns = [c for c in offline.columns if c.startswith("f_")]
    _assert_covers_every_pass(columns)
    for window_idx in scored:
        actual = recorder.seen[window_idx].set_index("service")
        expected = offline[offline["window_idx"] == window_idx].set_index("service")
        expected = expected.reindex(actual.index)
        for column in columns:
            assert np.allclose(actual[column].to_numpy(dtype="float64"),
                               expected[column].to_numpy(dtype="float64"),
                               rtol=1e-9, atol=1e-9, equal_nan=True), (window_idx, column)
    # The metric that only the carried state keeps alive is actually present.
    assert recorder.seen[scored[0]]["f_metrics_gc_pause_ms_mean"].notna().any()


# --- baseline refresh -------------------------------------------------------------------------
def test_baseline_refresh_is_off_by_default(replayed):
    detector = replayed.detector
    assert detector.refresh_windows == 0
    assert detector.n_baseline_refreshes == 0
    assert detector.state()["baseline_refreshes"] == 0


def test_baseline_refreshes_only_from_healthy_windows(model, exp_dir):
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest, refresh_windows=1)
    records = []
    for frame in source.frames():
        records.extend(detector.push(frame))

    assert detector.n_baseline_refreshes > 0
    # Nothing can be re-fitted before warmup_windows healthy windows have accumulated.
    assert detector.n_baseline_refreshes <= len(records) - detector.warmup_windows + 1

    during = {r["window_idx"] for r in detector.incidents[0].windows}
    healthy = {r["window_idx"] for r in records if not r["detect"]} - during
    pooled = {int(entry[0]["window_idx"].iloc[0]) for entry in detector._baseline_pool}
    assert pooled and pooled <= healthy
    assert len(detector._baseline_pool) == detector.warmup_windows


def test_baseline_refresh_skips_when_too_few_healthy_windows(model, exp_dir):
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest, refresh_windows=1)
    for frame in source.frames():
        if detector.push(frame):
            break
    source.stop()
    # One window scored, so at most one window is pooled: fewer than the three the
    # baseline needs, so the refresh was due but had to be skipped.
    assert len(detector._baseline_pool) < detector.warmup_windows
    assert detector.n_baseline_refreshes == 0


def test_live_source_defaults_to_periodic_refresh(model, exp_dir, tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    engine = Engine(model)
    assert engine._refresh_windows(ReplaySource(exp_dir, speed=0.0)) == 0
    assert engine._refresh_windows(
        OtlpFileSource(capture, poll_s=0.0)) == DEFAULT_REFRESH_WINDOWS
    assert Engine(model, refresh_windows=7)._refresh_windows(
        OtlpFileSource(capture, poll_s=0.0)) == 7


# --- bounded growth ---------------------------------------------------------------------------
def _record(index: int) -> dict:
    return {"window_idx": index, "detect": True, "window_start_ns": index,
            "window_end_ns": index + 1, "ranked_services": ["cart"], "root_scores": [1.0],
            "root_top1": "cart", "root_margin": 1.0, "detect_prob": 1.0,
            "fault_type_pred": "none", "fault_type_probs": {}, "fault_type_conf": 1.0,
            "evidence": {}}


def test_incident_window_history_is_capped():
    incident = Incident("inc-0001", "exp", opened_ns=0)
    for index in range(MAX_INCIDENT_WINDOWS + 100):
        incident.add(_record(index))
    assert len(incident.windows) == MAX_INCIDENT_WINDOWS
    assert incident.windows[-1]["window_idx"] == MAX_INCIDENT_WINDOWS + 99
    # Trimming the history must not corrupt the count.
    assert incident.summary()["n_windows"] == MAX_INCIDENT_WINDOWS + 100


def test_latency_samples_are_capped(replayed):
    detector = replayed.detector
    assert detector.latencies.maxlen == MAX_LATENCY_SAMPLES
    # The scored count is a running total, not the size of the rolling latency sample.
    assert detector.state()["windows_scored"] == len(replayed.records)


def test_store_keeps_open_incidents_and_caps_closed_ones():
    store = IncidentStore(max_closed=2)
    for index in range(5):
        store.save_incident(Incident(f"inc-{index:04d}", "exp", opened_ns=index,
                                     closed_ns=index + 1, windows=[_record(index)]))
    store.save_incident(Incident("inc-open", "exp", opened_ns=0, windows=[_record(0)]))
    kept = {incident.incident_id for incident in store.incidents()}
    assert kept == {"inc-0003", "inc-0004", "inc-open"}


# --- warm-up health gate ------------------------------------------------------------------
MID_FAULT_SECONDS = 70.0        # the replay starts here, well inside the fault


@pytest.fixture(scope="module")
def mid_fault_dir(tmp_path_factory):
    """An experiment whose fault is already running when the replay starts."""
    exp = generate_experiment(
        41, 240.0, FaultSpec("error_rate", "cart", 60.0, 60.0, 0.9),
        traffic=schema.TrafficProfile(base_rps=20.0, shape="steady"), warmup_s=WARMUP_S,
    )
    return schema.write_experiment(exp, tmp_path_factory.mktemp("mid-fault"))


def _health_frames(error_rate: float, requests: int, calls: int = 40):
    """Minimal pass-1 aggregates for one window of one service."""
    raw = pd.DataFrame({
        "window_idx": [0, 0],
        "service": ["cart", "frontend"],
        "f_traces_error_rate": [error_rate, 0.0],
        "f_traces_request_rate": [requests / 10.0, 4.0],
        "f_metrics_cpu_util_mean": [0.3, 0.3],
        "f_metrics_queue_depth_mean": [1.0, 1.0],
    })
    outbound = pd.DataFrame({
        "window_idx": [0] * calls,
        "service": ["frontend"] * calls,
        "peer": ["cart"] * calls,
        "duration_ns": [5_000_000] * calls,
        "status_error": [False] * calls,
    })
    return raw, outbound


def test_warmup_health_accepts_a_quiet_window():
    raw, outbound = _health_frames(error_rate=0.0, requests=40)
    assert warmup_health(raw, outbound, ["cart", "frontend"], 10.0) == []


def test_warmup_health_rejects_a_degraded_window():
    raw, outbound = _health_frames(error_rate=0.61, requests=40)
    reasons = warmup_health(raw, outbound, ["cart", "frontend"], 10.0)
    assert any("cart error_rate=0.61" in reason for reason in reasons)


def test_warmup_health_tolerates_a_healthy_systems_background_errors():
    """The gate must clear the fault-free maximum, or it defers the fit for ever."""
    raw, outbound = _health_frames(error_rate=0.19, requests=200)
    assert warmup_health(raw, outbound, ["cart", "frontend"], 10.0) == []


def test_warmup_health_ignores_rates_with_no_denominator():
    """Three failures out of four requests is 75 %, and means nothing."""
    raw, outbound = _health_frames(error_rate=0.75, requests=4, calls=4)
    assert warmup_health(raw, outbound, ["cart", "frontend"], 10.0) == []


def test_warmup_health_flags_a_service_that_stopped_answering():
    raw, outbound = _health_frames(error_rate=0.0, requests=0, calls=40)
    reasons = warmup_health(raw, outbound, ["cart", "frontend"], 10.0)
    assert any("cart unanswered_frac=1" in reason for reason in reasons)


def test_warmup_health_flags_hung_outbound_calls():
    raw, outbound = _health_frames(error_rate=0.0, requests=40, calls=40)
    outbound.loc[:9, "status_error"] = True
    outbound.loc[:9, "duration_ns"] = 5_000_000_000
    reasons = warmup_health(raw, outbound, ["cart", "frontend"], 10.0)
    assert any("frontend timeout_frac" in reason for reason in reasons)


def test_saturation_is_advisory_not_blocking():
    """Busy is not broken: a hot service must not stop the baseline from ever fitting."""
    raw, outbound = _health_frames(error_rate=0.0, requests=40)
    raw.loc[0, "f_metrics_cpu_util_mean"] = 0.97
    raw.loc[1, "f_metrics_queue_depth_mean"] = 900.0
    assert warmup_health(raw, outbound, ["cart", "frontend"], 10.0) == []
    advisory = warmup_health(raw, outbound, ["cart", "frontend"], 10.0,
                             include_advisory=True)
    assert any("cart cpu_util" in reason for reason in advisory)
    assert any("frontend queue_depth" in reason for reason in advisory)


def test_fit_is_deferred_until_the_incident_clears(model, mid_fault_dir):
    """A detector started mid-incident must not learn the incident as normal."""
    manifest = schema.read_manifest(mid_fault_dir)
    fault = manifest.faults[0]
    source = ReplaySource(mid_fault_dir, speed=0.0,
                          start_offset_ns=int(MID_FAULT_SECONDS * 1e9))
    assert source.manifest.start_ns == manifest.start_ns + int(MID_FAULT_SECONDS * 1e9)
    assert source.manifest.start_ns > fault.start_ns, "the replay must start mid-fault"

    detector = StreamingDetector(model, source.manifest)
    states, fitted_at = [], None
    for frame in source.frames():
        detector.push(frame)
        states.append(detector.baseline_state)
        if fitted_at is None and detector.baseline is not None:
            fitted_at = detector.now_ns
    source.stop()

    assert "untrusted" in states, "the degraded warm-up was accepted"
    assert fitted_at is not None, "the baseline was never fitted"
    assert fitted_at > fault.end_ns, "fitted while the fault was still running"
    # Every window it finally fitted on starts after the fault ended.
    fitted_windows = [int(entry[0]["window_idx"].iloc[0]) for entry, _ in detector._warmup_pool]
    starts = [detector.start_ns + index * detector.window_ns for index in fitted_windows]
    assert min(starts) >= fault.end_ns, (
        f"fitted on windows starting at {min(starts)}, fault ended at {fault.end_ns}")
    assert len(fitted_windows) == detector.warmup_windows


def test_healthy_warmup_still_fits_immediately(replayed):
    """The guard must not delay a normal start: same first window as before."""
    detector = replayed.detector
    assert detector.baseline_state == "ready"
    assert detector._warmup_cursor == detector.warmup_windows
    assert replayed.records[0]["window_idx"] == detector.warmup_windows


def test_stuck_positive_refits_on_healthy_telemetry(exp_dir, offline):
    """A permanently positive detector on clean telemetry re-fits its own baseline."""
    class AlwaysPositive(StubModel):
        name = "always-positive"

        def fit(self, windows, val=None):
            StubModel.fit(self, windows)
            self.detect_calibrator_ = _Squash(-1e9)     # everything crosses
            return self

    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(AlwaysPositive().fit(offline), source.manifest,
                                 refresh_windows=1000)
    assert detector.stuck_positive_limit == max(6, 3 * detector.warmup_windows)
    records = []
    for frame in source.frames():
        records.extend(detector.push(frame))
    source.stop()

    assert all(r["detect"] for r in records)
    assert len(records) <= detector.stuck_positive_limit, (
        "this fixture is too short to exceed the limit, so nothing was proven")
    assert detector.n_stuck_refits == 0


def test_stuck_positive_escape_hatch_fires(tmp_path):
    """Given enough consecutive positives on healthy telemetry, the baseline is replaced."""
    class AlwaysPositive(StubModel):
        name = "always-positive"

        def fit(self, windows, val=None):
            StubModel.fit(self, windows)
            self.detect_calibrator_ = _Squash(-1e9)
            return self

    exp = generate_experiment(
        51, 330.0, FaultSpec(FAULT_TYPE, FAULT_TARGET, 60.0, 20.0, 0.8),
        traffic=schema.TrafficProfile(base_rps=10.0, shape="steady"), warmup_s=WARMUP_S,
    )
    exp_dir = schema.write_experiment(exp, tmp_path)
    model = AlwaysPositive().fit(build_windows(exp))
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest, refresh_windows=1000)
    records = []
    for frame in source.frames():
        records.extend(detector.push(frame))
    source.stop()

    assert len(records) > detector.stuck_positive_limit
    assert detector.n_stuck_refits >= 1
    assert detector.state()["stuck_refits"] == detector.n_stuck_refits
    # The streak resets on the re-fit, so it does not re-fit on every window after.
    assert detector.n_stuck_refits <= len(records) // detector.stuck_positive_limit + 1
    assert detector.n_baseline_refreshes == 0, "the periodic refresh must not have fired"


# --- S3: pooled aggregates are one window each ------------------------------------------------
def test_pooled_aggregates_hold_exactly_one_window(model, exp_dir):
    """Pooling the whole buffer would duplicate rows and inflate the fitted statistics."""
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest, refresh_windows=1)
    records = []
    for frame in source.frames():
        records.extend(detector.push(frame))
    source.stop()

    for raw, peer_raw, outbound in detector._baseline_pool:
        assert raw["window_idx"].nunique() == 1
        assert peer_raw.empty or peer_raw["window_idx"].nunique() == 1
        assert outbound.empty or outbound["window_idx"].nunique() == 1
        assert set(raw["window_idx"]) == set(peer_raw["window_idx"]) or peer_raw.empty
        # One row per service, not one per service per buffered window.
        assert len(raw) == len(detector.services)

    incident_windows = {r["window_idx"] for r in detector.incidents[0].windows}
    pooled = {int(raw["window_idx"].iloc[0]) for raw, _, _ in detector._baseline_pool}
    assert not (pooled & incident_windows), "an incident's windows leaked into the pool"


def test_pooled_outbound_carries_what_a_fit_and_a_health_check_read(model, exp_dir):
    source = ReplaySource(exp_dir, speed=0.0)
    detector = StreamingDetector(model, source.manifest, refresh_windows=1)
    detector.run(source)
    _, _, outbound = next(iter(detector._baseline_pool))
    assert list(outbound.columns) == ["window_idx", "service", "peer", "duration_ns",
                                      "status_error"]


# --- S10: the tail must not re-deliver ------------------------------------------------------
def test_tail_does_not_redeliver_lines_written_during_the_read(tmp_path, monkeypatch):
    """A writer that appends between the stat and the read must not be read twice."""
    capture = tmp_path / "capture"
    capture.mkdir()
    for name in ("traces.jsonl", "metrics.jsonl", "logs.jsonl"):
        (capture / name).touch()
    path = capture / "logs.jsonl"
    line = json.dumps({"resourceLogs": [{
        "resource": {"attributes": [
            {"key": "service.name", "value": {"stringValue": "cart"}}]},
        "scopeLogs": [{"logRecords": [{"timeUnixNano": "1700000000000000000",
                                       "severityText": "ERROR",
                                       "body": {"stringValue": "boom"}}]}]}]})
    _append(path, line + "\n")

    source = OtlpFileSource(capture, poll_s=0.0, start_ns=1_700_000_000_000_000_000,
                            clock=lambda: 1_700_000_100_000_000_000)
    tail = dict(source._tails)["logs"]
    tail.offset = 0

    original_stat = Path.stat
    sneaked = []

    def stat_then_append(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)
        if self == path and not sneaked:
            sneaked.append(True)
            _append(path, line + "\n")      # a second line lands after the stat
        return result

    monkeypatch.setattr(Path, "stat", stat_then_append)
    first = tail.read(1_700_000_000_000_000_000, 1_700_000_100_000_000_000)
    monkeypatch.undo()
    second = tail.read(1_700_000_000_000_000_000, 1_700_000_100_000_000_000)

    assert sneaked, "the test did not exercise the race"
    assert len(first) == 2, "both lines present on disk should be read once"
    assert len(second) == 0, "the second poll re-delivered lines"


# --- S6: a bad frame must not end the run ------------------------------------------------------
class _AngryDetector:
    """Stands in for the detector: fails the frames whose index is in ``fail``."""

    def __init__(self, fail: set[int]):
        self.fail = fail
        self.seen = 0

    def push(self, frame):
        index = self.seen
        self.seen += 1
        if index in self.fail:
            raise ValueError(f"frame {index} is malformed")
        return []

    def state(self):
        return {"ready": True, "status": "ready"}


def test_one_bad_frame_does_not_end_detection(model, exp_dir):
    engine = Engine(model)
    engine.detector = _AngryDetector({1, 3})
    engine.mode, engine.source = "replay", ReplaySource(exp_dir, speed=0.0)
    frames = list(engine.source.frames())[:8]
    for frame in frames:
        assert engine._push(frame) is True
    assert engine.frame_errors == 2
    assert engine._consecutive_errors == 0
    assert engine.detector.seen == len(frames)
    assert engine.error is None
    assert engine.alive


def test_a_run_of_bad_frames_marks_the_engine_errored(model, exp_dir):
    engine = Engine(model)
    engine.detector = _AngryDetector(set(range(20)))
    engine.mode, engine.source = "replay", ReplaySource(exp_dir, speed=0.0)
    frames = list(engine.source.frames())[:10]
    outcomes = [engine._push(frame) for frame in frames]
    assert outcomes[:MAX_CONSECUTIVE_FRAME_ERRORS - 1] == [True] * (
        MAX_CONSECUTIVE_FRAME_ERRORS - 1)
    assert outcomes[MAX_CONSECUTIVE_FRAME_ERRORS - 1] is False
    assert engine.error and "consecutive frame failures" in engine.error
    assert not engine.alive


def test_health_and_ready_probes(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["frame_errors"] == 0
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["baseline_state"] == "ready"


def test_health_fails_when_the_pump_gave_up(model):
    app = create_app(model, links=Links())
    with TestClient(app) as test_client:
        # Idle is not unhealthy, but it is not ready either.
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/ready").status_code == 503
        app.state.engine.error = "5 consecutive frame failures; last was ValueError: boom"
        assert test_client.get("/health").status_code == 503
        assert test_client.get("/health").json()["status"] == "unhealthy"
        assert test_client.get("/ready").status_code == 503


# --- S13: evidence and incident lists stay bounded ------------------------------------------------
def test_evidence_is_kept_only_for_the_newest_windows():
    incident = Incident("inc-0001", "exp", opened_ns=0)
    for index in range(MAX_EVIDENCE_WINDOWS + 4):
        record = _record(index)
        record["evidence"] = {"cart": {"service": "cart", "spans": {"slow": [{}]}}}
        incident.add(record)
    with_evidence = [w["window_idx"] for w in incident.windows if w["evidence"]]
    assert with_evidence == list(range(4, MAX_EVIDENCE_WINDOWS + 4))
    # The verdict on show keeps its evidence however long the incident runs.
    assert incident.evidence and incident.full()["evidence"]


def test_stripping_evidence_does_not_mutate_the_caller_s_record():
    incident = Incident("inc-0001", "exp", opened_ns=0)
    first = _record(0)
    first["evidence"] = {"cart": {"service": "cart"}}
    incident.add(first)
    for index in range(1, MAX_EVIDENCE_WINDOWS + 2):
        record = _record(index)
        record["evidence"] = {"cart": {"service": "cart"}}
        incident.add(record)
    assert first["evidence"], "the record handed to push was mutated"
    assert not incident.windows[0]["evidence"]


def test_detector_incident_list_is_capped(replayed):
    detector = replayed.detector
    assert detector.incidents.maxlen == MAX_INCIDENTS
    assert detector.state()["n_incidents"] == detector.n_incidents_total


