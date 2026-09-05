"""Incident store: in-memory state plus optional JSONL persistence.

The API reads everything from here, so a run can be inspected live and replayed
afterwards. With an ``out_dir`` the store also appends two files:

    <out_dir>/windows.jsonl     one line per scored window (the prediction, no evidence)
    <out_dir>/incidents.jsonl   one line per incident state change (open / update / close)

Both are append-only event logs, so reading the first ``k`` lines answers "what did the
system say at time t" without any extra bookkeeping. The evidence lives on the incident
records, where it is worth its size.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from rca.serve.stream import Incident

WINDOWS_FILE = "windows.jsonl"
INCIDENTS_FILE = "incidents.jsonl"
MAX_CLOSED_INCIDENTS = 500      # open ones are never dropped


class IncidentStore:
    """Incidents and per-window predictions of one detector run."""

    def __init__(self, out_dir: Path | str | None = None, max_windows: int = 5000,
                 max_closed: int = MAX_CLOSED_INCIDENTS):
        self.out_dir = Path(out_dir) if out_dir is not None else None
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_closed = int(max_closed)
        self._windows: deque[dict] = deque(maxlen=max_windows)
        self._incidents: dict[str, Incident] = {}

    # --- writes -----------------------------------------------------------------------
    def add_window(self, record: dict) -> None:
        self._windows.append(record)
        self._append(WINDOWS_FILE, record)

    def save_incident(self, incident: Incident) -> None:
        self._incidents[incident.incident_id] = incident
        self._append(INCIDENTS_FILE, incident.summary())
        self._evict()

    def _evict(self) -> None:
        """Keep the newest ``max_closed`` closed incidents; the JSONL keeps them all."""
        closed = [i for i in self._incidents.values() if not i.is_open]
        for incident in sorted(closed, key=lambda i: i.opened_ns)[:-self.max_closed or None]:
            del self._incidents[incident.incident_id]

    # --- reads ------------------------------------------------------------------------
    def recent_windows(self, n: int = 100) -> list[dict]:
        windows = list(self._windows)
        return windows[-n:] if n > 0 else windows

    def incidents(self) -> list[Incident]:
        """Every incident of this run, newest first."""
        return sorted(self._incidents.values(), key=lambda i: i.opened_ns, reverse=True)

    def incident(self, incident_id: str) -> Incident | None:
        return self._incidents.get(incident_id)

    # --- persistence --------------------------------------------------------------------
    def _append(self, name: str, payload: dict) -> None:
        if self.out_dir is None:
            return
        with open(self.out_dir / name, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, default=_plain) + "\n")


def _plain(value):
    """Last-resort JSON encoder for numpy scalars that slipped through."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def read_jsonl(path: Path | str) -> list[dict]:
    """Read back a persisted event log."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
