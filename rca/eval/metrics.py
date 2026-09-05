"""Evaluation metrics for the three-stage system (``docs/MODELS.md`` -- Evaluation).

Everything is computed from the window table plus a model's ``predict`` frame, so the
same code evaluates any model. Incident-level quantities (detection delay, false alarms
per hour) need the experiment manifests to tell normal runs from traffic-only spikes.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from rca.models.aggregate import KEYS
from rca.models.base import sustained
from rca.models.calibration import expected_calibration_error, reliability_bins

NS_PER_S = 1e9
NS_PER_HOUR = 3.6e12

# Telemetry stays genuinely anomalous for a few seconds after a fault is lifted (queues
# drain, caches refill, restarted processes warm up), so the post-fault bucket is also
# reported with this much of the recovery ramp skipped.
RECOVERY_S = 30.0

# A fault window that starts this long after the fault began has had time to show
# symptoms; recall over those windows separates "slow to react" from "cannot see it".
SYMPTOMATIC_LAG_S = 10.0

# Requests reaching the root service while the fault ran. Below this an incident is
# barely exercised, and a miss says more about the traffic than about the model.
EXPOSURE_SPLIT = 10.0

# Metrics the cluster bootstrap puts a confidence interval on.
BOOTSTRAP_METRICS = (
    "detect_F1", "auroc", "top1_oracle", "top1_incident", "fa_all", "fault_macroF1",
)

TRUTH_COLUMNS = [
    "window_start_ns", "window_end_ns", "is_fault_window", "root_service", "fault_type",
    "since_fault_start_ns",
]


def window_truth(windows: pd.DataFrame) -> pd.DataFrame:
    """One ground-truth row per (experiment_id, window_idx)."""
    grouped = windows.groupby(KEYS, sort=True)
    truth = grouped[TRUTH_COLUMNS].first().reset_index()
    truth["root_service"] = truth["root_service"].astype(str)
    truth["fault_type"] = truth["fault_type"].astype(str)
    return truth


def joined(windows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Ground truth joined to predictions, one row per window."""
    dropped = [c for c in ("window_start_ns", "window_end_ns") if c in predictions.columns]
    return window_truth(windows).merge(predictions.drop(columns=dropped), on=KEYS, how="left")


def experiment_fault_ends(manifests: Iterable) -> dict[str, int]:
    """End of the last fault of each fault experiment, in absolute ns."""
    return {
        manifest.experiment_id: max(fault.end_ns for fault in manifest.faults)
        for manifest in manifests
        if manifest.faults
    }


def apply_debounce(frame: pd.DataFrame, windows_required: int) -> pd.DataFrame:
    """Require ``windows_required`` consecutive positive windows before flagging.

    A single noisy window stops being an alarm, at the price of one window of delay per
    step of ``K`` -- which is exactly the trade-off the operator is choosing. Only the
    ``detect`` flag moves; ``detect_prob`` is untouched, so AUROC and the stage-1
    reliability curve stay threshold-free.
    """
    if windows_required <= 1:
        return frame
    out = frame.sort_values(["experiment_id", "window_idx"]).copy()
    out["detect"] = sustained(
        out["experiment_id"].to_numpy(), out["detect"].astype(bool), windows_required
    )
    return out.loc[frame.index]


def experiment_categories(manifests: Iterable) -> dict[str, str]:
    """``fault`` | ``normal`` | ``traffic_spike`` per experiment.

    The simulator records this in ``manifest.extra["category"]``; without it, a run with
    no faults counts as a traffic-only spike when its traffic profile is a burst.
    """
    categories = {}
    for manifest in manifests:
        recorded = manifest.extra.get("category") if manifest.extra else None
        if recorded:
            categories[manifest.experiment_id] = str(recorded)
        elif manifest.faults:
            categories[manifest.experiment_id] = "fault"
        elif manifest.traffic.shape in ("bursty", "spike"):
            categories[manifest.experiment_id] = "traffic_spike"
        else:
            categories[manifest.experiment_id] = "normal"
    return categories


# --- stage 1 --------------------------------------------------------------------------
def detection_metrics(frame: pd.DataFrame) -> dict:
    """Per-window precision / recall / F1 / AUROC of the detector."""
    y = frame["is_fault_window"].to_numpy(bool)
    predicted = frame["detect"].to_numpy(bool)
    probability = frame["detect_prob"].to_numpy(float)
    true_positive = int((y & predicted).sum())
    precision = true_positive / int(predicted.sum()) if predicted.any() else float("nan")
    recall = true_positive / int(y.sum()) if y.any() else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if true_positive and precision + recall > 0 else 0.0)
    return {
        "n_windows": len(frame),
        "n_fault_windows": int(y.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((y == predicted).mean()),
        "auroc": float(roc_auc_score(y, probability)) if 0 < y.sum() < len(y) else float("nan"),
        "alarm_rate": float(predicted.mean()),
        "recall_symptomatic": symptomatic_recall(frame),
    }


def symptomatic_recall(frame: pd.DataFrame, lag_s: float = SYMPTOMATIC_LAG_S) -> float:
    """Recall over fault windows that began at least ``lag_s`` after the fault started.

    The onset window is half telemetry from before the fault; scoring it as a miss
    conflates reaction time with blindness, which is what the headline recall does.
    """
    mask = (frame["is_fault_window"].to_numpy(bool)
            & (frame["since_fault_start_ns"].to_numpy() >= lag_s * NS_PER_S))
    if not mask.any():
        return float("nan")
    return float(frame["detect"].to_numpy(bool)[mask].mean())


def detection_delay(frame: pd.DataFrame) -> dict:
    """Time to alarm per incident, in seconds from the fault start.

    The headline number is measured to the **end** of the first flagged window, which is
    the earliest moment the alarm could actually have fired: a windowed detector cannot
    emit a verdict for a window whose telemetry it has not finished collecting. Measuring
    to the window start, as this used to, credits the detector with up to a full window
    of clairvoyance. That older number is kept as ``onset_*`` for reference.

    Incidents never flagged while the fault is running are reported as undetected and
    excluded from the median / p90.
    """
    delays, onsets, undetected = [], [], 0
    for _, rows in frame[frame["is_fault_window"]].groupby("experiment_id", sort=True):
        rows = rows.sort_values("window_idx")
        fault_start = (int(rows["window_start_ns"].iloc[0])
                       - int(rows["since_fault_start_ns"].iloc[0]))
        positive = rows[rows["detect"].to_numpy(bool)]
        if positive.empty:
            undetected += 1
            continue
        delays.append(max(0.0, (int(positive["window_end_ns"].iloc[0]) - fault_start)
                          / NS_PER_S))
        onsets.append(max(0.0, (int(positive["window_start_ns"].iloc[0]) - fault_start)
                          / NS_PER_S))
    total = len(delays) + undetected
    return {
        "n_incidents": int(total),
        "n_detected": len(delays),
        "undetected_fraction": float(undetected / total) if total else float("nan"),
        "median_s": _quantile(delays, 0.5),
        "p90_s": _quantile(delays, 0.9),
        "delays_s": [round(value, 3) for value in sorted(delays)],
        "onset_median_s": _quantile(onsets, 0.5),
        "onset_p90_s": _quantile(onsets, 0.9),
        "onset_delays_s": [round(value, 3) for value in sorted(onsets)],
    }


def false_alarm_rates(
    frame: pd.DataFrame,
    categories: dict[str, str],
    fault_ends: dict[str, int] | None = None,
    recovery_s: float = RECOVERY_S,
) -> dict:
    """Debounced alarms per hour over fault-free time, split by the kind of quiet time.

    ``all_fault_free`` is every window in which no fault is active -- normal runs,
    traffic-only spikes, and both the pre-fault warm-up *and the post-fault recovery* of
    fault experiments. The post-fault tail is where a detector that latches misfires
    most, so leaving it out (as this function used to) flatters the headline number;
    ``post_fault`` reports it on its own and ``post_fault_excl_{recovery}s`` reports it
    again with the recovery ramp skipped, since telemetry right after a fault is lifted
    is legitimately still abnormal.

    An alarm is a negative-to-positive transition in the experiment's *full* window
    series, not within the bucket: a detector still firing as the fault ends and simply
    failing to clear is one alarm that has not reset, not a fresh false alarm at the
    bucket boundary.
    """
    category = frame["experiment_id"].map(categories)
    is_fault_experiment = (category == "fault").to_numpy(dtype=bool)
    fault = frame["is_fault_window"].to_numpy(bool)

    start = frame["experiment_id"].map(_fault_starts(frame))
    end = frame["experiment_id"].map(_fault_ends(frame, fault_ends))
    before = (frame["window_end_ns"] <= start).fillna(False).to_numpy(bool)
    after = (frame["window_start_ns"] >= end).fillna(False).to_numpy(bool) & ~fault
    recovered = (
        (frame["window_start_ns"] >= end + recovery_s * NS_PER_S).fillna(False).to_numpy(bool)
        & ~fault
    )

    buckets = {
        "normal": (category == "normal").to_numpy(dtype=bool),
        "traffic_spike": (category == "traffic_spike").to_numpy(dtype=bool),
        "pre_fault": is_fault_experiment & before,
        "post_fault": is_fault_experiment & after,
        f"post_fault_excl_{int(recovery_s)}s": is_fault_experiment & recovered,
        # Every window with no fault active, so nothing quiet can be left out by
        # construction (an experiment with a gap between two faults included).
        "all_fault_free": ~fault,
    }
    onsets = _alarm_onsets(frame)
    return {name: _alarm_rate(frame, mask, onsets) for name, mask in buckets.items()}


# --- stage 2 --------------------------------------------------------------------------
def localization_metrics(frame: pd.DataFrame) -> dict:
    """Top-1 / top-3 over fault windows, conditioned on detection and with an oracle."""
    fault = frame[frame["is_fault_window"]]
    detected = fault[fault["detect"].to_numpy(bool)]
    return {
        "oracle": _ranking_accuracy(fault),
        "detected": _ranking_accuracy(detected),
        "mean_margin": float(fault["root_margin"].mean()) if len(fault) else float("nan"),
        **incident_localization(frame),
        "per_fault_type": {
            str(name): _ranking_accuracy(rows)
            for name, rows in fault.groupby("fault_type", sort=True)
        },
    }


def incident_localization(frame: pd.DataFrame) -> dict:
    """Per-incident localization: does the *incident* get the right root, not the window.

    Window-level top-1 lets a long incident that is right in most of its windows hide a
    wrong first verdict, which is the one an on-call engineer actually acts on. An
    incident that was never detected counts as wrong -- you cannot localize what you did
    not alarm on.
    """
    first, majority, total = 0, 0, 0
    for _, rows in frame[frame["is_fault_window"]].groupby("experiment_id", sort=True):
        total += 1
        flagged = rows.sort_values("window_idx")
        flagged = flagged[flagged["detect"].to_numpy(bool)]
        if flagged.empty:
            continue
        truth = str(flagged["root_service"].iloc[0])
        if str(flagged["root_top1"].iloc[0]) == truth:
            first += 1
        votes = flagged["root_top1"].astype(str).value_counts()
        # Ties go to the alphabetically first service, matching predict()'s tie-break.
        if min(votes[votes == votes.max()].index) == truth:
            majority += 1
    return {
        "n_incidents": total,
        "top1_incident": float(first / total) if total else float("nan"),
        "top1_incident_majority": float(majority / total) if total else float("nan"),
    }


# --- stage 3 --------------------------------------------------------------------------
def classification_metrics(model, windows: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Fault-family P/R/F1 on the predicted root row and on the true root row."""
    fault = frame[frame["is_fault_window"]]
    on_predicted = _class_report(
        fault["fault_type"].astype(str), fault["fault_type_pred"].astype(str)
    )
    root_rows = windows[windows["is_root"]]
    if root_rows.empty:
        return {"predicted_root": on_predicted, "true_root": _class_report([], [])}
    probabilities = model.classify(root_rows)
    predicted = probabilities.columns.to_numpy()[probabilities.to_numpy().argmax(axis=1)]
    return {
        "predicted_root": on_predicted,
        "true_root": _class_report(root_rows["fault_type"].astype(str), predicted),
    }


# --- calibration ----------------------------------------------------------------------
def calibration_metrics(frame: pd.DataFrame, n_bins: int = 10) -> dict:
    """ECE and reliability bins for each stage."""
    fault = frame[frame["is_fault_window"]]
    # Stage 2 is calibrated on the number the operator is shown: the top-1 service's
    # score, over every window that raised an alarm. False alarms are included and count
    # as wrong, because a confident root guess on a window with no fault is exactly the
    # failure a calibration number should expose.
    flagged = frame[frame["detect"].astype(bool)]
    top1_score = [row[0] if len(row) else float("nan") for row in flagged["root_scores"]]
    top1_correct = (flagged["root_top1"].astype(str)
                    == flagged["root_service"].astype(str)).to_numpy(bool)
    stage3_correct = (fault["fault_type_pred"].astype(str)
                      == fault["fault_type"].astype(str)).to_numpy(bool)
    return {
        "detect": _calibration(frame["detect_prob"], frame["is_fault_window"], n_bins),
        "top1": _calibration(top1_score, top1_correct, n_bins),
        "fault_type": _calibration(fault["fault_type_conf"], stage3_correct, n_bins),
    }


# --- exposure ---------------------------------------------------------------------------
def target_exposure(windows: pd.DataFrame) -> pd.Series:
    """Requests that reached the root service while its fault ran, per experiment.

    A fault on a service nobody called during the incident leaves almost no trace, so a
    miss there is a property of the traffic, not of the detector. Stratifying on it keeps
    that from being silently averaged into the headline numbers.
    """
    root = windows[windows["is_root"]]
    if root.empty or "f_traces_request_rate" not in root.columns:
        return pd.Series(dtype="float64")
    seconds = (root["window_end_ns"] - root["window_start_ns"]).to_numpy() / NS_PER_S
    requests = root["f_traces_request_rate"].fillna(0.0).to_numpy() * seconds
    return pd.Series(requests, index=root["experiment_id"].to_numpy()).groupby(level=0).sum()


def exposure_metrics(
    frame: pd.DataFrame, windows: pd.DataFrame, split: float = EXPOSURE_SPLIT
) -> dict:
    """Detection recall, delay and localization for barely-exercised vs busy incidents."""
    exposure = target_exposure(windows)
    if exposure.empty:
        return {}
    strata = {f"exposure_lt_{int(split)}": exposure[exposure < split].index,
              f"exposure_ge_{int(split)}": exposure[exposure >= split].index}
    out = {}
    for name, ids in strata.items():
        rows = frame[frame["experiment_id"].isin(set(ids))]
        fault = rows[rows["is_fault_window"]]
        out[name] = {
            "n_experiments": len(set(ids)),
            "median_exposure": float(exposure[list(ids)].median()) if len(ids) else
            float("nan"),
            "recall": float(fault["detect"].astype(bool).mean()) if len(fault) else
            float("nan"),
            "recall_symptomatic": symptomatic_recall(rows),
            "delay_median_s": detection_delay(rows)["median_s"],
            "undetected": detection_delay(rows)["undetected_fraction"],
            "top1_oracle": _ranking_accuracy(fault)["top1"],
            "top1_incident": incident_localization(rows)["top1_incident"],
            "per_fault_type": {
                str(family): _ranking_accuracy(group)["top1"]
                for family, group in fault.groupby("fault_type", sort=True)
            },
        }
    return out


# --- bootstrap ----------------------------------------------------------------------------
def bootstrap_intervals(
    frame: pd.DataFrame, b: int = 500, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Percentile CIs from a cluster bootstrap whose resampling unit is the experiment.

    Windows inside one incident are anything but independent, so resampling windows would
    produce intervals several times too narrow. Every statistic here is either additive
    over experiments (counts) or recomputed from the resampled rows (AUROC), so a
    replicate is exact rather than approximated.
    """
    ids = sorted(frame["experiment_id"].unique())
    if len(ids) < 2 or b <= 0:
        return {}
    stats = _experiment_stats(frame, ids)
    rng = np.random.default_rng(seed)
    n = len(ids)
    draws: dict[str, list[float]] = {name: [] for name in BOOTSTRAP_METRICS}
    for _ in range(b):
        pick = rng.integers(0, n, n)
        for name, value in _replicate(stats, pick).items():
            draws[name].append(value)
    out = {}
    for name, values in draws.items():
        clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if clean.size == 0:
            continue
        out[name] = {
            "lo": float(np.quantile(clean, alpha / 2)),
            "hi": float(np.quantile(clean, 1 - alpha / 2)),
            "b": int(clean.size),
        }
    return out


def _experiment_stats(frame: pd.DataFrame, ids: list[str]) -> dict:
    """Per-experiment quantities the bootstrap replicates are assembled from."""
    onsets = _alarm_onsets(frame)
    classes = sorted(set(frame.loc[frame["is_fault_window"], "fault_type"].astype(str))
                     | set(frame.loc[frame["is_fault_window"], "fault_type_pred"].astype(str)))
    order = {name: position for position, name in enumerate(classes)}
    counts = np.zeros((len(ids), 3))              # detection tp, fp, fn
    oracle = np.zeros((len(ids), 2))              # top-1 hits, fault windows
    incident = np.zeros((len(ids), 2))            # incidents, first-window hits
    alarms = np.zeros((len(ids), 2))              # fault-free onsets, hours
    families = np.zeros((len(ids), len(classes), 3))
    probability, labels = [], []

    position = {name: index for index, name in enumerate(ids)}
    for name, rows in frame.groupby("experiment_id", sort=False):
        index = position[name]
        mask = frame["experiment_id"].to_numpy() == name
        y = rows["is_fault_window"].to_numpy(bool)
        predicted = rows["detect"].to_numpy(bool)
        counts[index] = [(y & predicted).sum(), (~y & predicted).sum(), (y & ~predicted).sum()]
        probability.append(rows["detect_prob"].to_numpy(float))
        labels.append(y)

        fault = rows[y]
        hit = (fault["root_top1"].astype(str) == fault["root_service"].astype(str)).to_numpy()
        oracle[index] = [hit.sum(), len(fault)]
        incident[index] = [
            float(len(fault) > 0),
            float(incident_localization(rows)["top1_incident"] or 0.0) if len(fault) else 0.0,
        ]
        quiet = rows[~y]
        alarms[index] = [
            onsets[mask][~y].sum(),
            (quiet["window_end_ns"] - quiet["window_start_ns"]).sum() / NS_PER_HOUR,
        ]
        for truth, guess in zip(fault["fault_type"].astype(str),
                                fault["fault_type_pred"].astype(str)):
            if truth == guess:
                families[index, order[truth], 0] += 1
            else:
                families[index, order[guess], 1] += 1
                families[index, order[truth], 2] += 1
    return {"counts": counts, "oracle": oracle, "incident": incident, "alarms": alarms,
            "families": families, "probability": probability, "labels": labels}


def _replicate(stats: dict, pick: np.ndarray) -> dict:
    counts = stats["counts"][pick].sum(axis=0)
    oracle = stats["oracle"][pick].sum(axis=0)
    incident = stats["incident"][pick].sum(axis=0)
    alarms = stats["alarms"][pick].sum(axis=0)
    families = stats["families"][pick].sum(axis=0)

    true_positive, false_positive, false_negative = counts
    denominator = 2 * true_positive + false_positive + false_negative
    y = np.concatenate([stats["labels"][i] for i in pick])
    probability = np.concatenate([stats["probability"][i] for i in pick])
    per_class = [
        2 * tp / (2 * tp + fp + fn)
        for tp, fp, fn in families if tp + fp + fn > 0
    ]
    return {
        "detect_F1": float(2 * true_positive / denominator) if denominator else float("nan"),
        "auroc": (float(roc_auc_score(y, probability))
                  if 0 < y.sum() < len(y) else float("nan")),
        "top1_oracle": float(oracle[0] / oracle[1]) if oracle[1] else float("nan"),
        "top1_incident": float(incident[1] / incident[0]) if incident[0] else float("nan"),
        "fa_all": float(alarms[0] / alarms[1]) if alarms[1] > 0 else float("nan"),
        "fault_macroF1": float(np.mean(per_class)) if per_class else float("nan"),
    }


# --- cost ------------------------------------------------------------------------------
def inference_profile(model, windows: pd.DataFrame, repeats: int = 1) -> dict:
    """Wall-clock of ``predict`` per window and the process's resident memory."""
    n_windows = int(windows.groupby(KEYS, sort=False).ngroups)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(windows)
    elapsed = (time.perf_counter() - start) / max(repeats, 1)
    return {
        "n_windows": n_windows,
        "predict_s": float(elapsed),
        "ms_per_window": float(elapsed * 1000 / n_windows) if n_windows else float("nan"),
        "rss_mb": resident_memory_mb(),
    }


def resident_memory_mb() -> float:
    """Process RSS in megabytes, without adding a dependency for it."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        # Without an explicit HANDLE restype the pseudo-handle is truncated to 32 bits.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return float("nan")
        return counters.WorkingSetSize / 1e6
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


# --- everything -------------------------------------------------------------------------
def evaluate(
    model,
    windows: pd.DataFrame,
    manifests: Iterable | None = None,
    predictions: pd.DataFrame | None = None,
    profile: bool = True,
    debounce: int = 1,
    bootstrap: int = 0,
) -> dict:
    """Full metric set for one model on one window table."""
    if predictions is None:
        predictions = model.predict(windows)
    frame = apply_debounce(joined(windows, predictions), debounce)
    manifests = list(manifests) if manifests is not None else []
    categories = experiment_categories(manifests)
    fault_ends = experiment_fault_ends(manifests)
    result = {
        "n_experiments": int(frame["experiment_id"].nunique()),
        "debounce": int(debounce),
        "detection": detection_metrics(frame),
        "detection_delay": detection_delay(frame),
        "false_alarms": false_alarm_rates(frame, categories, fault_ends),
        "localization": localization_metrics(frame),
        "classification": classification_metrics(model, windows, frame),
        "calibration": calibration_metrics(frame),
        "exposure": exposure_metrics(frame, windows),
    }
    if bootstrap:
        result["bootstrap"] = bootstrap_intervals(frame, bootstrap)
    if profile:
        result["cost"] = inference_profile(model, windows)
    return result


# --- helpers ---------------------------------------------------------------------------
def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def _fault_starts(frame: pd.DataFrame) -> pd.Series:
    fault = frame[frame["is_fault_window"]]
    if fault.empty:
        return pd.Series(dtype="int64")
    first = fault.sort_values("window_idx").groupby("experiment_id").first()
    return first["window_start_ns"] - first["since_fault_start_ns"]


def _fault_ends(frame: pd.DataFrame, fault_ends: dict[str, int] | None) -> pd.Series:
    """Fault end per experiment: exact from the manifests, else the last fault window."""
    fault = frame[frame["is_fault_window"]]
    derived = (fault.groupby("experiment_id")["window_end_ns"].max()
               if not fault.empty else pd.Series(dtype="int64"))
    if not fault_ends:
        return derived
    return pd.Series(fault_ends, dtype="int64").combine_first(derived)


def _alarm_onsets(frame: pd.DataFrame) -> np.ndarray:
    """Negative-to-positive transitions within each experiment's full window series."""
    ordered = frame.sort_values(["experiment_id", "window_idx"])
    flags = ordered["detect"].astype(bool)
    previous = flags.groupby(ordered["experiment_id"]).shift(1, fill_value=False)
    onset = pd.Series(flags.to_numpy() & ~previous.to_numpy(bool), index=ordered.index)
    return onset.reindex(frame.index).to_numpy(bool)


def _alarm_rate(frame: pd.DataFrame, mask: np.ndarray, onsets: np.ndarray) -> dict:
    """Alarm onsets per hour, plus how much of the bucket's quiet time sits flagged.

    The two say different things and both matter: ``per_hour`` counts how often the
    detector *starts* crying wolf, ``flagged_fraction`` how much of the quiet time it
    spends lit -- which is where an alarm that latches after the fault clears shows up.
    """
    empty = {"alarms": 0, "windows": 0, "hours": 0.0, "per_hour": float("nan"),
             "flagged_windows": 0, "flagged_fraction": float("nan")}
    if not mask.any():
        return empty
    rows = frame[mask]
    hours = float((rows["window_end_ns"] - rows["window_start_ns"]).sum() / NS_PER_HOUR)
    alarms = int(onsets[mask].sum())
    flagged = int(rows["detect"].astype(bool).sum())
    return {
        "alarms": alarms,
        "windows": len(rows),
        "hours": hours,
        "per_hour": float(alarms / hours) if hours > 0 else float("nan"),
        "flagged_windows": flagged,
        "flagged_fraction": float(flagged / len(rows)),
    }


def _ranking_accuracy(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"n": 0, "top1": float("nan"), "top3": float("nan")}
    top1 = (rows["root_top1"].astype(str) == rows["root_service"].astype(str)).mean()
    top3 = np.mean([
        truth in list(ranked)[:3]
        for truth, ranked in zip(rows["root_service"].astype(str), rows["ranked_services"])
    ])
    return {"n": len(rows), "top1": float(top1), "top3": float(top3)}


def _class_report(y_true, y_pred) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "per_class": {}}
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {
        name: {key: float(value) for key, value in scores.items()}
        for name, scores in report.items()
        if isinstance(scores, dict) and name not in ("macro avg", "weighted avg")
    }
    return {
        "n": len(y_true),
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": per_class,
    }


def _calibration(confidence, correct, n_bins: int) -> dict:
    confidence = np.asarray(list(confidence), dtype=float)
    correct = np.asarray(list(correct)).astype(float)
    return {
        "n": len(confidence),
        "ece": expected_calibration_error(confidence, correct, n_bins),
        "bins": reliability_bins(confidence, correct, n_bins),
    }
