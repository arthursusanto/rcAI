"""Window-table perturbations: a missing modality, and traces arriving a window late.

The same two operations serve both ends of the pipeline. At *test* time
(``rca.eval.robustness``) they stand in for a collector that died or fell behind. At
*training* time they are augmentation: a model whose training set never contained a
NaN'd modality has no idea what to do when one disappears, and answers with confident
nonsense -- so a fraction of the training windows is appended back in degraded form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rca.features.schema import MODALITIES, feature_columns, modality_of

KEYS = ["experiment_id", "window_idx"]
TRACE_MODALITY = "traces"

DEFAULT_DROPOUT = 0.3
AUGMENT_SEED = 0
# Copies are re-keyed so they aggregate as windows of their own instead of merging with
# the clean window they were made from.
DEGRADED_MARK = "#"


def modality_columns(windows: pd.DataFrame, modality: str) -> list[str]:
    return [c for c in feature_columns(windows.columns) if modality_of(c) == modality]


def with_modality_missing(windows: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Copy of the window table with every column of ``modality`` set to NaN."""
    out = windows.copy()
    out[modality_columns(windows, modality)] = np.nan
    return out


def with_delayed_traces(windows: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Trace features taken from ``lag`` windows earlier, per (experiment, service).

    The first ``lag`` windows of each series have nothing to shift in and become NaN,
    which is the same "not observable" convention the feature build uses. Row order and
    index are preserved.
    """
    columns = modality_columns(windows, TRACE_MODALITY)
    out = windows.copy()
    out["_position"] = np.arange(len(out))
    out = out.sort_values(["experiment_id", "service", "window_idx"])
    out[columns] = out.groupby(["experiment_id", "service"], sort=False)[columns].shift(lag)
    out = out.sort_values("_position").drop(columns="_position")
    return out.set_axis(windows.index)


def augment_training(
    windows: pd.DataFrame,
    fraction: float = DEFAULT_DROPOUT,
    modalities: list[str] | None = None,
    delayed_fraction: float | None = None,
    seed: int = AUGMENT_SEED,
) -> pd.DataFrame:
    """Append degraded copies of a random subset of the training windows.

    ``fraction`` of the windows come back with a single randomly chosen modality blanked
    -- for the *whole* window, every service of it, so the stage-1 aggregate sees the gap
    too and not just the per-service rows. Another ``delayed_fraction`` (half of
    ``fraction`` by default) come back with their trace features shifted in from the
    previous window. Labels are untouched: the copies teach the model what a fault looks
    like through a hole in the telemetry.
    """
    if fraction <= 0 or windows.empty:
        return windows
    modalities = list(modalities or MODALITIES)
    delayed_fraction = fraction / 2 if delayed_fraction is None else delayed_fraction
    rng = np.random.default_rng(seed)

    keys = windows[KEYS].drop_duplicates().reset_index(drop=True)
    index = pd.MultiIndex.from_frame(windows[KEYS])
    parts = [windows]

    picked = _sample(rng, len(keys), fraction)
    assigned = rng.integers(0, len(modalities), size=len(picked))
    for position, modality in enumerate(modalities):
        selected = keys.iloc[picked[assigned == position]]
        if selected.empty:
            continue
        rows = windows[index.isin(pd.MultiIndex.from_frame(selected))]
        parts.append(_marked(with_modality_missing(rows, modality), f"drop-{modality}"))

    delayed = _sample(rng, len(keys), delayed_fraction)
    if len(delayed):
        shifted = with_delayed_traces(windows, 1)
        selected = keys.iloc[delayed]
        parts.append(_marked(
            shifted[index.isin(pd.MultiIndex.from_frame(selected))], "delayed-traces"
        ))
    return pd.concat(parts, ignore_index=True)


def dropout_spec(value) -> tuple[float, list[str]]:
    """Normalise ``p`` or ``(p, modalities)`` into ``(p, modalities)``."""
    if isinstance(value, (int, float)):
        return float(value), list(MODALITIES)
    fraction, modalities = value
    return float(fraction), list(modalities or MODALITIES)


def _sample(rng: np.random.Generator, total: int, fraction: float) -> np.ndarray:
    count = min(round(fraction * total), total)
    return rng.choice(total, size=count, replace=False) if count > 0 else np.empty(0, int)


def _marked(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    out = rows.copy()
    out["experiment_id"] = out["experiment_id"].astype(str) + DEGRADED_MARK + label
    return out
