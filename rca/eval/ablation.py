"""Modality ablations: retrain the whole system on a subset of the telemetry.

Unlike ``rca.eval.robustness`` (a trained model meeting degraded input), an ablation
answers "how much of the result does this signal actually carry" -- so the dropped
columns are removed before training, not blanked at test time.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from rca.eval.metrics import evaluate
from rca.eval.robustness import summarize
from rca.features.schema import feature_columns, modality_of

# From docs/MODELS.md.
ALL_MODALITIES = ["metrics", "traces", "logs", "graph", "temporal"]

MODALITY_SETS: dict[str, list[str]] = {
    "metrics": ["metrics"],
    "traces": ["traces"],
    "logs": ["logs"],
    "metrics+traces": ["metrics", "traces"],
    "all-graph": [m for m in ALL_MODALITIES if m != "graph"],
    # The temporal family is derived from the others, so "all - temporal" measures what
    # the trailing-window view adds over the same signals read one window at a time.
    "all-temporal": [m for m in ALL_MODALITIES if m != "temporal"],
    "all": list(ALL_MODALITIES),
}


def restrict_modalities(windows: pd.DataFrame, modalities: Iterable[str]) -> pd.DataFrame:
    """Drop every feature column outside ``modalities``; keys and labels are kept."""
    keep = set(modalities)
    dropped = [c for c in feature_columns(windows.columns) if modality_of(c) not in keep]
    return windows.drop(columns=dropped)


def ablation_report(
    make_model: Callable[[], object],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    manifests: Iterable | None = None,
    sets: dict[str, list[str]] | None = None,
    debounce: int = 1,
) -> dict:
    """Retrain and evaluate once per modality subset."""
    sets = MODALITY_SETS if sets is None else sets
    report = {}
    for name, modalities in sets.items():
        model = make_model().fit(
            restrict_modalities(train, modalities), restrict_modalities(val, modalities)
        )
        report[name] = summarize(evaluate(
            model, restrict_modalities(test, modalities), manifests, profile=False,
            debounce=debounce,
        ))
    return report
