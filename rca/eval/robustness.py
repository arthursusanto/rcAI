"""Robustness of a *trained* model to degraded telemetry.

Both perturbations happen at test time only -- the model is not retrained -- which is the
question an operator actually has: the collector for one signal dies, or the trace
pipeline falls a window behind, and the incident is still running. The perturbations
themselves live in ``rca.models.augment``, because training uses them too.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from rca.eval.metrics import evaluate
from rca.features.schema import feature_columns, modality_of
from rca.models.augment import (
    TRACE_MODALITY,
    modality_columns,
    with_delayed_traces,
    with_modality_missing,
)

__all__ = [
    "TRACE_MODALITY",
    "modality_columns",
    "robustness_report",
    "summarize",
    "with_delayed_traces",
    "with_modality_missing",
]


def robustness_report(
    model,
    windows: pd.DataFrame,
    manifests: Iterable | None = None,
    lag: int = 1,
    debounce: int = 1,
) -> dict:
    """Headline metrics for intact telemetry and for each degraded condition."""
    conditions = {"intact": windows}
    for modality in sorted({modality_of(c) for c in feature_columns(windows.columns)}):
        conditions[f"missing_{modality}"] = with_modality_missing(windows, modality)
    conditions[f"traces_delayed_{lag}"] = with_delayed_traces(windows, lag)
    return {
        name: summarize(evaluate(model, table, manifests, profile=False, debounce=debounce))
        for name, table in conditions.items()
    }


def summarize(result: dict) -> dict:
    """The handful of numbers worth putting in a comparison row."""
    return {
        "detect_f1": result["detection"]["f1"],
        "detect_recall": result["detection"]["recall"],
        "recall_sympt": result["detection"]["recall_symptomatic"],
        "detect_precision": result["detection"]["precision"],
        "auroc": result["detection"]["auroc"],
        "top1_incident": result["localization"]["top1_incident"],
        "top1_detected": result["localization"]["detected"]["top1"],
        "top1_oracle": result["localization"]["oracle"]["top1"],
        "top3_oracle": result["localization"]["oracle"]["top3"],
        "fault_macro_f1": result["classification"]["predicted_root"]["macro_f1"],
        "median_delay_s": result["detection_delay"]["median_s"],
        "undetected": result["detection_delay"]["undetected_fraction"],
        "fa_all_per_hour": result["false_alarms"]["all_fault_free"]["per_hour"],
        "fa_post_per_hour": result["false_alarms"]["post_fault"]["per_hour"],
        "fa_all_flagged": result["false_alarms"]["all_fault_free"]["flagged_fraction"],
    }
