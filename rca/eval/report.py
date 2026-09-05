"""Results serialisation: JSON, markdown tables, and MLflow tracking."""
from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

MLFLOW_EXPERIMENT = "rca"
MLFLOW_DIR = "mlruns"

# Comparison table: column label -> path into one model's result dict.
SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "detect_P": ("detection", "precision"),
    "detect_R": ("detection", "recall"),
    "detect_F1": ("detection", "f1"),
    "recall_sympt": ("detection", "recall_symptomatic"),
    "auroc": ("detection", "auroc"),
    # Time to alarm is measured to the end of the flagged window; delay_onset_med_s is
    # the older window-start number, kept so the two can be compared.
    "delay_med_s": ("detection_delay", "median_s"),
    "delay_p90_s": ("detection_delay", "p90_s"),
    "delay_onset_med_s": ("detection_delay", "onset_median_s"),
    "undetected": ("detection_delay", "undetected_fraction"),
    "fa_all": ("false_alarms", "all_fault_free", "per_hour"),
    "fa_post": ("false_alarms", "post_fault", "per_hour"),
    "top1_incident": ("localization", "top1_incident"),
    "top1_det": ("localization", "detected", "top1"),
    "top1_oracle": ("localization", "oracle", "top1"),
    "top3_oracle": ("localization", "oracle", "top3"),
    "fault_macroF1": ("classification", "predicted_root", "macro_f1"),
    "ece_detect": ("calibration", "detect", "ece"),
    "ece_top1": ("calibration", "top1", "ece"),
    "ece_fault": ("calibration", "fault_type", "ece"),
    "ms_per_window": ("cost", "ms_per_window"),
    "rss_mb": ("cost", "rss_mb"),
}


def dig(result: dict, path: Iterable[str]):
    node = result
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return float("nan")
        node = node[key]
    return node


def dig_dict(result: dict, path: Iterable[str]) -> dict:
    """``dig`` for a sub-dict; a missing branch is an empty one (NaN is truthy)."""
    node = dig(result, path)
    return node if isinstance(node, dict) else {}


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model, one column per headline metric."""
    rows = [
        {"model": name, **{label: dig(result, path) for label, path in SUMMARY_FIELDS.items()}}
        for name, result in results.items()
    ]
    return pd.DataFrame(rows).set_index("model")


# Canonical bucket order; the post-fault recovery cut carries its window count in its
# name, so anything unrecognised is appended rather than dropped.
BUCKET_ORDER = ["normal", "traffic_spike", "pre_fault", "post_fault", "all_fault_free"]


def comparison_display(results: dict[str, dict], decimals: int = 3) -> pd.DataFrame:
    """``comparison_table`` with bootstrap intervals folded into the cells as strings.

    A point estimate with no interval invites reading a 0.02 gap between two models as a
    result; ``0.807 [0.74, 0.85]`` makes the resolution of the experiment visible in the
    same glance.
    """
    table = comparison_table(results)
    display = table.copy().astype(object)
    for name, result in results.items():
        intervals = dig_dict(result, ("bootstrap",))
        for metric, bounds in intervals.items():
            if metric in display.columns and isinstance(bounds, dict):
                display.loc[name, metric] = (
                    f"{_cell(table.loc[name, metric], decimals)} "
                    f"[{bounds['lo']:.{decimals - 1}f}, {bounds['hi']:.{decimals - 1}f}]"
                )
    return display


def exposure_table(results: dict[str, dict]) -> pd.DataFrame:
    """Detection and localization split by how much traffic the root service saw."""
    rows = []
    for name, result in results.items():
        for stratum, values in dig_dict(result, ("exposure",)).items():
            rows.append({
                "model / exposure": f"{name} [{stratum.replace('exposure_', '')}]",
                "n_exp": values.get("n_experiments", float("nan")),
                "median_requests": values.get("median_exposure", float("nan")),
                "recall": values.get("recall", float("nan")),
                "recall_sympt": values.get("recall_symptomatic", float("nan")),
                "delay_med_s": values.get("delay_median_s", float("nan")),
                "undetected": values.get("undetected", float("nan")),
                "top1_oracle": values.get("top1_oracle", float("nan")),
                "top1_incident": values.get("top1_incident", float("nan")),
            })
    return pd.DataFrame(rows).set_index("model / exposure") if rows else pd.DataFrame()


def false_alarm_buckets(results: dict[str, dict]) -> list[str]:
    found = {name for result in results.values()
             for name in dig_dict(result, ("false_alarms",))}
    ordered = [name for name in BUCKET_ORDER if name in found]
    return ordered + sorted(found - set(ordered))


def false_alarm_table(results: dict[str, dict]) -> pd.DataFrame:
    buckets = false_alarm_buckets(results)
    rows = []
    for name, result in results.items():
        row = {"model": name}
        for bucket in buckets:
            row[f"{bucket}_/h"] = dig(result, ("false_alarms", bucket, "per_hour"))
            row[f"{bucket}_alarms"] = dig(result, ("false_alarms", bucket, "alarms"))
            row[f"{bucket}_flagged"] = dig(
                result, ("false_alarms", bucket, "flagged_fraction"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def per_family_table(results: dict[str, dict]) -> pd.DataFrame:
    """Oracle top-1 localization per fault family, and per family x target exposure.

    The exposure rows are what tell a genuinely hard family apart from one that happens
    to land on services nothing was calling.
    """
    families = sorted({
        family
        for result in results.values()
        for family in dig_dict(result, ("localization", "per_fault_type"))
    })
    rows = []
    for name, result in results.items():
        per_family = dig_dict(result, ("localization", "per_fault_type"))
        rows.append({"model": name, **{
            family: per_family.get(family, {}).get("top1", float("nan"))
            for family in families
        }})
        for stratum, values in dig_dict(result, ("exposure",)).items():
            by_family = values.get("per_fault_type", {})
            rows.append({
                "model": f"{name} [{stratum.replace('exposure_', '')}]",
                **{family: by_family.get(family, float("nan")) for family in families},
            })
    return pd.DataFrame(rows).set_index("model")


def nested_table(report: dict[str, dict]) -> pd.DataFrame:
    """Table from a {condition: {metric: value}} report (robustness / ablation)."""
    return pd.DataFrame(report).T


def write_results(results: dict, out_dir: Path | str, name: str = "results.json") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(jsonable(results), indent=2) + "\n", newline="\n")
    return path


def write_markdown(
    results: dict[str, dict],
    out_dir: Path | str,
    title: str,
    sections: dict[str, pd.DataFrame] | None = None,
    name: str = "results.md",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [f"# {title}", "",
             "## Model comparison", "",
             markdown_table(comparison_display(results)), "",
             ("_Bracketed ranges are 95 % percentile intervals from a cluster "
              "bootstrap over test experiments._"), "",
             "## False alarms per hour (debounced, fault-free time)", "",
             markdown_table(false_alarm_table(results)), "",
             "## Detection and localization by target exposure", "",
             markdown_table(exposure_table(results)), "",
             "## Oracle top-1 localization per fault family", "",
             markdown_table(per_family_table(results)), ""]
    for heading, table in (sections or {}).items():
        parts += [f"## {heading}", "", markdown_table(table), ""]
    path = out_dir / name
    path.write_text("\n".join(parts), newline="\n")
    return path


def markdown_table(table: pd.DataFrame, decimals: int = 3) -> str:
    """A GitHub markdown table (hand-rolled; ``to_markdown`` needs ``tabulate``)."""
    if table.empty:
        return "_(no rows)_"
    header = [str(table.index.name or "")] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for key, row in table.iterrows():
        cells = [str(key)] + [_cell(value, decimals) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(value, decimals: int) -> str:
    if isinstance(value, (float, np.floating)):
        return "-" if math.isnan(float(value)) else f"{float(value):.{decimals}f}"
    return str(value)


def log_mlflow(
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts: Iterable[Path] = (),
    experiment: str = MLFLOW_EXPERIMENT,
    tracking_dir: Path | str = MLFLOW_DIR,
) -> None:
    """Log one run to the local file-backed MLflow store; never fatal to a run."""
    # MLflow 3 refuses the file store unless the caller opts in explicitly.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    tracking = Path(tracking_dir).absolute()
    tracking.mkdir(parents=True, exist_ok=True)
    try:
        mlflow.set_tracking_uri(tracking.as_uri())
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({key: str(value) for key, value in params.items()})
            mlflow.log_metrics({
                key: float(value) for key, value in metrics.items()
                if isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))
            })
            for artifact in artifacts:
                if Path(artifact).exists():
                    mlflow.log_artifact(str(artifact))
    except Exception as error:                                  # noqa: BLE001
        print(f"warning: MLflow logging failed ({error})", file=sys.stderr)


def flat_metrics(result: dict, prefix: str = "") -> dict[str, float]:
    """Headline metrics of one model's result, flattened for MLflow."""
    return {
        f"{prefix}{label}": dig(result, path) for label, path in SUMMARY_FIELDS.items()
    }


def jsonable(value):
    """Recursively convert numpy scalars, arrays and NaN into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value if value is None or isinstance(value, str) else str(value)
