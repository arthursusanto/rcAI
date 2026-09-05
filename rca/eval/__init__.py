"""Evaluation: metrics, robustness, ablations and reporting."""
from rca.eval.ablation import MODALITY_SETS, ablation_report
from rca.eval.metrics import evaluate
from rca.eval.report import comparison_table, log_mlflow, write_markdown, write_results
from rca.eval.robustness import robustness_report

__all__ = [
    "MODALITY_SETS",
    "ablation_report",
    "comparison_table",
    "evaluate",
    "log_mlflow",
    "robustness_report",
    "write_markdown",
    "write_results",
]
