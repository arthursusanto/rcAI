"""Build the window dataset of a whole experiment directory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rca.data import schema
from rca.features.windows import DEFAULT_WINDOW_NS, build_windows, empty_table


def build_dataset(
    experiments_root: Path | str,
    out_path: Path | str,
    window_ns: float = DEFAULT_WINDOW_NS,
    stride_ns: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Window every experiment under ``experiments_root`` and write one parquet file."""
    directories = schema.list_experiments(Path(experiments_root))[:limit]
    tables = [
        build_windows(schema.read_experiment(d), window_ns=window_ns, stride_ns=stride_ns)
        for d in directories
    ]
    windows = pd.concat(tables, ignore_index=True) if tables else empty_table()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_path, index=False)
    return windows
