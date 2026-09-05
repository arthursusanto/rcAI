"""Leakage-free train / val / test splits.

Every split is *by experiment*: an experiment id lands in exactly one set, so windows of
the same incident can never appear on both sides of a split. Besides the random split,
the deliberately hard generalization splits hold out whole services, fault intensities,
traffic profiles or time ranges, which is where a root-cause model's claimed accuracy
usually falls apart.

All functions take manifests (``rca.data.schema.read_manifest``) and return
``{"train": [...], "val": [...], "test": [...]}`` of experiment ids.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from rca.data.schema import INFRA_OWNER, NO_FAULT, Manifest

Split = dict[str, list[str]]

DEFAULT_FRACS = (0.6, 0.2, 0.2)
DEFAULT_VAL_FRAC = 0.15


def fault_type_of(manifest: Manifest) -> str:
    """Fault family of an experiment, ``"none"`` for a fault-free / traffic-only run."""
    fault = _first_fault(manifest)
    return fault.fault_type if fault else NO_FAULT


def target_of(manifest: Manifest) -> str:
    """Root-cause service of an experiment, folded onto its owner; ``""`` if fault-free."""
    fault = _first_fault(manifest)
    if fault is None:
        return ""
    return INFRA_OWNER.get(fault.target, fault.target)


def split_by_experiment(
    manifests: Iterable[Manifest], seed: int = 0, fracs: Sequence[float] = DEFAULT_FRACS
) -> Split:
    """Random split over experiment ids, stratified by fault type (including "none")."""
    strata: dict[str, list[str]] = defaultdict(list)
    for manifest in manifests:
        strata[fault_type_of(manifest)].append(manifest.experiment_id)

    rng = np.random.default_rng(seed)
    split: Split = {"train": [], "val": [], "test": []}
    for fault_type in sorted(strata):
        ids = sorted(set(strata[fault_type]))
        rng.shuffle(ids)
        n = len(ids)
        n_train = _round(n * fracs[0])
        n_val = _round(n * (fracs[0] + fracs[1])) - n_train
        split["train"] += ids[:n_train]
        split["val"] += ids[n_train:n_train + n_val]
        split["test"] += ids[n_train + n_val:]
    return _sorted(split)


def split_holdout_service(
    manifests: Iterable[Manifest],
    holdout_services: Iterable[str],
    seed: int = 0,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Split:
    """Test = experiments whose root-cause service is held out; nothing else sees them."""
    holdout = set(holdout_services)
    return _holdout(manifests, lambda m: target_of(m) in holdout, seed, val_frac)


def split_holdout_intensity(
    manifests: Iterable[Manifest],
    test_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 0,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Split:
    """Test = experiments whose fault intensity falls in ``test_range`` (inclusive)."""
    low, high = test_range

    def is_test(manifest: Manifest) -> bool:
        fault = _first_fault(manifest)
        return fault is not None and low <= fault.intensity <= high

    return _holdout(manifests, is_test, seed, val_frac)


def split_holdout_traffic(
    manifests: Iterable[Manifest],
    test_shapes: Iterable[str] | None = None,
    rps_threshold: float | None = None,
    seed: int = 0,
    val_frac: float = DEFAULT_VAL_FRAC,
) -> Split:
    """Test = experiments with a held-out traffic shape or above an rps threshold."""
    shapes = set(test_shapes or ())

    def is_test(manifest: Manifest) -> bool:
        if manifest.traffic.shape in shapes:
            return True
        return rps_threshold is not None and manifest.traffic.base_rps >= rps_threshold

    return _holdout(manifests, is_test, seed, val_frac)


def split_by_time(
    manifests: Iterable[Manifest], frac: float = 0.2, val_frac: float = 0.1
) -> Split:
    """Chronological split: the last ``frac`` of experiments is test, the block before val."""
    ordered = [m.experiment_id for m in sorted(manifests, key=lambda m: (m.start_ns,
                                                                        m.experiment_id))]
    n = len(ordered)
    n_test = _round(n * frac)
    n_val = _round(n * (frac + val_frac)) - n_test
    cut_val = n - n_test - n_val
    return _sorted({
        "train": ordered[:cut_val],
        "val": ordered[cut_val:n - n_test],
        "test": ordered[n - n_test:],
    })


def describe(split: Split, manifests: Iterable[Manifest]) -> pd.DataFrame:
    """Print, and return, the fault-type and root-target composition of each set."""
    by_id = {m.experiment_id: m for m in manifests}
    rows = []
    for name in ("train", "val", "test"):
        for experiment_id in split.get(name, []):
            manifest = by_id.get(experiment_id)
            if manifest is None:
                continue
            rows.append({
                "set": name,
                "fault_type": fault_type_of(manifest),
                "target": target_of(manifest) or "-",
            })
    frame = pd.DataFrame(rows, columns=["set", "fault_type", "target"])
    order = [name for name in ("train", "val", "test") if split.get(name)]
    print(f"experiments per set: "
          f"{ {name: len(split.get(name, [])) for name in ('train', 'val', 'test')} }")
    if frame.empty:
        return frame
    for column in ("fault_type", "target"):
        table = pd.crosstab(frame[column], frame["set"]).reindex(columns=order, fill_value=0)
        print(f"\nby {column}:")
        print(table.to_string())
    return frame


def _round(value: float) -> int:
    """Half-up rounding, so cumulative fraction boundaries never round the wrong way."""
    return int(np.floor(value + 0.5))


def _first_fault(manifest: Manifest):
    if not manifest.faults:
        return None
    return min(manifest.faults, key=lambda f: f.start_ns)


def _holdout(manifests, is_test, seed: int, val_frac: float) -> Split:
    manifests = list(manifests)
    test = [m for m in manifests if is_test(m)]
    rest = [m for m in manifests if not is_test(m)]
    split = split_by_experiment(rest, seed=seed, fracs=(1.0 - val_frac, val_frac, 0.0))
    split["test"] = [m.experiment_id for m in test]
    return _sorted(split)


def _sorted(split: Split) -> Split:
    return {name: sorted(set(ids)) for name, ids in split.items()}
