"""Tests for the experiment-level splits."""
from __future__ import annotations

import pandas as pd
import pytest
from conftest import make_manifest

from rca.data import schema
from rca.data.splits import (
    DEFAULT_FRACS,
    describe,
    split_by_experiment,
    split_by_time,
    split_holdout_intensity,
    split_holdout_service,
    split_holdout_traffic,
)

SERVICES = ["cart", "checkout", "payment", "shipping", "ad"]
SHAPES = ["steady", "ramp", "diurnal", "bursty"]
STRATUM_SIZE = 10                              # experiments per fault type in the fixture
STRATA = len(schema.FAULT_TYPES) + 1           # ... plus the fault-free stratum


@pytest.fixture
def manifests() -> list[schema.Manifest]:
    """80 faulty experiments (8 families x 10) plus 10 fault-free ones."""
    out = []
    for index, fault_type in enumerate(schema.FAULT_TYPES):
        for repeat in range(10):
            out.append(make_manifest(
                f"exp-{fault_type}-{repeat}",
                fault_type,
                target=SERVICES[repeat % len(SERVICES)],
                intensity=0.1 * (repeat + 1),
                shape=SHAPES[(index + repeat) % len(SHAPES)],
                base_rps=10.0 * (repeat + 1),
                start_ns=1_700_000_000_000_000_000 + (index * 10 + repeat) * 10**12,
            ))
    for repeat in range(10):
        out.append(make_manifest(
            f"exp-none-{repeat}", None, shape=SHAPES[repeat % len(SHAPES)],
            base_rps=10.0 * (repeat + 1),
            start_ns=1_700_000_000_000_000_000 + (100 + repeat) * 10**12,
        ))
    return out


def assert_partition(split, manifests, expect_complete=True):
    ids = [set(split[name]) for name in ("train", "val", "test")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    if expect_complete:
        assert set().union(*ids) == {m.experiment_id for m in manifests}


def test_split_by_experiment_is_a_partition(manifests):
    split = split_by_experiment(manifests, seed=1)
    assert_partition(split, manifests)
    # every set gets its requested share, up to one experiment of rounding per stratum
    for name, frac in zip(("train", "val", "test"), DEFAULT_FRACS, strict=True):
        assert abs(len(split[name]) - STRATA * STRATUM_SIZE * frac) <= STRATA


def test_split_by_experiment_is_deterministic_and_seed_dependent(manifests):
    assert split_by_experiment(manifests, seed=1) == split_by_experiment(manifests, seed=1)
    assert split_by_experiment(manifests, seed=2) != split_by_experiment(manifests, seed=1)


def test_split_by_experiment_stratifies_by_fault_type(manifests):
    """Every fault family, "none" included, is split in the same proportions."""
    split = split_by_experiment(manifests, seed=3)
    by_id = {m.experiment_id: m for m in manifests}
    for name, frac in zip(("train", "val", "test"), DEFAULT_FRACS, strict=True):
        counts = {}
        for experiment_id in split[name]:
            fault_type = by_id[experiment_id].faults[0].fault_type if by_id[
                experiment_id].faults else "none"
            counts[fault_type] = counts.get(fault_type, 0) + 1
        assert set(counts) == set(schema.FAULT_TYPES) | {"none"}
        for count in counts.values():
            assert abs(count - STRATUM_SIZE * frac) <= 1


def test_holdout_service_keeps_targets_out_of_train(manifests):
    holdout = {"payment", "ad"}
    split = split_holdout_service(manifests, holdout)
    assert_partition(split, manifests)
    by_id = {m.experiment_id: m for m in manifests}

    assert {by_id[i].faults[0].target for i in split["test"]} == holdout
    for name in ("train", "val"):
        for experiment_id in split[name]:
            faults = by_id[experiment_id].faults
            assert not faults or faults[0].target not in holdout
    # fault-free experiments have no target and stay available for training
    assert any(not by_id[i].faults for i in split["train"])


def test_holdout_intensity(manifests):
    split = split_holdout_intensity(manifests, test_range=(0.8, 1.0))
    assert_partition(split, manifests)
    by_id = {m.experiment_id: m for m in manifests}
    assert all(by_id[i].faults[0].intensity >= 0.8 - 1e-9 for i in split["test"])
    assert all(
        not by_id[i].faults or by_id[i].faults[0].intensity < 0.8 - 1e-9
        for i in split["train"] + split["val"]
    )
    assert len(split["test"]) == 8 * 3   # intensities 0.8, 0.9, 1.0 of each family


def test_holdout_traffic(manifests):
    split = split_holdout_traffic(manifests, test_shapes=["bursty"])
    assert_partition(split, manifests)
    by_id = {m.experiment_id: m for m in manifests}
    assert all(by_id[i].traffic.shape == "bursty" for i in split["test"])
    assert all(by_id[i].traffic.shape != "bursty" for i in split["train"] + split["val"])

    by_rps = split_holdout_traffic(manifests, rps_threshold=80.0)
    assert_partition(by_rps, manifests)
    assert all(by_id[i].traffic.base_rps >= 80.0 for i in by_rps["test"])


def test_split_by_time_is_chronological(manifests):
    split = split_by_time(manifests, frac=0.2, val_frac=0.1)
    assert_partition(split, manifests)
    by_id = {m.experiment_id: m for m in manifests}
    latest_train = max(by_id[i].start_ns for i in split["train"])
    earliest_val = min(by_id[i].start_ns for i in split["val"])
    latest_val = max(by_id[i].start_ns for i in split["val"])
    earliest_test = min(by_id[i].start_ns for i in split["test"])
    assert latest_train < earliest_val <= latest_val < earliest_test
    assert len(split["test"]) == 18


def test_describe_reports_composition(manifests, capsys):
    split = split_holdout_service(manifests, ["payment"])
    frame = describe(split, manifests)
    printed = capsys.readouterr().out
    assert "by fault_type:" in printed and "by target:" in printed
    assert len(frame) == sum(len(ids) for ids in split.values())
    assert set(frame.loc[frame["set"] == "test", "target"]) == {"payment"}


def test_manifest_is_the_completion_marker(tmp_path):
    """An interrupted write must be skipped, not read back as an empty experiment."""
    from rca.data import schema as sch

    exp = sch.Experiment(
        manifest=sch.Manifest(
            experiment_id="exp-0", source="sim", seed=1, start_ns=0, end_ns=1,
            traffic=sch.TrafficProfile(base_rps=1.0, shape="steady"), faults=[],
            services=list(sch.SERVICES), edges=list(sch.DEPENDENCY_EDGES),
            metric_interval_ns=1, warmup_ns=0),
        metrics=pd.DataFrame(columns=list(sch.METRICS_COLUMNS)),
        spans=pd.DataFrame(columns=list(sch.SPANS_COLUMNS)),
        logs=pd.DataFrame(columns=list(sch.LOGS_COLUMNS)))
    written = sch.write_experiment(exp, tmp_path)
    assert sch.list_experiments(tmp_path) == [written]

    # Killed after the parquets but before the manifest.
    (written / "manifest.json").unlink()
    assert sch.list_experiments(tmp_path) == []
    # ... or with the manifest present but a parquet lost.
    (written / "manifest.json").write_text("{}", newline="\n")
    (written / "spans.parquet").unlink()
    assert sch.list_experiments(tmp_path) == []
