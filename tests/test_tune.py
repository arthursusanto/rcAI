"""Tests for the cross-validated hyperparameter search."""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
import pytest
from synthetic_windows import Spec, make_windows

from rca.models.baselines import MODELS
from rca.models.tune import (
    LOGREG_SPACE,
    SELECTION,
    STAGES,
    XGB_SPACE,
    Trial,
    build_folds,
    configurations,
    final_params,
    grouped_folds,
    read_params,
    select,
    stage_frame,
    subsample,
    trials_frame,
    tune_model,
    write_tuning,
)

# Sixteen experiments, so a four-fold split still leaves both fault families and a
# fault-free run on either side of every fold.
SPECS = (
    [Spec(f"exp-cpu-{i}", "cpu_saturation", ["cart", "payment", "shipping"][i % 3])
     for i in range(6)]
    + [Spec(f"exp-err-{i}", "error_rate", ["payment", "cart", "shipping"][i % 3])
       for i in range(6)]
    + [Spec(f"exp-none-{i}", None) for i in range(4)]
)
STRATA = {spec.experiment_id: spec.fault_type or "none" for spec in SPECS}
FAMILIES = sorted(set(STRATA.values()))
# How each model's stage estimator names its hyperparameters: XGBClassifier is the
# estimator, the other two sit at the end of an imputation pipeline.
PREFIX = {"xgb": "", "rf": "randomforestclassifier__", "logreg": "logisticregression__"}


@pytest.fixture(scope="module")
def windows() -> pd.DataFrame:
    return make_windows(SPECS)


@pytest.fixture(scope="module")
def folds(windows):
    return build_folds(windows, STRATA, k=2, seed=0)


# --- grouped folds ------------------------------------------------------------------------
@pytest.mark.parametrize("k", [2, 3, 4])
def test_grouped_folds_partition_the_experiments(k):
    parts = grouped_folds(STRATA, STRATA, k=k, seed=0)
    assert len(parts) == k
    assert sorted(itertools.chain(*parts)) == sorted(STRATA)   # every id used exactly once
    assert all(parts)                                        # and no fold left empty


def test_grouped_folds_spread_every_fault_family():
    """Four folds over four experiments of each family: one apiece, nothing missing."""
    for part in grouped_folds(STRATA, STRATA, k=4, seed=0):
        assert sorted({STRATA[experiment_id] for experiment_id in part}) == FAMILIES


def test_grouped_folds_are_seeded():
    same = grouped_folds(STRATA, STRATA, k=4, seed=0)
    assert same == grouped_folds(STRATA, STRATA, k=4, seed=0)
    assert same != grouped_folds(STRATA, STRATA, k=4, seed=1)


@pytest.mark.parametrize("k", [2, 4])
def test_folds_never_split_an_experiment(windows, k):
    """No experiment id is on both sides of a fold, and each is validated exactly once."""
    built = build_folds(windows, STRATA, k=k, seed=0)
    validated = []
    for fold in built:
        assert not set(fold.val_ids) & set(fold.train_ids)
        assert sorted(fold.val_ids + fold.train_ids) == sorted(STRATA)
        validated += fold.val_ids
    assert sorted(validated) == sorted(STRATA)


@pytest.mark.parametrize("k", [2, 4])
def test_validation_rows_partition_the_pool_exactly_once(windows, k):
    """A row-level check that grouping held: nothing is validated twice or dropped.

    The training side is augmented with degraded copies, so this also catches an
    augmented copy leaking into a validation matrix -- the counts would no longer add up.
    """
    built = build_folds(windows, STRATA, k=k, seed=0)
    for stage in STAGES:
        total = sum(len(fold.stages[stage].x_val) for fold in built)
        assert total == len(stage_frame(windows, stage))


def test_training_side_is_augmented(windows, folds):
    """The fold trains on degraded copies as `TwoStageModel.fit` does, and validates clean."""
    fold = folds[0]
    clean = windows[windows["experiment_id"].isin(set(fold.train_ids))]
    for stage in STAGES:
        assert len(fold.stages[stage].x_all) > len(stage_frame(clean, stage))


def test_early_stopping_holdout_is_carved_out_of_the_training_side(folds):
    """`fit` + `eval` is exactly `all`, so early stopping never watches the scored fold."""
    for fold in folds:
        for stage in STAGES:
            data = fold.stages[stage]
            assert len(data.x_fit) < len(data.x_all)
            assert len(data.x_eval) > 0


def test_subsample_keeps_the_family_mix():
    picked = subsample(list(STRATA), STRATA, 8, seed=0)
    assert 6 <= len(picked) <= 10
    assert sorted({STRATA[experiment_id] for experiment_id in picked}) == FAMILIES
    assert picked == subsample(list(STRATA), STRATA, 8, seed=0)
    assert subsample(list(STRATA), STRATA, 0, seed=0) == sorted(STRATA)


# --- search space --------------------------------------------------------------------------
def test_a_small_discrete_space_is_enumerated_not_sampled():
    rng = np.random.default_rng(0)
    grid = configurations(LOGREG_SPACE, 12, rng)
    assert [config["C"] for config in grid] == LOGREG_SPACE["C"][1]


def test_random_search_draws_distinct_configurations_inside_the_bounds():
    configs = configurations(XGB_SPACE, 39, np.random.default_rng(0))
    assert len(configs) == 39
    assert len({json.dumps(c, sort_keys=True) for c in configs}) == 39
    for config in configs:
        assert set(config) == set(XGB_SPACE)
        for name, (kind, low, high) in XGB_SPACE.items():
            assert low <= config[name] <= high
            assert isinstance(config[name], int if kind == "int" else float)


def test_random_search_is_seeded():
    first = configurations(XGB_SPACE, 10, np.random.default_rng(3))
    assert first == configurations(XGB_SPACE, 10, np.random.default_rng(3))
    assert first != configurations(XGB_SPACE, 10, np.random.default_rng(4))


# --- the search ---------------------------------------------------------------------------
def test_search_is_deterministic(folds):
    """Same seed, same folds -> the same trials and the same winner, value for value."""
    first, best = tune_model("xgb", folds, trials=3, seed=0, early_stopping=0)
    again, best_again = tune_model("xgb", folds, trials=3, seed=0, early_stopping=0)
    assert best == best_again
    assert [t.params for t in first] == [t.params for t in again]
    assert [t.per_fold for t in first] == [t.per_fold for t in again]


def test_trial_zero_is_the_model_defaults(folds):
    log, _ = tune_model("logreg", folds, trials=3, seed=0)
    for stage in STAGES:
        assert next(t for t in log if t.stage == stage).params == {}


def test_the_winner_is_the_best_cv_mean(folds):
    log, best = tune_model("logreg", folds, trials=4, seed=0)
    for stage in STAGES:
        stage_log = [t for t in log if t.stage == stage]
        assert select(stage_log).score == max(t.score for t in stage_log)
        assert best[stage] == final_params(select(stage_log))


def test_every_trial_is_scored_on_every_fold(folds):
    log, _ = tune_model("logreg", folds, trials=2, seed=0)
    for trial in log:
        assert len(trial.per_fold) == len(folds)
        assert SELECTION[trial.stage] in trial.metrics


def test_trials_frame_records_params_time_and_spread(folds):
    log, _ = tune_model("logreg", folds, trials=2, seed=0)
    frame = trials_frame(log)
    assert len(frame) == len(log)
    assert {"model", "stage", "trial", "cv_mean", "cv_std", "seconds"} <= set(frame.columns)
    assert (frame["seconds"] >= 0).all()
    assert frame.loc[frame["p_C"].notna(), "p_C"].isin(LOGREG_SPACE["C"][1]).all()


def test_select_breaks_ties_towards_the_earlier_trial():
    """Ties are common (a saturated metric) and must not be broken on anything measured."""
    tied = [Trial("xgb", "root", index, {"max_depth": index}, None, {"top1": 0.9},
                  {"top1": 0.0}, [0.9], seconds, None)
            for index, seconds in enumerate([9.0, 0.1, 5.0])]
    assert select(tied).index == 0
    better = Trial("xgb", "root", 3, {}, None, {"top1": 0.91}, {"top1": 0.0}, [0.91], 9.0,
                   None)
    assert select([*tied, better]).index == 3


def test_early_stopping_winner_carries_its_stopping_round_as_n_estimators():
    trial = Trial("xgb", "root", 7, {"n_estimators": 800, "max_depth": 4}, 50,
                  {"top1": 0.9}, {"top1": 0.01}, [0.9, 0.9], 1.0, 123)
    assert final_params(trial)["n_estimators"] == 123
    assert "early_stopping_rounds" not in final_params(trial)
    plain = Trial("xgb", "root", 7, {"n_estimators": 800}, None,
                  {"top1": 0.9}, {"top1": 0.01}, [0.9, 0.9], 1.0, None)
    assert final_params(plain)["n_estimators"] == 800


# --- round trip ----------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["xgb", "rf", "logreg"])
def test_best_json_round_trips_into_a_fitted_model(windows, folds, tmp_path, name):
    """The tuned values reach the estimator that `rca train` actually fits."""
    log, best = tune_model(name, folds, trials=3, seed=0, early_stopping=0)
    path = write_tuning(tmp_path, log, {name: best})
    assert path.name == "best.json"
    assert (tmp_path / f"trials-{name}.csv").exists()
    assert (tmp_path / f"trials-{name}.json").exists()

    tuned = read_params(path, name)
    assert tuned == best
    train = windows[windows["experiment_id"].isin(set(folds[0].train_ids))]
    val = windows[windows["experiment_id"].isin(set(folds[0].val_ids))]
    model = MODELS[name](params=tuned).fit(train, val)
    assert len(model.predict(val)) == val.groupby(["experiment_id", "window_idx"]).ngroups

    for stage, attribute in (("detect", "detect_model_"), ("root", "root_model_"),
                             ("fault", "fault_model_")):
        settings = getattr(model, attribute).get_params()
        for key, value in tuned[stage].items():
            found = settings[PREFIX[name] + key]
            if value == "auto":       # resolved to this fold's class ratio at fit time
                assert isinstance(found, float)
            else:
                assert found == value


def test_best_json_merges_models_tuned_in_separate_runs(folds, tmp_path):
    """Searching one model at a time is how a big search is kept inside a time budget."""
    xgb_log, xgb_best = tune_model("xgb", folds, trials=2, seed=0, early_stopping=0)
    write_tuning(tmp_path, xgb_log, {"xgb": xgb_best})
    rf_log, rf_best = tune_model("rf", folds, trials=2, seed=0)
    path = write_tuning(tmp_path, rf_log, {"rf": rf_best})

    assert read_params(path, "xgb") == xgb_best
    assert read_params(path, "rf") == rf_best
    assert {p.name for p in tmp_path.glob("trials-*.csv")} == {"trials-xgb.csv",
                                                               "trials-rf.csv"}


def test_read_params_accepts_both_shapes_and_shrugs_at_a_missing_model(tmp_path):
    path = tmp_path / "best.json"
    path.write_text(json.dumps({"xgb": {"root": {"max_depth": 3}}}), newline="\n")
    assert read_params(path, "xgb") == {"root": {"max_depth": 3}}
    assert read_params(path, "rf") == {}                     # untuned model -> defaults
    assert read_params(None, "xgb") == {}

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"root": {"max_depth": 3}}), newline="\n")
    assert read_params(bare, "anything") == {"root": {"max_depth": 3}}


def test_untuned_model_is_unchanged_by_an_empty_params_file(windows, folds):
    train = windows[windows["experiment_id"].isin(set(folds[0].train_ids))]
    val = windows[windows["experiment_id"].isin(set(folds[0].val_ids))]
    default = MODELS["xgb"]().fit(train, val).predict(val)
    empty = MODELS["xgb"](params={}).fit(train, val).predict(val)
    pd.testing.assert_frame_equal(default, empty)


def test_scale_pos_weight_auto_matches_the_untuned_default(windows, folds):
    """"auto" is the ratio XGBModel computes, so picking it must change nothing."""
    train = windows[windows["experiment_id"].isin(set(folds[0].train_ids))]
    val = windows[windows["experiment_id"].isin(set(folds[0].val_ids))]
    default = MODELS["xgb"]().fit(train, val)
    auto = MODELS["xgb"](params={"root": {"scale_pos_weight": "auto"}}).fit(train, val)
    assert (default.root_model_.get_params()["scale_pos_weight"]
            == auto.root_model_.get_params()["scale_pos_weight"])
    assert np.allclose(default.score_root(val), auto.score_root(val))


# --- budget -----------------------------------------------------------------------------------
def test_tuning_a_tiny_table_stays_within_its_budget(windows):
    """A guard against an accidental blow-up (a fold rebuilt per trial, threads thrashing).

    The bound is deliberately loose -- it is a smoke alarm, not a benchmark -- but three
    configurations of three stages over two folds of 768 rows is a second's work.
    """
    start = time.perf_counter()
    built = build_folds(windows, STRATA, k=2, seed=0)
    log, best = tune_model("xgb", built, trials=3, seed=0, early_stopping=50,
                           early_stopping_top=1)
    elapsed = time.perf_counter() - start
    assert len(log) == 3 * (3 + 1)                            # three stages, +1 early-stopped
    assert set(best) == set(STAGES)
    assert elapsed < 120.0, f"tuning the tiny table took {elapsed:.1f}s"
