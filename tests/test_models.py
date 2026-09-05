"""Tests for the three-stage models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic_windows import DEFAULT_SPECS, SERVICES, make_windows

from rca.features.schema import feature_columns
from rca.models.aggregate import aggregate_windows
from rca.models.augment import augment_training, modality_columns
from rca.models.base import (
    assert_no_service_identity,
    best_f1_threshold,
    sustained,
    threshold_for_target_fpr,
)
from rca.models.baselines import (
    MODELS,
    RULES,
    SIGNATURE_COMPONENTS,
    SignatureModel,
    _signature_score,
)
from rca.models.calibration import fit_binary_calibrator

TRAIN = [spec.experiment_id for spec in DEFAULT_SPECS if "train" in spec.experiment_id]
VAL = [spec.experiment_id for spec in DEFAULT_SPECS if "val" in spec.experiment_id]
TEST = [spec.experiment_id for spec in DEFAULT_SPECS if "test" in spec.experiment_id]


@pytest.fixture(scope="module")
def windows() -> pd.DataFrame:
    return make_windows()


def folds(windows: pd.DataFrame):
    return tuple(
        windows[windows["experiment_id"].isin(ids)] for ids in (TRAIN, VAL, TEST)
    )


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_fits_and_predicts_one_row_per_window(windows, name):
    train, val, test = folds(windows)
    model = MODELS[name]().fit(train, val)
    predictions = model.predict(test)

    expected = test.groupby(["experiment_id", "window_idx"]).ngroups
    assert len(predictions) == expected
    assert predictions["detect_prob"].between(0.0, 1.0).all()
    assert predictions["detect"].dtype == bool
    for ranked, scores in zip(predictions["ranked_services"], predictions["root_scores"]):
        assert sorted(ranked) == sorted(SERVICES)
        assert len(scores) == len(SERVICES)
        assert scores == sorted(scores, reverse=True)
    assert (predictions["root_top1"] == [r[0] for r in predictions["ranked_services"]]).all()
    assert (predictions["root_margin"] >= 0).all()
    for probabilities in predictions["fault_type_probs"]:
        assert set(probabilities) == set(model.classes_)
        assert np.isclose(sum(probabilities.values()), 1.0)
    assert predictions["fault_type_pred"].isin(model.classes_).all()


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb"])
def test_learned_models_find_the_planted_root(windows, name):
    """The test experiment's root service never carries this fault family in training."""
    train, val, test = folds(windows)
    model = MODELS[name]().fit(train, val)
    predictions = model.predict(test).set_index(["experiment_id", "window_idx"])
    truth = test[test["is_root"]].set_index(["experiment_id", "window_idx"])["service"]
    assert (predictions.loc[truth.index, "root_top1"] == truth).all()


def test_stage_two_refuses_a_service_identity_feature(windows):
    poisoned = windows.copy()
    poisoned["f_graph_is_cart"] = (poisoned["service"] == "cart").astype(float)
    features = [c for c in poisoned.columns if c.startswith("f_")]
    with pytest.raises(ValueError, match="service-identity"):
        assert_no_service_identity(poisoned, features)


def test_depth_from_entry_is_allowed(windows):
    """Constant per service, but structural -- the guard must let it through."""
    features = [c for c in windows.columns if c.startswith("f_")]
    assert "f_graph_depth_from_entry" in features
    assert_no_service_identity(windows, features)


def test_save_and_load_round_trip(windows, tmp_path):
    train, val, test = folds(windows)
    model = MODELS["logreg"]().fit(train, val)
    model.save(tmp_path / "model")
    reloaded = type(model).load(tmp_path / "model")
    pd.testing.assert_frame_equal(model.predict(test), reloaded.predict(test))


def test_aggregate_takes_max_mean_and_count_over_services(windows):
    aggregate = aggregate_windows(windows).set_index(["experiment_id", "window_idx"])
    rows = windows[
        (windows["experiment_id"] == "exp-train-cpu") & (windows["window_idx"] == 5)
    ]
    column = rows["f_traces_latency_z"]
    key = ("exp-train-cpu", 5)
    assert aggregate.loc[key, "a_traces_latency_z_max"] == pytest.approx(column.max())
    assert aggregate.loc[key, "a_traces_latency_z_mean"] == pytest.approx(column.mean())
    # The root (9.0) and the entry point (6.0) are both above the z = 3 threshold.
    assert aggregate.loc[key, "a_traces_latency_z_n_over"] == 2.0


def test_aggregate_count_is_nan_when_nothing_is_observable(windows):
    blinded = windows.copy()
    blinded["f_traces_latency_z"] = np.nan
    aggregate = aggregate_windows(blinded)
    assert aggregate["a_traces_latency_z_n_over"].isna().all()
    assert aggregate["a_traces_latency_z_max"].isna().all()


def test_best_f1_threshold_picks_the_perfect_split():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = best_f1_threshold(labels, probabilities)
    assert (probabilities >= threshold).tolist() == [False, False, True, True]


def test_target_false_alarm_rate_bounds_the_negatives():
    labels = np.array([0] * 10 + [1] * 4)
    probabilities = np.concatenate([np.linspace(0.0, 0.5, 10), np.linspace(0.4, 0.9, 4)])
    threshold = threshold_for_target_fpr(labels, probabilities, 0.1)
    assert (probabilities[labels == 0] >= threshold).mean() <= 0.1


def test_calibrator_falls_back_to_platt_on_few_positives():
    rng = np.random.default_rng(0)
    scores = rng.random(400)
    few = fit_binary_calibrator(scores, scores > 0.95)          # 20 positives
    many = fit_binary_calibrator(scores, scores > 0.5)          # ~200 positives
    assert few.method == "platt"
    assert many.method == "isotonic"
    assert np.all((few.transform(scores) >= 0) & (few.transform(scores) <= 1))


def test_calibrator_handles_a_single_class_fold():
    calibrator = fit_binary_calibrator(np.linspace(0, 1, 20), np.zeros(20, dtype=bool))
    assert calibrator.method == "constant"
    assert calibrator.transform(np.array([0.0, 1.0, np.nan])).tolist() == [0.0, 0.0, 0.0]


# --- degraded-telemetry augmentation ---------------------------------------------------
def test_augmentation_appends_one_copy_per_sampled_window(windows):
    augmented = augment_training(windows, fraction=0.5)
    n_windows = windows.groupby(["experiment_id", "window_idx"]).ngroups
    n_services = windows["service"].nunique()
    # 50 % of windows lose a modality, half that many get late traces; each copy is a
    # whole window, so it brings one row per service.
    expected = len(windows) + (round(0.5 * n_windows) + round(0.25 * n_windows)) * n_services
    assert len(augmented) == expected
    pd.testing.assert_frame_equal(
        augmented.iloc[:len(windows)].reset_index(drop=True), windows.reset_index(drop=True)
    )


def test_augmentation_is_disabled_at_zero(windows):
    pd.testing.assert_frame_equal(augment_training(windows, fraction=0.0), windows)


def test_dropped_modality_covers_the_whole_window_not_single_rows(windows):
    augmented = augment_training(windows, fraction=0.5)
    copies = augmented[augmented["experiment_id"].str.contains("#drop-")]
    assert not copies.empty
    for experiment_id, rows in copies.groupby("experiment_id"):
        modality = experiment_id.split("#drop-")[1]
        dropped = modality_columns(windows, modality)
        kept = [c for c in feature_columns(windows.columns) if c not in dropped]
        for _, window in rows.groupby("window_idx"):
            # every service of the window, and the modality gone for all of them
            assert sorted(window["service"]) == sorted(SERVICES)
            assert window[dropped].isna().all().all()
            # something else is still observable, so this is a modality gap not a hole
            assert window[kept].notna().any().any()


def test_delayed_copy_carries_the_previous_window_traces(windows):
    augmented = augment_training(windows, fraction=0.5)
    copies = augmented[augmented["experiment_id"].str.endswith("#delayed-traces")]
    assert not copies.empty
    traces = modality_columns(windows, "traces")
    source = windows.set_index(["experiment_id", "window_idx", "service"])
    for row in copies.itertuples():
        origin = row.experiment_id.split("#")[0]
        if row.window_idx == 0:
            assert pd.isna([getattr(row, c) for c in traces]).all()
            continue
        previous = source.loc[(origin, row.window_idx - 1, row.service), traces]
        # nan_ok: a column nothing ever observes is NaN on both sides.
        assert [getattr(row, c) for c in traces] == pytest.approx(
            previous.to_numpy(), nan_ok=True)
    # non-trace columns still come from the window itself
    assert copies["f_metrics_cpu_util_z"].notna().all()


def test_augmented_copies_aggregate_as_windows_of_their_own(windows):
    augmented = augment_training(windows, fraction=0.5)
    n_windows = windows.groupby(["experiment_id", "window_idx"]).ngroups
    expected = n_windows + round(0.5 * n_windows) + round(0.25 * n_windows)
    assert aggregate_windows(augmented).shape[0] == expected


def test_dropout_survives_a_fit_and_leaves_calibration_on_clean_data(windows):
    train, val, test = folds(windows)
    model = MODELS["xgb"](modality_dropout=0.3).fit(train, val)
    clean, degraded = model.training_rows_
    assert clean == len(train) and degraded > clean
    assert len(model.predict(test)) == test.groupby(
        ["experiment_id", "window_idx"]).ngroups


# --- the untrained baselines ---------------------------------------------------------------
def test_rules_alarm_needs_two_consecutive_windows(windows):
    """The burn-rate requirement lives in the model, not in the evaluation."""
    train, val, test = folds(windows)
    model = MODELS["rules"]().fit(train, val)
    assert model.detect_sustain == 2
    predictions = model.predict(test).sort_values(["experiment_id", "window_idx"])
    above = predictions["detect_prob"].to_numpy() >= model.detect_threshold_
    flagged = predictions["detect"].to_numpy(bool)
    # Never fires without the previous window also being above threshold ...
    assert not flagged[0]
    assert (flagged[1:] <= (above[1:] & above[:-1])).all()
    # ... and a sustained run does fire, so the requirement is not just suppressing it.
    assert flagged.any()


def test_rules_use_the_latency_ratio_not_a_z_score():
    assert RULES["latency"] == ("f_traces_latency_ratio", 2.0)
    assert set(RULES) == {"latency", "error_rate", "cpu", "memory",
                          "inbound_errors", "inbound_unanswered"}


def test_signature_score_is_a_normalised_mean(windows):
    """A row at every component's full scale scores 1; a quiet row scores near 0."""
    row = windows.iloc[[0]].copy()
    for column, scale in SIGNATURE_COMPONENTS.values():
        row[column] = scale
    assert _signature_score(row)[0] == pytest.approx(1.0)
    for column, scale in SIGNATURE_COMPONENTS.values():
        row[column] = 0.0
    assert _signature_score(row)[0] == pytest.approx(0.0)
    # Values beyond the scale are clipped, so one huge signal cannot carry the mean.
    row = windows.iloc[[0]].copy()
    for column, _ in SIGNATURE_COMPONENTS.values():
        row[column] = 0.0
    column, scale = SIGNATURE_COMPONENTS["latency"]
    row[column] = 1e6 * scale
    assert _signature_score(row)[0] == pytest.approx(1 / len(SIGNATURE_COMPONENTS))


def test_signature_localization_discounts_a_failing_dependency(windows):
    """Two equally anomalous services: the one explained by its callee ranks lower."""
    rows = windows.iloc[[0, 1]].copy()
    for column, scale in SIGNATURE_COMPONENTS.values():
        rows[column] = scale
    rows["f_graph_downstream_explained"] = [0.0, 30.0]
    scores = SignatureModel().score_root(rows)
    assert scores[0] > scores[1]
    assert scores[1] == pytest.approx(scores[0] - 1.0)


def test_sustained_needs_a_run_inside_one_experiment():
    flags = [True, True, False, True, True, True, False]
    groups = ["a", "a", "a", "a", "b", "b", "b"]
    # The run at index 3-5 straddles the a/b boundary and does not carry across it.
    assert sustained(groups, flags, 2).tolist() == [
        False, True, False, False, False, True, False
    ]
    assert sustained(groups, flags, 1).tolist() == flags


def test_identity_guard_ignores_sparse_binary_fractions():
    """A 0/1 column that is constant per service only because it is mostly NaN is not
    a service one-hot (real captures have such columns on small datasets)."""
    import numpy as np
    import pandas as pd

    from rca.models.base import assert_no_service_identity
    windows = pd.DataFrame({
        "service": ["a"] * 4 + ["b"] * 4,
        "f_logs_error_trace_frac": [np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, 0.0, np.nan],
        "f_identity_a": [1.0] * 4 + [0.0] * 4,
    })
    assert_no_service_identity(windows, ["f_logs_error_trace_frac"])
    import pytest
    with pytest.raises(ValueError):
        assert_no_service_identity(windows, ["f_identity_a"])
