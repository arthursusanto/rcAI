"""Tests for the evaluation metrics, robustness perturbations and ablations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic_windows import (
    DEFAULT_SPECS,
    SERVICES,
    Spec,
    make_manifests,
    make_windows,
)

from rca.eval.ablation import restrict_modalities
from rca.eval.metrics import (
    apply_debounce,
    bootstrap_intervals,
    detection_delay,
    evaluate,
    experiment_categories,
    experiment_fault_ends,
    false_alarm_rates,
    incident_localization,
    joined,
    symptomatic_recall,
    target_exposure,
    window_truth,
)
from rca.eval.report import comparison_table, write_markdown, write_results
from rca.eval.robustness import with_delayed_traces, with_modality_missing
from rca.features.schema import feature_columns, modality_of
from rca.models.baselines import MODELS
from rca.models.calibration import expected_calibration_error

FAULT_START_S = 45.0
DELAY_SPECS = [
    Spec("exp-a", "cpu_saturation", "cart", FAULT_START_S, 85.0),
    Spec("exp-b", "cpu_saturation", "cart", FAULT_START_S, 85.0),
    Spec("exp-c", "cpu_saturation", "cart", FAULT_START_S, 85.0),
    Spec("exp-n", None),
    Spec("exp-s", None, traffic_shape="bursty", category="traffic_spike"),
]
# For a fault running 45 s -> 85 s over 10 s windows: pre-fault is windows 0..3, fault
# windows are 4..8, post-fault recovery is 9..11.
DETECT = {
    #        pre-fault    |  fault window   | post-fault
    "exp-a": [True, True, False, True] + [False, False, True, True, True] + [True, False, True],
    "exp-b": [False] * 10 + [True, True],
    "exp-c": [False, False, False, False, True] + [False] * 7,
    "exp-n": [False, True, True, False, True] + [False] * 7,
    "exp-s": [True] * 4 + [False] * 8,
}
FAULT_ENDS = experiment_fault_ends(make_manifests(DELAY_SPECS))
CATEGORIES = experiment_categories(make_manifests(DELAY_SPECS))


def fake_predictions(windows: pd.DataFrame, detect: dict[str, list[bool]]) -> pd.DataFrame:
    """A prediction frame with the given detection flags and a fixed, wrong ranking."""
    truth = window_truth(windows)
    flags = [detect[e][i] for e, i in zip(truth["experiment_id"], truth["window_idx"])]
    return pd.DataFrame({
        "experiment_id": truth["experiment_id"],
        "window_idx": truth["window_idx"],
        "detect_prob": [1.0 if flag else 0.0 for flag in flags],
        "detect": flags,
        "ranked_services": [list(SERVICES) for _ in flags],
        "root_scores": [[0.4, 0.3, 0.2, 0.1] for _ in flags],
        "root_top1": [SERVICES[0]] * len(flags),
        "root_margin": [0.1] * len(flags),
        "fault_type_pred": ["cpu_saturation"] * len(flags),
        "fault_type_probs": [{"cpu_saturation": 1.0} for _ in flags],
        "fault_type_conf": [1.0] * len(flags),
    })


@pytest.fixture(scope="module")
def delay_frame() -> pd.DataFrame:
    windows = make_windows(DELAY_SPECS)
    return joined(windows, fake_predictions(windows, DETECT))


def test_detection_delay_is_measured_to_the_end_of_the_alarming_window(delay_frame):
    result = detection_delay(delay_frame)
    # The alarm cannot fire before the window's telemetry is complete, so the delay runs
    # to the window *end*: exp-a first fires in window 6 (ends at 70 s, fault started at
    # 45 s) -> 25 s; exp-c fires in window 4 (ends at 50 s) -> 5 s. exp-b never fires
    # while the fault runs -> undetected.
    assert result["delays_s"] == [5.0, 25.0]
    assert result["n_incidents"] == 3
    assert result["n_detected"] == 2
    assert result["undetected_fraction"] == pytest.approx(1 / 3)
    assert result["median_s"] == pytest.approx(15.0)
    assert result["p90_s"] == pytest.approx(23.0)
    # The old window-start number is kept for reference and is one window earlier.
    assert result["onset_delays_s"] == [0.0, 15.0]
    assert result["onset_median_s"] == pytest.approx(7.5)
    assert result["onset_p90_s"] == pytest.approx(13.5)


def test_false_alarms_are_debounced_per_bucket(delay_frame):
    rates = false_alarm_rates(delay_frame, CATEGORIES, FAULT_ENDS)

    # exp-n: F T T F T ... -> two negative-to-positive transitions over 12 x 10 s,
    # and three of the twelve windows sit flagged.
    assert rates["normal"] == {
        "alarms": 2, "windows": 12, "hours": pytest.approx(120 / 3600),
        "per_hour": pytest.approx(60.0),
        "flagged_windows": 3, "flagged_fraction": pytest.approx(0.25),
    }
    # exp-s: one run of four positives -> a single alarm.
    assert rates["traffic_spike"]["alarms"] == 1
    assert rates["traffic_spike"]["per_hour"] == pytest.approx(30.0)
    # Pre-fault time is windows 0..3 (the fault starts inside window 4). exp-a fires
    # T T F T there (two alarms), exp-c and exp-b none, over 3 x 4 x 10 s.
    assert rates["pre_fault"]["windows"] == 12
    assert rates["pre_fault"]["alarms"] == 2
    assert rates["pre_fault"]["per_hour"] == pytest.approx(60.0)


def test_post_fault_recovery_counts_as_fault_free_time(delay_frame):
    rates = false_alarm_rates(delay_frame, CATEGORIES, FAULT_ENDS)

    # Windows 9..11 of the three fault experiments start at or after the 85 s fault end.
    assert rates["post_fault"]["windows"] == 9
    # exp-a is still firing in window 9, but it was already firing in window 8 while the
    # fault ran: an alarm that has not cleared, not a new one. Its window 11 and exp-b's
    # window 10 are genuine onsets.
    assert rates["post_fault"]["alarms"] == 2
    assert rates["post_fault"]["per_hour"] == pytest.approx(80.0)
    # ... but four of those nine windows sit flagged, which is what the latch costs and
    # what the onset count deliberately does not double-charge.
    assert rates["post_fault"]["flagged_windows"] == 4
    assert rates["post_fault"]["flagged_fraction"] == pytest.approx(4 / 9)

    # Every window with no fault active: 2 x 12 fault-free experiments + 3 x 7 quiet
    # windows of the fault experiments (12 each, minus the 5 the fault runs through).
    assert rates["all_fault_free"]["windows"] == 45
    assert rates["all_fault_free"]["alarms"] == 7
    assert rates["all_fault_free"]["per_hour"] == pytest.approx(56.0)
    # The buckets tile the fault-free time: nothing counted twice, nothing dropped.
    assert rates["all_fault_free"]["alarms"] == sum(
        rates[bucket]["alarms"]
        for bucket in ("normal", "traffic_spike", "pre_fault", "post_fault")
    )
    # The old definition stopped at pre-fault and would have missed the tail entirely.
    assert rates["all_fault_free"]["alarms"] > (
        rates["normal"]["alarms"] + rates["traffic_spike"]["alarms"]
        + rates["pre_fault"]["alarms"]
    )


def test_post_fault_bucket_can_skip_the_recovery_ramp(delay_frame):
    # Default 30 s ramp reaches past the end of these 120 s experiments.
    rates = false_alarm_rates(delay_frame, CATEGORIES, FAULT_ENDS)
    assert rates["post_fault_excl_30s"]["windows"] == 0
    assert np.isnan(rates["post_fault_excl_30s"]["per_hour"])

    # With a 10 s ramp only windows starting at or after 95 s survive: 10 and 11.
    short = false_alarm_rates(delay_frame, CATEGORIES, FAULT_ENDS, recovery_s=10.0)
    assert short["post_fault_excl_10s"]["windows"] == 6
    assert short["post_fault_excl_10s"]["alarms"] == 2
    assert short["post_fault_excl_10s"]["per_hour"] == pytest.approx(120.0)


def test_fault_end_falls_back_to_the_last_fault_window(delay_frame):
    """Without manifests the last fault window's end stands in for the fault end."""
    assert (false_alarm_rates(delay_frame, CATEGORIES)["post_fault"]
            == false_alarm_rates(delay_frame, CATEGORIES, FAULT_ENDS)["post_fault"])


def test_missing_modality_nulls_only_that_modality():
    windows = make_windows()
    for modality in ("metrics", "traces", "logs", "graph"):
        degraded = with_modality_missing(windows, modality)
        for column in feature_columns(windows.columns):
            if modality_of(column) == modality:
                assert degraded[column].isna().all(), column
            else:
                pd.testing.assert_series_equal(degraded[column], windows[column])


def test_delayed_traces_take_the_previous_window_of_the_same_service():
    windows = make_windows()
    delayed = with_delayed_traces(windows, lag=1)
    assert (delayed.index == windows.index).all()
    key = (windows["experiment_id"] == "exp-train-cpu") & (windows["service"] == "cart")
    original = windows[key].sort_values("window_idx")["f_traces_latency_z"].to_numpy()
    shifted = delayed[key].sort_values("window_idx")["f_traces_latency_z"].to_numpy()
    assert np.isnan(shifted[0])
    assert shifted[1:] == pytest.approx(original[:-1])
    # A non-trace column is untouched.
    pd.testing.assert_series_equal(delayed["f_metrics_cpu_util_z"],
                                   windows["f_metrics_cpu_util_z"])


def test_ablation_keeps_only_the_requested_modalities():
    windows = make_windows()
    restricted = restrict_modalities(windows, ["metrics", "traces"])
    assert {modality_of(c) for c in feature_columns(restricted.columns)} == {
        "metrics", "traces"}
    assert "experiment_id" in restricted.columns and "is_root" in restricted.columns


def test_ece_of_a_perfectly_calibrated_vector_is_zero():
    confidence, correct = [], []
    for bin_index in range(10):
        probability = bin_index / 10 + 0.05
        hits = round(probability * 100)
        confidence += [probability] * 100
        correct += [True] * hits + [False] * (100 - hits)
    assert expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(0.0)


def test_ece_of_an_overconfident_vector_is_large():
    confidence = [0.95] * 100
    correct = [True] * 50 + [False] * 50
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.45)


def test_evaluate_reports_every_section(tmp_path):
    windows = make_windows()
    manifests = make_manifests()
    train = windows[windows["experiment_id"].str.contains("train")]
    val = windows[windows["experiment_id"].str.contains("val")]
    test = windows[windows["experiment_id"].str.contains("test")]
    model = MODELS["logreg"]().fit(train, val)

    result = evaluate(model, test, manifests)
    assert set(result) >= {"detection", "detection_delay", "false_alarms", "localization",
                           "classification", "calibration", "cost"}
    assert 0.0 <= result["detection"]["f1"] <= 1.0
    assert result["cost"]["ms_per_window"] > 0.0
    assert result["cost"]["rss_mb"] > 0.0
    # Stage 2 calibration is scored over alarming windows, false alarms included.
    assert result["calibration"]["top1"]["n"] == int(
        result["detection"]["alarm_rate"] * result["detection"]["n_windows"] + 0.5
    )

    table = comparison_table({"logreg": result})
    assert list(table.index) == ["logreg"]
    assert write_results({"logreg": result}, tmp_path).exists()
    assert write_markdown({"logreg": result}, tmp_path, "t").read_text().startswith("# t")


def test_experiment_categories_falls_back_to_the_traffic_profile():
    manifests = make_manifests(DEFAULT_SPECS)
    for manifest in manifests:
        manifest.extra = {}
    categories = experiment_categories(manifests)
    assert categories["exp-train-cpu"] == "fault"
    assert categories["exp-train-normal"] == "normal"
    assert categories["exp-val-normal"] == "traffic_spike"


# --- debounce --------------------------------------------------------------------------
def test_debounce_requires_consecutive_positive_windows(delay_frame):
    debounced = apply_debounce(delay_frame, 2)
    flags = debounced.set_index(["experiment_id", "window_idx"])["detect"]
    # exp-a raw:       T T F T | F F T T T | T F T
    # exp-a at K = 2:  F T F F | F F F T T | T F F
    assert list(flags.loc["exp-a"]) == [
        False, True, False, False, False, False, False, True, True, True, False, False
    ]
    # exp-c fires exactly once, so at K = 2 it never alarms at all.
    assert not any(flags.loc["exp-c"])
    assert (apply_debounce(delay_frame, 1)["detect"] == delay_frame["detect"]).all()


def test_debounce_trades_delay_for_false_alarms(delay_frame):
    debounced = apply_debounce(delay_frame, 2)
    delay = detection_delay(debounced)
    # exp-a's first sustained window is index 7, which ends at 80 s.
    assert delay["delays_s"] == [35.0]
    assert delay["onset_delays_s"] == [25.0]
    assert delay["n_detected"] == 1
    assert delay["undetected_fraction"] == pytest.approx(2 / 3)

    rates = false_alarm_rates(debounced, CATEGORIES, FAULT_ENDS)
    assert rates["normal"]["alarms"] == 1                 # was 2 at K = 1
    assert rates["normal"]["per_hour"] == pytest.approx(30.0)
    assert rates["traffic_spike"]["alarms"] == 1
    assert rates["pre_fault"]["alarms"] == 1              # was 2 at K = 1
    assert rates["post_fault"]["alarms"] == 1             # was 2 at K = 1
    assert rates["all_fault_free"]["alarms"] == 4         # was 7 at K = 1


def test_evaluate_records_the_debounce_it_used():
    windows = make_windows()
    train = windows[windows["experiment_id"].str.contains("train")]
    val = windows[windows["experiment_id"].str.contains("val")]
    test = windows[windows["experiment_id"].str.contains("test")]
    model = MODELS["logreg"]().fit(train, val)
    strict = evaluate(model, test, make_manifests(), debounce=3, profile=False)
    assert strict["debounce"] == 3
    assert strict["detection"]["alarm_rate"] <= evaluate(
        model, test, make_manifests(), debounce=1, profile=False
    )["detection"]["alarm_rate"]


# --- per-incident localization, symptomatic recall, exposure, bootstrap ------------------
def test_incident_localization_uses_the_first_alarming_window(delay_frame):
    """The fake predictions always name SERVICES[0]; the planted root is "cart"."""
    frame = delay_frame.copy()
    result = incident_localization(frame)
    assert result["n_incidents"] == 3
    # frontend is always ranked first, so no incident is localized correctly...
    assert result["top1_incident"] == pytest.approx(0.0)

    # ... unless the first alarming window names the truth. Give exp-a a correct first
    # verdict and a wrong majority, and the two numbers separate.
    fault = frame["is_fault_window"] & (frame["experiment_id"] == "exp-a")
    first = frame.index[fault & (frame["window_idx"] == 6)]
    frame.loc[first, "root_top1"] = "cart"
    result = incident_localization(frame)
    assert result["top1_incident"] == pytest.approx(1 / 3)
    assert result["top1_incident_majority"] == pytest.approx(0.0)
    # An undetected incident counts as wrong rather than being skipped.
    assert result["n_incidents"] == 3


def test_symptomatic_recall_excludes_the_onset_window(delay_frame):
    # exp-a fault windows are 4..8 with the fault at 45 s: windows 6, 7, 8 start at least
    # 10 s in, and exp-a is flagged in all three.
    only_a = delay_frame[delay_frame["experiment_id"] == "exp-a"]
    assert symptomatic_recall(only_a) == pytest.approx(1.0)
    # Window 4 and 5 are within the first 10 s, and exp-a is flagged in neither.
    assert only_a[only_a["is_fault_window"]]["detect"].mean() == pytest.approx(3 / 5)


def test_target_exposure_counts_requests_to_the_root_during_the_fault():
    windows = make_windows()
    exposure = target_exposure(windows)
    # 5 rps x 10 s windows x 5 fault windows for a normally-loaded incident.
    assert exposure["exp-test-cpu"] == pytest.approx(250.0)
    assert exposure["exp-test-quiet"] == pytest.approx(2.5)
    assert "exp-test-normal" not in exposure          # no root rows at all


def test_exposure_strata_split_the_incidents(tmp_path):
    windows = make_windows()
    manifests = make_manifests()
    train = windows[windows["experiment_id"].str.contains("train")]
    val = windows[windows["experiment_id"].str.contains("val")]
    test = windows[windows["experiment_id"].str.contains("test")]
    result = evaluate(MODELS["logreg"]().fit(train, val), test, manifests, profile=False)
    assert set(result["exposure"]) == {"exposure_lt_10", "exposure_ge_10"}
    assert result["exposure"]["exposure_lt_10"]["n_experiments"] == 1
    assert result["exposure"]["exposure_ge_10"]["n_experiments"] == 1
    assert set(result["exposure"]["exposure_ge_10"]["per_fault_type"]) == {"cpu_saturation"}


def test_bootstrap_intervals_bracket_the_point_estimate(delay_frame):
    intervals = bootstrap_intervals(delay_frame, b=200, seed=0)
    assert set(intervals) == {"detect_F1", "auroc", "top1_oracle", "top1_incident",
                              "fa_all", "fault_macroF1"}
    for bounds in intervals.values():
        assert bounds["lo"] <= bounds["hi"]
    # Seeded, so two runs agree exactly.
    assert bootstrap_intervals(delay_frame, b=200, seed=0) == intervals
    assert bootstrap_intervals(delay_frame, b=200, seed=1) != intervals


def test_ablation_sets_cover_the_temporal_family():
    from rca.eval.ablation import MODALITY_SETS
    assert "temporal" in MODALITY_SETS["all"]
    assert "temporal" not in MODALITY_SETS["all-temporal"]
    assert set(MODALITY_SETS["all"]) - set(MODALITY_SETS["all-temporal"]) == {"temporal"}
    assert set(MODALITY_SETS["all"]) - set(MODALITY_SETS["all-graph"]) == {"graph"}


def test_missing_temporal_is_a_robustness_condition():
    windows = make_windows()
    degraded = with_modality_missing(windows, "temporal")
    assert degraded["f_temporal_traces_latency_z_mean3"].isna().all()
    assert degraded["f_traces_latency_z"].notna().any()
