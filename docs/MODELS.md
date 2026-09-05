# Modelling and evaluation design

## Problem framing

Input: the window table (`rca/features/schema.py`) — one row per (experiment, window,
service). The system answers, per window:

1. **Detection** — is a fault active? Stage 1 operates on a *system-level* window
   vector (the per-service rows aggregated: max/mean of key z-scores, counts of
   anomalous services) so it is independent of which service is failing.
2. **Localization** — which service is the root? Stage 2 scores every service row of a
   flagged window with a *candidate scorer* (binary: is_root) and ranks services by
   score. This is service-agnostic by construction: the model sees the service's own
   features + graph-relative features, never a service-identity one-hot. That is what
   makes held-out-service evaluation meaningful and what the real demo needs when a
   service is added.
3. **Fault classification** — multiclass over the 8 families, from the top-ranked
   service's row (train on true-root rows; at inference, on the predicted root's row).
4. **Confidence** — calibrated probabilities for each stage (isotonic / Platt fit on the
   validation fold), plus the ranking margin between top-1 and top-2 for localization.

## Baselines

- `rules`: the thresholds a monitoring stack ships with, on raw per-service features —
  latency ratio vs. the experiment's own warm-up > 2× (`f_traces_latency_ratio`), error
  rate > 5 %, cpu > 90 %, memory > 90 % of limit — plus the two dependency-side rules any
  service mesh gives you: inbound client error rate > 5 % and inbound unanswered fraction
  > 20 %. Detection = any rule holds for **2 consecutive windows** (the burn-rate pattern
  real alerting uses), applied inside the model so the comparison is on equal terms;
  localization = service with the most rules fired, tie-broken by "most downstream";
  fault class = rule → family mapping. Expressing latency as a *ratio* rather than a
  z-score matters: a z threshold fires on any tight baseline, which fired on 48 % of
  fault-free windows and made this a straw man.
- `signature`: an untrained normalized anomaly score — each of latency z, inbound client
  latency z, error rate / 5 %, cpu z, memory z, queue-depth z, log-error z and inbound
  unanswered / 20 % divided by the level at which it is clearly abnormal and clipped to
  [0, 1], then averaged. Detection = max over services, thresholded on the validation
  fold like `stat`; localization = argmax over services of that score minus the
  downstream explanation (on the same [0, 1] scale); fault class from a fixed
  component → family signature table.
- `stat`: robust z-score detector (any |z| > k on latency/error/cpu/mem/queue), with
  localization by largest z after subtracting the downstream explanation
  (`graph_explained_by_downstream`).
- `logreg`, `rf`, `xgb`: the same three-stage structure with learned stage models.

## Evaluation protocol

Splits are by experiment (`rca/data/splits.py`) and never random over windows.
Reported per split type: by-experiment (main), hold-out service, hold-out intensity
(high and low), hold-out traffic, by-time.

Metrics:

- Detection: per-window precision/recall/F1/AUROC; **incident detection delay** =
  time from fault start to the first window with a positive detection (per
  experiment, measured to the **end** of the first flagged window — the earliest moment a
  windowed detector could actually have emitted the verdict; the window-*start* number is
  kept alongside as `delay_onset_*` for reference; undetected incidents counted
  separately); **false alarms per hour**
  computed over *fault-free time*, meaning every window in which no fault is active:
  normal and traffic-only experiments, plus both the pre-fault warm-up **and the
  post-fault recovery** of fault experiments. Reported as `all_fault_free` and broken
  down per bucket (`normal`, `traffic_spike`, `pre_fault`, `post_fault`), because the
  post-fault tail is where a detector that latches misfires most and omitting it
  flatters the headline rate. Telemetry stays genuinely abnormal for a few seconds
  after a fault is lifted (queues drain, caches refill, restarted processes warm up),
  so the tail is also reported as `post_fault_excl_30s`, skipping that recovery ramp.
  Debounce policy: an alarm is a transition from negative to positive in the
  experiment's full window series, so consecutive positives are one alarm and a
  detector still firing as the fault ends is an alarm that has not cleared rather than
  a fresh false alarm. Detection itself can additionally require K consecutive positive
  windows (`--debounce K`), which trades detection delay against this rate.
- Localization: **per incident** — `top1_incident` is the fraction of incidents whose
  *first alarming window* names the true root (an undetected incident counts as wrong),
  with `top1_incident_majority` the majority verdict over the incident's flagged windows.
  That is the number an on-call engineer experiences; window-level top-1 lets a long
  incident hide a wrong first verdict. Also reported per window: top-1 / top-3 over fault
  windows where detection fired, and over all fault windows (oracle detection), per fault
  family.
- Detection is additionally reported as `recall_symptomatic`, over fault windows starting
  at least 10 s after the fault began, so reaction time is not scored as blindness; and
  stratified by **target exposure** — the number of requests that reached the root service
  while its fault ran — because a fault on a service nothing was calling leaves almost no
  trace and a miss there is a property of the traffic, not of the model.
- Classification: per-class P/R/F1 and macro-F1 on true-root rows and on predicted-root
  rows.
- Calibration: ECE and reliability bins per stage. Stage 2 is scored as `ece_top1` — the
  top-1 root score against top-1 correctness over every window that raised an alarm,
  false alarms included and counted as wrong — because that is the number an operator is
  shown, and a confident root guess on a window with no fault is exactly the failure a
  calibration metric should expose.
- Every headline number carries a 95 % percentile confidence interval from a **cluster
  bootstrap whose resampling unit is the test experiment** (B = 500, seeded). Windows
  inside one incident are anything but independent, so resampling windows would produce
  intervals several times too narrow.
- Latency: inference wall time per window (feature build + model), resident memory.
- Robustness: evaluation with each modality's columns set to NaN (missing telemetry),
  and with traces delayed by one window (features computed from the previous window's
  traces).
- Ablations: retrain with modality subsets {metrics}, {traces}, {logs},
  {metrics+traces}, {all − graph}, {all − temporal}, {all}.

All runs are logged to MLflow (local `mlruns/`) with params, metrics, and the split
description as an artifact.
