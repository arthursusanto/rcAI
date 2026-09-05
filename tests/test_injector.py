"""Dry-run command generation for the eight fault families."""
from __future__ import annotations

import json
import sys

import pytest

from rca.benchmark import injector as inj
from rca.data import schema

HOLD_NS = 120_000_000_000


def fault(fault_type: str, target: str, intensity: float = 0.5) -> schema.Fault:
    return schema.Fault(fault_type=fault_type, target=target,
                        start_ns=1_700_000_000_000_000_000,
                        end_ns=1_700_000_000_000_000_000 + HOLD_NS,
                        intensity=intensity)


@pytest.fixture
def dry():
    return inj.FaultInjector(dry_run=True)


@pytest.fixture
def quota():
    """The squeeze on its own, kept selectable for comparison against the default."""
    return inj.FaultInjector(dry_run=True, cpu_mechanism="quota-only")


@pytest.fixture
def hog_only():
    """The burner on its own, likewise."""
    return inj.FaultInjector(dry_run=True, cpu_mechanism="hog-only")


# --- allowed targets ------------------------------------------------------------------
@pytest.mark.parametrize("fault_type", schema.FAULT_TYPES)
def test_allowed_targets_are_real_services(fault_type):
    targets = inj.allowed_targets(fault_type)
    assert targets, f"{fault_type} has no injectable target"
    assert set(targets) <= set(schema.SERVICES)


def test_allowed_targets_reflect_what_the_demo_supports():
    assert inj.allowed_targets("error_rate") == ["cart", "payment", "ad"]
    assert inj.allowed_targets("queue_backlog") == ["checkout", "accounting", "fraud-detection"]
    assert inj.allowed_targets("cache_slowdown") == [
        "cart", "product-catalog", "recommendation"]
    # accounting shares astronomy-db with product-catalog, which serves far more
    # traffic, so netem on the store degrades product-catalog and the label would name
    # the wrong service.
    assert "accounting" not in inj.allowed_targets("cache_slowdown")
    # Entry points are excluded: pausing them leaves an experiment with no telemetry.
    assert "frontend-proxy" not in inj.allowed_targets("dependency_failure")
    # The Kafka-only consumers are excluded too: `docker pause` on them is exactly the
    # queue_backlog injection, so the two labels would share identical telemetry.
    assert "accounting" not in inj.allowed_targets("dependency_failure")
    assert "fraud-detection" not in inj.allowed_targets("dependency_failure")
    # Distroless images have no shell, so no balloon can be exec'd into them.
    assert "checkout" not in inj.allowed_targets("memory_leak")
    # Both hog mechanisms run a burner inside the target with `docker exec`, so they
    # reach exactly the services that have a shell...
    assert set(inj.allowed_targets("cpu_saturation")) == set(inj.SHELL_SERVICES)
    assert set(inj.allowed_targets("cpu_saturation", "hog-only")) == set(inj.SHELL_SERVICES)
    assert "checkout" not in inj.allowed_targets("cpu_saturation")
    # ... while the squeeze on its own is a daemon operation and reaches all 15.
    assert set(inj.allowed_targets("cpu_saturation", "quota-only")) == set(schema.SERVICES)


@pytest.mark.parametrize("fault_type", schema.FAULT_TYPES)
def test_every_family_plans_a_reversible_action(dry, fault_type):
    target = inj.allowed_targets(fault_type)[0]
    f = fault(fault_type, target)
    assert dry.plan(f), f"{fault_type} planned no command"
    assert dry.plan_clear(f), f"{fault_type} planned no clear command"
    assert "mechanism" in dry.params(f)


def test_rejects_a_target_the_family_cannot_reach(dry):
    with pytest.raises(ValueError, match="cannot be injected"):
        dry.plan(fault("error_rate", "shipping"))


# --- per family -----------------------------------------------------------------------
def test_cpu_hog_burns_a_duty_cycled_share_inside_the_target(hog_only):
    dry = hog_only
    command = dry.plan(fault("cpu_saturation", "quote", 0.3))[0]
    # `docker exec` puts the burner in the target's own cgroup, so its CPU is charged
    # to the target and comes out of the same 1.0 CPU the service has.
    assert command.startswith("docker exec -d quote sh -c ")
    assert inj.HOG_MARKER in command and inj.HOG_BURN_MARKER in command
    # Exiting on INT/TERM, not merely handling them: a shell that only handles SIGTERM
    # carries on running, which left the controller alive after a clear.
    assert "trap 'exit 143' INT TERM" in command
    assert "trap 'kill -CONT $p 2>/dev/null; kill $p 2>/dev/null' EXIT" in command
    # CONT before TERM: half of every duty cycle leaves the burner SIGSTOPped, and a
    # stopped process cannot take a SIGTERM.
    assert dry.plan_clear(fault("cpu_saturation", "quote", 0.3)) == [
        f"docker exec quote pkill -CONT -f {inj.HOG_BURN_MARKER}",
        f"docker exec quote pkill -f {inj.HOG_MARKER}",
        f"docker exec quote pkill -f {inj.HOG_BURN_MARKER}"]


def test_cpu_hog_duty_cycle_spans_the_intensity_range(hog_only):
    dry = hog_only
    duties = {i: dry.params(fault("cpu_saturation", "quote", i))["duty"]
              for i in (0.0, 0.3, 0.5, 1.0)}
    assert duties[0.3] == pytest.approx(0.39, abs=0.01)     # ~40% of one core
    assert duties[1.0] == pytest.approx(0.95, abs=0.01)
    assert list(duties.values()) == sorted(duties.values())
    # The period is split between running and stopped, and neither half vanishes.
    for intensity, duty in duties.items():
        params = dry.params(fault("cpu_saturation", "quote", intensity))
        assert params["on_s"] + params["off_s"] == pytest.approx(inj.HOG_PERIOD_S)
        assert params["on_s"] > 0 and params["off_s"] > 0, intensity
        assert params["on_s"] == pytest.approx(inj.HOG_PERIOD_S * duty, abs=1e-3)


def test_cpu_hog_cannot_outlive_its_experiment(hog_only):
    dry = hog_only
    # A clear that never runs (crash, Ctrl-C) must not leave a core burning. The burner
    # watches its own clock rather than trusting anything else to stop it.
    f = fault("cpu_saturation", "quote", 1.0)
    hold_s = (f.end_ns - f.start_ns) // 1_000_000_000
    params = dry.params(f)
    assert params["duration_s"] == hold_s
    assert params["deadline_s"] == hold_s + inj.HOG_GRACE_S
    burner = dry.plan(f)[0].split(inj.HOG_BURN_MARKER, 1)[1]
    assert f"$(date +%s)+{hold_s + inj.HOG_GRACE_S}" in burner
    assert f"$i -lt {inj.HOG_SPIN_ITERATIONS}" in burner


def test_cpu_quota_squeezes_the_limit_relative_to_measured_usage(quota):
    # dry_run cannot sample a container, so it plans against a nominal 0.2 cores.
    assert quota.plan(fault("cpu_saturation", "frontend", 0.0)) == [
        "docker update --cpus 0.16 frontend"]        # 80% of 0.2
    assert quota.plan(fault("cpu_saturation", "frontend", 1.0)) == [
        "docker update --cpus 0.02 frontend"]        # 10% of 0.2
    # Clearing restores the compose baseline. `--cpus 0` is ignored by the daemon, so
    # there is no way back to "unlimited" -- and a constant denominator is the point.
    assert quota.plan_clear(fault("cpu_saturation", "frontend", 1.0)) == [
        "docker update --cpus 1 frontend"]


def test_hog_plus_quota_pins_the_cgroup_at_the_limit(dry):
    # The hog alone moves cpu_util but does not slow a service that has spare headroom;
    # the squeeze alone is a no-op on one that is idle. The default does both: intensity
    # picks the limit and the burner runs continuously inside it, so the cgroup sits at
    # the limit whatever the service was doing -- which is what makes cpu_util ~1.
    busy = fault("cpu_saturation", "cart", 0.3)
    busy.params["measured_cpu_cores"] = 0.30
    params = dry.params(busy)
    assert params["cpus"] == 0.575                           # lerp(0.8, 0.05, 0.3)
    assert params["duty"] == 1.0                             # continuous
    assert params["expected_service_share_cores"] == 0.2875  # Q shared at equal weight
    assert dry.plan(busy)[0] == "docker update --cpus 0.575 cart"
    assert dry.plan(busy)[1].startswith("docker exec -d cart sh -c ")

    # No duty cycle on the squeeze path: a SIGSTOP/SIGCONT fraction is a share of wall
    # time, but the limit already caps the burner while it runs, so a duty of d would
    # deliver d x Q cores and cpu_util would land at d, not 1.
    burn = dry.plan(busy)[1]
    assert inj.HOG_BURN_MARKER in burn
    assert inj.HOG_MARKER not in burn
    assert "kill -STOP" not in burn
    assert "period_s" not in params


def test_hog_plus_quota_tightens_the_limit_as_intensity_rises(dry):
    # The whole point of this parameterisation: the stall a request can hit is
    # (1 - Q) x one CFS period, so Q has to *fall* with intensity. Driving the duty
    # instead made Q rise, and the hardest faults produced the mildest traces.
    rows = []
    for intensity in (0.0, 0.3, 0.75, 1.0):
        f = fault("cpu_saturation", "quote", intensity)
        f.params["measured_cpu_cores"] = 0.001
        p = dry.params(f)
        rows.append((p["cpus"], p["expected_service_share_cores"],
                     p["expected_max_stall_ms"]))
    quotas, shares, stalls = zip(*rows)
    assert list(quotas) == sorted(quotas, reverse=True), quotas
    assert list(shares) == sorted(shares, reverse=True), shares
    assert list(stalls) == sorted(stalls), stalls
    assert stalls == (20.0, 42.5, 76.2, 95.0)
    # Q is never squeezed below what docker will accept.
    hardest = fault("cpu_saturation", "quote", 1.0)
    hardest.params["measured_cpu_cores"] = 0.001
    assert dry.params(hardest)["cpus"] >= inj.MIN_CPUS


def test_the_hardest_setting_starves_a_busy_service(dry):
    busy = fault("cpu_saturation", "cart", 1.0)
    busy.params["measured_cpu_cores"] = 0.30
    params = dry.params(busy)
    assert params["cpus"] == 0.05
    # 0.025 cores against the 0.30 it was using: a 12x starvation, plus a 95 ms stall.
    assert params["expected_service_share_cores"] == 0.025
    assert params["expected_max_stall_ms"] == 95.0
    assert dry.plan(busy)[0] == "docker update --cpus 0.05 cart"
    assert dry.plan_clear(busy)[-1] == "docker update --cpus 1 cart"


def test_hog_plus_quota_unwinds_in_the_opposite_order(dry):
    f = fault("cpu_saturation", "quote", 0.75)
    f.params["measured_cpu_cores"] = 0.001
    # Set the box, then fill it.
    assert dry.plan(f)[0] == "docker update --cpus 0.238 quote"
    # Empty it, then take the box away: restoring the limit first would just feed the
    # burner the CPU the service is meant to get back.
    clear = dry.plan_clear(f)
    assert clear[0] == f"docker exec quote pkill -CONT -f {inj.HOG_BURN_MARKER}"
    assert clear[-1] == "docker update --cpus 1 quote"


def test_hog_only_leaves_the_limit_alone(hog_only):
    f = fault("cpu_saturation", "quote", 0.75)
    assert not any("update --cpus" in c for c in hog_only.plan(f))
    assert not any("update --cpus" in c for c in hog_only.plan_clear(f))
    assert hog_only.params(f)["cpus"] == inj.BASELINE_CPUS


def test_an_unknown_cpu_mechanism_is_rejected():
    with pytest.raises(ValueError, match="cpu_mechanism"):
        inj.FaultInjector(cpu_mechanism="stress-ng")


def test_cpu_saturation_scales_with_the_sampled_usage_and_has_a_floor(quota):
    busy = fault("cpu_saturation", "payment", 1.0)
    busy.params["measured_cpu_cores"] = 0.65        # what `docker stats` reported
    assert quota.plan(busy) == ["docker update --cpus 0.065 payment"]      # 0.65 x 0.10
    assert quota.params(busy)["measured_cpu_cores"] == 0.65
    # An almost idle container is squeezed as if it wanted CPU_FLOOR_CORES. Without that
    # the limit would land on MIN_CPUS at every intensity above ~0.3 and the fault would
    # stop varying with its own intensity label -- 11 of the 15 services idle that low.
    idle = fault("cpu_saturation", "shipping", 0.0)
    idle.params["measured_cpu_cores"] = 0.003
    assert quota.plan(idle) == ["docker update --cpus 0.12 shipping"]      # 0.15 x 0.80
    assert quota.params(idle)["squeeze_from_cores"] == inj.CPU_FLOOR_CORES
    # ... and only the very top of the range still reaches the floor docker imposes.
    hardest = fault("cpu_saturation", "shipping", 1.0)
    hardest.params["measured_cpu_cores"] = 0.003
    assert quota.plan(hardest) == [f"docker update --cpus {inj.MIN_CPUS:g} shipping"]


def test_cpu_saturation_intensity_actually_varies_the_limit(quota):
    # The defect this guards: with a bare `max(0.02, used x lerp)` every intensity above
    # about 0.3 produced an identical 0.02 CPU limit for an idle service.
    limits = []
    for intensity in (0.0, 0.25, 0.5, 0.75, 1.0):
        f = fault("cpu_saturation", "shipping", intensity)
        f.params["measured_cpu_cores"] = 0.01
        limits.append(quota.params(f)["cpus"])
    assert limits == sorted(limits, reverse=True)
    assert len(set(limits)) >= 4, limits


def test_reset_puts_everything_back_without_being_told_what_broke(tmp_path):
    path = tmp_path / "demo.flagd.json"
    path.write_text(json.dumps(FLAGS, indent=2), newline="\n")
    live = inj.FaultInjector(flags_path=path, dry_run=True)
    live.reset(vus=25)
    commands = live.commands

    # Every container unpaused and every application service back to the 1.0 CPU
    # baseline, in one invocation each rather than 15 round trips.
    assert commands[0].startswith("docker unpause ")
    assert "payment" in commands[0] and "valkey-cart" in commands[0]
    restore = next(c for c in commands if c.startswith("docker update --cpus 1 "))
    for service in schema.SERVICES:
        assert f" {service}" in restore
    # Hogs die before the CPU is handed back, or the restore just feeds them.
    assert commands.index(restore) > commands.index(
        f"docker exec quote pkill -f {inj.HOG_BURN_MARKER}")
    # A netem sidecar per container: netem lives in one netns at a time.
    deletes = [c for c in commands if "tc qdisc del" in c]
    assert len(deletes) == len(set(inj.CONTAINERS.values()))
    # Any cpu_saturation hog left behind by an interrupted experiment dies here rather
    # than burning a core into the next one until its `timeout` fires.
    hogs = [c for c in commands if f"pkill -f {inj.HOG_MARKER}" in c]
    assert len(hogs) == len(inj.SHELL_SERVICES)
    conts = [c for c in commands if f"pkill -CONT -f {inj.HOG_BURN_MARKER}" in c]
    assert len(conts) == len(inj.SHELL_SERVICES)
    assert commands[-1].startswith("flag reset ")


def test_reset_writes_every_flag_back_to_its_baseline(tmp_path):
    path = tmp_path / "demo.flagd.json"
    path.write_text(json.dumps(FLAGS, indent=2), newline="\n")
    live = inj.FaultInjector(flags_path=path)
    live.inject(fault("error_rate", "cart", 0.5))
    live.set_traffic(vus=5, enabled=False)

    live.reset(vus=25)
    flags = json.loads(path.read_text())["flags"]
    assert flags["cartFailure"]["defaultVariant"] == "off"
    # Traffic must come back on, and the VU level is whatever the campaign asked for.
    assert flags["loadGeneratorTraffic"]["defaultVariant"] == "on"
    assert flags["loadGeneratorVUs"]["defaultVariant"] == "25"
    # Called with no vus, the configured level is left alone.
    live.reset()
    assert json.loads(path.read_text())["flags"]["loadGeneratorVUs"]["defaultVariant"] == "25"


def test_flag_write_is_atomic_and_keeps_the_file_shape(tmp_path):
    path = tmp_path / "demo.flagd.json"
    path.write_text(json.dumps(FLAGS, indent=4), newline="\n")
    live = inj.FaultInjector(flags_path=path)
    live.inject(fault("error_rate", "cart", 0.5))

    text = path.read_text()
    assert json.loads(text)["flags"]["cartFailure"]["defaultVariant"] == "50%"
    # The 4-space indent of the original survives the rewrite.
    assert '\n    "flags"' in text
    assert "\r" not in text
    # os.replace leaves no debris beside the file for flagd to try to parse.
    assert [p.name for p in tmp_path.iterdir()] == ["demo.flagd.json"]


def test_clear_records_commands_that_failed(tmp_path):
    # A docker binary that does not exist fails every command; clearing must not raise,
    # but it must say so rather than leave the caller believing the stack is clean.
    live = inj.FaultInjector(docker=(sys.executable, "-c", "import sys; sys.exit(3)"))
    f = fault("dependency_failure", "shipping")
    live.clear(f)
    failures = f.params["clear_failures"]
    assert len(failures) == 1
    assert failures[0]["returncode"] == 3
    assert "unpause shipping" in failures[0]["command"]


def test_clear_reuses_the_inject_time_cpu_measurement():
    # inject() records the measurement on the fault, so clear() re-plans without
    # sampling the already-throttled container: this injector is not dry-run and its
    # docker binary does not exist, so any real `docker stats` call would raise.
    live = inj.FaultInjector(docker=("definitely-not-docker",), cpu_mechanism="quota-only")
    f = fault("cpu_saturation", "quote", 1.0)
    f.params["measured_cpu_cores"] = 0.4
    assert live.plan_clear(f) == ["definitely-not-docker update --cpus 1 quote"]
    assert live.plan(f) == ["definitely-not-docker update --cpus 0.04 quote"]


def test_memory_leak_uses_the_demo_flag_for_email(dry):
    f = fault("memory_leak", "email", 1.0)
    assert dry.plan(f) == ["flag emailMemoryLeak=10000x"]
    assert dry.plan_clear(f) == ["flag emailMemoryLeak=off"]
    assert dry.plan(fault("memory_leak", "email", 0.0)) == ["flag emailMemoryLeak=1x"]


def test_memory_leak_balloons_shell_capable_services(dry):
    f = fault("memory_leak", "recommendation", 1.0)
    command = dry.plan(f)[0]
    assert command.startswith("docker exec -d recommendation sh -c ")
    assert inj.BALLOON_MARKER in command
    params = dry.params(f)
    # Headroom is (500 MiB limit - 6 MiB floor) / 2 = 247, and intensity 1.0 takes all
    # of it, so 2x the balloon plus the service floor still fits under the limit.
    assert params["target_mb"] == 247
    assert 2 * params["target_mb"] + inj.BALLOON_FLOOR_MB <= inj.MEMORY_LIMIT_MB[
        "recommendation"]
    assert params["chunk_bytes"] == 247 * 1024 * 1024 // 120
    assert dry.plan_clear(f) == [f"docker exec recommendation pkill -f {inj.BALLOON_MARKER}"]


def test_balloon_never_exceeds_a_small_service_memory_limit(dry):
    # currency's limit is 20 MiB: an uncapped 45%-of-limit balloon would peak at 18 MiB
    # of a 20 MiB cgroup and OOM-kill it.
    for service in inj.allowed_targets("memory_leak"):
        if service == "email":
            continue        # flag-driven, no balloon
        limit = inj.MEMORY_LIMIT_MB[service]
        for intensity in (0.0, 0.5, 1.0):
            target = dry.params(fault("memory_leak", service, intensity))["target_mb"]
            assert 2 * target + inj.BALLOON_FLOOR_MB <= limit, (service, intensity)


def test_network_latency_uses_a_netem_sidecar_by_default(dry):
    assert dry.plan(fault("network_latency", "cart", 0.0)) == [
        ("docker run --rm --network container:cart --cap-add NET_ADMIN "
         "nicolaka/netshoot:latest tc qdisc replace dev eth0 root netem "
         "delay 50ms 5ms distribution normal")]
    assert dry.plan(fault("network_latency", "cart", 1.0))[0].endswith(
        "netem delay 800ms 80ms distribution normal")
    assert dry.plan_clear(fault("network_latency", "cart"))[0].endswith(
        "tc qdisc del dev eth0 root")


def test_network_latency_exec_mode_needs_no_sidecar():
    exec_injector = inj.FaultInjector(dry_run=True, netem_mode="exec")
    assert exec_injector.plan(fault("network_latency", "cart", 0.0)) == [
        ("docker exec cart tc qdisc replace dev eth0 root netem "
         "delay 50ms 5ms distribution normal")]


def test_packet_loss_scales_from_2_to_40_percent(dry):
    assert dry.plan(fault("packet_loss", "shipping", 0.0))[0].endswith("netem loss 2.0%")
    assert dry.plan(fault("packet_loss", "shipping", 1.0))[0].endswith("netem loss 40.0%")


def test_dependency_failure_prefers_the_flag_then_falls_back_to_pause(dry):
    assert dry.plan(fault("dependency_failure", "payment")) == ["flag paymentUnreachable=on"]
    assert dry.plan(fault("dependency_failure", "shipping")) == ["docker pause shipping"]
    assert dry.plan_clear(fault("dependency_failure", "shipping")) == ["docker unpause shipping"]


def test_error_rate_picks_a_fractional_variant(dry):
    assert dry.plan(fault("error_rate", "cart", 0.5)) == ["flag cartFailure=50%"]
    assert dry.plan(fault("error_rate", "payment", 1.0)) == ["flag paymentFailure=100%"]
    # adFailure is boolean, so intensity has no parameter to drive.
    assert dry.plan(fault("error_rate", "ad", 0.2)) == ["flag adFailure=on"]
    assert dry.params(fault("error_rate", "cart", 0.5))["error_fraction"] == 0.5


def test_queue_backlog(dry):
    assert dry.plan(fault("queue_backlog", "checkout")) == ["flag kafkaQueueProblems=on"]
    assert dry.plan(fault("queue_backlog", "accounting")) == ["docker pause accounting"]


def test_cache_slowdown_targets_the_backing_store(dry):
    assert "container:valkey-cart" in dry.plan(fault("cache_slowdown", "cart"))[0]
    # Only the service that dominates a shared store may claim it as its own.
    with pytest.raises(ValueError, match="cannot be injected"):
        dry.plan(fault("cache_slowdown", "accounting"))
    assert "container:astronomy-db" in dry.plan(fault("cache_slowdown", "product-catalog"))[0]
    assert dry.plan(fault("cache_slowdown", "recommendation")) == [
        "flag recommendationCacheFailure=on"]


# --- recording and flag writing --------------------------------------------------------
def test_dry_run_records_commands_and_touches_nothing(tmp_path, dry):
    f = fault("cpu_saturation", "quote", 0.3)
    dry.inject(f)
    dry.clear(f)
    assert len(dry.commands) == 6
    assert dry.commands[0].startswith("docker update --cpus ")
    assert dry.commands[1].startswith("docker exec -d quote sh -c ")
    assert dry.commands[2] == f"docker exec quote pkill -CONT -f {inj.HOG_BURN_MARKER}"
    assert dry.commands[-1] == "docker update --cpus 1 quote"
    # Injecting records the concrete parameters on the fault for reproducibility.
    assert f.params["cpus"] == 0.575                  # lerp(0.8, 0.05, 0.3)
    assert f.params["duty"] == 1.0                   # continuous inside the limit
    assert f.params["mechanism"] == "exec-cpu-hog-quota"


def test_quota_run_records_commands_and_touches_nothing(quota):
    f = fault("cpu_saturation", "quote", 1.0)
    quota.inject(f)
    quota.clear(f)
    assert quota.commands == ["docker update --cpus 0.02 quote",
                              "docker update --cpus 1 quote"]
    assert f.params["cpus"] == 0.02
    assert f.params["measured_cpu_cores"] == inj.NOMINAL_CPU_CORES
    assert f.params["mechanism"] == "docker-update-cpus"


FLAGS = {"flags": {
    "cartFailure": {"defaultVariant": "off", "state": "ENABLED",
                    "variants": {"off": 0, "50%": 0.5, "100%": 1}},
    "loadGeneratorVUs": {"defaultVariant": "5", "state": "ENABLED",
                         "variants": {"5": 5, "10": 10, "25": 25, "50": 50}},
    "loadGeneratorTraffic": {"defaultVariant": "on", "state": "ENABLED",
                             "variants": {"off": 0, "on": 1}},
}}


def test_flag_write_updates_the_default_variant(tmp_path):
    path = tmp_path / "demo.flagd.json"
    path.write_text(json.dumps(FLAGS), newline="\n")
    live = inj.FaultInjector(flags_path=path)

    live.inject(fault("error_rate", "cart", 0.5))
    assert json.loads(path.read_text())["flags"]["cartFailure"]["defaultVariant"] == "50%"
    live.clear(fault("error_rate", "cart", 0.5))
    assert json.loads(path.read_text())["flags"]["cartFailure"]["defaultVariant"] == "off"

    live.set_traffic(vus=25, enabled=True)
    flags = json.loads(path.read_text())["flags"]
    assert flags["loadGeneratorVUs"]["defaultVariant"] == "25"
    assert flags["loadGeneratorTraffic"]["defaultVariant"] == "on"


def test_unknown_variant_is_rejected(tmp_path):
    path = tmp_path / "demo.flagd.json"
    path.write_text(json.dumps(FLAGS), newline="\n")
    live = inj.FaultInjector(flags_path=path)
    with pytest.raises(ValueError, match="no variant"):
        live.inject(fault("error_rate", "cart", 0.1))     # "10%" is absent from FLAGS
    with pytest.raises(ValueError, match="loadGeneratorVUs variants"):
        live.set_traffic(vus=7)
