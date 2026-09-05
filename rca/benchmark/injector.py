"""Fault injection against a running OpenTelemetry Demo docker compose stack.

Three mechanisms are used, chosen per fault family because the demo's images are
heterogeneous (distroless, alpine, debian) and none of them ship ``stress-ng`` or
``tc``:

``docker``
    ``docker update --cpus`` (throttle), ``docker pause``/``unpause`` (make a
    dependency unreachable). Works on every container regardless of image.
``exec``
    ``docker exec <c> sh -c '<busy loop>'`` for in-container CPU burn and a memory
    balloon. Only usable where the image has a shell (see ``SHELL_SERVICES``);
    ``stress-ng`` is not present in any demo image, so a POSIX-sh loop is the
    fallback the plan calls for.
``netem``
    ``tc qdisc replace dev eth0 root netem ...``. No demo image contains ``tc``,
    so the default ``netem_mode="sidecar"`` runs a throwaway container joined to the
    target's network namespace (``--network container:<target> --cap-add NET_ADMIN``).
    ``netem_mode="exec"`` uses ``docker exec`` instead and needs both the NET_ADMIN
    capability from ``deploy/compose/docker-compose.override.yml`` and images
    rebuilt with iproute2.
``flag``
    Rewrites the demo's ``src/flagd/demo.flagd.json`` (bind-mounted into flagd,
    which watches the file). flagd's OFREP endpoint is evaluation-only -- the demo's
    own load generator reads flags through ``POST /ofrep/v1/evaluate/flags/<key>``
    -- so there is no HTTP way to *set* a flag; the file is the control surface,
    exactly as the flagd-ui does it.

Intensity is 0..1 and maps to concrete parameters as documented on each
``_plan_*`` method; the concrete values are written back into ``Fault.params``.

Every command is recorded in ``FaultInjector.commands``. With ``dry_run=True``
nothing is executed and no file is written, which is what the unit tests use.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from rca.data import schema

log = logging.getLogger(__name__)

# --- demo topology ------------------------------------------------------------------
# Container names from the demo's compose.yaml / compose.full.yaml (they equal the
# service names except for the two datastores).
CONTAINERS: dict[str, str] = {s: s for s in schema.SERVICES} | {
    "kafka": "kafka",
    "valkey": "valkey-cart",
    "postgresql": "astronomy-db",
    "flagd": "flagd",
    "load-generator": "load-generator",
}

# Services whose image has a POSIX shell, from the final FROM line of each
# src/<service>/Dockerfile. The rest are distroless and cannot be exec'd into at all.
SHELL_SERVICES: frozenset[str] = frozenset({
    "accounting",       # mcr.microsoft.com/dotnet/aspnet (debian)
    "ad",               # eclipse-temurin:*-jre (debian)
    "cart",             # mcr.microsoft.com/dotnet/runtime-deps (alpine)
    "currency",         # alpine
    "email",            # ruby (alpine)
    "frontend-proxy",   # envoyproxy/envoy (ubuntu)
    "image-provider",   # nginxinc/nginx-unprivileged (alpine)
    "quote",            # php:cli (alpine)
    "recommendation",   # python (alpine)
})

# deploy.resources.limits.memory from the demo's compose files, in MiB. Used to size
# the memory balloon so it does not OOM-kill the container it is meant to degrade.
MEMORY_LIMIT_MB: dict[str, int] = {
    "ad": 300, "cart": 160, "checkout": 20, "currency": 20, "email": 100,
    "frontend": 250, "frontend-proxy": 90, "image-provider": 120, "payment": 140,
    "product-catalog": 20, "quote": 40, "recommendation": 500, "shipping": 20,
    "accounting": 160, "fraud-detection": 300,
}

# Entry points: pausing them yields an experiment with no telemetry at all.
_NO_PAUSE = frozenset({"frontend", "frontend-proxy"})

# Kafka-only consumers. Pausing them is exactly how queue_backlog is injected, so
# offering the identical command as dependency_failure too would put two different
# labels on byte-identical telemetry. They stay with queue_backlog, which mirrors the
# simulator's constraint.
_CONSUMER_ONLY = frozenset({"accounting", "fraud-detection"})

# The backing store whose latency a service's cache_slowdown is injected on.
#
# accounting is deliberately absent even though it does use astronomy-db. netem is
# applied to the *store*, and astronomy-db is shared with product-catalog, which serves
# far more traffic: in a real campaign, cache_slowdown@accounting made product-catalog
# the visibly degraded service (infra latency z 17, latency z 7) while accounting barely
# moved -- so the experiment carried the wrong root-cause label. A shared backing store
# can only be a valid target for the service that dominates its traffic.
CACHE_BACKEND: dict[str, str] = {
    "cart": "valkey",                   # valkey-cart is cart's alone
    "product-catalog": "postgresql",    # dominates astronomy-db's traffic
}

DEFAULT_NETEM_IMAGE = "nicolaka/netshoot:latest"

# CPU limit every application service is started with by
# deploy/compose/docker-compose.override.yml (`deploy.resources.limits.cpus: "1"`, which
# Compose writes to HostConfig.NanoCpus -- the same field `docker update --cpus` sets).
# Clearing a cpu_saturation fault restores exactly this value: `docker update --cpus 0`
# does *not* remove a limit (the daemon ignores zero-valued fields), so there is no
# "unlimited" to go back to, and going back to a constant is the point anyway -- it keeps
# ingest's cpu_util denominator identical before, during and after the fault.
BASELINE_CPUS = 1.0

# Cores assumed for the squeeze when nothing can be measured (dry_run).
NOMINAL_CPU_CORES = 0.2

# `docker update` rejects --cpus below 0.01; stay clear of the edge.
MIN_CPUS = 0.02

# Floor on the usage the squeeze is computed *from*. 11 of the 15 services idle below
# 0.025 cores, so scaling their measured usage lands on MIN_CPUS at every intensity
# above ~0.3 and the fault stops varying with its own label. Pretending an idle service
# wants at least this much spreads the limit over 0.12 .. 0.02 cores instead.
CPU_FLOOR_CORES = 0.15

# `docker stats` is a one-second delta and a bursty service can read near zero in any
# single sample, so the squeeze is computed from an average of a few.
CPU_SAMPLES = 3
CPU_SAMPLE_GAP_S = 1.0

# --- cpu_saturation mechanisms -------------------------------------------------------
# "hog+quota"  a CPU burner inside the target's cgroup *and* a limit just above what the
#              burner takes, so the service is left with about half of what it needs and
#              CFS throttles it. The default: see _plan_cpu_hog for why neither half
#              works alone.
# "hog-only"   the burner without the squeeze -- moves cpu_util, but a service with spare
#              headroom is not actually slowed down.
# "quota-only" the squeeze without the burner -- a no-op on anything that idles below
#              the smallest limit docker will accept.
CPU_MECHANISMS = ("hog+quota", "hog-only", "quota-only")
DEFAULT_CPU_MECHANISM = "hog+quota"

# What the service keeps once the limit is set: half of what it was measured using, so
# it is throttled to roughly half speed, with a floor for a service that measured ~0.
HOG_HEADROOM_FRACTION = 0.5

# hog+quota: the CFS limit by intensity. Intensity drives the *limit*, and the burner's
# duty follows it, because the stall a request can hit is (1 - Q) x one CFS period -- so
# a Q that rose with intensity made the fault gentler the harder it was set, which is
# what the first parameterisation did (measured: quote p95 +7x at intensity 0.75 but
# unchanged at 1.0).
HOG_QUOTA_RANGE = (0.8, 0.05)

# The kernel's default CFS period, i.e. how long a cgroup that has spent its quota stays
# throttled at worst. Used only to record the stall the parameters imply.
CFS_PERIOD_MS = 100

# hog-only: share of one core the burner takes, by intensity. Unused by the default.
HOG_DUTY_RANGE = (0.15, 0.95)
HOG_PERIOD_S = 0.5              # one stop/cont cycle; 2 `sleep` forks per period
HOG_GRACE_S = 10                # hard stop, past the planned hold
# Two distinct markers so a clear can find each half on its own. Neither is a prefix of
# the other, and each pkill is its own argv (no `sh -c`), so the only process carrying a
# marker at clear time is pkill itself -- which excludes its own pid.
HOG_MARKER = "RCA_FAULT=hogctl"          # the duty-cycle controller
HOG_BURN_MARKER = "RCA_FAULT=hogburn"    # the process that actually burns the core
# Shell arithmetic iterations between wall-clock checks in the burner. Big enough that
# the `date` fork is noise, small enough to bound the overshoot past the deadline.
HOG_SPIN_ITERATIONS = 100_000

# Appears in the balloon process's command line so `pkill -f` can find it. `pkill`
# skips its own process, so passing the marker as its argument is safe.
BALLOON_MARKER = "RCA_FAULT=balloon"

# MiB left for the service itself when sizing a balloon against its memory limit.
BALLOON_FLOOR_MB = 6

# --- flagd feature flags (verified against src/flagd/demo.flagd.json) ----------------
FLAG_OFF = "off"
# Shared variant names of the fractional failure flags (cartFailure, paymentFailure).
_PCT_VARIANTS: tuple[tuple[str, float], ...] = (
    ("10%", 0.10), ("25%", 0.25), ("50%", 0.50),
    ("75%", 0.75), ("90%", 0.90), ("100%", 1.00),
)
_LEAK_VARIANTS: tuple[str, ...] = ("1x", "10x", "100x", "1000x", "10000x")
VU_VARIANTS: tuple[int, ...] = (5, 10, 25, 50)

# error_rate is only injectable where a failure flag exists. productCatalogFailure is
# deliberately absent: its targeting rule returns "off" in both branches, so setting
# defaultVariant has no effect, and it only fails one product id anyway.
ERROR_RATE_FLAGS: dict[str, str] = {
    "cart": "cartFailure",          # fractional
    "payment": "paymentFailure",    # fractional
    "ad": "adFailure",              # boolean
}


@dataclass(frozen=True)
class Action:
    """One reversible step: a subprocess invocation or a feature-flag write."""

    kind: str                       # "run" | "flag"
    argv: tuple[str, ...] = ()
    flag: str = ""
    variant: str = ""
    # Exit codes that mean "done", not "broken". `pkill` returns 1 when nothing matched,
    # which for a best-effort clear is the normal outcome -- the hog stops itself at its
    # own deadline, so a clear that arrives just after it has nothing left to kill.
    # Without this every cpu_saturation manifest would carry a clear_failures entry and
    # the field would stop meaning anything.
    ok_codes: tuple[int, ...] = (0,)

    def render(self) -> str:
        if self.kind == "flag":
            return f"flag {self.flag}={self.variant}"
        return " ".join(self.argv)


def _json_indent(text: str) -> int:
    """Indent width of the first indented line, so a rewrite keeps the file's shape."""
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if stripped and stripped != line:
            return len(line) - len(stripped)
    return 2


def _lerp(lo: float, hi: float, intensity: float) -> float:
    return lo + (hi - lo) * min(max(intensity, 0.0), 1.0)


def _pick(options: tuple, intensity: float):
    """Pick from an ordered ladder of variants by intensity (0 -> first, 1 -> last)."""
    idx = round(min(max(intensity, 0.0), 1.0) * (len(options) - 1))
    return options[idx]


def container_ip_map(docker: tuple[str, ...] = ("docker",)) -> dict[str, str]:
    """``{container ip: canonical service}`` for the containers in the label space.

    Go's gRPC instrumentation writes the *resolved address* into ``server.address``
    (``172.18.0.10``) where the JS SDK writes the hostname, so ingest cannot name the
    callee of a checkout -> payment / cart / currency / product-catalog span without
    this. The runner records it per experiment in ``manifest.extra["ip_map"]`` and
    ``ingest.normalize_service`` consults it.

    Containers that are not running are simply absent: `docker inspect` reports them on
    stderr and still prints the rest, so a partial map is better than an exception here.
    """
    by_container = {container: service for service, container in CONTAINERS.items()}
    result = subprocess.run(
        [*docker, "inspect", "--format",
         "{{.Name}} {{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
         *sorted(by_container)],
        check=False, capture_output=True, text=True)
    ip_map: dict[str, str] = {}
    for line in result.stdout.splitlines():
        name, *ips = line.split() or [""]
        service = by_container.get(name.lstrip("/"))
        if service:
            ip_map.update({ip: service for ip in ips})
    return ip_map


def allowed_targets(fault_type: str,
                    cpu_mechanism: str = DEFAULT_CPU_MECHANISM) -> list[str]:
    """Services a fault family can actually be injected on, in schema.SERVICES order."""
    if fault_type == "cpu_saturation":
        # A hog is started with `docker exec`, so it needs a shell in the image; the
        # quota squeeze is a daemon-level operation and reaches every container.
        allowed = (set(schema.SERVICES) if cpu_mechanism == "quota-only"
                   else set(SHELL_SERVICES))
    elif fault_type in ("network_latency", "packet_loss"):
        allowed = set(schema.SERVICES)
    elif fault_type == "memory_leak":
        allowed = set(SHELL_SERVICES) | {"email"}
    elif fault_type == "dependency_failure":
        allowed = set(schema.SERVICES) - _NO_PAUSE - _CONSUMER_ONLY
    elif fault_type == "error_rate":
        allowed = set(ERROR_RATE_FLAGS)
    elif fault_type == "queue_backlog":
        allowed = {"checkout", "accounting", "fraud-detection"}
    elif fault_type == "cache_slowdown":
        allowed = set(CACHE_BACKEND) | {"recommendation"}
    else:
        raise ValueError(f"unknown fault type {fault_type!r}")
    return [s for s in schema.SERVICES if s in allowed]


class FaultInjector:
    """Injects and clears the eight fault families on a running demo stack."""

    def __init__(
        self,
        flags_path: Path | str | None = None,
        dry_run: bool = False,
        netem_mode: str = "sidecar",
        netem_image: str = DEFAULT_NETEM_IMAGE,
        netem_interface: str = "eth0",
        docker: tuple[str, ...] = ("docker",),
        cpu_mechanism: str = DEFAULT_CPU_MECHANISM,
    ) -> None:
        if netem_mode not in ("sidecar", "exec"):
            raise ValueError(f"netem_mode must be 'sidecar' or 'exec', got {netem_mode!r}")
        if cpu_mechanism not in CPU_MECHANISMS:
            raise ValueError(f"cpu_mechanism must be one of {CPU_MECHANISMS}, "
                             f"got {cpu_mechanism!r}")
        self.flags_path = Path(flags_path) if flags_path is not None else None
        self.dry_run = dry_run
        self.netem_mode = netem_mode
        self.netem_image = netem_image
        self.netem_interface = netem_interface
        self.docker = tuple(docker)
        self.cpu_mechanism = cpu_mechanism
        self.commands: list[str] = []

    # --- public API ------------------------------------------------------------
    def inject(self, fault: schema.Fault) -> None:
        params, actions, _ = self._plan(fault)
        fault.params.update(params)
        self._apply(actions, check=True)

    def clear(self, fault: schema.Fault) -> None:
        _, _, actions = self._plan(fault)
        # Clearing is best-effort: deleting a qdisc that is not there, or unpausing a
        # container that is already running, must not abort the experiment. A clear that
        # really did fail is recorded on the fault (and so reaches the manifest) instead
        # of leaving the next experiment quietly contaminated.
        failures = self._apply(actions, check=False)
        if failures:
            fault.params["clear_failures"] = failures
            log.warning("clearing %s on %s left %d command(s) failing: %s",
                        fault.fault_type, fault.target, len(failures),
                        "; ".join(f["command"] for f in failures))

    def reset(self, vus: int | None = None) -> None:
        """Force the stack back to its fault-free baseline, whatever state it is in.

        Unconditional and idempotent, because it has to run after a crash or a Ctrl-C
        when there is no fault object left to clear: every container is unpaused, every
        netem qdisc is deleted, every application service gets its 1.0 CPU baseline back
        and every feature flag goes to its off/baseline variant. Without it, one
        interrupted experiment silently mislabels every experiment after it.

        ``vus`` pins ``loadGeneratorVUs``; leaving it None keeps whatever is configured.
        """
        containers = sorted(set(CONTAINERS.values()))
        # One invocation each: the daemon takes a list, and 15 round trips per
        # experiment is not free.
        self._apply([self._run("unpause", *containers)], check=False)
        # Hogs first: restoring the limit while one is still burning just feeds it.
        self._apply_parallel([
            action for service in sorted(SHELL_SERVICES)
            for action in self._hog_kill(CONTAINERS[service])])
        self._apply([self._run("update", "--cpus", f"{BASELINE_CPUS:g}",
                               *[CONTAINERS[s] for s in schema.SERVICES])], check=False)
        # netem lives in one network namespace at a time, so this really is one sidecar
        # per container -- run them together or a reset costs half a minute.
        self._apply_parallel([self._netem_del(c) for c in containers])
        self.reset_flags(vus)

    def reset_flags(self, vus: int | None = None) -> None:
        """Every flag to its baseline variant, in a single rewrite of the file."""
        if self.flags_path is None:
            return
        data = json.loads(self.flags_path.read_text())
        variants = {}
        for name, flag in data["flags"].items():
            if name == "loadGeneratorTraffic":
                variants[name] = "on"
            elif name == "loadGeneratorVUs":
                if vus is not None:
                    variants[name] = str(vus)
            elif FLAG_OFF in flag["variants"]:
                variants[name] = FLAG_OFF
        self.commands.append("flag reset " + " ".join(
            f"{k}={v}" for k, v in sorted(variants.items())))
        if not self.dry_run:
            self._write_flags(variants)

    def set_traffic(self, vus: int | None = None, enabled: bool | None = None) -> None:
        """Drive the k6 load generator through its two flagd flags.

        ``loadGeneratorVUs`` (5/10/25/50) is read by the load generator's entrypoint
        every 10 s and restarts k6 when it changes; ``loadGeneratorTraffic`` (on/off)
        is checked every iteration and pauses traffic without a restart.
        """
        actions = []
        if enabled is not None:
            actions.append(Action("flag", flag="loadGeneratorTraffic",
                                  variant="on" if enabled else "off"))
        if vus is not None:
            if vus not in VU_VARIANTS:
                raise ValueError(f"loadGeneratorVUs variants are {VU_VARIANTS}, got {vus}")
            actions.append(Action("flag", flag="loadGeneratorVUs", variant=str(vus)))
        self._apply(actions, check=True)

    # --- planning --------------------------------------------------------------
    def _plan(self, fault: schema.Fault) -> tuple[dict, list[Action], list[Action]]:
        if fault.fault_type not in schema.FAULT_TYPES:
            raise ValueError(f"unknown fault type {fault.fault_type!r}")
        allowed = allowed_targets(fault.fault_type, self.cpu_mechanism)
        if fault.target not in allowed:
            raise ValueError(
                f"{fault.fault_type} cannot be injected on {fault.target!r}; "
                f"allowed: {allowed}")
        builder = {
            "cpu_saturation": self._plan_cpu_saturation,
            "memory_leak": self._plan_memory_leak,
            "network_latency": self._plan_network_latency,
            "packet_loss": self._plan_packet_loss,
            "dependency_failure": self._plan_dependency_failure,
            "error_rate": self._plan_error_rate,
            "queue_backlog": self._plan_queue_backlog,
            "cache_slowdown": self._plan_cache_slowdown,
        }[fault.fault_type]
        return builder(fault)

    def plan(self, fault: schema.Fault) -> list[str]:
        """Rendered inject commands, for inspection and tests."""
        return [a.render() for a in self._plan(fault)[1]]

    def plan_clear(self, fault: schema.Fault) -> list[str]:
        return [a.render() for a in self._plan(fault)[2]]

    def params(self, fault: schema.Fault) -> dict:
        return self._plan(fault)[0]

    # --- one builder per fault family ------------------------------------------
    def _plan_cpu_saturation(self, fault):
        if self.cpu_mechanism == "quota-only":
            return self._plan_cpu_quota(fault)
        return self._plan_cpu_hog(fault, squeeze=self.cpu_mechanism == "hog+quota")

    def _plan_cpu_hog(self, fault, squeeze: bool = True):
        """Burn CPU *inside* the target's cgroup, and cap the cgroup just above the burn.

        Neither half works alone, which is why the default does both:

        * The **quota squeeze alone** is unobservable on an idle service. A CFS limit
          only bites something that already wants more CPU than the limit, and most of
          the demo idles near 0.01 cores, so any limit docker will accept is a no-op and
          the experiment is labelled cpu_saturation with no saturation in its telemetry.
        * The **hog alone** moves `cpu_util` on any target -- a process started with
          `docker exec` lives in the target's cgroup, so its CPU is charged to the target
          -- but measured live it did *not* slow the service down: at duty 0.95 the
          median server-span latency of `quote` and `cart` barely moved, because the 5%
          of a core left over still hugely exceeds what a handler needing a few hundred
          microseconds asks for.

        Together they saturate. Intensity picks the limit
        ``Q = lerp(*HOG_QUOTA_RANGE, intensity)`` and the burner then runs *continuously*
        inside it. The cgroup is pinned at Q, so:

        * ``cpu_util`` is ``used / limit`` and used is Q -- it reads ~1.0 at every
          intensity, on an idle target as much as a busy one.
        * The service shares Q with the burner at equal CFS weight, so it gets about
          Q/2 -- against Q = 0.05 at full intensity, that is a hard starvation whatever
          it was using before.
        * The cgroup exhausts Q early in every 100 ms CFS period and *everything in it*
          stalls until the next one, up to ``(1 - Q) x 100 ms``, which grows with
          intensity.

        Two parameterisations were tried and measured before this one, and both failed
        for reasons worth not repeating:

        * Intensity picking the *duty* with Q following it (``Q = duty + headroom``)
          makes Q rise with intensity, so the stall shrinks and the hardest faults
          produce the mildest traces -- measured: quote p95 +7x at intensity 0.75 and
          unchanged at 1.0.
        * Keeping the duty cycle once Q is in charge silently scales the burn twice. A
          SIGSTOP/SIGCONT duty is a fraction of *wall time*, but while the burner runs
          the cgroup already caps it at Q, so a duty of d delivers ``d x Q`` cores, not
          d. At intensity 1.0 that left the burner running 3% of the time and cpu_util
          at 0.58 instead of 1.0.

        The share is a duty cycle rather than a raw busy loop: `dd if=/dev/zero
        of=/dev/null` burns one whole core, and a controller loop SIGSTOPs and SIGCONTs
        it on a fixed period, so the fraction is set by two `sleep` calls and needs no
        calibration for the host's speed. The controller traps its own death to CONT and
        kill the burner (a stopped process cannot take a SIGTERM), and the whole thing
        runs under `timeout`, so a clear that never happens still cannot leave the hog
        burning: it is the hold plus HOG_GRACE_S and no more.
        """
        c = CONTAINERS[fault.target]
        dur_s = self._duration_s(fault)
        duty = round(_lerp(*HOG_DUTY_RANGE, fault.intensity), 3)
        used, samples, quota = 0.0, [], BASELINE_CPUS
        if squeeze:
            used, samples = self._measure_cpu_cores(c, fault)
            quota = round(max(MIN_CPUS, _lerp(*HOG_QUOTA_RANGE, fault.intensity)), 3)
            # Continuous: the limit does the shaping, so a duty cycle on top of it would
            # only scale the burn a second time.
            duty = 1.0
        on_s = round(HOG_PERIOD_S * duty, 3)
        off_s = round(HOG_PERIOD_S - on_s, 3)
        deadline = dur_s + HOG_GRACE_S
        # The burner watches its own clock instead of trusting anything to kill it: a
        # `dd` under `timeout` looked simpler but leaves a SIGSTOPped process behind when
        # the controller dies without running its trap, which busybox sh does.
        burner = (
            f"{HOG_BURN_MARKER}; end=$(($(date +%s)+{deadline})); "
            "while [ $(date +%s) -lt $end ]; do i=0; "
            f"while [ $i -lt {HOG_SPIN_ITERATIONS} ]; do i=$((i+1)); done; done"
        )
        script = (
            f"{HOG_MARKER}; sh -c '{burner}' & p=$!; "
            # Two traps, not one: a shell that merely *handles* SIGTERM carries on
            # running afterwards, so a single combined trap left the controller looping
            # (harmlessly, but still there) until its own deadline. INT/TERM exit, and
            # exiting is what fires the EXIT trap that takes the burner with it.
            "trap 'kill -CONT $p 2>/dev/null; kill $p 2>/dev/null' EXIT; "
            "trap 'exit 143' INT TERM; "
            f"end=$(($(date +%s)+{dur_s})); "
            "while [ $(date +%s) -lt $end ]; do "
            f"kill -CONT $p 2>/dev/null; sleep {on_s}; "
            f"kill -STOP $p 2>/dev/null; sleep {off_s}; done; "
            "kill -CONT $p 2>/dev/null; kill $p 2>/dev/null"
        )
        params = {"mechanism": "exec-cpu-hog" + ("-quota" if squeeze else ""),
                  "container": c, "duty": duty, "duration_s": dur_s,
                  "deadline_s": deadline, "measured_cpu_cores": used,
                  "cpu_samples": samples, "cpus": quota,
                  "baseline_cpus": BASELINE_CPUS,
                  # What the service is left with once it shares the limit with the
                  # burner at equal CFS weight, and the worst stall a request can hit.
                  "expected_service_share_cores": round(quota / 2, 4) if squeeze else None,
                  "expected_max_stall_ms": round(
                      (BASELINE_CPUS - quota) * CFS_PERIOD_MS, 1)}
        if not squeeze:
            params.update({"period_s": HOG_PERIOD_S, "on_s": on_s, "off_s": off_s})
        # hog+quota runs the burner bare; hog-only wraps it in the duty-cycle controller,
        # which is the only shaping available when the limit is left alone.
        inject = [self._run("exec", "-d", c, "sh", "-c",
                            burner if squeeze else script)]
        clear = self._hog_kill(c)
        if squeeze:
            # Set the box before filling it, and empty it before taking the box away:
            # restoring the limit while the burner still runs would hand it the CPU the
            # service is supposed to get back.
            inject.insert(0, self._run("update", "--cpus", f"{quota:g}", c))
            clear.append(self._run("update", "--cpus", f"{BASELINE_CPUS:g}", c))
        return params, inject, clear

    def _hog_kill(self, container: str) -> list[Action]:
        """Take down both halves of a hog. CONT first: a SIGSTOPped burner -- which is
        what half of every duty cycle leaves it as -- cannot receive SIGTERM."""
        return [self._pkill(container, "-CONT", HOG_BURN_MARKER),
                self._pkill(container, None, HOG_MARKER),
                self._pkill(container, None, HOG_BURN_MARKER)]

    def _pkill(self, container: str, signal: str | None, pattern: str) -> Action:
        """One `docker exec ... pkill -f <pattern>`, tolerating "nothing matched".

        Its own argv is the only command line carrying the pattern at that moment, and
        pkill skips its own pid -- which is why this is not wrapped in `sh -c`.
        """
        args = ("exec", container, "pkill", *( (signal,) if signal else () ), "-f", pattern)
        return Action("run", argv=(*self.docker, *args), ok_codes=(0, 1))

    def _plan_cpu_quota(self, fault):
        """Squeeze the CPU limit to a fraction of what the target is *actually* using.

        A fixed absolute limit is not a fixed amount of pressure: the demo's services
        idle anywhere between 0.003 and 0.7 cores, so one absolute number starves some
        of them and is a no-op on the rest. The new limit is therefore relative to the
        container's usage sampled at inject time -- 80% of it at intensity 0 (mild
        throttling) down to 15% at intensity 1 -- which makes the *degree* of saturation
        comparable across targets. Normalised cpu_util (cores used divided by
        container.cpu.limit) then sits near 1.0 while the fault is active, which is what
        the simulator models.

        Uniform across all 15 services because it is a daemon-level operation. A
        shell-based busy loop would only work on 9 of them and would produce a
        different metric signature, which would leak the target into the label.
        """
        c = CONTAINERS[fault.target]
        used, samples = self._measure_cpu_cores(c, fault)
        squeeze_from = max(used, CPU_FLOOR_CORES)
        cpus = round(max(MIN_CPUS, squeeze_from * _lerp(0.8, 0.10, fault.intensity)), 3)
        params = {"mechanism": "docker-update-cpus", "container": c,
                  "measured_cpu_cores": used, "cpu_samples": samples,
                  "squeeze_from_cores": round(squeeze_from, 4), "cpus": cpus,
                  "baseline_cpus": BASELINE_CPUS}
        return (params,
                [self._run("update", "--cpus", f"{cpus:g}", c)],
                [self._run("update", "--cpus", f"{BASELINE_CPUS:g}", c)])

    def _measure_cpu_cores(self, container: str,
                           fault: schema.Fault) -> tuple[float, list[float]]:
        """Mean cores the container is using, and the samples it came from.

        ``docker stats`` reports percent of one core, so 100% is 1.0 core and the value
        can exceed 100% on a multi-core host. It is a one-second delta, so a request-
        driven service reads near zero whenever the sample lands between requests --
        hence CPU_SAMPLES of them, a second apart.

        The measurement is cached on the fault, so ``clear()`` -- which re-plans the
        whole fault -- reuses the pre-squeeze number instead of sampling the container
        while it is throttled, and repeated ``params()`` calls stay consistent.
        """
        cached = fault.params.get("measured_cpu_cores")
        if cached is not None:
            return float(cached), list(fault.params.get("cpu_samples") or [])
        if self.dry_run:
            return NOMINAL_CPU_CORES, []
        samples = []
        for i in range(CPU_SAMPLES):
            if i:
                time.sleep(CPU_SAMPLE_GAP_S)
            samples.append(self._cpu_sample(container))
        return round(sum(samples) / len(samples), 4), samples

    def _cpu_sample(self, container: str) -> float:
        argv = [*self.docker, "stats", "--no-stream", "--format", "{{.CPUPerc}}", container]
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
        text = result.stdout.strip().rstrip("%")
        if result.returncode or not text:
            raise RuntimeError(f"{' '.join(argv)} failed ({result.returncode}): "
                               f"{result.stderr.strip()}")
        return round(float(text) / 100.0, 4)

    def _plan_memory_leak(self, fault):
        """Growing resident memory, 25%..100% of the container's safe balloon headroom.

        email uses the demo's own ``emailMemoryLeak`` flag; the other shell-capable
        services grow a shell variable by a fixed chunk every second, which is
        anonymous memory and therefore shows up in container.memory.usage.total.

        ``s="$s$b"`` transiently holds both the old and the new string, so the balloon
        costs 2x its target at the moment it grows. The headroom is therefore
        ``(limit - 6 MiB) / 2``, leaving a 6 MiB floor for the service itself; without
        that cap a 45%-of-limit balloon OOM-kills the 20 MiB services (currency,
        shipping-class limits) instead of degrading them.
        """
        dur_s = self._duration_s(fault)
        if fault.target == "email":
            variant = _pick(_LEAK_VARIANTS, fault.intensity)
            params = {"mechanism": "flag", "flag": "emailMemoryLeak", "variant": variant}
            return (params,
                    [Action("flag", flag="emailMemoryLeak", variant=variant)],
                    [Action("flag", flag="emailMemoryLeak", variant=FLAG_OFF)])
        c = CONTAINERS[fault.target]
        limit_mb = MEMORY_LIMIT_MB[fault.target]
        headroom_mb = max(1.0, (limit_mb - BALLOON_FLOOR_MB) / 2)
        target_mb = max(1, round(headroom_mb * _lerp(0.25, 1.0, fault.intensity)))
        chunk_bytes = max(4096, (target_mb * 1024 * 1024) // dur_s)
        script = (
            f"{BALLOON_MARKER}; end=$(($(date +%s)+{dur_s})); "
            f'b=$(head -c {chunk_bytes} /dev/zero | tr "\\0" "x"); s=""; '
            'while [ $(date +%s) -lt $end ]; do s="$s$b"; sleep 1; done'
        )
        params = {"mechanism": "exec-balloon", "container": c,
                  "target_mb": target_mb, "chunk_bytes": chunk_bytes,
                  "duration_s": dur_s, "memory_limit_mb": limit_mb,
                  "headroom_mb": headroom_mb}
        inject = [self._run("exec", "-d", c, "sh", "-c", script)]
        # The loop is self-limiting on duration; pkill only matters for an early clear
        # and is best-effort (some images have no pkill), hence check=False in _apply.
        clear = [self._pkill(c, None, BALLOON_MARKER)]
        return params, inject, clear

    def _plan_network_latency(self, fault):
        """netem delay 50 ms (intensity 0) .. 800 ms (intensity 1), jitter 10%."""
        c = CONTAINERS[fault.target]
        delay_ms = round(_lerp(50, 800, fault.intensity))
        jitter_ms = max(1, delay_ms // 10)
        params = {"mechanism": f"netem-{self.netem_mode}", "container": c,
                  "delay_ms": delay_ms, "jitter_ms": jitter_ms,
                  "interface": self.netem_interface}
        return (params,
                [self._netem_add(c, ("delay", f"{delay_ms}ms", f"{jitter_ms}ms",
                                     "distribution", "normal"))],
                [self._netem_del(c)])

    def _plan_packet_loss(self, fault):
        """netem loss 2% (intensity 0) .. 40% (intensity 1)."""
        c = CONTAINERS[fault.target]
        loss_pct = round(_lerp(2.0, 40.0, fault.intensity), 1)
        params = {"mechanism": f"netem-{self.netem_mode}", "container": c,
                  "loss_pct": loss_pct, "interface": self.netem_interface}
        return (params,
                [self._netem_add(c, ("loss", f"{loss_pct}%"))],
                [self._netem_del(c)])

    def _plan_dependency_failure(self, fault):
        """The service stops answering: ``paymentUnreachable`` flag, else pause.

        Binary in both cases -- intensity is recorded but has no parameter to drive.
        """
        if fault.target == "payment":
            params = {"mechanism": "flag", "flag": "paymentUnreachable", "variant": "on"}
            return (params,
                    [Action("flag", flag="paymentUnreachable", variant="on")],
                    [Action("flag", flag="paymentUnreachable", variant=FLAG_OFF)])
        c = CONTAINERS[fault.target]
        params = {"mechanism": "docker-pause", "container": c}
        return params, [self._run("pause", c)], [self._run("unpause", c)]

    def _plan_error_rate(self, fault):
        """Fractional failure flags where the demo has them, boolean otherwise."""
        flag = ERROR_RATE_FLAGS[fault.target]
        if flag == "adFailure":
            variant, fraction = "on", 1.0      # boolean flag: intensity not applicable
        else:
            variant, fraction = _pick(_PCT_VARIANTS, fault.intensity)
        params = {"mechanism": "flag", "flag": flag, "variant": variant,
                  "error_fraction": fraction}
        return (params,
                [Action("flag", flag=flag, variant=variant)],
                [Action("flag", flag=flag, variant=FLAG_OFF)])

    def _plan_queue_backlog(self, fault):
        """Kafka lag: the demo's ``kafkaQueueProblems`` flag, or pausing a consumer."""
        if fault.target == "checkout":
            params = {"mechanism": "flag", "flag": "kafkaQueueProblems", "variant": "on"}
            return (params,
                    [Action("flag", flag="kafkaQueueProblems", variant="on")],
                    [Action("flag", flag="kafkaQueueProblems", variant=FLAG_OFF)])
        c = CONTAINERS[fault.target]
        params = {"mechanism": "docker-pause", "container": c, "consumer_group": fault.target}
        return params, [self._run("pause", c)], [self._run("unpause", c)]

    def _plan_cache_slowdown(self, fault):
        """Slow backing store: netem on valkey/postgres, or the recommendation flag."""
        if fault.target == "recommendation":
            params = {"mechanism": "flag", "flag": "recommendationCacheFailure",
                      "variant": "on"}
            return (params,
                    [Action("flag", flag="recommendationCacheFailure", variant="on")],
                    [Action("flag", flag="recommendationCacheFailure", variant=FLAG_OFF)])
        backend = CACHE_BACKEND[fault.target]
        c = CONTAINERS[backend]
        delay_ms = round(_lerp(20, 400, fault.intensity))
        params = {"mechanism": f"netem-{self.netem_mode}", "container": c,
                  "backend": backend, "delay_ms": delay_ms,
                  "interface": self.netem_interface}
        return (params,
                [self._netem_add(c, ("delay", f"{delay_ms}ms"))],
                [self._netem_del(c)])

    # --- command construction --------------------------------------------------
    def _duration_s(self, fault: schema.Fault) -> int:
        return max(1, (fault.end_ns - fault.start_ns) // 1_000_000_000)

    def _run(self, *args: str) -> Action:
        return Action("run", argv=(*self.docker, *args))

    def _netem_add(self, container: str, netem_args: tuple[str, ...]) -> Action:
        return self._tc(container, ("replace", "dev", self.netem_interface, "root",
                                    "netem", *netem_args))

    def _netem_del(self, container: str) -> Action:
        return self._tc(container, ("del", "dev", self.netem_interface, "root"))

    def _tc(self, container: str, qdisc_args: tuple[str, ...]) -> Action:
        if self.netem_mode == "exec":
            return self._run("exec", container, "tc", "qdisc", *qdisc_args)
        return self._run("run", "--rm", "--network", f"container:{container}",
                         "--cap-add", "NET_ADMIN", self.netem_image,
                         "tc", "qdisc", *qdisc_args)

    # --- execution -------------------------------------------------------------
    def _apply(self, actions: list[Action], check: bool) -> list[dict]:
        """Run actions; return the ones that failed (empty unless ``check`` is False)."""
        failures: list[dict] = []
        for action in actions:
            self.commands.append(action.render())
            if self.dry_run:
                continue
            if action.kind == "flag":
                self._write_flag(action.flag, action.variant)
                continue
            # check is handled below so the failure message can carry stderr.
            try:
                result = subprocess.run(list(action.argv), check=False,
                                        capture_output=True, text=True)
            except OSError as exc:
                # No docker CLI at all. A best-effort clear or reset has to record that
                # and carry on -- raising here is how a half-cleared stack happens.
                if check:
                    raise
                failures.append({"command": action.render(), "returncode": -1,
                                 "stderr": str(exc)[:400]})
                continue
            if result.returncode not in action.ok_codes:
                if check:
                    raise RuntimeError(f"{action.render()} failed "
                                       f"({result.returncode}): {result.stderr.strip()}")
                failures.append({"command": action.render(),
                                 "returncode": result.returncode,
                                 "stderr": result.stderr.strip()[:400]})
        return failures

    def _apply_parallel(self, actions: list[Action]) -> None:
        """Run actions at once, best effort. Only ``reset`` needs this: it is one netem
        sidecar per container and doing them in sequence costs half a minute."""
        for action in actions:
            self.commands.append(action.render())
        if self.dry_run:
            return
        running = []
        for action in actions:
            try:
                running.append(subprocess.Popen(list(action.argv),
                                                stdout=subprocess.DEVNULL,
                                                stderr=subprocess.DEVNULL))
            except OSError as exc:
                log.warning("could not run %s: %s", action.render(), exc)
        for process in running:
            process.wait()

    def _write_flag(self, name: str, variant: str) -> None:
        self._write_flags({name: variant})

    def _write_flags(self, variants: dict[str, str]) -> None:
        """Set several ``defaultVariant``s in one atomic rewrite of the flag file."""
        if self.flags_path is None:
            raise RuntimeError("flags_path is required to change a feature flag")
        text = self.flags_path.read_text()
        data = json.loads(text)
        for name, variant in variants.items():
            flag = data["flags"][name]
            if variant not in flag["variants"]:
                raise ValueError(f"flag {name} has no variant {variant!r}; "
                                 f"available: {sorted(flag['variants'])}")
            flag["defaultVariant"] = variant
        # Write a sibling and rename over the original. A plain in-place rewrite is not
        # atomic: flagd re-reads on the first write event and can see a truncated file,
        # which makes it drop every flag until the next change. flagd watches the
        # directory, so it sees the rename. The original indent is reused so the file
        # the flagd UI shows keeps its shape.
        temporary = self.flags_path.with_name(f".{self.flags_path.name}.rca")
        temporary.write_text(json.dumps(data, indent=_json_indent(text)) + "\n",
                             newline="\n")
        os.replace(temporary, self.flags_path)
