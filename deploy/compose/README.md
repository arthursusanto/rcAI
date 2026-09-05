# Running the OpenTelemetry Demo as the RCA benchmark

Everything here was checked against `open-telemetry/opentelemetry-demo` at commit
`60a6322` (demo `IMAGE_VERSION=3.0.0`, collector contrib `0.159.0`, flagd `v0.16.0`).
The demo restructured its compose files: there is no `docker-compose.yml` any more, the
base file is `compose.yaml` and optional layers are added with `-f`.

## 1. Clone the demo

```bash
git clone https://github.com/open-telemetry/opentelemetry-demo.git
cd opentelemetry-demo
```

## 2. Copy these two files in

```bash
cp <rca-repo>/deploy/compose/otelcol-config-extras.yml src/otel-collector/otelcol-config-extras.yml
cp <rca-repo>/deploy/compose/docker-compose.override.yml ./docker-compose.override.yml
mkdir -p rca-capture            # host directory for the file exporter output
docker pull nicolaka/netshoot   # the netem sidecar image the injector uses
```

`src/otel-collector/otelcol-config-extras.yml` is the demo's own customization seam:
`compose.yaml` starts the collector with
`--config=/etc/otelcol-config.yml --config=/etc/otelcol-config-extras.yml`, so the extras
file is always loaded last and merged over the base config. It is an empty stub upstream.

## 3. Start it

```bash
docker compose -f compose.yaml -f compose.full.yaml -f docker-compose.override.yml up -d
```

* `compose.full.yaml` is **required**: `accounting`, `fraud-detection` and `kafka` live
  only in that layer and all three are part of the RCA label space. It also swaps the
  collector's command to load `otelcol-config-full.yml` (which adds the `kafkametrics`
  receiver) *before* the extras file, so the extras override still wins.
* `compose.observability.yaml` is deliberately **not** used. The benchmark reads
  telemetry from files, and Jaeger + Prometheus + OpenSearch + Grafana would compete for
  CPU with the services under test. If you do add it, re-add its exporters by hand in
  `otelcol-config-extras.yml` (`otlp_grpc/jaeger`, `otlp_http/prometheus`, `opensearch`):
  the collector replaces arrays on merge, so the `exporters:` lists in the extras file
  are full replacements.
* Override the capture directory with `RCA_CAPTURE_DIR=/some/path docker compose ...`.
* The override gives all 15 application services a **baseline limit of 1.0 CPU** (see
  "`cpu_util` normalisation" below). Confirm it took after `up -d`:
  `docker inspect --format '{{.HostConfig.NanoCpus}}' currency` must print
  `1000000000`. Re-running `up -d` after editing the override recreates exactly the
  services whose config changed.

Check the stack: `http://localhost:8080` is the store, `http://localhost:8080/feature`
is the flagd UI.

## 4. Where the output lands

The collector writes three append-only files, one JSON object per line, in OTLP-JSON:

| host path                    | container path          | content                     |
| ---------------------------- | ----------------------- | --------------------------- |
| `./rca-capture/traces.jsonl` | `/capture/traces.jsonl` | `{"resourceSpans": [...]}`  |
| `./rca-capture/metrics.jsonl`| `/capture/metrics.jsonl`| `{"resourceMetrics": [...]}`|
| `./rca-capture/logs.jsonl`   | `/capture/logs.jsonl`   | `{"resourceLogs": [...]}`   |

Rotation is off (`append: true`, no `rotation:` block), so the files grow for the whole
campaign and each experiment is a `[start_ns, end_ns]` window inside them. The collector
container runs as `user: 0:0`, so it can write to a plain host directory. Truncate the
files between campaigns; a long campaign produces tens of GB of traces.

## 5. Run a campaign

From the RCA repo, with the demo running:

```bash
rca bench run-campaign --n 20 --seed 1 \
    --out-root data/experiments \
    --capture-dir /path/to/opentelemetry-demo/rca-capture \
    --flags-path /path/to/opentelemetry-demo/src/flagd/demo.flagd.json
```

`--warmup-s`, `--fault-min-s`, `--fault-max-s` and `--cooldown-s` shorten the shape for a
pilot run; keep `--warmup-s` at 30 s or more, because the feature baseline is fitted on
that period and needs at least three 10 s windows.

Add `--dry-run` to print the plan and every command without touching the stack.
`rca bench ingest --capture-dir ... --start-ns ... --end-ns ...` re-parses one window;
pass `--ip-map` as well, or checkout's gRPC dependency edges are lost (see below).

## Fault families: mechanism and reachable targets

`stress-ng` and `tc` are in none of the demo images (they are distroless, alpine or
slim), so the injector never depends on them being present.

| fault | mechanism | targets |
| --- | --- | --- |
| `cpu_saturation` | duty-cycled CPU hog inside the target's cgroup via `docker exec` (default); `docker update --cpus` squeeze with `cpu_mechanism="quota"` | the 9 with a shell (hog); all 15 (quota) |
| `memory_leak` | `emailMemoryLeak` flag for email; POSIX-sh balloon via `docker exec` elsewhere, sized to `(limit - 6 MiB) / 2` scaled by intensity | the 9 services with a shell, plus email |
| `network_latency` | `tc netem delay` 50–800 ms in the target's netns | all 15 |
| `packet_loss` | `tc netem loss` 2–40 % | all 15 |
| `dependency_failure` | `paymentUnreachable` flag for payment; `docker pause` elsewhere | all 15 except frontend, frontend-proxy, accounting and fraud-detection |
| `error_rate` | `cartFailure` / `paymentFailure` (fractional), `adFailure` (boolean) | cart, payment, ad |
| `queue_backlog` | `kafkaQueueProblems` flag for checkout; `docker pause` the consumer otherwise | checkout, accounting, fraud-detection |
| `cache_slowdown` | `tc netem delay` on valkey-cart / astronomy-db; `recommendationCacheFailure` flag | cart, product-catalog, recommendation |

### netem

Default `netem_mode="sidecar"`: a throwaway container joins the target's network
namespace and configures the qdisc there.

```
docker run --rm --network container:cart --cap-add NET_ADMIN nicolaka/netshoot \
    tc qdisc replace dev eth0 root netem delay 300ms 30ms distribution normal
```

`netem_mode="exec"` runs `docker exec <target> tc qdisc ...` instead. That needs both the
`NET_ADMIN` capability this override adds *and* images rebuilt with `iproute2`; no demo
image ships `tc` today, which is why the sidecar is the default.

### `cpu_util` normalisation and the constant CPU denominator

`schema.METRIC_CPU_UTIL` is a fraction of the container's *allotted* CPU, but
`container.cpu.utilization` is cores used (100 == one core), so ingest has to divide by
the allotment. The receiver (contrib `receiver/dockerstatsreceiver`, `v0.159.0`) only
emits `container.cpu.limit` when the computed limit is `> 0`, and `calculateCPULimit`
reads `HostConfig` in the order `NanoCPUs` -> `CpusetCpus` -> `CPUQuota / CPUPeriod`.

The demo itself sets no CPU limits at all, and that is a problem this override fixes.
Measured on the running stack before the fix: an unconstrained container emits no limit
datapoint, so ingest fell back to `container.cpu.logical.count` (16 on the machine these
measurements were taken on) and baseline `cpu_util` sat around 0.002 -- and then the
*denominator* changed the instant a `cpu_saturation` fault set a quota. That step is
perfectly correlated with the label, so a model would learn the artefact rather than the
signal.

The fix is in two halves:

* `docker-compose.override.yml` gives all 15 application services
  `deploy.resources.limits.cpus: "1"`, which Compose writes to `HostConfig.NanoCpus`.
  `container.cpu.limit` is then always present and equal to 1.0 outside a fault.
  Verify after `up -d` with
  `docker inspect --format '{{.HostConfig.NanoCpus}}' <service>` -> `1000000000`.
* the `quota` cpu_saturation mechanism uses `docker update --cpus <x>`, which writes
  the same `NanoCpus` field, and **clears to `docker update --cpus 1`** rather than to
  "unlimited". `docker update --cpus 0` does *not* remove a limit -- the daemon skips
  zero-valued fields, and `NanoCpus` stays at whatever it was -- so restoring an
  explicit baseline is the only way back. The default `hog` mechanism never touches the
  limit at all: it competes *inside* it.

The receiver's event loop subscribes to the Docker `update` action and re-inspects the
container on every non-`destroy` event, and this was confirmed live: squeezing `payment`
to `--cpus 0.02` moved `container.cpu.limit` from `1.0` to `0.02` on the very next 5 s
scrape, and back to `1.0` one scrape after the clear.

The numerator is deliberately *not* the metric whose name suggests it — see
"The numerator is `container.cpu.usage.total`" below.

`IngestReport.cpu_denominator` counts which denominator each point used, and
`IngestReport.cpu_limit_missing` counts, per service, the points that had no
`container.cpu.limit`. On a correctly deployed stack that second counter is empty; if it
is not, the named container is missing its baseline limit and its `cpu_util` is not
comparable with the rest.

### cpu_saturation: a hog inside the cgroup, plus a limit just above it

Neither half of this works on its own, which is why the default (`cpu_mechanism =
"hog+quota"`) is both.

**A quota squeeze alone is a no-op on an idle service.** A CFS limit only bites something
that already wants more CPU than the limit, and eleven of the fifteen demo services idle
below 0.025 cores, so any limit `docker update` will accept (it refuses below 0.01, and
`MIN_CPUS` keeps a margin at 0.02) changes nothing. The experiment gets the
`cpu_saturation` label and its telemetry contains no saturation at all -- a mislabelled
sample, which is worse than a weak one, and why per-family top-1 collapses for it.

**A hog alone moves `cpu_util` but does not slow the service down.** A process started
with `docker exec` lives in the target container's cgroup, so the kernel charges its CPU
to the target -- that part works on any target at any traffic level. But measured live at
duty 0.95, the median server-span latency of `quote` and `cart` barely moved: the 5% of a
core left over still vastly exceeds what a handler needing a few hundred microseconds
asks for. And `cpu_util` only reached ~0.76, because it was still being divided by the
untouched 1.0 limit.

**Together they saturate.** The limit is set to `Q = min(1.0, duty + max(MIN_CPUS,
0.5 x measured_usage))`, so once the burner has taken its share the service keeps about
half of what it was using, and `cpu_util` -- `used / limit`, with both near Q -- reads
~1.0. `docker stats` is sampled three times a second apart at inject time to get
`measured_usage`; the duty, Q and the measurement all go into `Fault.params`.

The share is a duty cycle. A burner process spins on shell arithmetic -- one whole core
when it is running -- and a controller SIGSTOPs and SIGCONTs it on a fixed 0.5 s period,
so the share is set by two `sleep` calls and needs no calibration against the host's
speed (a counted busy loop would). Intensity maps to duty over
`HOG_DUTY_RANGE = (0.15, 0.95)`: intensity 0.3 gives 0.39 of a core, 1.0 gives 0.95.

```
# the burner: watches its own clock, so nothing has to survive to stop it
sh -c 'RCA_FAULT=hogburn; end=$(($(date +%s)+55));
       while [ $(date +%s) -lt $end ]; do
           i=0; while [ $i -lt 100000 ]; do i=$((i+1)); done
       done' &
p=$!
trap 'kill -CONT $p; kill $p' EXIT     # exiting is what stops the burner
trap 'exit 143' INT TERM               # ... and this is what makes the shell exit
while [ $(date +%s) -lt $end ]; do
    kill -CONT $p; sleep 0.375         # on_s  = period x duty
    kill -STOP $p; sleep 0.125         # off_s = period - on_s
done
```

Two details in there were bugs found by running it, not theory:

* **The burner watches its own clock** instead of being a `dd` under `timeout`. `timeout`
  only kills its direct child, so when the controller died without cleaning up, the
  `dd` was left behind *SIGSTOPped* -- forever, since a stopped process never reaches
  any deadline of its own. Verified after the change: kill the controller mid-cycle and
  call no clear at all, and 30 s later the container is back to 0.000 cores with no
  leftover processes.
* **`trap 'exit 143' INT TERM` is separate from the EXIT trap.** A shell that merely
  *handles* SIGTERM carries on running afterwards, so a single combined trap killed the
  burner but left the controller looping until its own deadline.

Three details that matter:

* **The hard stop is the burner's own deadline (`hold + HOG_GRACE_S`), not the clear.**
  `clear()` is three `pkill`s -- `-CONT` the burner first, because half of every duty
  cycle leaves it SIGSTOPped and a stopped process cannot receive SIGTERM, then the
  controller, then the burner. Each is its own argv rather than one `sh -c`, so the only
  process carrying a marker at clear time is `pkill` itself, which skips its own pid.
  `FaultInjector.reset()` runs the same three in every shell-capable container.
* **Reachable targets are the 9 services with a shell**, not all 15. `docker exec` needs
  something to exec. See "Known limitations".
* `cpu_mechanism="hog-only"` and `"quota-only"` keep each half selectable on its own for
  comparison. `"quota-only"` samples the container
  with `docker stats` (3 samples, 1 s apart), then set
  `max(0.02, max(used, 0.15) * lerp(0.8, 0.10, intensity))`. The `max(used, 0.15)` floor
  exists because scaling an idle service's own usage lands on `MIN_CPUS` at every
  intensity above ~0.3, which stops the fault varying with its own label. It reaches all
  15 services and is the only option for the distroless ones.

#### Measured

`loadGeneratorVUs = 25`, 45 s hold, at three intensities on an idle target (`quote`,
called only by `shipping`) and a busy one (`frontend-proxy`, the entry point --
`frontend` itself is distroless). `cpu_util` is normalised by the squeezed limit Q.

| target | i | Q | stall | phase | `cpu_util` mean/max | spans | median | mean | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `quote` | 0.3 | 0.575 | 42.5 ms | before | 0.011 / 0.012 | 9 | 0.30 ms | 0.32 ms | 0.47 ms |
| | | | | **during** | **1.002 / 1.010** | 19 | 0.30 ms | 1.19 ms | 2.20 ms |
| `quote` | 0.75 | 0.238 | 76.2 ms | before | 0.013 / 0.016 | 8 | 0.29 ms | 0.29 ms | 0.33 ms |
| | | | | **during** | **1.000 / 1.003** | 15 | 0.33 ms | 0.34 ms | 0.49 ms |
| `quote` | 1.0 | 0.05 | 95.0 ms | before | 0.010 / 0.012 | 5 | 0.29 ms | 0.32 ms | 0.38 ms |
| | | | | **during** | **0.997 / 1.011** | 14 | 0.31 ms | **7.49 ms** | **35.50 ms** |
| `frontend-proxy` | 0.3 | 0.575 | 42.5 ms | before | 0.021 / 0.027 | 249 | 2.69 ms | 25.69 ms | 11.85 ms |
| | | | | **during** | **0.999 / 1.009** | 477 | 3.32 ms | 31.31 ms | **41.03 ms** |
| `frontend-proxy` | 0.75 | 0.238 | 76.2 ms | before | 0.020 / 0.021 | 227 | 2.79 ms | 25.76 ms | 15.32 ms |
| | | | | **during** | **1.001 / 1.003** | 428 | 3.34 ms | 42.04 ms | **79.15 ms** |
| `frontend-proxy` | 1.0 | 0.05 | 95.0 ms | before | 0.025 / 0.036 | 249 | 3.02 ms | 24.50 ms | 16.76 ms |
| | | | | **during** | **0.991 / 1.027** | 480 | 6.77 ms | **91.26 ms** | **299.87 ms** |

**`cpu_util` is 1.00 at every intensity, on both targets**, against baselines of
0.010-0.025. That is what pinning the cgroup at Q buys: the burner alone saturates the
limit, so `used / limit` is 1 by construction whether the service was idle or busy.

**Latency now rises with intensity**, and on the target with enough traffic to measure
it the trend is unambiguous -- `frontend-proxy` p95 goes 41 -> 79 -> 300 ms and its mean
31 -> 42 -> 91 ms across the three intensities, from a ~25 ms baseline. Its median moves
too (2.7 -> 6.8 ms at full intensity).

`quote` is the honest caveat: it serves 5-19 spans in a window, so its p95 is effectively
the second-largest of a handful of sub-millisecond spans and is dominated by sampling
noise -- 2.20 ms at intensity 0.3 but 0.49 ms at 0.75 is not a real inversion. At full
intensity even quote is unmistakable (p95 0.38 -> 35.50 ms, mean 0.32 -> 7.49 ms). Read
`cpu_util` for this family on low-traffic targets; the trace side needs traffic.

One operational cost: at Q = 0.05 the clear takes ~5-6 s instead of ~0.6 s, because
`clear()` `docker exec`s into a cgroup that now has 5% of a core. The fault therefore
overruns its nominal window by a few seconds at the highest intensity. Restoring the
limit *before* killing the burner would avoid it, at the price of a ~1 s window where the
burner has the full core back; the current order is the conservative one.

### Feature flags

flagd is started with `--uri file:./etc/flagd/demo.flagd.json` and `./src/flagd` is
bind-mounted into it, so the file on the host is the control surface — the same one the
flagd UI edits. The injector rewrites `defaultVariant` in place (same inode) so flagd's
file watch survives; a rewrite is picked up in about a second.

**The file is the only interface, and not just because OFREP is read-only.**
`compose.yaml` publishes flagd as `ports: ["${FLAGD_PORT}", "${FLAGD_OFREP_PORT}"]` —
container port only, no host side — so Docker assigns an *ephemeral* host port that
changes on every recreate (58952/58953 in one run here). There is no stable
`localhost:8013` / `localhost:8016` to talk to from the host; `docker port flagd` is the
only way to find today's, and it is stale after the next `up -d`. Even if you do find
it, OFREP (`POST /ofrep/v1/evaluate/flags/<key>`) only *evaluates* a flag, it cannot set
one. Nothing in the injector or the runner needs a flagd port.

### Traffic

The load generator is **k6** (it was locust in older versions; there is no
`loadgeneratorFloodHomepage` flag any more — `floodHome` is one of the weighted tasks in
`script.js`). Two flags control it, and nothing else is controllable at runtime:

* `loadGeneratorTraffic` — `on`/`off`, checked every iteration, pauses all traffic.
* `loadGeneratorVUs` — `5` / `10` / `25` / `50`. The container's `entrypoint.sh` polls
  flagd over OFREP every 10 s and restarts k6 when the value changes, because k6 v2's
  `constant-vus` executor cannot resize a running VU pool. Expect up to ~10 s of lag and
  a brief traffic gap on every change; the runner therefore changes VUs before the
  recorded window (`settle_s`) and only again for a deliberate traffic-only spike.

The `LOAD_GENERATOR_VUS` env var only sets the starting value. Never set `K6_VUS`,
`K6_DURATION`, `K6_ITERATIONS` or `K6_STAGES` — k6 discards the script's scenarios if any
of them is present.

## What each family actually does to the telemetry (measured)

Every family was injected once against the live stack (`loadGeneratorVUs = 10`, 45 s
hold, 30 s recovery between probes) and the capture window was ingested and compared
against the 20 s immediately before injection. "during" is the last 30 s of the hold, so
the effect has had 15 s to land.

| family (target, parameter) | measurement | before | during |
| --- | --- | --- | --- |
| `cpu_saturation` (quote, hog+quota at intensity 1.0) | `cpu_util` mean | 0.010 | **0.997** |
| | quote server spans, p95 | 0.38 ms | **35.50 ms** |
| `cpu_saturation` (frontend-proxy, hog+quota at intensity 1.0) | `cpu_util` mean | 0.025 | **0.991** |
| | frontend-proxy server spans, p95 | 16.76 ms | **299.87 ms** |
| `network_latency` (currency, 350 ms) | callers' client spans -> currency, mean | 1.43 ms | **351.43 ms** |
| | currency's own server spans, mean | 0.035 ms | 0.034 ms |
| `packet_loss` (cart, 24.8 %) | client spans -> cart, mean | 1.47 ms | **341.30 ms** |
| | client spans -> cart, p95 | 2.69 ms | **974.34 ms** |
| | client spans -> cart, error rate | 0.000 | 0.000 |
| `dependency_failure` (shipping, `docker pause`) | client spans -> shipping, mean | 4.20 ms | **15 626 ms** |
| | shipping server spans in window | 1 | **0** |
| `dependency_failure` (payment, `paymentUnreachable`) | payment server spans in window | 2 | **0** |
| `error_rate` (cart, `cartFailure=50%`) | cart `EmptyCart` server spans, error rate | 0.000 | **0.500** |
| | cart server spans, error rate (all RPCs) | 0.000 | 0.028 |
| `queue_backlog` (checkout, `kafkaQueueProblems`) | accounting `queue_depth` max | 1 | **101** |
| | fraud-detection `queue_depth` max | 0 | **606** |
| `queue_backlog` (accounting, `docker pause`) | accounting `queue_depth` max | 1 | 5 |
| `cache_slowdown` (cart, 210 ms on valkey-cart) | cart -> valkey client spans, mean | 0.19 ms | **293.90 ms** |
| | cart's own server spans, mean | 0.58 ms | **517.73 ms** |
| `memory_leak` (currency, balloon, 4 MiB) | currency `mem_bytes`, max | 3.93 MiB | **8.28 MiB** |
| `memory_leak` (email, `emailMemoryLeak=100x`) | email `mem_bytes` | 56.09 MiB | **57.71 MiB** |

The two `cpu_saturation` rows are from a separate pair of 45 s probes at 25 VUs with the
current hog+quota mechanism (see "cpu_saturation: a hog inside the cgroup"); the rest of
the table is from the eight-family sweep, whose `cpu_saturation` probe predates it.
Watch out for one receiver hiccup seen once during that sweep: `docker_stats` emitted no
datapoints at all for `payment` for the first 35 s of a capture while every other
container had a point every 5 s from the first second.

All eight families move the telemetry, and the netem, pause, exec-balloon and flag
mechanisms all worked on the first attempt. Five things the run turned up are worth
knowing; the first and the last were pipeline bugs and are fixed, the middle three are
properties of the demo and are not going away.

### The numerator is `container.cpu.usage.total`

`docker_stats` publishes a metric called `container.cpu.utilization`, and it is the wrong
one: it **does not respond to a CPU fault at all**. Squeezing `payment` from 1.0 to
0.02 CPU for 55 s:

```
             docker stats (instantaneous)   container.cpu.utilization   container.cpu.limit
before                    0.43 %                      9.263                   1.0
+10 s                     0.43 %                      9.236                   0.02
+30 s                     1.22 %                      9.167                   0.02
+50 s                     2.02 %                      9.114                   0.02
after clear                  —                        9.114                   1.0
```

The container really is throttled — `docker stats` shows it climbing to 2.02 %, i.e.
saturating its new 0.02-core quota — but the gauge drifts by ~0.0025/s and ignores the
fault completely. It behaves as `total_cpu_time / total_system_time` over the container's
whole lifetime, not as a per-interval rate; a `docker pause` does not dent it either, and
`load-generator` reads 40–55 there while `docker stats` reports 100–400 %.

`METRIC_MAP` therefore maps **`container.cpu.usage.total`**, the cumulative CPU-nanosecond
counter the same receiver emits. `_differenced` divides its increment by the elapsed time
(the `_RATE_DIFFERENCED` set) to get cores used, carrying `(ts_ns, value)` per series in
the same `state` dict the streaming tail already passes, so a one-point-per-poll live tail
produces exactly what one offline pass does. `_normalize_cpu` then divides by
`container.cpu.limit`.

Confirmed live on a 45 s squeeze of `frontend`, measured at 0.0918 cores and squeezed to
`--cpus 0.044`:

| phase | `cpu_util` mean | min | max | samples |
| --- | --- | --- | --- | --- |
| before | 0.042 | 0.037 | 0.050 | 3 |
| during | **0.792** | 0.587 | 0.987 | 7 |
| after | 0.039 | 0.027 | 0.059 | 4 |

`container.cpu.utilization` is still read, but only to be counted: if a capture has it and
no usage counter, the affected services are named in
`IngestReport.cpu_utilization_only` instead of silently producing no `cpu_util`.

### `cartFailure` only wraps `EmptyCart`

`src/cart/src/services/CartService.cs` reads the flag inside `EmptyCart` and nowhere
else, routing that one call to a deliberately broken store. At 50 % the *EmptyCart*
error rate really is 0.5, but `EmptyCart` is called once per checkout, so cart's
service-wide server error rate only reached 0.028 (1 error in 36 spans). `error_rate` on
cart is therefore a low-rate label unless the traffic is checkout-heavy; `paymentFailure`
(also fractional) is worth preferring where a strong error signal is wanted.

### `packet_loss` reads as latency, not as errors

At 24.8 % loss nothing errored: TCP retransmitted and the calls into cart went from
1.5 ms to 341 ms mean and 974 ms p95. Expect `packet_loss` to look like a heavy-tailed
`network_latency`, not like `error_rate`.

### `queue_backlog` by `docker pause` is traffic-limited

Pausing `accounting` can only accumulate the messages produced while it is paused —
about 3 orders in 45 s at 10 VUs, and lag peaked at 5. The `kafkaQueueProblems` flag
produces 100 extra messages *per order* and drove lag to 101 (accounting) and 606
(fraud-detection) over the same hold. Prefer the flag, or a long hold, or more VUs.

### checkout's gRPC client spans carry an IP, so peers need an IP map

Go's gRPC instrumentation puts the *resolved address* in `server.address`
(`'172.18.0.10'`), while the JS frontend puts the hostname (`'product-catalog'`) and
checkout's own HTTP calls put `server.address: 'shipping'`. A bare IP is not a service
name, so without help every checkout -> payment / cart / currency / product-catalog
client span is ingested with no peer and those four dependency edges simply do not exist.

`injector.container_ip_map()` reads the addresses out of `docker inspect` and the runner
stores them per experiment in `manifest.extra["ip_map"]`; `ingest.normalize_service`
consults the map whenever a peer attribute is not already a known service name. IPs
change on every recreate, which is why the map is captured per experiment rather than
written down. For a hand-driven window, `rca bench ingest --ip-map <file.json>` takes the
same dict.

Measured over one 140 s capture at 25 VUs, on checkout's 350 client spans:

| | resolved peers | unresolved |
| --- | --- | --- |
| without the map | shipping 68, email 33 | **249** |
| with the map | currency 91, shipping 68, cart 67, product-catalog 57, payment 34, email 33 | **0** |

Distinct `service -> peer` edges across the whole capture went from 16 to 20 — the four
that appear are exactly checkout's gRPC callees. Anything still unresolved is named in
`IngestReport.peer_unresolved`, so a stale or missing map shows up as container IPs in
the report rather than as quietly missing edges.

## Known limitations

* `stress-ng` is unavailable, and 6 of the 15 services (`checkout`, `frontend`,
  `payment`, `product-catalog`, `shipping`, `fraud-detection`) are distroless and cannot
  be `docker exec`'d into at all. CPU saturation therefore squeezes the daemon-level CPU
  limit everywhere (uniform signature, no shell needed) and memory leaks are limited to
  the services that do have a shell.
* `cpu_saturation`'s latency signal needs traffic. On a service serving a handful of
  sub-millisecond spans per window (`quote`) the p95 is sampling noise below full
  intensity; `cpu_util` is the reliable channel there. See "Measured" above.
* `cpu_saturation` reaches **9 of the 15 services**, not all of them. The hog is
  started with `docker exec`, and `checkout`, `frontend`, `payment`, `product-catalog`,
  `shipping` and `fraud-detection` are distroless -- there is no shell to exec. Running
  the burner in a *helper container* that writes its own PID into the target's
  `cgroup.procs` would cover all 15, but that needs a **privileged container with
  `/sys/fs/cgroup` bind-mounted read-write** -- a capability this benchmark deliberately
  does not take, since it hands the injector host-wide cgroup control. Without that
  privileged helper the options are: leave those six services out of the family, or fall
  back to `cpu_mechanism="quota"` for them and accept that the CPU signature then differs
  by target -- which partly encodes the target in the label, and is exactly what the
  uniform mechanism was chosen to avoid.
* With `cpu_mechanism="quota"`, intensity saturates on a lightly loaded stack: the
  squeeze is floored at `MIN_CPUS = 0.02` cores, so on a service idling below 0.025
  cores every intensity above ~0.3 produces the same limit. This is the defect the hog
  replaces.
* `cache_slowdown` cannot target `accounting`, although it does query astronomy-db.
  netem is applied to the *store*, and astronomy-db is shared with `product-catalog`,
  which serves far more traffic. Measured in a real campaign: `cache_slowdown@accounting`
  left `product-catalog` as the visibly degraded service (infra latency z 17, latency
  z 7) while accounting itself barely moved, so the experiment carried the wrong
  root-cause label. A shared backing store is only a valid target for the service that
  dominates its traffic -- which is why `cart` (valkey-cart is its alone) and
  `product-catalog` are fine.
* `productCatalogFailure` is not used for `error_rate`: its targeting rule returns `off`
  in both branches, so changing `defaultVariant` has no effect, and it only fails one
  product id.
* `dependency_failure` and `queue_backlog` are binary (a flag or `docker pause`);
  intensity is recorded in the manifest but has no parameter to drive.
* `accounting` and `fraud-detection` are not `dependency_failure` targets: pausing them
  is exactly the `queue_backlog` injection, so the two families would carry identical
  telemetry under different labels.
* `docker pause` makes callers hang until their timeout rather than get a connection
  refused, so it reads as a timeout-shaped dependency failure.
