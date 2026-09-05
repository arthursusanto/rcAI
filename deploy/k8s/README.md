# Running the RCA inference service on Kubernetes, next to the demo

This directory deploys the OpenTelemetry Demo from its Helm chart with the RCA capture
seam wired in, plus the `rca serve` inference server reading that capture live.

| file | what it is |
| --- | --- |
| `../Dockerfile` | image for the `rca` package, model baked in |
| `values-demo.yaml` | Helm values override for the `opentelemetry-demo` chart |
| `rca-serve.yaml` | PVC + Deployment + Service for the inference server |

Everything here was checked against chart **`opentelemetry-demo` 0.41.0**
(`appVersion: 3.0.0`, the same demo version `deploy/compose` targets) and its
**`opentelemetry-collector` 0.165.0** subchart. **None of it has ever been run against a
cluster.** Every claim below is either read out of the chart source (cited inline) or
explicitly marked unverified — treat this directory as a reviewed design, not as a
tested deployment. The docker-compose path in `deploy/compose` is the one that has
actually produced data.

## 1. A cluster

Summing `resources.limits.memory` across chart 0.41.0's enabled components gives
~5.5 GiB, plus 400 MiB collector, 600 MiB Jaeger, 300 MiB Grafana, 1100 MiB OpenSearch
and Prometheus on top — call it 8 GiB of memory limits before `rca-serve`'s own 1 GiB.
Give Docker Desktop / the VM 10–12 GB and 6 CPUs, or switch off the components listed
at the end of `values-demo.yaml`.

```bash
kind create cluster --name rca
# or
minikube start --cpus 6 --memory 12288
```

Use a **single-node** cluster. The capture PVC is `ReadWriteOnce` (see the comment in
`rca-serve.yaml`: kind's default `standard` StorageClass rejects `ReadWriteMany`
outright), and RWO means one *node* — the collector pod and the `rca-serve` pod share
it happily as long as they are co-located, which on one node they always are.

## 2. Build and load the rca image

The model is **baked into the image**, not mounted. A ConfigMap cannot carry it — the
Kubernetes ConfigMap limit is 1 MiB and the trained models are 3.8 MB (`artifacts/xgb`)
and 8.8 MB (`artifacts/rf-fpr`) — and a PVC would need a separate copy-in step before
the server could start. Baking makes the image the versioned unit: one image, one model.

```bash
# from the repo root; the build context is the repo root, not deploy/
docker build -f deploy/Dockerfile -t rca-serve:local \
    --build-arg MODEL_DIR=artifacts/xgb .

kind load docker-image rca-serve:local --name rca
# or
minikube image load rca-serve:local
```

`MODEL_DIR` is a path inside the build context. `artifacts/` is gitignored, so it is
whatever the last training run wrote on the build machine; the root `.dockerignore`
deliberately does *not* exclude it. Tag the image after the model
(`rca-serve:xgb-v1`) rather than reusing `:local` if you keep more than one.

## 3. Install the chart

```bash
kubectl apply -f deploy/k8s/rca-serve.yaml        # PVC first

helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update
helm install rca-demo open-telemetry/opentelemetry-demo \
    --version 0.41.0 \
    -f deploy/k8s/values-demo.yaml
```

Apply `rca-serve.yaml` first: the collector mounts the `rca-capture` claim and its pod
stays `Pending` until the claim exists. Install both into the same namespace — the
`JAEGER_URL`/`GRAFANA_URL` values use bare service names.

What the values file changes, and nothing else:

* **File exporters.** `file/rca_traces`, `file/rca_metrics`, `file/rca_logs` writing
  `/capture/{traces,metrics,logs}.jsonl`, byte-for-byte the same config as
  `deploy/compose/otelcol-config-extras.yml`, appended to the three pipelines'
  exporter lists. Helm replaces lists rather than merging them, so each list repeats
  the chart's own exporters (`otlp_grpc/jaeger`, `otlp_http/prometheus`, `opensearch`,
  `debug`, `span_metrics`) alongside the new one.
* **The capture volume**, via the collector subchart's `extraVolumes` /
  `extraVolumeMounts`, and `securityContext.runAsUser: 0` so the collector can write it
  (the compose stack runs the collector as `user: 0:0` for the same reason).
* **`NET_ADMIN`** on the 15 application components plus `valkey-cart`, `astronomy-db`
  and `kafka`, through `components.<name>.securityContext.capabilities.add` — which the
  chart renders as the *container* security context
  (`templates/_objects.tpl`).
* **`enabled: true` pinned** on `kafka`, `accounting` and `fraud-detection`.

### On the "full components" flag

There isn't one. The chart has **no** equivalent of `compose.full.yaml`: every
component is gated individually by `components.<name>.enabled`, and in chart 0.41.0 all
of them default to `true` except `firepit`. `kafka`, `accounting` and `fraud-detection`
are therefore already on; `values-demo.yaml` only pins them so a future chart default
cannot silently delete three members of the RCA label space. (The chart *also* enables
`agent`, `chatbot`, `mcp`, `telemetry-docs` and `opamp-server`, which are not in the
label space; `values-demo.yaml` has a commented-out block to switch them off if the
laptop is tight. Their telemetry is harmless either way — `rca.benchmark.ingest`
drops any service outside `KNOWN_SERVICES`.)

## 4. Reach the UIs

```bash
kubectl port-forward svc/frontend-proxy 8080:8080   # store, /feature, /jaeger/ui, /grafana
kubectl port-forward svc/rca-serve 8000:8000        # incident API + UI at /
```

`rca-serve.yaml` sets `JAEGER_URL=http://jaeger:16686/jaeger/ui` and
`GRAFANA_URL=http://grafana/grafana` — the chart's in-cluster services (verified:
frontend-proxy is configured with `JAEGER_HOST=jaeger`, `JAEGER_UI_PORT=16686`,
`GRAFANA_HOST=grafana`, `GRAFANA_PORT=80`; jaeger_query is configured with
`base_path: /jaeger/ui`; `grafana.ini` sets `serve_from_sub_path: true` under
`/grafana`). Those URLs end up in links your **browser** follows, and a browser outside
the cluster cannot resolve `jaeger` or `grafana`. If you reach the stack through the
port-forward above, delete both env entries — `rca.serve.api` then falls back to
`http://localhost:8080/jaeger/ui` and `http://localhost:8080/grafana`, which are exactly
the proxied paths. The Grafana datasource UIDs the code defaults to
(`webstore-metrics`, `webstore-logs`) match the ones the chart provisions.

## 5. Where the capture lands

Same three append-only OTLP-JSON files as the compose path, one JSON object per line:
`/capture/traces.jsonl`, `/capture/metrics.jsonl`, `/capture/logs.jsonl`, on the
`rca-capture` PVC. Rotation is off, so they grow for the whole campaign and each
experiment is a `[start_ns, end_ns]` window inside them. To pull one out for offline
work:

```bash
kubectl cp <collector-pod>:/capture/traces.jsonl ./traces.jsonl
```

**Known gap — resource metrics.** There is no `docker_stats` receiver in Kubernetes.
Per-container CPU/memory come from the chart's `kubeletMetrics` preset (the
`kubeletstats` receiver), under different metric names and keyed by
`k8s.container.name` rather than `container.name`. `rca/benchmark/ingest.py`'s
`METRIC_MAP` maps only the `docker_stats` names, and `normalize_service` reads
`service.name` or `container.name`, so **resource metrics from a Kubernetes capture do
not ingest today**. Traces, logs, `kafkametrics` and the SDK metrics do. Closing that
gap is an `ingest.py` change, not a deployment change, and is out of scope here.

## 6. Fault injection on Kubernetes

**There is no Kubernetes backend in the injector, and this section does not add one.**
`rca/benchmark/injector.py` shells out to `docker` for every non-flag action
(`FaultInjector.docker = ("docker",)`, every `Action(kind="run")` is a `docker …` argv)
and writes feature flags by rewriting a local `demo.flagd.json` in place. `rca bench
run-campaign` therefore works against docker compose only. What follows is the manual
translation of each mechanism, so a future `KubernetesFaultInjector` has a verified
target to aim at — and so the gaps are visible before someone writes it.

The demo images are identical to the compose ones (`ghcr.io/open-telemetry/demo:…`), so
the shell/distroless split carries over unchanged: the 6 distroless services
(`checkout`, `frontend`, `payment`, `product-catalog`, `shipping`, `fraud-detection`)
cannot be `kubectl exec`'d into any more than they could be `docker exec`'d into, and
no image ships `tc` or `stress-ng`.

### cpu_saturation — `docker update --cpus` → in-place pod resize

```bash
kubectl patch pod <pod> --subresource resize --patch \
  '{"spec":{"containers":[{"name":"cart","resources":{"limits":{"cpu":"50m"}}}]}}'
```

This is the only faithful equivalent: it changes the running container's CFS quota
without restarting it. In-place pod resize (`InPlacePodVerticalScaling`) is **stable in
Kubernetes 1.35**, and the `--subresource=resize` flag needs **kubectl ≥ 1.32** — both
straight from the Kubernetes "Resize CPU and Memory Resources assigned to Containers"
task page. It was beta and on by default before that; *unverified* exactly which
release, so check the feature gate on an older cluster before relying on it.

Patching the *Deployment* instead (`kubectl patch deploy/cart …`) rewrites the pod
template and rolls the pod, which is a restart, not a throttle — a completely different
telemetry signature, and one the model would be right to call something else. Without
in-place resize there is no faithful mechanism; skip the family rather than substitute
a restart.

Note the chart sets only `resources.limits.memory` on the application components, so a
CPU limit is a new field rather than a change to an existing one.

### memory_leak — `docker exec -d` balloon → `kubectl exec`

The `email` case is unchanged: it is the `emailMemoryLeak` feature flag (see the flagd
section below). For the nine shell-capable services, the same POSIX-sh balloon script
runs under `kubectl exec`:

```bash
kubectl exec deploy/cart -- sh -c 'nohup sh -c "<balloon script>" >/dev/null 2>&1 &'
```

`kubectl exec` has no `-d`; the process must be detached inside the shell or the exec
stream holds the invocation open for the whole fault. The balloon is self-limiting on
duration exactly as in compose, and `pkill -f RCA_FAULT=balloon` for an early clear
works the same way.

### network_latency / packet_loss / cache_slowdown — netshoot sidecar → ephemeral container

```bash
kubectl debug -it <pod> --image=nicolaka/netshoot --profile=netadmin -- \
    tc qdisc replace dev eth0 root netem delay 300ms 30ms distribution normal
# clear
kubectl debug -it <pod> --image=nicolaka/netshoot --profile=netadmin -- \
    tc qdisc del dev eth0 root
```

This is the direct analogue of `docker run --network container:<c> --cap-add
NET_ADMIN`: every container in a pod already shares the pod's network namespace, so an
ephemeral container configures the qdisc the application container is using. The
`netadmin` profile adds exactly the capability needed — verified in kubectl's
`pkg/cmd/debug/profiles.go`, where `allowNetadminCapability` adds `NET_ADMIN` and
`NET_RAW` to the debug container.

Two caveats. Ephemeral containers **cannot be removed** from a pod once added, so a long
campaign accumulates terminated debug containers in the pod spec; recreate the target
pod periodically. And `kubectl debug` will not help if the pod has a pod-level
`runAsNonRoot` — the chart does not set one by default, but if you add one, the
capability is granted and then immediately unusable.

`cache_slowdown` targets `valkey-cart` and `astronomy-db` the same way; the
`recommendationCacheFailure` case is a flag.

### dependency_failure and queue_backlog — `docker pause` → scale to 0, or a NetworkPolicy

Neither Kubernetes option reproduces `docker pause` exactly, and they differ from each
other in a way the model can see:

* `kubectl scale deploy/<svc> --replicas=0` removes the Service's endpoints, so callers
  get a **connection refused** almost immediately. Recovery (`--replicas=1`) brings up a
  *new* pod, which adds a cold-start signature that compose's `unpause` does not have.
* A default-deny NetworkPolicy on the target **drops** packets, so callers hang until
  their own timeout — much closer to `docker pause`'s semantics, which is what
  `deploy/compose/README.md` records as the observed behaviour.

```bash
kubectl apply -f - <<'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: rca-isolate-cart
spec:
  podSelector:
    matchLabels:
      opentelemetry.io/name: cart
  policyTypes: [Ingress]
EOF
# clear
kubectl delete networkpolicy rca-isolate-cart
```

A NetworkPolicy is only enforced if the CNI implements it. kind's bundled `kindnetd`
does since **kind v0.25.0**, where it started vendoring
`sigs.k8s.io/kube-network-policies` (verified by reading
`images/kindnetd/cmd/kindnetd/main.go` at tags v0.23.0 … v0.30.0: absent at v0.23.0,
present and ungated from v0.25.0). Older kind releases ignore NetworkPolicy *silently*,
which would produce experiments labelled as faults with no fault in the telemetry.
minikube's default CNI does not enforce it either — start with `--cni=calico`.

The `podSelector` label is verified: `otel-demo.selectorLabels` in the chart's
`_helpers.tpl` is `opentelemetry.io/name: <component>`, and it is applied to every
component's pod template. (`app.kubernetes.io/component` and `app.kubernetes.io/name`
carry the same value and would work too.) Component Deployments are named after the
component, so `kubectl exec deploy/cart` and `kubectl scale deploy/cart` are correct.

`queue_backlog` on `accounting` / `fraud-detection` is the same scale-to-0 move against
the Kafka consumer; on `checkout` it is the `kafkaQueueProblems` flag.

### error_rate and every other flag-driven fault — edit the flagd ConfigMap

The chart's flagd wiring is **not** the compose bind-mount, and this is the part most
likely to trip up a Kubernetes injector. Verified from the chart:

* `templates/flagd-config.yaml` creates a ConfigMap named `flagd-config` from
  `charts/opentelemetry-demo/flagd/*.json`, i.e. `demo.flagd.json`.
* `components.flagd` mounts that ConfigMap at `/config-ro` (`additionalVolumes`) and an
  **emptyDir** named `config-rw` at `/etc/flagd` (`mountedEmptyDirs`).
* An init container `init-config` copies `/config-ro/demo.flagd.json` to
  `/config-rw/demo.flagd.json`.
* flagd is started with `--uri file:./etc/flagd/demo.flagd.json`, i.e. it reads the
  **emptyDir copy**, not the ConfigMap.
* The `flagd-ui` sidecar mounts the same emptyDir at `/app/data` — which is how the UI
  edits flags at all.

So editing the ConfigMap alone changes nothing until the pod restarts and the init
container re-copies:

```bash
kubectl edit configmap flagd-config          # set defaultVariant on the flag
kubectl rollout restart deployment/flagd
```

The restart is the cost: flagd is unavailable for a few seconds and every service
re-resolves its flags, which is itself a small perturbation at the start of the fault
window. Restarting flagd also restarts the `flagd-ui` sidecar.

The closer analogue — rewriting the file in place inside the running pod, which is what
the compose injector does and what flagd's file watch is designed for — would be
`kubectl exec deploy/flagd -c flagd-ui -- …` against `/app/data/demo.flagd.json` on the
shared emptyDir. **Unverified**: whether the `flagd-ui` image has a shell and the
utilities to do it. The `flagd` container itself is a distroless-style flagd build and
almost certainly cannot.

Flags this affects, unchanged from compose: `cartFailure`, `paymentFailure`,
`adFailure` (error_rate), `paymentUnreachable` (dependency_failure),
`kafkaQueueProblems` (queue_backlog), `emailMemoryLeak` (memory_leak),
`recommendationCacheFailure` (cache_slowdown), and `loadGeneratorTraffic` /
`loadGeneratorVUs` for traffic control.

### Summary

| fault | compose | Kubernetes | faithful? |
| --- | --- | --- | --- |
| `cpu_saturation` | `docker update --cpus` | `kubectl patch pod --subresource resize` | yes, where in-place resize is available |
| `memory_leak` | `docker exec -d` balloon / flag | `kubectl exec` + `nohup … &` / flag | yes |
| `network_latency` | netshoot sidecar + `tc netem` | `kubectl debug --profile=netadmin` + `tc netem` | yes |
| `packet_loss` | same | same | yes |
| `dependency_failure` | `docker pause` / flag | NetworkPolicy (timeout) or scale to 0 (refused) / flag | approximate |
| `error_rate` | flag file rewrite | ConfigMap edit + `rollout restart` | approximate (restart) |
| `queue_backlog` | `docker pause` / flag | scale to 0 / flag | approximate |
| `cache_slowdown` | netem on backend / flag | `kubectl debug` netem / flag | yes |

## Open items

* No `KubernetesFaultInjector`. `rca bench run-campaign` is docker-compose-only.
* `rca/benchmark/ingest.py` does not map `kubeletstats` metric names, so resource
  metrics from a Kubernetes capture are dropped (§5).
* Multi-node clusters need an RWX capture volume or a single-replica collector
  (`opentelemetry-collector.mode: deployment`); see the comments in `rca-serve.yaml`.
