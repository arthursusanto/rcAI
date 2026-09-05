"""Static topology of the simulated OpenTelemetry Demo.

Service profiles (CPU cost, concurrency, memory, runtime) plus the call trees of the
request types the load generator drives. Everything here is deterministic data; the
random parts of the simulation live in :mod:`rca.sim.generate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rca.data.schema import INFRA_OWNER, SERVICES

INFRA: list[str] = list(INFRA_OWNER)          # kafka, valkey, postgresql, flagd, load-generator
ALL_COMPONENTS: list[str] = SERVICES + INFRA

# Reference entry rate the capacity constants below are calibrated at: the mean of the
# generated traffic range, so the healthy cpu_util *distribution* matches the measured
# OTel Demo runs rather than only its top end.
REF_RPS = 8.25

CLIENT_OVERHEAD_MS = 0.05       # caller-side serialisation work per outbound call


@dataclass(frozen=True)
class Profile:
    base_ms: float              # mean own service time per call (== span self-time)
    hop_ms: float               # network + serialisation a caller pays to reach this one
    cores: float                # concurrency for the M/M/c waiting-time approximation
    base_util: float            # CPU utilisation at REF_RPS (calibrates cpu capacity)
    idle_util: float            # utilisation with no traffic at all
    runtime: str
    mem_base_mb: float
    mem_limit_mb: float
    threads_base: int


# base_ms and base_util are calibrated against the measured OTel Demo runs: base_ms is
# the median span self-time per service and base_util the median cpu_util (fraction of a
# 1-core allotment) on healthy windows at REF_RPS. hop_ms is the per-call overhead a
# caller sees on top of the callee's server span, which for the cheap gRPC services is
# larger than their own service time -- that is what makes inbound/server ratios big.
PROFILES: dict[str, Profile] = {
    #                     base_ms  hop_ms cores base_util idle_util runtime  mem_mb limit thr
    "frontend":        Profile(9.60, 1.20, 4, 0.062, 0.015, "node",   180, 512, 12),
    "frontend-proxy":  Profile(0.22, 1.60, 6, 0.043, 0.011, "native",  48, 256, 10),
    "cart":            Profile(0.28, 1.10, 4, 0.015, 0.004, "dotnet", 140, 384, 20),
    "checkout":        Profile(15.70, 1.30, 4, 0.019, 0.005, "go",     70, 256, 14),
    "currency":        Profile(0.04, 0.90, 4, 0.015, 0.003, "cpp",     32, 128,  6),
    "email":           Profile(6.40, 1.80, 2, 0.041, 0.010, "ruby",   110, 256,  5),
    "payment":         Profile(0.93, 2.30, 3, 0.252, 0.050, "node",   150, 384,  9),
    "product-catalog": Profile(2.35, 1.90, 4, 0.035, 0.008, "go",      80, 256, 16),
    "recommendation":  Profile(3.90, 3.20, 2, 0.037, 0.009, "python", 190, 512,  4),
    "shipping":        Profile(0.23, 1.10, 4, 0.012, 0.003, "rust",    28, 128,  8),
    "ad":              Profile(1.89, 2.00, 3, 0.018, 0.006, "java",   260, 640, 24),
    "quote":           Profile(0.48, 1.80, 2, 0.019, 0.005, "php",     60, 192,  4),
    "accounting":      Profile(49.00, 1.00, 2, 0.005, 0.002, "dotnet", 160, 384, 10),
    "fraud-detection": Profile(23.50, 1.00, 2, 0.008, 0.003, "kotlin", 240, 512, 14),
    "image-provider":  Profile(0.05, 0.80, 6, 0.012, 0.004, "native",  36, 128,  8),
    # infra components: telemetry only, never a root-cause label; their base_ms
    # mirrors INFRA_CALL_MS, which is what the client spans into them cost
    "kafka":           Profile(0.50, 0.30, 4, 0.030, 0.010, "jvm",    420, 1024, 30),
    "valkey":          Profile(0.15, 0.15, 2, 0.010, 0.003, "native",  96, 256,  4),
    "postgresql":      Profile(0.90, 0.35, 4, 0.040, 0.008, "native", 320, 768, 18),
    "flagd":           Profile(0.12, 0.15, 4, 0.008, 0.002, "go",      40, 128,  6),
    "load-generator":  Profile(0.50, 0.50, 8, 0.030, 0.008, "python", 120, 256, 16),
}

# Runtimes with a stop-the-world collector: they show a memory sawtooth and gc pauses.
GC_RUNTIMES = {"node", "dotnet", "java", "kotlin", "ruby", "jvm"}

# Per-operation overrides of Profile.base_ms.
OP_BASE_MS: dict[tuple[str, str], float] = {
    ("product-catalog", "ListProducts"): 3.4,
    ("product-catalog", "GetProduct"): 1.5,
    ("cart", "GetCart"): 0.24,
    ("cart", "AddItem"): 0.36,
    ("cart", "EmptyCart"): 0.20,
    ("shipping", "ShipOrder"): 0.34,
}

# Infra calls: the base_ms of the client span is the work done inside the component.
INFRA_CALL_MS = {"valkey": 0.15, "postgresql": 0.90, "flagd": 0.12, "kafka": 0.50}

# cache_slowdown is only meaningful where the service has a cache/store in front of it.
CACHE_PEER: dict[str, str] = {
    "cart": "valkey",
    "product-catalog": "postgresql",
    "recommendation": "",           # in-process cache, emitted as an internal span
    # accounting is deliberately absent: netem for a postgresql slowdown goes on the
    # shared astronomy-db, and product-catalog is the service that visibly suffers, so
    # labelling the experiment "accounting" would be labelling the wrong service.
}

KAFKA_CONSUMERS = ["accounting", "fraud-detection"]
KAFKA_PRODUCER = "checkout"


# --- call trees ---------------------------------------------------------------------

@dataclass
class Node:
    service: str
    operation: str
    kind: str                       # server | client | producer | consumer | internal
    peer: str = ""
    group: int = 0                  # 0 = sequential; equal consecutive >0 run in parallel
    optional: bool = False          # parent ignores an error from this subtree
    cache: bool = False             # affected by cache_slowdown on the emitting service
    is_async: bool = False          # does not contribute to the parent duration
    prob: float = 1.0               # fraction of traces this node is present in
    children: list[Node] = field(default_factory=list)


def call(caller: str, callee: str, op: str, *children: Node, group: int = 0,
         optional: bool = False, prob: float = 1.0) -> Node:
    """A synchronous RPC: a client span in caller wrapping a server span in callee."""
    server = Node(callee, op, "server", children=list(children))
    return Node(caller, callee + "/" + op, "client", peer=callee, group=group,
                optional=optional, prob=prob, children=[server])


def infra_call(service: str, component: str, op: str, group: int = 0, cache: bool = False,
               prob: float = 1.0) -> Node:
    """A call to an infra component: a single client span, no server span."""
    return Node(service, component + "/" + op, "client", peer=component, group=group,
                cache=cache, prob=prob)


def flag(service: str, prob: float = 0.15) -> Node:
    return infra_call(service, "flagd", "resolve", prob=prob)


def db(service: str, op: str, group: int = 0) -> Node:
    return infra_call(service, "postgresql", op, group=group, cache=True)


def kv(op: str, group: int = 0) -> Node:
    return infra_call("cart", "valkey", op, group=group, cache=True)


def produce(*consumers: Node) -> Node:
    return Node(KAFKA_PRODUCER, "orders publish", "producer", peer="kafka",
                children=list(consumers))


def consume(service: str, *children: Node) -> Node:
    return Node(service, "orders process", "consumer", peer="kafka", is_async=True,
                children=list(children))


def _browse() -> Node:
    return call(
        "load-generator", "frontend-proxy", "GET /api/products",
        flag("frontend-proxy"),
        call(
            "frontend-proxy", "frontend", "GET /api/products",
            call("frontend", "product-catalog", "ListProducts",
                 db("product-catalog", "SELECT products")),
            call("frontend", "recommendation", "ListRecommendations",
                 Node("recommendation", "cache lookup", "internal", cache=True),
                 call("recommendation", "product-catalog", "GetProduct",
                      db("product-catalog", "SELECT product")),
                 group=1, optional=True, prob=0.10),
            call("frontend", "ad", "GetAds", flag("ad"), group=1, optional=True, prob=0.06),
            call("frontend", "currency", "Convert", group=1, prob=0.06),
            flag("frontend")))


def _view_cart() -> Node:
    return call(
        "load-generator", "frontend-proxy", "GET /api/cart",
        call(
            "frontend-proxy", "frontend", "GET /api/cart",
            call("frontend", "cart", "GetCart", kv("HGET"), flag("cart")),
            call("frontend", "product-catalog", "GetProduct",
                 db("product-catalog", "SELECT product"), group=1, prob=0.16),
            call("frontend", "currency", "Convert", group=1, prob=0.20),
            call("frontend", "shipping", "GetQuote",
                 call("shipping", "quote", "GetQuote"), group=1, prob=0.03)))


def _add_to_cart() -> Node:
    return call(
        "load-generator", "frontend-proxy", "POST /api/cart",
        call(
            "frontend-proxy", "frontend", "POST /api/cart",
            call("frontend", "product-catalog", "GetProduct",
                 db("product-catalog", "SELECT product")),
            call("frontend", "cart", "AddItem", kv("HSET"))))


def _checkout() -> Node:
    return call(
        "load-generator", "frontend-proxy", "POST /api/checkout",
        call(
            "frontend-proxy", "frontend", "POST /api/checkout",
            call(
                "frontend", "checkout", "PlaceOrder",
                call("checkout", "cart", "GetCart", kv("HGET")),
                call("checkout", "product-catalog", "GetProduct",
                     db("product-catalog", "SELECT product"), group=1),
                call("checkout", "currency", "Convert", group=1),
                call("checkout", "shipping", "GetQuote",
                     call("shipping", "quote", "GetQuote"), group=1),
                call("checkout", "payment", "Charge", flag("payment")),
                call("checkout", "shipping", "ShipOrder",
                     call("shipping", "quote", "GetQuote", prob=0.10)),
                call("checkout", "email", "SendOrderConfirmation", optional=True),
                produce(
                    consume("accounting", db("accounting", "INSERT order")),
                    consume("fraud-detection", flag("fraud-detection", prob=0.3))),
                call("checkout", "cart", "EmptyCart", kv("DEL")),
                flag("checkout"))))


def _recommendations() -> Node:
    return call(
        "load-generator", "frontend-proxy", "GET /api/recommendations",
        call(
            "frontend-proxy", "frontend", "GET /api/recommendations",
            call("frontend", "recommendation", "ListRecommendations",
                 Node("recommendation", "cache lookup", "internal", cache=True),
                 call("recommendation", "product-catalog", "ListProducts",
                      db("product-catalog", "SELECT products")),
                 flag("recommendation"))))


def _ad_request() -> Node:
    return call(
        "load-generator", "frontend-proxy", "GET /api/data",
        call(
            "frontend-proxy", "frontend", "GET /api/data",
            call("frontend", "ad", "GetAds", flag("ad", prob=0.4))))


def _image_fetch() -> Node:
    return call(
        "load-generator", "frontend-proxy", "GET /images/product.jpg",
        call("frontend-proxy", "image-provider", "GET /image"))


REQUEST_TYPES: dict[str, Node] = {
    "browse_product": _browse(),
    "view_cart": _view_cart(),
    "add_to_cart": _add_to_cart(),
    "checkout": _checkout(),
    "recommendations": _recommendations(),
    "ad_request": _ad_request(),
    "image_fetch": _image_fetch(),
}

# Calibrated so the per-service request share, normalised to the frontend-proxy rate,
# matches the measured OTel Demo k6 profile: the listing page dominates and checkout is
# a rare event, so payment/email/quote/accounting see only a few spans per window.
DEFAULT_MIX: dict[str, float] = {
    "browse_product": 0.52,
    "view_cart": 0.19,
    "add_to_cart": 0.06,
    "checkout": 0.03,
    "recommendations": 0.02,
    "ad_request": 0.02,
    "image_fetch": 0.16,
}

TYPE_NAMES = list(REQUEST_TYPES)


# --- flattening ---------------------------------------------------------------------

@dataclass
class FlatNode:
    idx: int
    parent: int                     # -1 for the trace root
    service: str
    operation: str
    kind: str
    peer: str
    optional: bool
    cache: bool
    is_async: bool
    prob: float
    base_ms: float                  # mean own CPU demand
    queue_service: str              # component whose waiting factor inflates this work
    cpu_service: str                # component the CPU demand is charged to
    blocks: list[list[int]]         # synchronous children, grouped into parallel blocks
    async_children: list[int]


def _base_ms(node: Node) -> float:
    if node.kind in ("server", "consumer"):
        return OP_BASE_MS.get((node.service, node.operation), PROFILES[node.service].base_ms)
    if node.kind == "internal":
        return 0.3
    if node.peer in INFRA_CALL_MS:
        return INFRA_CALL_MS[node.peer]
    return CLIENT_OVERHEAD_MS


def _queue_service(node: Node) -> str:
    if node.kind in ("server", "consumer"):
        return node.service
    if node.peer in INFRA_CALL_MS:
        return node.peer
    return ""


def _cpu_service(node: Node) -> str:
    if node.kind in ("client", "producer") and node.peer in INFRA_CALL_MS:
        return node.peer
    return node.service


def flatten(root: Node) -> list[FlatNode]:
    """Depth-first flatten; a parent always has a lower index than its children."""
    flat: list[FlatNode] = []

    def visit(node: Node, parent: int) -> int:
        idx = len(flat)
        flat.append(FlatNode(idx, parent, node.service, node.operation, node.kind, node.peer,
                             node.optional, node.cache, node.is_async, node.prob,
                             _base_ms(node), _queue_service(node), _cpu_service(node), [], []))
        blocks: list[list[int]] = []
        prev_group = 0
        for child in node.children:
            cidx = visit(child, idx)
            if child.is_async:
                flat[idx].async_children.append(cidx)
                continue
            if child.group != 0 and child.group == prev_group and blocks:
                blocks[-1].append(cidx)
            else:
                blocks.append([cidx])
            prev_group = child.group
        flat[idx].blocks = blocks
        return idx

    visit(root, -1)
    return flat


FLAT_TYPES: dict[str, list[FlatNode]] = {n: flatten(t) for n, t in REQUEST_TYPES.items()}

# (caller, callee) pairs that actually occur in the call trees.
CALL_EDGES: set[tuple[str, str]] = {
    (n.service, n.peer) for flat in FLAT_TYPES.values() for n in flat
    if n.kind in ("client", "producer") and n.peer
}


def reachability(flat: list[FlatNode]) -> list[float]:
    """Probability each node is present in a trace: its own prob times its ancestors'."""
    out: list[float] = []
    for n in flat:                      # a parent always precedes its children
        out.append(n.prob if n.parent < 0 else out[n.parent] * n.prob)
    return out


def reference_demand_ms(rps: float, mix: dict[str, float]) -> dict[str, float]:
    """Expected CPU demand in ms/s per component at rps entry requests per second."""
    out = {c: 0.0 for c in ALL_COMPONENTS}
    for name, flat in FLAT_TYPES.items():
        share = rps * mix.get(name, 0.0)
        for n, reach in zip(flat, reachability(flat), strict=True):
            out[n.cpu_service] += share * reach * n.base_ms
    return out


_REF_DEMAND = reference_demand_ms(REF_RPS, DEFAULT_MIX)

# ms of CPU available per wall-clock second, chosen so each component sits at its
# Profile.base_util while the system runs at REF_RPS.
CPU_CAPACITY_MS: dict[str, float] = {
    c: max(1.0, _REF_DEMAND[c] / max(1e-6, PROFILES[c].base_util - PROFILES[c].idle_util))
    for c in ALL_COMPONENTS
}

# Bytes moved per inbound call, used for the cumulative network counters.
NET_RX_BYTES = {c: 900.0 + 210.0 * (i % 7) for i, c in enumerate(ALL_COMPONENTS)}
NET_TX_BYTES = {c: 2600.0 + 640.0 * (i % 5) for i, c in enumerate(ALL_COMPONENTS)}
