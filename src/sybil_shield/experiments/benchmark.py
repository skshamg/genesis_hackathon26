"""
benchmark.py -- Sybil benchmark with a realistic threat model.

Attack edges (honest<->sybil) are the scarce resource and are an explicit,
controlled parameter. The attacker can freely add sybil<->sybil edges, so
"degree matching" is done INSIDE the sybil region (mimicry), never by
wiring sybils to honest nodes.
"""
import networkx as nx
import numpy as np
from sybil_shield.core.generate_graphs import _make_region, _largest_connected_component

HONEST_FAMILIES = ["ba", "ws", "sbm"]        # training distribution
ALL_FAMILIES = ["ba", "ws", "er", "sbm"]


def make_sybil_graph(n_honest=300, honest_family="ba", sybil_family="er",
                     n_farms=3, farm_size_range=(25, 60), attack_edges=20,
                     mimic=True, target_honest_hubs=False, seed=None):
    rng = np.random.default_rng(seed)
    honest = _largest_connected_component(
        _make_region(n_honest, honest_family, int(rng.integers(0, 2**31))))
    G = nx.Graph()
    G.add_nodes_from((n, {"label": 0}) for n in honest.nodes())
    G.add_edges_from(honest.edges())
    honest_nodes = list(honest.nodes())
    h_avg_deg = 2 * honest.number_of_edges() / honest.number_of_nodes()
    hdeg = np.array([honest.degree(n) for n in honest_nodes], float)

    next_id = max(honest_nodes) + 1
    farms = []
    for _ in range(n_farms):
        size = int(rng.integers(farm_size_range[0], farm_size_range[1] + 1))
        farm = _largest_connected_component(
            _make_region(size, sybil_family, int(rng.integers(0, 2**31))))
        if farm.number_of_nodes() == 0:
            continue
        farm = nx.relabel_nodes(farm, {o: next_id + i for i, o in enumerate(farm.nodes())})
        next_id += farm.number_of_nodes()
        nodes = list(farm.nodes())
        if mimic and len(nodes) >= 2:  # attacker densifies the farm until avg degree matches honest
            tries = 0
            while 2 * farm.number_of_edges() / len(nodes) < h_avg_deg and tries < 20000:
                u, v = rng.choice(nodes, 2, replace=False)
                farm.add_edge(int(u), int(v)); tries += 1
        G.add_nodes_from((n, {"label": 1}) for n in nodes)
        G.add_edges_from(farm.edges())
        farms.append(nodes)

    # attack edges: at least one per farm (keeps graph connected), rest random
    p_h = hdeg / hdeg.sum() if target_honest_hubs else None
    placed = 0
    def add(farm_nodes):
        nonlocal placed
        for _ in range(50):
            s = int(rng.choice(farm_nodes)); h = int(rng.choice(honest_nodes, p=p_h))
            if not G.has_edge(s, h):
                G.add_edge(s, h); placed += 1; return
    for f in farms: add(f)
    while placed < max(attack_edges, len(farms)):
        add(farms[int(rng.integers(len(farms)))])
    G.graph.update(honest_family=honest_family, sybil_family=sybil_family,
                   attack_edges=placed)
    return G


def count_attack_edges(G):
    return sum(1 for u, v in G.edges() if G.nodes[u]["label"] != G.nodes[v]["label"])


def make_configurable_sybil_graph(total_nodes, sybil_fraction, rng):
    """
    make_sybil_graph() takes n_honest/farm_size_range/n_farms directly and
    has no notion of "total nodes" or "sybil fraction" -- this wraps it so
    train_model.py's frontend-facing knobs (node-count range, sybil-fraction
    range) can be honored without changing make_sybil_graph's own contract.
    sybil_fraction is a TARGET: farm sizes are randomized per-farm around
    sybil_total/n_farms, so the realized fraction on any one graph will be
    close but not exact -- fine for training-set diversity, don't rely on
    it for precise labeling.
    """
    sybil_fraction = float(min(max(sybil_fraction, 0.01), 0.9))
    total_nodes = max(20, int(total_nodes))
    n_honest = max(10, int(round(total_nodes * (1 - sybil_fraction))))
    sybil_total = max(6, total_nodes - n_honest)

    n_farms = int(rng.integers(2, 5))
    per_farm = max(2, sybil_total // n_farms)
    lo = max(2, int(per_farm * 0.7))
    hi = max(lo + 1, int(per_farm * 1.3))

    max_attack = max(4, min(300, sybil_total))
    attack_edges = int(np.exp(rng.uniform(np.log(3), np.log(max_attack))))

    return make_sybil_graph(
        n_honest=n_honest,
        honest_family=str(rng.choice(HONEST_FAMILIES)),
        sybil_family=str(rng.choice(ALL_FAMILIES)),
        n_farms=n_farms,
        farm_size_range=(lo, hi),
        attack_edges=attack_edges,
        mimic=bool(rng.integers(0, 2)),
        target_honest_hubs=bool(rng.integers(0, 2)),
        seed=int(rng.integers(0, 2**31)),
    )


def make_configurable_train_set(num_graphs=50, min_nodes=100, max_nodes=500,
                                 sybil_fraction_min=0.05, sybil_fraction_max=0.30,
                                 seed=1):
    """Synthetic training set driven by the same knobs the Train tab's
    form exposes: graph count, a node-count range, and a sybil-fraction
    range (randomized per graph within that range). Used by
    train_model.py; make_train() in eval_robust.py stays as the
    fixed-distribution generator existing callers (facebook eval's
    fresh-train fallback, tests) already depend on."""
    if min_nodes > max_nodes:
        min_nodes, max_nodes = max_nodes, min_nodes
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        total = int(rng.integers(min_nodes, max_nodes + 1))
        frac = float(rng.uniform(sybil_fraction_min, sybil_fraction_max))
        graphs.append(make_configurable_sybil_graph(total, frac, rng))
    return graphs


def make_transfer_train_set(num_graphs=40, small_nodes=(150, 500),
                             large_nodes=(1500, 4000), large_frac=0.15,
                             sybil_fraction_range=(0.10, 0.50),
                             attack_edges_range=(3, 400), seed=1):
    """Synthetic training set aimed at real-dataset transfer, not at
    train_model.py's frontend knobs -- see docs/ conversation history:
    make_configurable_train_set() (above) and eval_robust.make_train()
    both measured avg degree ~9.7-9.8 and (for the CLI-default path)
    mean sybil fraction ~13.5%, against a real Facebook eval graph built
    as an exactly-50/50 benign_test/sybil_test split with avg degree
    ~43.7 -- two independent, compounding distribution gaps. This
    generator targets both directly:

      - sybil_fraction drawn from sybil_fraction_range, not capped at
        0.30 -- every real eval you actually run constructs a
        balanced-ish test population.
      - attack_edges drawn independently of sybil population size
        (log-uniform over attack_edges_range), not
        min(300, sybil_total) -- make_configurable_sybil_graph's cap
        silently starves attack-edge exposure whenever sybil_fraction
        happens to be small. A small sybil population can still face a
        dense, many-edges attack.
      - `large_frac` of graphs are drawn from `large_nodes` instead of
        `small_nodes`, to give the model at least some exposure to
        graphs an order of magnitude closer to real deployment scale.
        Kept as a MINORITY of graphs (default 15%) because build_features/
        trust_scores materialize dense NxN arrays -- a handful of
        large graphs per training run is affordable, uniformly scaling
        every graph to that size is not.

    Existing callers (make_configurable_train_set, make_train) are left
    exactly as they were; this is purely additive.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        is_large = rng.random() < large_frac
        lo, hi = large_nodes if is_large else small_nodes
        total = int(rng.integers(lo, hi + 1))
        frac = float(rng.uniform(*sybil_fraction_range))
        n_honest = max(10, int(round(total * (1 - frac))))
        sybil_total = max(6, total - n_honest)

        n_farms = int(rng.integers(2, 5))
        per_farm = max(2, sybil_total // n_farms)
        farm_lo = max(2, int(per_farm * 0.7))
        farm_hi = max(farm_lo + 1, int(per_farm * 1.3))

        attack_edges = int(np.exp(rng.uniform(
            np.log(attack_edges_range[0]), np.log(attack_edges_range[1]))))

        graphs.append(make_sybil_graph(
            n_honest=n_honest,
            honest_family=str(rng.choice(HONEST_FAMILIES)),
            sybil_family=str(rng.choice(ALL_FAMILIES)),
            n_farms=n_farms,
            farm_size_range=(farm_lo, farm_hi),
            attack_edges=attack_edges,
            mimic=bool(rng.integers(0, 2)),
            target_honest_hubs=bool(rng.integers(0, 2)),
            seed=int(rng.integers(0, 2**31)),
        ))
    return graphs
