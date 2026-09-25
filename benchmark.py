"""
benchmark.py -- Sybil benchmark with a realistic threat model.

Attack edges (honest<->sybil) are the scarce resource and are an explicit,
controlled parameter. The attacker can freely add sybil<->sybil edges, so
"degree matching" is done INSIDE the sybil region (mimicry), never by
wiring sybils to honest nodes.
"""
import networkx as nx
import numpy as np
from generate_graphs import _make_region, _largest_connected_component

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
        farm = nx.relabel_nodes(farm, {o: next_id + i for i, o in enumerate(farm.nodes())})
        next_id += farm.number_of_nodes()
        nodes = list(farm.nodes())
        if mimic:  # attacker densifies the farm until avg degree matches honest
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
