"""Pipeline-level (feature-recomputing) evasion attacks: add sybil->honest edges."""
import numpy as np
from detector import trust_scores, pick_seeds

def attack_graph(G, n_edges, mode="random", rng=None, seed_knowledge=0):
    """
    mode: 'random'  - uniform honest targets
          'hubs'    - honest nodes sampled proportional to degree
          'trusted' - attacker knows seeds, targets top-decile ppr/deg honest nodes
    Returns an attacked COPY (labels untouched).
    """
    rng = rng or np.random.default_rng(0)
    G = G.copy()
    nodes = list(G.nodes())
    honest = [n for n in nodes if G.nodes[n]["label"] == 0]
    sybil = [n for n in nodes if G.nodes[n]["label"] == 1]
    if mode == "random":
        p = None; pool = honest
    elif mode == "hubs":
        d = np.array([G.degree(n) for n in honest], float); p = d / d.sum(); pool = honest
    elif mode == "trusted":
        ts = trust_scores(G, nodes, pick_seeds(G, seed_knowledge))["ppr0.8/deg"]
        sc = {n: ts[i] for i, n in enumerate(nodes)}
        pool = sorted(honest, key=lambda n: -sc[n])[: max(10, len(honest) // 10)]; p = None
    else:
        raise ValueError(mode)
    added = 0; tries = 0
    while added < n_edges and tries < 50 * n_edges + 100:
        tries += 1
        s = int(rng.choice(sybil)); h = int(rng.choice(pool, p=p))
        if not G.has_edge(s, h):
            G.add_edge(s, h); added += 1
    return G
