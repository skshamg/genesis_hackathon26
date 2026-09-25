"""Feature extraction and graph scoring utilities."""

import math

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata


def _adj(G, nodes):
    return nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")


def pick_seeds(G, seed=0, frac=0.05):
    rng = np.random.RandomState(seed)
    honest = [n for n, d in G.nodes(data=True) if d.get("label") == 0]
    k = max(3, int(len(honest) * frac))
    return list(rng.choice(honest, size=min(k, len(honest)), replace=False))


def ppr_vector(A, deg, s, alpha, iters=200):
    inv = 1.0 / np.maximum(deg, 1)
    pi = s.copy()
    for _ in range(iters):
        pi = alpha * (A @ (inv * pi)) + (1 - alpha) * s
    return pi


def early_walk(A, deg, s, steps):
    inv = 1.0 / np.maximum(deg, 1)
    pi = s.copy()
    for _ in range(steps):
        pi = A @ (inv * pi)
    return pi


def trust_scores(G, nodes, seeds, alphas=(0.5, 0.8, 0.95)):
    """Returns dict name -> per-node trust score (higher = more honest)."""
    A = _adj(G, nodes)
    deg = np.asarray(A.sum(1)).ravel()
    idx = {n: i for i, n in enumerate(nodes)}
    s = np.zeros(len(nodes))
    s[[idx[x] for x in seeds]] = 1.0 / len(seeds)
    out = {}
    for a in alphas:
        pi = ppr_vector(A, deg, s, a)
        out[f"ppr{a}"] = pi
        out[f"ppr{a}/deg"] = pi / np.maximum(deg, 1)
    t = int(math.ceil(math.log2(len(nodes))))
    out["sybilrank"] = early_walk(A, deg, s, t) / np.maximum(deg, 1)
    return out


def _rank(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def build_features(G, seed=0, use_trust=True, use_struct=True, seed_frac=0.05):
    nodes = list(G.nodes())
    seeds = pick_seeds(G, seed, frac=seed_frac)
    cols = []
    if use_trust:
        ts = trust_scores(G, nodes, seeds)
        for k in ("ppr0.5/deg", "ppr0.8/deg", "ppr0.95/deg", "sybilrank"):
            cols.append(_rank(np.log(ts[k] + 1e-12)))
    if use_struct:
        cl = nx.clustering(G)
        core = nx.core_number(G)
        cols.append(_rank([G.degree(n) for n in nodes]))
        cols.append(_rank([cl[n] for n in nodes]))
        cols.append(_rank([core[n] for n in nodes]))
        cols.extend(_rank(feature) for feature in extra_structural_features(G, nodes, seeds, seed=seed))
    X = np.column_stack(cols)
    y = np.array([G.nodes[n].get("label", 0) for n in nodes])
    A = _adj(G, nodes).toarray()
    A += np.eye(len(nodes))
    d = A.sum(1) ** -0.5
    A_hat = d[:, None] * A * d[None, :]
    return X, y, A_hat, nodes, seeds


def sweep_cut(G, nodes, score, min_frac=0.4, max_frac=0.97):
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    order = np.argsort(-score, kind="stable")
    deg = np.array([G.degree(v) for v in nodes], float)
    total_vol = deg.sum()
    nbrs = [[idx[u] for u in G.neighbors(v)] for v in nodes]
    inset = np.zeros(n, bool)
    cut = 0.0
    vol = 0.0
    best = (np.inf, None)
    for k, v in enumerate(order, 1):
        e_in = sum(inset[u] for u in nbrs[v])
        cut += deg[v] - 2 * e_in
        vol += deg[v]
        inset[v] = True
        if min_frac * n <= k <= max_frac * n:
            phi = cut / max(min(vol, total_vol - vol), 1e-9)
            if phi < best[0]:
                best = (phi, k)
    k = best[1]
    honest_side = np.zeros(n, bool)
    honest_side[order[:k]] = True
    cut_edges = [(nodes[i], nodes[j]) for i in range(n) if honest_side[i] for j in nbrs[i] if not honest_side[j]]
    return ~honest_side, best[0], cut_edges


def extra_structural_features(G, nodes, trusted_seeds, seed=0):
    """Return five topology/trust-robustness features used by SYBIL-SHIELD.

    1. mean neighbor-overlap (edge-level camouflage/community support)
    2. approximate betweenness centrality (bridge/choke structure)
    3. variance of PPR trust across disjoint benign-seed subsets
    4. degree / mean-neighbor-degree ratio
    5. shortest-path distance to the nearest trusted benign seed
    """
    nodes = list(nodes)
    node_set = set(nodes)
    degree = dict(G.degree())

    # 1) Mean Jaccard overlap over incident edges.
    overlap = []
    nbr_sets = {n: set(G.neighbors(n)) for n in nodes}
    for n in nodes:
        vals = []
        for v in G.neighbors(n):
            if v not in node_set:
                continue
            a, b = nbr_sets[n], nbr_sets[v]
            union = len(a | b)
            vals.append(len(a & b) / union if union else 0.0)
        overlap.append(float(np.mean(vals)) if vals else 0.0)

    # 2) Sampled betweenness to keep this feasible on larger graphs.
    k = min(100, max(1, len(nodes)))
    try:
        btw = nx.betweenness_centrality(G, k=k, normalized=True, seed=seed)
    except Exception:
        btw = nx.betweenness_centrality(G, normalized=True)
    betweenness = [btw.get(n, 0.0) for n in nodes]

    # 3) PPR disagreement across 2 disjoint subsets of trusted seeds.
    rng = np.random.RandomState(seed)
    seeds = [s for s in trusted_seeds if s in node_set]
    rng.shuffle(seeds)
    if len(seeds) >= 2:
        mid = max(1, len(seeds) // 2)
        subsets = [seeds[:mid], seeds[mid:]]
        subsets = [s for s in subsets if s]
        pprs = []
        A = _adj(G, nodes)
        deg = np.asarray(A.sum(1)).ravel()
        idx = {n: i for i, n in enumerate(nodes)}
        for subset in subsets:
            s = np.zeros(len(nodes))
            for node in subset:
                if node in idx:
                    s[idx[node]] = 1.0 / len(subset)
            pprs.append(ppr_vector(A, deg, s, 0.85))
        if len(pprs) >= 2:
            ppr_variance = np.var(np.vstack(pprs), axis=0)
        else:
            ppr_variance = np.zeros(len(nodes))
    else:
        ppr_variance = np.zeros(len(nodes))

    # 4) Local degree ratio against the mean degree of neighbors.
    degree_ratio = []
    for n in nodes:
        nbrs = [v for v in G.neighbors(n) if v in node_set]
        mean_nbr_deg = np.mean([degree[v] for v in nbrs]) if nbrs else 1.0
        degree_ratio.append(float(degree[n]) / max(float(mean_nbr_deg), 1e-12))

    # 5) Distance to the nearest trusted seed.
    seed_dist = {n: np.inf for n in nodes}
    frontier = list(seeds)
    for s in seeds:
        if s in seed_dist:
            seed_dist[s] = 0
    head = 0
    while head < len(frontier):
        u = frontier[head]
        head += 1
        du = seed_dist[u]
        for v in G.neighbors(u):
            if v in seed_dist and seed_dist[v] == np.inf:
                seed_dist[v] = du + 1
                frontier.append(v)
    finite = [d for d in seed_dist.values() if np.isfinite(d)]
    fallback = (max(finite) + 1) if finite else float(len(nodes))
    nearest_seed_distance = [float(seed_dist[n] if np.isfinite(seed_dist[n]) else fallback) for n in nodes]

    return [overlap, betweenness, ppr_variance.tolist(), degree_ratio, nearest_seed_distance]
