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


def build_features(G, seed=0, use_trust=True, use_struct=True):
    nodes = list(G.nodes())
    seeds = pick_seeds(G, seed)
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
