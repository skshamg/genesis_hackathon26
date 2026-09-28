"""Topology-only features with an explicit PPR ablation switch."""

import networkx as nx
import numpy as np


def personalized_pagerank_from_seeds(G, seeds, alpha=0.85):
    personalization = {n: (1.0 / len(seeds) if n in seeds else 0.0) for n in G.nodes()}
    try:
        return nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=300)
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=1000, tol=1e-4)


def egonet_density(G, node):
    ego_nodes = set(G.neighbors(node)) | {node}
    ego = G.subgraph(ego_nodes)
    n = ego.number_of_nodes()
    if n < 2:
        return 0.0
    return ego.number_of_edges() / (n * (n - 1) / 2)


def extract_features(G, seed_fraction=0.05, seed=0, include_ppr=True):
    """Return structural features; PPR can be disabled for a clean ablation."""
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    honest_nodes = [n for n, d in G.nodes(data=True) if d.get("label", 0) == 0]
    n_seeds = max(3, int(len(honest_nodes) * seed_fraction))
    seeds = list(rng.choice(honest_nodes, size=min(n_seeds, len(honest_nodes)), replace=False))

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    core_num = nx.core_number(G)
    max_deg = max(degree.values()) or 1
    max_core = max(core_num.values()) or 1

    feats = []
    for n in nodes:
        row = [
            degree[n] / max_deg,
            clustering[n],
            core_num[n] / max_core,
            egonet_density(G, n),
        ]
        feats.append(row)

    X = np.asarray(feats, dtype=np.float64)

    if include_ppr:
        ppr = personalized_pagerank_from_seeds(G, seeds)
        p = np.array([ppr[n] for n in nodes], dtype=np.float64)
        p = np.log1p(p * len(nodes))
        span = (p.max() - p.min()) or 1.0
        p = (p - p.min()) / span
        X = np.column_stack([X, p])

    y = np.array([G.nodes[n].get("label", 0) for n in nodes], dtype=np.int64)
    node_index = {n: i for i, n in enumerate(nodes)}
    return X, y, node_index, seeds


def extract_features_with_seeds(G, node_index, seeds, include_ppr=True):
    """
    Same feature computation as extract_features(), but takes a FIXED
    node_index and a FIXED trusted-seed set explicitly instead of rebuilding
    node_index from G.nodes() and resampling seeds from label data on every
    call.

    Needed for feature-aware attacks / adversarial training: as the
    attacker mutates the graph, features must be recomputed against the
    SAME trusted seeds throughout (a real deployed pipeline's seeds don't
    move just because an attacker added edges), and against the SAME node
    ordering the caller's A_hat/y/model already use. Safe whenever only
    edges are added or removed -- never nodes, so node_index stays valid.
    """
    nodes = [None] * len(node_index)
    for n, i in node_index.items():
        nodes[i] = n

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    core_num = nx.core_number(G)
    max_deg = max(degree.values()) or 1
    max_core = max(core_num.values()) or 1

    feats = []
    for n in nodes:
        row = [
            degree[n] / max_deg,
            clustering[n],
            core_num[n] / max_core,
            egonet_density(G, n),
        ]
        feats.append(row)

    X = np.asarray(feats, dtype=np.float64)

    if include_ppr:
        ppr = personalized_pagerank_from_seeds(G, seeds)
        p = np.array([ppr[n] for n in nodes], dtype=np.float64)
        p = np.log1p(p * len(nodes))
        span = (p.max() - p.min()) or 1.0
        p = (p - p.min()) / span
        X = np.column_stack([X, p])

    return X


def normalized_adjacency(G, node_index):
    n = len(node_index)
    A = np.zeros((n, n), dtype=np.float64)
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        A[i, j] = A[j, i] = 1.0
    A += np.eye(n)
    deg = A.sum(axis=1)
    inv = np.zeros_like(deg)
    mask = deg > 0
    inv[mask] = np.power(deg[mask], -0.5)
    D = np.diag(inv)
    return D @ A @ D