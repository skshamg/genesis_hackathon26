from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from sybil_shield.core import model_store
from sybil_shield.experiments.learned_gcn import LearnedTopologyGCN, decode_seed_mask
from sybil_shield.experiments.benchmark import make_transfer_train_set


def evaluate_learned_gcn(
    *,
    model_type: str = "learned_gcn",
    model_id: str | None = None,
    num_graphs: int = 2,
    min_nodes: int = 80,
    max_nodes: int = 120,
    sybil_fraction_min: float = 0.10,
    sybil_fraction_max: float = 0.35,
    seed: int = 7,
    epochs: int = 1,
    learning_rate: float = 0.01,
    hidden_dim: int = 8,
    n_prop_layers: int = 2,
):
    train_graphs = make_transfer_train_set(
        num_graphs=num_graphs,
        small_nodes=(max(50, min_nodes), max_nodes),
        large_nodes=(max(800, min_nodes), max(1500, max_nodes)),
        sybil_fraction_range=(sybil_fraction_min, sybil_fraction_max),
        seed=seed,
    )

    if model_id is not None:
        model = model_store.load_model(model_type, model_id)
    else:
        model = LearnedTopologyGCN(hidden_dim=hidden_dim, n_prop_layers=n_prop_layers, seed=seed)
        for G in train_graphs:
            nodes = list(G.nodes())
            node_index = {n: i for i, n in enumerate(nodes)}
            y = np.array([G.nodes[n].get("label", 0) for n in nodes], dtype=np.int64)
            seeds = [n for n in nodes if G.nodes[n].get("label", 0) == 0][: max(1, len(nodes) // 20)]
            X = np.zeros((len(nodes), 1), dtype=np.float64)
            for s in seeds:
                if s in node_index:
                    X[node_index[s], 0] = 1.0
            A = nx_to_adj(G, nodes)
            A += np.eye(len(nodes))
            d = A.sum(1) ** -0.5
            A_hat = d[:, None] * A * d[None, :]
            class_weight = cw(y)
            model.fit_step(A_hat, X, y, class_weight=class_weight, lr=learning_rate)

    all_y = []
    all_scores = []
    for G in train_graphs:
        nodes = list(G.nodes())
        node_index = {n: i for i, n in enumerate(nodes)}
        y = np.array([G.nodes[n].get("label", 0) for n in nodes], dtype=np.int64)
        seeds = [n for n in nodes if G.nodes[n].get("label", 0) == 0][: max(1, len(nodes) // 20)]
        X = np.zeros((len(nodes), 1), dtype=np.float64)
        for s in seeds:
            if s in node_index:
                X[node_index[s], 0] = 1.0
        A = nx_to_adj(G, nodes)
        A += np.eye(len(nodes))
        d = A.sum(1) ** -0.5
        A_hat = d[:, None] * A * d[None, :]
        probs = model.predict_proba(A_hat, X)[:, 1]
        all_y.append(y)
        all_scores.append(probs)

    y = np.concatenate(all_y)
    scores = np.concatenate(all_scores)
    preds = (scores >= 0.5).astype(int)
    auc = float(roc_auc_score(y, scores))
    acc = float(accuracy_score(y, preds))

    print(f"learned_gcn eval: auc={auc:.4f} accuracy={acc:.4f}")
    return {"auc": auc, "accuracy": acc}


def nx_to_adj(G, nodes):
    import networkx as nx

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr").toarray()
    return A


def cw(y):
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    counts = counts + 1e-8
    return np.array([1.0 / counts[0], 1.0 / counts[1]], dtype=np.float64)


if __name__ == "__main__":
    evaluate_learned_gcn()
