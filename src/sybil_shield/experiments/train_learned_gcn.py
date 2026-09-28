from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np
from scipy.sparse import eye
from sklearn.metrics import accuracy_score, roc_auc_score

from sybil_shield.core import model_store
from sybil_shield.experiments.benchmark import make_transfer_train_set
from sybil_shield.experiments.learned_gcn import LearnedTopologyGCN, seed_mask_features


def _normalized_adjacency(G, nodes):
    import networkx as nx

    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr").toarray()
    A = A + np.eye(len(nodes))
    d = A.sum(1) ** -0.5
    return d[:, None] * A * d[None, :]


def _class_weights(y):
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    counts = counts + 1e-8
    return np.array([1.0 / counts[0], 1.0 / counts[1]], dtype=np.float64)


def train_learned_gcn(
    *,
    model_type: str = "learned_gcn",
    num_graphs: int = 20,
    min_nodes: int = 80,
    max_nodes: int = 120,
    sybil_fraction_min: float = 0.10,
    sybil_fraction_max: float = 0.35,
    epochs: int = 30,
    learning_rate: float = 0.01,
    hidden_dim: int = 16,
    n_prop_layers: int = 2,
    seed: int = 7,
    model_id: str | None = None,
):
    train_graphs = make_transfer_train_set(
        num_graphs=num_graphs,
        small_nodes=(max(50, min_nodes), max_nodes),
        large_nodes=(max(800, min_nodes), max(1500, max_nodes)),
        sybil_fraction_range=(sybil_fraction_min, sybil_fraction_max),
        seed=seed,
    )

    model = LearnedTopologyGCN(
        hidden_dim=hidden_dim,
        n_prop_layers=n_prop_layers,
        n_classes=2,
        seed=seed,
    )

    last_loss = None
    for _ in range(epochs):
        for G in train_graphs:
            nodes = list(G.nodes())
            node_index = {n: i for i, n in enumerate(nodes)}
            y = np.array([G.nodes[n].get("label", 0) for n in nodes], dtype=np.int64)
            honest_nodes = [n for n in nodes if G.nodes[n].get("label", 0) == 0]
            seed_set = honest_nodes[: max(1, len(nodes) // 20)]
            X = seed_mask_features(G, node_index, seed_set)
            A_hat = _normalized_adjacency(G, nodes)
            last_loss = model.fit_step(A_hat, X, y, class_weight=_class_weights(y), lr=learning_rate)

    all_y = []
    all_scores = []
    for G in train_graphs:
        nodes = list(G.nodes())
        node_index = {n: i for i, n in enumerate(nodes)}
        y = np.array([G.nodes[n].get("label", 0) for n in nodes], dtype=np.int64)
        honest_nodes = [n for n in nodes if G.nodes[n].get("label", 0) == 0]
        seed_set = honest_nodes[: max(1, len(nodes) // 20)]
        X = seed_mask_features(G, node_index, seed_set)
        A_hat = _normalized_adjacency(G, nodes)
        probs = model.predict_proba(A_hat, X)[:, 1]
        all_y.append(y)
        all_scores.append(probs)

    y = np.concatenate(all_y)
    scores = np.concatenate(all_scores)
    preds = (scores >= 0.5).astype(int)
    auc = float(roc_auc_score(y, scores))
    acc = float(accuracy_score(y, preds))

    params = {
        "num_graphs": num_graphs,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "sybil_fraction_min": sybil_fraction_min,
        "sybil_fraction_max": sybil_fraction_max,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "n_prop_layers": n_prop_layers,
        "seed": seed,
        "feature_source": "learned",
    }
    metrics = {
        "train_loss": float(last_loss) if last_loss is not None else None,
        "auc": auc,
        "accuracy": acc,
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "n_prop_layers": int(n_prop_layers),
    }

    meta = model_store.save_model(
        model,
        model_type=model_type,
        params=params,
        metrics=metrics,
        model_id=model_id,
    )
    print(f"saved model: {meta['model_id']} ({model_type})")
    print(f"architecture: {model.get_architecture()}")
    print(f"AUC={auc:.4f}  accuracy={acc:.4f}  last_loss={metrics['train_loss']:.4f}")
    return meta


def _cli():
    parser = argparse.ArgumentParser(description="Train and evaluate a learned-feature GCN on transfer graphs.")
    parser.add_argument("--model-type", default="learned_gcn")
    parser.add_argument("--num-graphs", type=int, default=20)
    parser.add_argument("--min-nodes", type=int, default=80)
    parser.add_argument("--max-nodes", type=int, default=120)
    parser.add_argument("--sybil-fraction-min", type=float, default=0.10)
    parser.add_argument("--sybil-fraction-max", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--n-prop-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()

    train_learned_gcn(
        model_type=args.model_type,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        sybil_fraction_min=args.sybil_fraction_min,
        sybil_fraction_max=args.sybil_fraction_max,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        n_prop_layers=args.n_prop_layers,
        seed=args.seed,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    _cli()
