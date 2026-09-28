"""
train_model.py

Canonical training entrypoint. Trains a model against configurable
synthetic-graph parameters, evaluates it on held-out synthetic graphs,
and saves it into the model registry (sybil_shield.core.model_store)
under <model_type>/<model_id>.

Every other entry point should now LOAD from that registry instead of
training its own throwaway copy:
  - the API's zero-shot crypto-scan path (predict_graph in api/app.py)
  - real_facebook_eval.py (via its model_id/model_type params)
  - future scripts

Only model_type == "gradient_adversarial" is implemented today. An
"unsupervised" variant is expected later; train() below already routes
on model_type so adding it is a matter of adding a branch, not
restructuring the registry or the API.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    for candidate in (str(ROOT), str(SRC)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

import time
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from sybil_shield.core import model_store
from sybil_shield.core.detector import build_features
from sybil_shield.core.gradient_attack_detector import train_gradient_adversarial
from sybil_shield.experiments.benchmark import (
    make_configurable_train_set,
    make_transfer_train_set,
)
from sybil_shield.experiments.train_transfer_model import (
    train_gradient_adversarial_transfer,
)

ProgressCallback = Callable[[Dict[str, Any]], None]

SUPPORTED_MODEL_TYPES = ("gradient_adversarial",)


def evaluate_model(model, graphs: Sequence, seed_frac: float) -> Dict[str, Any]:
    """AUC/accuracy/precision/recall/confusion matrix/ROC curve, pooled
    across every node in every held-out graph (not averaged per-graph --
    graphs vary in size, so pooling weights larger graphs correctly)."""
    all_y, all_scores = [], []
    for i, G in enumerate(graphs):
        X, y, A_hat, _, _ = build_features(G, seed=i, seed_frac=seed_frac)
        probs = model.predict_proba(A_hat, X)[:, 1]
        all_y.append(y)
        all_scores.append(probs)
    y = np.concatenate(all_y)
    scores = np.concatenate(all_scores)
    preds = (scores >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y, scores)
    step = max(1, len(fpr) // 200)  # thin the curve so meta.json stays small
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    return {
        "auc": float(roc_auc_score(y, scores)),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "confusion_matrix": {
            "true_positive": int(tp), "false_positive": int(fp),
            "true_negative": int(tn), "false_negative": int(fn),
        },
        "roc_curve": [
            {"fpr": float(f), "tpr": float(t)}
            for f, t in zip(fpr[::step], tpr[::step])
        ],
    }


def train_model(
    model_type: str = "gradient_adversarial",
    training_mode: str = "baseline",
    num_graphs: int = 50,
    min_nodes: int = 100,
    max_nodes: int = 500,
    sybil_fraction_min: float = 0.05,
    sybil_fraction_max: float = 0.30,
    epochs: int = 120,
    learning_rate: float = 0.02,
    attack_every: int = 2,
    attack_budget: int = 8,
    recompute_every: int = 3,
    hidden_dims: Sequence[int] = (16, 16),
    betweenness_k: int = 30,
    seed_frac: float = 0.05,
    seed: int = 1,
    model_id: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    bootstrap_seed_prob: float = 0.5,
    attack_mode_weights: Sequence[float] = (0.5, 0.25, 0.25),
) -> Dict[str, Any]:
    """Train, evaluate, and save one model. Returns the saved meta.json
    dict (model_id, architecture, params, metrics).

    progress_cb, if given, is called once per (epoch, graph) pair with a
    dict matching the Train tab's websocket/poll message shape:
    {status, current_graph_index, total_graphs, epoch, total_epochs,
    loss, honest_count, sybil_count, current_graph}, plus a final call
    with status="completed" carrying model_id and metrics. When
    progress_cb is None, progress is printed instead (see
    train_gradient_adversarial's verbose param).
    """
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: {SUPPORTED_MODEL_TYPES}. "
            "(An unsupervised variant isn't implemented yet.)"
        )
    if training_mode not in {"baseline", "transfer"}:
        raise ValueError("training_mode must be one of {'baseline', 'transfer'}")

    t0 = time.time()
    hidden_dims = tuple(hidden_dims)
    attack_mode_weights = tuple(float(x) for x in attack_mode_weights)

    if training_mode == "transfer":
        train_graphs = make_transfer_train_set(
            num_graphs=num_graphs, small_nodes=(max(50, min_nodes), max_nodes),
            large_nodes=(max(800, min_nodes), max(1500, max_nodes)),
            sybil_fraction_range=(sybil_fraction_min, sybil_fraction_max),
            seed=seed,
        )
    else:
        train_graphs = make_configurable_train_set(
            num_graphs=num_graphs, min_nodes=min_nodes, max_nodes=max_nodes,
            sybil_fraction_min=sybil_fraction_min, sybil_fraction_max=sybil_fraction_max,
            seed=seed,
        )

    def _epoch_cb(epoch, graph_index, loss, honest_count=None, sybil_count=None, current_graph=None):
        if progress_cb is None:
            return
        progress_cb({
            "status": "running",
            "current_graph_index": graph_index,
            "total_graphs": len(train_graphs),
            "epoch": epoch,
            "total_epochs": epochs,
            "loss": loss,
            "honest_count": honest_count,
            "sybil_count": sybil_count,
            "current_graph": current_graph,
        })

    if training_mode == "transfer":
        model = train_gradient_adversarial_transfer(
            train_graphs=train_graphs, seed_frac=seed_frac, n_epochs=epochs,
            lr=learning_rate, attack_every=attack_every, attack_budget=attack_budget,
            recompute_every=recompute_every, hidden_dims=hidden_dims,
            betweenness_k=betweenness_k, bootstrap_seed_prob=bootstrap_seed_prob,
            attack_mode_weights=attack_mode_weights,
            rng=np.random.default_rng(seed),
            verbose=(progress_cb is None),
            epoch_cb=_epoch_cb if progress_cb is not None else None,
        )
    else:
        model = train_gradient_adversarial(
            train_graphs, seed_frac=seed_frac, n_epochs=epochs, lr=learning_rate,
            attack_every=attack_every, attack_budget=attack_budget,
            recompute_every=recompute_every, hidden_dims=hidden_dims,
            betweenness_k=betweenness_k, rng=np.random.default_rng(seed),
            verbose=(progress_cb is None),
            epoch_cb=_epoch_cb if progress_cb is not None else None,
        )

    training_graphs = []
    for idx, G in enumerate(train_graphs):
        nodes = []
        for node, attrs in G.nodes(data=True):
            node_id = str(node)
            nodes.append({
                "id": node_id,
                "node_type": attrs.get("node_type", "wallet"),
                "label": int(attrs.get("label", 0) or 0),
            })
        training_graphs.append({
            "graph_id": f"train_graph_{idx + 1}",
            "label": f"Graph {idx + 1}",
            "nodes": nodes,
            "edges": [[str(u), str(v)] for u, v in G.edges()],
        })

    # Held-out synthetic graphs, same knobs, disjoint seed -- not a
    # rigorous train/test split by distribution, but enough to catch a
    # broken training run and to populate the Inspect Model tab.
    if training_mode == "transfer":
        holdout_graphs = make_transfer_train_set(
            num_graphs=max(3, num_graphs // 6), small_nodes=(max(50, min_nodes), max_nodes),
            large_nodes=(max(800, min_nodes), max(1500, max_nodes)),
            sybil_fraction_range=(sybil_fraction_min, sybil_fraction_max),
            seed=seed + 999,
        )
    else:
        holdout_graphs = make_configurable_train_set(
            num_graphs=max(3, num_graphs // 6), min_nodes=min_nodes, max_nodes=max_nodes,
            sybil_fraction_min=sybil_fraction_min, sybil_fraction_max=sybil_fraction_max,
            seed=seed + 999,
        )
    metrics = evaluate_model(model, holdout_graphs, seed_frac=seed_frac)
    metrics["train_seconds"] = round(time.time() - t0, 1)

    params = dict(
        training_mode=training_mode,
        num_graphs=num_graphs, min_nodes=min_nodes, max_nodes=max_nodes,
        sybil_fraction_min=sybil_fraction_min, sybil_fraction_max=sybil_fraction_max,
        epochs=epochs, learning_rate=learning_rate, attack_every=attack_every,
        attack_budget=attack_budget, recompute_every=recompute_every,
        hidden_dims=list(hidden_dims), betweenness_k=betweenness_k,
        seed_frac=seed_frac, seed=seed,
        bootstrap_seed_prob=bootstrap_seed_prob,
        attack_mode_weights=list(attack_mode_weights),
    )
    meta = model_store.save_model(
        model, model_type=model_type, params=params, metrics=metrics,
        model_id=model_id, training_graphs=training_graphs,
    )

    if progress_cb is not None:
        progress_cb({
            "status": "completed",
            "current_graph_index": len(train_graphs),
            "total_graphs": len(train_graphs),
            "epoch": epochs,
            "total_epochs": epochs,
            "loss": None,
            "honest_count": None,
            "sybil_count": None,
            "current_graph": None,
            "model_id": meta["model_id"],
            "metrics": metrics,
        })

    return meta


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Train and register a Sybil-detection model.")
    parser.add_argument("--model-type", default="gradient_adversarial", choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--training-mode", default="baseline", choices=("baseline", "transfer"),
                        help="Use the transfer-focused training distribution and seed strategy.")
    parser.add_argument("--num-graphs", type=int, default=50)
    parser.add_argument("--min-nodes", type=int, default=100)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--sybil-fraction-min", type=float, default=0.05)
    parser.add_argument("--sybil-fraction-max", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--attack-every", type=int, default=2)
    parser.add_argument("--attack-budget", type=int, default=8)
    parser.add_argument("--recompute-every", type=int, default=3)
    parser.add_argument("--hidden-dims", default="16,16",
                         help="Comma-separated layer widths, e.g. 16,16 or 32,32")
    parser.add_argument("--betweenness-k", type=int, default=30)
    parser.add_argument("--seed-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-id", default=None, help="Custom id; default is timestamp-based")
    parser.add_argument("--bootstrap-seed-prob", type=float, default=0.5,
                        help="Probability of using the community bootstrap when training in transfer mode.")
    parser.add_argument("--attack-mode-weights", default="0.5,0.25,0.25",
                        help="Three comma-separated weights for gradient/random/hubs attacks in transfer mode.")
    args = parser.parse_args()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    attack_mode_weights = tuple(
        float(x.strip()) for x in args.attack_mode_weights.split(",") if x.strip()
    )

    def _print_progress(msg):
        if msg["status"] == "completed":
            print(f"\nDone in {msg['metrics']['train_seconds']}s -> "
                  f"model_id={msg['model_id']} AUC={msg['metrics']['auc']:.4f} "
                  f"acc={msg['metrics']['accuracy']:.4f}")
            return
        if msg["current_graph_index"] == 0 and msg["epoch"] % 20 == 0:
            print(f"  epoch {msg['epoch']:3d}/{msg['total_epochs']} "
                  f"graph {msg['current_graph_index']}/{msg['total_graphs']} "
                  f"loss={msg['loss']:.4f}")

    meta = train_model(
        model_type=args.model_type, training_mode=args.training_mode,
        num_graphs=args.num_graphs, min_nodes=args.min_nodes, max_nodes=args.max_nodes,
        sybil_fraction_min=args.sybil_fraction_min,
        sybil_fraction_max=args.sybil_fraction_max,
        epochs=args.epochs, learning_rate=args.learning_rate,
        attack_every=args.attack_every, attack_budget=args.attack_budget,
        recompute_every=args.recompute_every, hidden_dims=hidden_dims,
        betweenness_k=args.betweenness_k, seed_frac=args.seed_frac,
        seed=args.seed, model_id=args.model_id, progress_cb=_print_progress,
        bootstrap_seed_prob=args.bootstrap_seed_prob,
        attack_mode_weights=attack_mode_weights,
    )
    return meta


if __name__ == "__main__":
    _cli()
