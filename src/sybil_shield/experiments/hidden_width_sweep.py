"""
hidden_width_sweep.py

Sweeps GCN hidden-layer WIDTH (16/32/64, depth pinned at 2 layers -- GCNs
oversmooth badly past 2-3 layers, so width is the lever here, not depth)
for the gradient-adversarial model, and logs AUC/accuracy per config
against the REAL Facebook transfer benchmark (real_facebook_eval.py),
not just a synthetic holdout -- that's the actual evidence needed to
answer "why 32?" instead of citing the hardcoded (16,16) with no
justification.

Every width is trained on the exact same synthetic training set (same
seed) so differences in the logged AUC come from width, not train-set
noise. Each trained model is saved into the registry (core.model_store)
with its Facebook AUC/accuracy attached to its metrics, so the winner
is immediately usable elsewhere (predict_graph model_id, facebook eval
--model-id) without retraining.
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from sybil_shield.core import model_store
from sybil_shield.core.gradient_attack_detector import train_gradient_adversarial
from sybil_shield.experiments.eval_robust import make_train
from sybil_shield.experiments import real_facebook_eval as facebook_eval

DEFAULT_WIDTHS = (16, 32, 64)


def run_sweep(
    widths: Sequence[int] = DEFAULT_WIDTHS,
    num_graphs: int = 36,
    epochs: int = 120,
    seed: int = 1,
    betweenness_k: int = 30,
    seed_frac: Optional[float] = None,
    dataset_dir: Optional[str] = None,
    save_to_registry: bool = True,
    model_type: str = "gradient_adversarial",
) -> List[Dict[str, Any]]:
    seed_frac = facebook_eval.SEED_FRAC if seed_frac is None else seed_frac

    if dataset_dir:
        graph_file = os.path.join(dataset_dir, "graph.txt")
        train_file = os.path.join(dataset_dir, "train.txt")
        test_file = os.path.join(dataset_dir, "test.txt")
    else:
        graph_file, train_file, test_file = (
            facebook_eval.GRAPH_FILE, facebook_eval.TRAIN_FILE, facebook_eval.TEST_FILE,
        )

    print("Loading Facebook graph and labels...")
    G = facebook_eval.load_graph(graph_file)
    benign_train, _sybil_train = facebook_eval.load_labels_file(train_file)  # sybil_train unused
    benign_test, sybil_test = facebook_eval.load_labels_file(test_file)
    print(f"Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  "
          f"(benign_train={len(benign_train)}, benign_test={len(benign_test)}, "
          f"sybil_test={len(sybil_test)})")

    nodes = sorted(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    test_nodes = benign_test + sybil_test
    y_test = np.array([0] * len(benign_test) + [1] * len(sybil_test))
    test_idx = [node_index[n] for n in test_nodes]

    print(f"Generating {num_graphs} shared synthetic training graphs "
          f"(seed={seed}, reused across every width)...")
    train_graphs = make_train(num_graphs, seed=seed)

    results: List[Dict[str, Any]] = []
    for w in widths:
        hidden_dims = (w, w)
        print(f"\n=== hidden_dims=({w},{w}) ===")
        t0 = time.time()
        model = train_gradient_adversarial(
            train_graphs, seed_frac=seed_frac, n_epochs=epochs,
            hidden_dims=hidden_dims, betweenness_k=betweenness_k,
            rng=np.random.default_rng(seed), verbose=True,
        )
        elapsed = time.time() - t0

        X, A_hat = facebook_eval.facebook_features(G, nodes, benign_train)
        probs = model.predict_proba(A_hat, X)[:, 1]
        scores = probs[test_idx]
        auc = float(roc_auc_score(y_test, scores))
        acc = float(accuracy_score(y_test, (scores >= 0.5).astype(int)))
        print(f"  hidden_dims=({w},{w})  AUC={auc:.4f}  acc={acc:.4f}  ({elapsed:.0f}s)")

        entry: Dict[str, Any] = {
            "hidden_dims": [w, w], "auc": auc, "accuracy": acc,
            "train_seconds": round(elapsed, 1),
        }

        if save_to_registry:
            meta = model_store.save_model(
                model, model_type=model_type,
                params=dict(
                    num_graphs=num_graphs, epochs=epochs, hidden_dims=[w, w],
                    betweenness_k=betweenness_k, seed_frac=seed_frac, seed=seed,
                    source="hidden_width_sweep",
                ),
                # facebook_* prefix distinguishes this from the synthetic-
                # holdout auc/accuracy keys train_model.py writes, since
                # both land in the same meta.json["metrics"] shape.
                metrics={"facebook_auc": auc, "facebook_accuracy": acc},
            )
            entry["model_id"] = meta["model_id"]
        results.append(entry)

    print("\n" + "=" * 60)
    print("HIDDEN-WIDTH SWEEP SUMMARY (real Facebook transfer AUC)")
    print("=" * 60)
    print(f"{'hidden_dims':>14s}  {'AUC':>7s}  {'acc':>7s}  {'model_id':>26s}")
    for r in results:
        print(f"{str(tuple(r['hidden_dims'])):>14s}  {r['auc']:>7.4f}  "
              f"{r['accuracy']:>7.4f}  {r.get('model_id', ''):>26s}")

    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest: hidden_dims={tuple(best['hidden_dims'])}  AUC={best['auc']:.4f}"
          + (f"  model_id={best['model_id']}" if "model_id" in best else ""))

    return results


def _cli():
    parser = argparse.ArgumentParser(
        description="Sweep GCN hidden width (2 layers, width varies) against real Facebook AUC."
    )
    parser.add_argument("--widths", default="16,32,64",
                         help="Comma-separated widths, e.g. 16,32,64")
    parser.add_argument("--num-graphs", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--betweenness-k", type=int, default=30)
    parser.add_argument("--seed-frac", type=float, default=None,
                         help="Default: real_facebook_eval.SEED_FRAC (0.05)")
    parser.add_argument("--dataset-dir", default=None,
                         help="Defaults to real_facebook_eval.DATA_DIR")
    parser.add_argument("--no-registry", action="store_true",
                         help="Don't save trained models into the registry")
    parser.add_argument("--model-type", default="gradient_adversarial")
    args = parser.parse_args()

    widths = tuple(int(x) for x in args.widths.split(",") if x.strip())
    run_sweep(
        widths=widths, num_graphs=args.num_graphs, epochs=args.epochs,
        seed=args.seed, betweenness_k=args.betweenness_k, seed_frac=args.seed_frac,
        dataset_dir=args.dataset_dir, save_to_registry=not args.no_registry,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    _cli()