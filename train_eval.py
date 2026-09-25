"""Train/evaluate harder Track 4 GCN variants.

Runs a clean structural-only GCN and a structural+PPR GCN, while retaining PPR
as a separate baseline. This makes it possible to tell whether message passing
adds information beyond the trust-propagation feature itself.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from generate_graphs import make_dataset, make_cross_distribution_test
from features import extract_features, normalized_adjacency
from gcn import GCN
from ppr_baseline import ppr_trust_scores


def class_weights(y, n_classes=2):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return len(y) / (n_classes * counts)


def prepare_graph(G, seed=0, include_ppr=False):
    X, y, node_index, seeds = extract_features(G, seed=seed, include_ppr=include_ppr)
    return X, y, normalized_adjacency(G, node_index), node_index, seeds


def train(model, train_graphs, include_ppr=False, n_epochs=80, lr=0.015, verbose=True):
    prepared = [prepare_graph(g, seed=i, include_ppr=include_ppr)
                for i, (g, _hard) in enumerate(train_graphs)]
    for epoch in range(n_epochs):
        losses = []
        for X, y, A_hat, _, _ in prepared:
            losses.append(model.fit_step(
                A_hat, X, y, class_weight=class_weights(y), lr=lr
            ))
        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d} avg_loss={np.mean(losses):.4f}")
    return model


def evaluate(model, G, label, include_ppr=False, seed=0):
    X, y, A_hat, node_index, seeds = prepare_graph(G, seed=seed, include_ppr=include_ppr)
    score = model.predict_proba(A_hat, X)[:, 1]
    auc = roc_auc_score(y, score)
    ap = average_precision_score(y, score)
    acc = accuracy_score(y, score > 0.5)

    ppr = ppr_trust_scores(G, seeds, node_index)
    ppr_auc = roc_auc_score(y, -ppr)
    print(f"{label:30s} GCN AUC={auc:.3f} AP={ap:.3f} acc={acc:.3f} | PPR AUC={ppr_auc:.3f}")
    return {"label": label, "y": y, "gcn_score": score,
            "gcn_auc": auc, "gcn_ap": ap, "gcn_acc": acc, "ppr_auc": ppr_auc,
            "embedding": model.embedding(A_hat, X)}


def run_variant(name, include_ppr, train_graphs, tests):
    n_features = 5 if include_ppr else 4
    print(f"\n=== {name} ({n_features} features) ===")
    model = GCN(n_features=n_features, hidden_dims=(16, 16), n_classes=2, seed=0)
    train(model, train_graphs, include_ppr=include_ppr)
    return [evaluate(model, g, label, include_ppr=include_ppr, seed=seed)
            for g, label, seed in tests]


def main():
    print("Building diverse training set...")
    train_graphs = make_dataset(n_graphs=20, seed=1, difficulty="mixed")

    tests = []
    for i, (g, _) in enumerate(make_dataset(3, seed=101, difficulty="easy")):
        tests.append((g, f"held-out easy #{i}", 200 + i))
    for i, (g, _) in enumerate(make_dataset(3, seed=202, difficulty="hard")):
        tests.append((g, f"held-out hard #{i}", 300 + i))
    for i, (g, _) in enumerate(make_cross_distribution_test(3, seed=303)):
        tests.append((g, f"cross-distribution #{i}", 400 + i))

    structural = run_variant("STRUCTURAL ONLY", False, train_graphs, tests)
    with_ppr = run_variant("STRUCTURAL + PPR", True, train_graphs, tests)

    print("\nAblation summary (same test graphs):")
    for a, b in zip(structural, with_ppr):
        print(f"{a['label']:30s} structural={a['gcn_auc']:.3f}  +PPR={b['gcn_auc']:.3f}  PPR={a['ppr_auc']:.3f}")

    return structural, with_ppr


if __name__ == "__main__":
    main()
