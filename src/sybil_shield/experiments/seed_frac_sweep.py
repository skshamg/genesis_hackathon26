"""
seed_frac_generalization_check.py

Separates two explanations for the earlier seed_frac sweep's huge AUC
gains (up to ~0.999 at seed_frac=1.0):

  (a) higher training-time seed_frac produces a genuinely better decision
      rule under scarce-seed conditions, vs
  (b) the earlier eval was picking up training-time leakage -- pick_seeds
      only samples HONEST nodes, and PPR re-injects seed mass every
      iteration, so at high frac a seeded honest node's PPR score is
      largely a re-encoding of its own known label. Sybils are never
      seeds, so they never get this boost. That leak happened to still
      transfer to the real Facebook eval because that eval's fixed
      100-seed setup naturally produces a similarly sharp honest/sybil
      PPR split on a dense graph -- not because the model learned to
      reason well under scarcity.

To tell these apart: evaluate each seed_frac-trained model on a held-out
SYNTHETIC "hard" test set (disjoint seed, never seen in training) using a
FIXED, small ABSOLUTE seed count at inference -- not scaled by training
frac, and the SAME seed nodes reused across every model. If (a) is right,
higher training frac should still help here. If it's (b), the advantage
should shrink or vanish once test-time seed scarcity is pinned to
something realistic and leakage-free.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from sybil_shield.core.detector import build_features
from sybil_shield.experiments.eval_robust import make_train, cw
from sybil_shield.core.gcn import GCN
from sybil_shield.core.generate_graphs import make_dataset
from sybil_shield.core.gradient_attack_detector import (
    build_features_with_seeds,
    build_features_normalized_adjacency,
)


def train_on(graphs, seed_frac, epochs=120):
    prep = [build_features(g, seed=i, seed_frac=seed_frac) for i, g in enumerate(graphs)]
    model = GCN(n_features=prep[0][0].shape[1], hidden_dims=(16, 16), seed=0)
    for ep in range(epochs):
        for X, y, A, *_ in prep:
            model.fit_step(A, X, y, class_weight=cw(y), lr=0.02)
    return model


def eval_features(G, seeds):
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    X = build_features_with_seeds(G, node_index, seeds)
    A_hat = build_features_normalized_adjacency(G, node_index)
    y = np.array([G.nodes[n].get("label", 0) for n in nodes])
    return X, A_hat, y


def fixed_seed_set(G, n_seeds, rng):
    honest = [n for n, d in G.nodes(data=True) if d.get("label", 0) == 0]
    k = min(n_seeds, len(honest))
    return list(rng.choice(honest, size=k, replace=False))


def main():
    print("Generating shared synthetic training graphs...")
    train_graphs = make_train(36, seed=1)

    print("Generating held-out HARD test graphs (disjoint seed, unseen in training)...")
    hard_test = [g for g, _hard in make_dataset(n_graphs=15, difficulty="hard", seed=4242)]

    # Fixed, small absolute seed counts -- same seeds reused across every
    # model so only training-time frac varies between rows.
    seed_counts = [10, 20]
    fixed_seeds_per_graph = {
        n_seeds: [
            fixed_seed_set(G, n_seeds, np.random.RandomState(5000 + gi))
            for gi, G in enumerate(hard_test)
        ]
        for n_seeds in seed_counts
    }

    train_fracs = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30]

    header = f"{'seed_frac':>10s}" + "".join(f"{'AUC@'+str(n)+'seeds':>16s}" for n in seed_counts)
    print(f"\n{header}")
    print("-" * len(header))

    for frac in train_fracs:
        model = train_on(train_graphs, seed_frac=frac)
        row = f"{frac:>10.4f}"
        for n_seeds in seed_counts:
            aucs = []
            for gi, G in enumerate(hard_test):
                seeds = fixed_seeds_per_graph[n_seeds][gi]
                X, A_hat, y = eval_features(G, seeds)
                probs = model.predict_proba(A_hat, X)[:, 1]
                aucs.append(roc_auc_score(y, probs))
            row += f"{np.mean(aucs):>16.4f}"
        print(row)


if __name__ == "__main__":
    main()