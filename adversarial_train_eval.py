"""
adversarial_train_eval.py

Compares a normally-trained GCN vs an adversarially-trained GCN under the
same gradient-based edge attack.

IMPORTANT:
After the attacker modifies the graph, we RECOMPUTE the topology-derived
features on the attacked graph. This means the evaluation attacks the
complete detector pipeline rather than only the adjacency matrix.

Adversarial training:
    clean graph
        ↓
    clean training step
        ↓
    attack a fresh COPY
        ↓
    recompute attacked features
        ↓
    adversarial training step

The original clean graphs are never mutated.
"""

from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score

from generate_graphs import (
    make_dataset,
    make_cross_distribution_test,
)
from features import extract_features, extract_features_with_seeds, normalized_adjacency
from gcn import GCN
from adversarial import greedy_gradient_attack


def class_weights(y, n_classes=2):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    return len(y) / (n_classes * counts)


def prepare(G, seed=0):
    """
    Extract all detector inputs from a graph.

    Returns:
        X          node features
        y          labels
        A_hat      normalized adjacency
        node_index node ordering used by X/A_hat
        seeds      trusted seeds used by PPR
    """
    X, y, node_index, seeds = extract_features(G, seed=seed)
    A_hat = normalized_adjacency(G, node_index)

    return X, y, A_hat, node_index, seeds


def train_baseline(model, train_graphs, n_epochs=80, lr=0.02):
    """
    Standard training on clean graphs only.
    """

    prepared = [
        prepare(g, seed=i)
        for i, (g, _h) in enumerate(train_graphs)
    ]

    for epoch in range(n_epochs):

        for X, y, A_hat, _idx, _seeds in prepared:

            cw = class_weights(y)

            model.fit_step(
                A_hat,
                X,
                y,
                class_weight=cw,
                lr=lr,
            )

    return model


def train_adversarial(
    model,
    train_graphs,
    n_epochs=80,
    lr=0.02,
    attack_budget=8,
    attack_every=2,
    rng=None,
):
    """
    Adversarial training.

    Every `attack_every` epochs:

        1. Train on the clean graph.
        2. Create a fresh attacked COPY.
        3. Recompute features on that attacked graph.
        4. Train on the attacked graph.

    Clean training graphs are never mutated.
    """

    rng = rng or np.random.RandomState(0)

    # Clean representations.
    prepared = [
        prepare(g, seed=i)
        for i, (g, _h) in enumerate(train_graphs)
    ]

    for epoch in range(n_epochs):

        for gi, (
            X,
            y,
            A_hat,
            node_index,
            seeds,
        ) in enumerate(prepared):

            cw = class_weights(y)

            # ---------------------------------------------------------
            # 1. Normal clean training step
            # ---------------------------------------------------------

            model.fit_step(
                A_hat,
                X,
                y,
                class_weight=cw,
                lr=lr,
            )

            # ---------------------------------------------------------
            # 2. Generate adversarial graph
            # ---------------------------------------------------------

            if epoch % attack_every == 0:

                G_clean = train_graphs[gi][0]

                # Feature-aware: X is recomputed against these SAME fixed
                # seeds/node_index as the graph is perturbed, every
                # `recompute_every` steps -- not just A_hat. This is what
                # keeps the attack honest about what it's actually fooling.
                feature_fn = partial(
                    extract_features_with_seeds, seeds=seeds
                )

                G_atk, A_atk, X_atk, _added, _hist = (
                    greedy_gradient_attack(
                        model,
                        G_clean.copy(),
                        node_index,
                        y,
                        X,
                        seeds,
                        budget=attack_budget,
                        class_weight=cw,
                        recompute_every=max(1, attack_budget // 3),
                        rng=rng,
                        feature_fn=feature_fn,
                    )
                )

                # Only edges changed -- node set/order and labels (y) are
                # identical to the clean graph by construction, so there's
                # nothing to re-derive or sanity-check here anymore.

                cw_atk = class_weights(y)

                # -----------------------------------------------------
                # 3. Adversarial training step
                # -----------------------------------------------------

                model.fit_step(
                    A_atk,
                    X_atk,
                    y,
                    class_weight=cw_atk,
                    lr=lr,
                )

    return model


def attack_eval(
    model,
    G,
    seed=900,
    budget=25,
    rng=None,
):
    """
    Evaluate one graph.

    Returns:
        clean_auc
        attacked_auc
        attack_history

    IMPORTANT:
    The attacked graph gets completely new topology-derived features.
    """

    # -------------------------------------------------------------
    # CLEAN GRAPH
    # -------------------------------------------------------------

    X, y, A_hat, node_index, seeds = prepare(
        G,
        seed=seed,
    )

    cw = class_weights(y)

    probs, _ = model.forward(
        A_hat,
        X,
    )

    clean_auc = roc_auc_score(
        y,
        probs[:, 1],
    )

    # -------------------------------------------------------------
    # ATTACK -- feature-aware: X recomputed against these fixed
    # seeds/node_index as edges are added, every few steps.
    # -------------------------------------------------------------

    feature_fn = partial(extract_features_with_seeds, seeds=seeds)

    G_atk, A_atk, X_atk, added, hist = (
        greedy_gradient_attack(
            model,
            G.copy(),
            node_index,
            y,
            X,
            seeds,
            budget=budget,
            class_weight=cw,
            recompute_every=3,
            rng=rng,
            feature_fn=feature_fn,
        )
    )

    # Only edges changed -- labels/node order are identical to the clean
    # graph by construction.

    # -------------------------------------------------------------
    # ATTACKED PREDICTION
    # -------------------------------------------------------------

    probs_atk, _ = model.forward(
        A_atk,
        X_atk,
    )

    attacked_auc = roc_auc_score(
        y,
        probs_atk[:, 1],
    )

    return clean_auc, attacked_auc, hist


def main():

    # =============================================================
    # TRAINING DISTRIBUTION
    # =============================================================

    train_graphs = make_dataset(
        n_graphs=10,
        difficulty="mixed",
        seed=1,
    )

    # =============================================================
    # HELD-OUT HARD DISTRIBUTION
    # =============================================================

    hard_test = make_dataset(
        n_graphs=10,
        difficulty="hard",
        seed=700,
    )

    # =============================================================
    # CROSS-DISTRIBUTION TEST
    # =============================================================

    cross_test = make_cross_distribution_test(
        n_graphs=10,
        seed=900,
    )

    # Normalize both collections into:
    #     (graph, label)
    test_graphs = (
        [(g, "hard") for g, _ in hard_test]
        + [(g, "cross") for g in cross_test]
    )

    # =============================================================
    # BASELINE
    # =============================================================

    print("Training baseline model...")

    baseline = GCN(
        n_features=5,
        hidden_dims=(16, 16),
        n_classes=2,
        seed=0,
    )

    train_baseline(
        baseline,
        train_graphs,
        n_epochs=80,
        lr=0.02,
    )

    # =============================================================
    # ADVERSARIAL MODEL
    # =============================================================

    print("Training adversarially-hardened model...")

    hardened = GCN(
        n_features=5,
        hidden_dims=(16, 16),
        n_classes=2,
        seed=0,
    )

    train_adversarial(
        hardened,
        train_graphs,
        n_epochs=80,
        lr=0.02,
        attack_budget=8,
        attack_every=2,
    )

    # =============================================================
    # EVALUATION
    # =============================================================

    print(
        "\nAttack comparison on held-out "
        "stress-test graphs (budget=25 each):"
    )

    baseline_drops = []
    hardened_drops = []

    hard_baseline_drops = []
    hard_hardened_drops = []

    cross_baseline_drops = []
    cross_hardened_drops = []

    for i, (g, kind) in enumerate(test_graphs):

        # ---------------------------------------------------------
        # Baseline
        # ---------------------------------------------------------

        b_clean, b_atk, _ = attack_eval(
            baseline,
            g,
            seed=900 + i,
            budget=25,
        )

        # ---------------------------------------------------------
        # Hardened
        # ---------------------------------------------------------

        h_clean, h_atk, _ = attack_eval(
            hardened,
            g,
            seed=900 + i,
            budget=25,
        )

        b_drop = b_clean - b_atk
        h_drop = h_clean - h_atk

        baseline_drops.append(b_drop)
        hardened_drops.append(h_drop)

        if kind == "hard":
            hard_baseline_drops.append(b_drop)
            hard_hardened_drops.append(h_drop)
        else:
            cross_baseline_drops.append(b_drop)
            cross_hardened_drops.append(h_drop)

        print(f"{kind:5s} #{i:02d}")

        print(
            f"  baseline   clean={b_clean:.3f} "
            f"attacked={b_atk:.3f} "
            f"drop={b_drop:.3f}"
        )

        print(
            f"  hardened   clean={h_clean:.3f} "
            f"attacked={h_atk:.3f} "
            f"drop={h_drop:.3f}"
        )

    # =============================================================
    # SUMMARY
    # =============================================================

    print("\n=== OVERALL SUMMARY ===")

    print(
        f"baseline mean drop = "
        f"{np.mean(baseline_drops):.3f} ± "
        f"{np.std(baseline_drops):.3f}"
    )

    print(
        f"hardened mean drop = "
        f"{np.mean(hardened_drops):.3f} ± "
        f"{np.std(hardened_drops):.3f}"
    )

    print("\n=== HARD TEST SUMMARY ===")

    print(
        f"baseline mean drop = "
        f"{np.mean(hard_baseline_drops):.3f} ± "
        f"{np.std(hard_baseline_drops):.3f}"
    )

    print(
        f"hardened mean drop = "
        f"{np.mean(hard_hardened_drops):.3f} ± "
        f"{np.std(hard_hardened_drops):.3f}"
    )

    print("\n=== CROSS-DISTRIBUTION SUMMARY ===")

    print(
        f"baseline mean drop = "
        f"{np.mean(cross_baseline_drops):.3f} ± "
        f"{np.std(cross_baseline_drops):.3f}"
    )

    print(
        f"hardened mean drop = "
        f"{np.mean(cross_hardened_drops):.3f} ± "
        f"{np.std(cross_hardened_drops):.3f}"
    )


if __name__ == "__main__":
    main()