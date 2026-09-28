import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from sybil_shield.experiments.benchmark import make_sybil_graph, HONEST_FAMILIES, ALL_FAMILIES
from sybil_shield.core.detector import build_features, trust_scores, pick_seeds
from sybil_shield.core.attacks import attack_graph
from sybil_shield.experiments.eval_robust import make_train, cw
from sybil_shield.core.gcn import GCN


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

def train_on(graphs, epochs=120):
    """
    Train a GCN on a fixed collection of graphs.
    """
    prep = [
        build_features(g, seed=i)
        for i, g in enumerate(graphs)
    ]

    model = GCN(
        n_features=prep[0][0].shape[1],
        hidden_dims=(16, 16),
        seed=0
    )

    for ep in range(epochs):
        for X, y, A, *_ in prep:
            model.fit_step(
                A,
                X,
                y,
                class_weight=cw(y),
                lr=0.02
            )

    return model


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def evaluate_model(model, G, seed):
    """
    Run the COMPLETE detector pipeline on the attacked graph.

    Important:
    Features are recomputed after the attack.
    """

    nodes = list(G.nodes())
    y = np.array([
        1 if G.nodes[n]["label"] == 1 else 0
        for n in nodes
    ])

    # Rebuild all topology/trust features after attack
    X, _, A, *_ = build_features(G, seed=seed)

    pred = model.predict_proba(A, X)[:, 1]

    return roc_auc_score(y, pred)


# ------------------------------------------------------------
# TEST GRAPH GENERATION
# ------------------------------------------------------------

def make_test_graphs(n_graphs=30, seed=777):
    """
    Fixed test set.

    Every attack intensity will be evaluated on these SAME graphs.
    This makes the comparison much cleaner.
    """

    rng = np.random.default_rng(seed)

    graphs = []

    for r in range(n_graphs):

        G = make_sybil_graph(
            n_honest=int(rng.integers(250, 340)),
            honest_family=str(
                rng.choice(HONEST_FAMILIES)
            ),
            sybil_family=str(
                rng.choice(ALL_FAMILIES)
            ),
            n_farms=3,

            # Small baseline number of cross-community edges
            attack_edges=10,

            # Make Sybil topology structurally less obvious
            mimic=True,

            seed=int(
                rng.integers(0, 2**31)
            )
        )

        graphs.append((G, r))

    return graphs


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

def main():

    t0 = time.time()

    print("=" * 70)
    print("ATTACK INTENSITY ROBUSTNESS EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. TRAINING DATA
    # --------------------------------------------------------

    print("\n[1/4] Generating training graphs...")

    train_graphs = make_train(
        36,
        seed=1
    )

    # --------------------------------------------------------
    # 2. TRAIN TWO MODELS
    # --------------------------------------------------------

    print("[2/4] Training standard GCN...")

    standard_model = train_on(train_graphs)

    print("[2/4] Generating attack-augmented training set...")

    rng = np.random.default_rng(5)

    augmented_graphs = []

    modes = [
        "random",
        "hubs",
        "trusted"
    ]

    for i, G in enumerate(train_graphs):

        mode = modes[i % len(modes)]

        # Variable attack intensity during training
        extra_edges = int(
            rng.integers(20, 150)
        )

        attacked = attack_graph(
            G,
            extra_edges,
            mode,
            rng,
            seed_knowledge=i
        )

        augmented_graphs.append(attacked)

    print("[2/4] Training attack-augmented GCN...")

    augmented_model = train_on(
        augmented_graphs
    )

    # --------------------------------------------------------
    # 3. FIXED TEST SET
    # --------------------------------------------------------

    print("[3/4] Generating fixed test set...")

    tests = make_test_graphs(
        n_graphs=30,
        seed=777
    )

    # Attack intensity.

    # Remember:
    # The graph already contains 10 base attack edges.
    # These are EXTRA edges added by the attacker.

    attack_levels = [
        0,
        10,
        20,
        40,
        60,
        80,
        100,
        150,
        200,
        300
    ]

    modes = [
        "random",
        "hubs",
        "trusted"
    ]

    results = []

    # --------------------------------------------------------
    # 4. RUN EXPERIMENT
    # --------------------------------------------------------

    print("[4/4] Running attack-intensity sweep...\n")

    for mode in modes:

        print("\n" + "=" * 70)
        print(f"ATTACK MODE: {mode}")
        print("=" * 70)

        print(
            f"{'Extra edges':>12} | "
            f"{'Standard GCN':>18} | "
            f"{'Augmented GCN':>18} | "
            f"{'Difference':>12}"
        )

        print("-" * 70)

        for extra_edges in attack_levels:

            standard_scores = []
            augmented_scores = []

            for G, seed in tests:

                # ------------------------------------------------
                # IMPORTANT:
                # Every model sees EXACTLY the same attacked graph.
                # ------------------------------------------------

                if extra_edges == 0:

                    Ga = G

                else:

                    attack_rng = np.random.default_rng(
                        100000
                        + seed * 1000
                        + extra_edges
                    )

                    Ga = attack_graph(
                        G.copy(),
                        extra_edges,
                        mode,
                        attack_rng,
                        seed_knowledge=seed
                    )

                # Recompute features after attack
                standard_auc = evaluate_model(
                    standard_model,
                    Ga,
                    seed
                )

                augmented_auc = evaluate_model(
                    augmented_model,
                    Ga,
                    seed
                )

                standard_scores.append(
                    standard_auc
                )

                augmented_scores.append(
                    augmented_auc
                )

            # ----------------------------------------------------
            # Statistics
            # ----------------------------------------------------

            standard_mean = np.mean(
                standard_scores
            )

            standard_std = np.std(
                standard_scores
            )

            augmented_mean = np.mean(
                augmented_scores
            )

            augmented_std = np.std(
                augmented_scores
            )

            difference = (
                augmented_mean
                - standard_mean
            )

            print(
                f"{extra_edges:>12} | "
                f"{standard_mean:.3f} ± {standard_std:.3f} | "
                f"{augmented_mean:.3f} ± {augmented_std:.3f} | "
                f"{difference:+.3f}"
            )

            results.append({
                "mode": mode,
                "extra_edges": extra_edges,
                "standard_mean": standard_mean,
                "standard_std": standard_std,
                "augmented_mean": augmented_mean,
                "augmented_std": augmented_std,
                "difference": difference
            })

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    with open(
        "attack_intensity_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "extra_edges",
                "standard_mean",
                "standard_std",
                "augmented_mean",
                "augmented_std",
                "difference"
            ]
        )

        writer.writeheader()
        writer.writerows(results)
        

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Test graphs: {len(tests)}"
    )

    print(
        f"Attack levels: {len(attack_levels)}"
    )

    print(
        f"Total runtime: {time.time() - t0:.1f}s"
    )

    print(
        "\nSaved: attack_intensity_results.csv"
    )


    # --------------------------------------------------------
    # PLOT ROBUSTNESS CURVES
    # --------------------------------------------------------

    for mode in modes:

        mode_results = [
            r for r in results
            if r["mode"] == mode
        ]

        x = [
            r["extra_edges"]
            for r in mode_results
        ]

        standard_mean = [
            r["standard_mean"]
            for r in mode_results
        ]

        standard_std = [
            r["standard_std"]
            for r in mode_results
        ]

        augmented_mean = [
            r["augmented_mean"]
            for r in mode_results
        ]

        augmented_std = [
            r["augmented_std"]
            for r in mode_results
        ]

        plt.figure(figsize=(9, 6))

        # Standard GCN
        plt.plot(
            x,
            standard_mean,
            marker="o",
            label="Standard GCN"
        )

        plt.fill_between(
            x,
            np.array(standard_mean) - np.array(standard_std),
            np.array(standard_mean) + np.array(standard_std),
            alpha=0.15
        )

        # Attack-augmented GCN
        plt.plot(
            x,
            augmented_mean,
            marker="o",
            label="Attack-Augmented GCN"
        )

        plt.fill_between(
            x,
            np.array(augmented_mean) - np.array(augmented_std),
            np.array(augmented_mean) + np.array(augmented_std),
            alpha=0.15
        )

        plt.axhline(
            0.5,
            linestyle="--",
            alpha=0.6,
            label="Random baseline"
        )

        plt.xlabel("Additional Sybil → Honest Edges")
        plt.ylabel("ROC-AUC")

        plt.title(
            f"Sybil Detection Robustness — {mode.capitalize()} Attack"
        )

        plt.ylim(0, 1.05)
        plt.grid(alpha=0.25)
        plt.legend()

        plt.tight_layout()

        filename = f"robustness_{mode}.png"

        plt.savefig(
            filename,
            dpi=200
        )

        plt.show()

        print(f"Saved plot: {filename}")

if __name__ == "__main__":
    main()