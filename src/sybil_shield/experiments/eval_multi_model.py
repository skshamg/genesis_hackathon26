import numpy as np
import time
import csv

from sklearn.metrics import roc_auc_score

from sybil_shield.experiments.benchmark import make_sybil_graph, HONEST_FAMILIES, ALL_FAMILIES
from sybil_shield.core.attacks import attack_graph
from sybil_shield.core.detector import build_features
from sybil_shield.experiments.eval_robust import make_train, cw
from sybil_shield.core.gcn import GCN


def train_on(graphs, epochs=120):
    prep = [build_features(g, seed=i) for i, g in enumerate(graphs)]

    model = GCN(
        n_features=prep[0][0].shape[1],
        hidden_dims=(16, 16),
        seed=0
    )

    for ep in range(epochs):
        for X, y, A, *_ in prep:
            model.fit_step(
                A, X, y,
                class_weight=cw(y),
                lr=0.02
            )

    return model


def make_augmented_training_set(
    base_graphs, mode, seed=1234,
    min_edges=20, max_edges=150
):
    rng = np.random.default_rng(seed)

    modes = ["random", "hubs", "trusted"] if mode == "mixed" else [mode]
    attacked_graphs = []

    for i, G in enumerate(base_graphs):
        attack_mode = modes[i % len(modes)]
        extra_edges = int(rng.integers(min_edges, max_edges + 1))

        attack_rng = np.random.default_rng(seed + i * 10007)

        attacked_graphs.append(
            attack_graph(
                G,
                extra_edges,
                attack_mode,
                attack_rng,
                seed_knowledge=i
            )
        )

    return attacked_graphs


def make_test_graphs(n_graphs=30, seed=777):
    rng = np.random.default_rng(seed)
    graphs = []

    for i in range(n_graphs):
        G = make_sybil_graph(
            n_honest=int(rng.integers(250, 340)),
            honest_family=str(rng.choice(HONEST_FAMILIES)),
            sybil_family=str(rng.choice(ALL_FAMILIES)),
            n_farms=3,
            attack_edges=10,
            mimic=True,
            seed=int(rng.integers(0, 2**31))
        )
        graphs.append((G, i))

    return graphs


def evaluate_model(model, G, seed):
    nodes = list(G.nodes())

    y = np.array([
        1 if G.nodes[n]["label"] == 1 else 0
        for n in nodes
    ])

    X, _, A, *_ = build_features(G, seed=seed)
    probabilities = model.predict_proba(A, X)[:, 1]

    return y, probabilities


def evaluate_ensemble(models, G, seed):
    y, p0 = evaluate_model(models[0], G, seed)
    scores = [p0]

    for model in models[1:]:
        _, p = evaluate_model(model, G, seed)
        scores.append(p)

    ensemble_score = np.mean(np.vstack(scores), axis=0)

    return roc_auc_score(y, ensemble_score)


def main():
    t0 = time.time()

    print("=" * 78)
    print("MULTI-MODEL ADVERSARIAL ROBUSTNESS EXPERIMENT")
    print("=" * 78)

    print("\n[1/6] Generating common training graphs...")
    base_train = make_train(36, seed=1)

    print("[2/6] Training standard GCN...")
    standard = train_on(base_train)

    print("[2/6] Training random-robust GCN...")
    random_model = train_on(
        make_augmented_training_set(
            base_train, "random", seed=100
        )
    )

    print("[2/6] Training trusted-robust GCN...")
    trusted_model = train_on(
        make_augmented_training_set(
            base_train, "trusted", seed=200
        )
    )

    print("[2/6] Training mixed-robust GCN...")
    mixed_model = train_on(
        make_augmented_training_set(
            base_train, "mixed", seed=300
        )
    )

    models = {
        "standard": standard,
        "random": random_model,
        "trusted": trusted_model,
        "mixed": mixed_model
    }

    print("[3/6] Generating fixed test set...")
    tests = make_test_graphs(30, seed=777)

    attack_levels = [0, 10, 20, 40, 60, 80, 100, 150, 200, 300]
    attack_modes = ["random", "hubs", "trusted"]

    results = []

    print("[4/6] Running multi-model attack sweep...\n")

    for attack_mode in attack_modes:
        print("\n" + "=" * 78)
        print(f"TEST ATTACK: {attack_mode.upper()}")
        print("=" * 78)

        header = (
            f"{'Edges':>7} | {'Standard':>10} | "
            f"{'Random':>10} | {'Trusted':>10} | "
            f"{'Mixed':>10} | {'Ensemble':>10}"
        )
        print(header)
        print("-" * len(header))

        for extra_edges in attack_levels:
            model_scores = {name: [] for name in models}
            ensemble_scores = []

            for G, graph_id in tests:
                if extra_edges == 0:
                    attacked = G
                else:
                    mode_offset = {
                        "random": 0,
                        "hubs": 1000000,
                        "trusted": 2000000
                    }[attack_mode]

                    attack_rng = np.random.default_rng(
                        100000
                        + graph_id * 1000
                        + extra_edges
                        + mode_offset
                    )

                    attacked = attack_graph(
                        G.copy(),
                        extra_edges,
                        attack_mode,
                        attack_rng,
                        seed_knowledge=graph_id
                    )

                y = None

                for name, model in models.items():
                    y, p = evaluate_model(
                        model, attacked, graph_id
                    )
                    model_scores[name].append(
                        roc_auc_score(y, p)
                    )

                ensemble_scores.append(
                    evaluate_ensemble(
                        list(models.values()),
                        attacked,
                        graph_id
                    )
                )

            means = {
                name: float(np.mean(scores))
                for name, scores in model_scores.items()
            }

            ensemble_mean = float(np.mean(ensemble_scores))

            print(
                f"{extra_edges:>7} | "
                f"{means['standard']:.3f} | "
                f"{means['random']:.3f} | "
                f"{means['trusted']:.3f} | "
                f"{means['mixed']:.3f} | "
                f"{ensemble_mean:.3f}"
            )

            for name, scores in model_scores.items():
                results.append({
                    "attack_mode": attack_mode,
                    "extra_edges": extra_edges,
                    "model": name,
                    "mean_auc": float(np.mean(scores)),
                    "std_auc": float(np.std(scores))
                })

            results.append({
                "attack_mode": attack_mode,
                "extra_edges": extra_edges,
                "model": "ensemble",
                "mean_auc": ensemble_mean,
                "std_auc": float(np.std(ensemble_scores))
            })

    print("\n[5/6] Saving results...")

    with open("multi_model_robustness_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack_mode", "extra_edges", "model",
                "mean_auc", "std_auc"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("[6/6] Generating plots...")

    try:
        import matplotlib.pyplot as plt

        for attack_mode in attack_modes:
            plt.figure(figsize=(10, 6))

            for model_name in [
                "standard", "random", "trusted",
                "mixed", "ensemble"
            ]:
                rows = [
                    r for r in results
                    if r["attack_mode"] == attack_mode
                    and r["model"] == model_name
                ]

                x = np.array([r["extra_edges"] for r in rows])
                mean = np.array([r["mean_auc"] for r in rows])
                std = np.array([r["std_auc"] for r in rows])

                plt.plot(
                    x, mean,
                    marker="o",
                    label=model_name
                )

                plt.fill_between(
                    x,
                    np.maximum(0, mean - std),
                    np.minimum(1, mean + std),
                    alpha=0.10
                )

            plt.axhline(
                0.5,
                linestyle="--",
                alpha=0.5,
                label="Random baseline"
            )

            plt.xlabel("Additional Sybil → Honest Edges")
            plt.ylabel("ROC-AUC")
            plt.title(
                f"Multi-Model Robustness — "
                f"{attack_mode.capitalize()} Attack"
            )
            plt.ylim(0, 1.05)
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()

            filename = f"multi_model_{attack_mode}.png"
            plt.savefig(filename, dpi=200)
            plt.show()

            print(f"Saved plot: {filename}")

    except ImportError:
        print("Matplotlib not installed; CSV results were still saved.")

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPLETE")
    print("=" * 78)
    print(f"Test graphs: {len(tests)}")
    print(f"Attack modes: {len(attack_modes)}")
    print(f"Attack levels: {len(attack_levels)}")
    print(f"Runtime: {time.time() - t0:.1f}s")
    print("\nSaved: multi_model_robustness_results.csv")


if __name__ == "__main__":
    main()
