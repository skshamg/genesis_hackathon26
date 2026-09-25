"""
real_facebook_eval.py

Zero-shot transfer evaluation of the 4 multi-model-robustness GCN variants
from eval_multi_model.py -- standard / random-augmented / trusted-augmented /
mixed-augmented -- plus a probability-averaged ensemble, on the real
Undirected_Facebook benchmark.

All four models are trained EXACTLY as in eval_multi_model.py: on
detector.build_features(...), on the same synthetic training graphs /
attack-augmentation pipeline. build_features() normally derives its own
trusted seeds from ground-truth node labels (pick_seeds), which the real
Facebook graph doesn't have. So for the Facebook graph we reproduce
build_features's exact feature construction (same 7 columns, same order,
same rank-normalization) but seed trust propagation with the real,
known-honest benign_train nodes from train.txt instead.

No test labels are used to construct features, and sybil_train is
deliberately unused -- only benign_train nodes act as trusted seeds.
"""

import os
import numpy as np
import networkx as nx
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, accuracy_score

from detector import trust_scores
from eval_multi_model import train_on, make_augmented_training_set
from eval_robust import make_train
from gradient_attack_detector import train_gradient_adversarial


# detector.py (the top-level compat wrapper) only re-exports
# build_features/early_walk/pick_seeds/ppr_vector/sweep_cut/trust_scores,
# not the private _rank/_adj helpers build_features uses internally.
# Inlined here, verbatim, so facebook_features() below stays byte-for-byte
# consistent with bad_people_finder/core/detector.py's own build_features().

def _rank(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def _adj(G, nodes):
    return nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")

# ---------------------------------------------------------
# CONFIG -- point this at your local dataset
# ---------------------------------------------------------

DATA_DIR = r"C:\Users\saman\OneDrive\Desktop\Undirected_Facebook\Undirected_Facebook"

GRAPH_FILE = os.path.join(DATA_DIR, "graph.txt")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")


# ---------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------

def load_labels_file(path):
    """
    train.txt / test.txt share the same format:
        line 1 = benign nodes
        line 2 = sybil nodes
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    benign = [int(x) for x in lines[0].split()]
    sybil = [int(x) for x in lines[1].split()]
    return benign, sybil


def load_graph(path):
    """
    graph.txt contains undirected edges, each stored twice; NetworkX
    collapses the duplicates automatically.
    """
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            u, v = map(int, line.split()[:2])
            if u != v:
                G.add_edge(u, v)
    return G


# ---------------------------------------------------------
# FEATURES -- mirrors detector.build_features EXACTLY,
# but with explicit trusted seeds instead of pick_seeds().
# ---------------------------------------------------------

def facebook_features(G, nodes, trusted_seeds):
    """
    Reproduces build_features(G, use_trust=True, use_struct=True)'s
    7-column feature layout and rank-normalization, seeded from
    known-honest nodes rather than sampled-from-labels.

    Columns (in order):
        rank(log ppr0.5/deg), rank(log ppr0.8/deg), rank(log ppr0.95/deg),
        rank(log sybilrank), rank(degree), rank(clustering), rank(core number)
    """
    ts = trust_scores(G, nodes, trusted_seeds)
    cols = []
    for k in ("ppr0.5/deg", "ppr0.8/deg", "ppr0.95/deg", "sybilrank"):
        cols.append(_rank(np.log(ts[k] + 1e-12)))

    cl = nx.clustering(G)
    core = nx.core_number(G)
    cols.append(_rank([G.degree(n) for n in nodes]))
    cols.append(_rank([cl[n] for n in nodes]))
    cols.append(_rank([core[n] for n in nodes]))

    X = np.column_stack(cols)

    A = _adj(G, nodes).toarray()
    A += np.eye(len(nodes))
    d = A.sum(1) ** -0.5
    A_hat = d[:, None] * A * d[None, :]
    return X, A_hat


# ---------------------------------------------------------
# TRAIN THE 4 VARIANTS -- identical to eval_multi_model.py
# ---------------------------------------------------------

def train_all_models(seed=1):
    print("[1/3] Generating shared synthetic training graphs...")
    base_train = make_train(36, seed=seed)

    print("[2/3] Training the 4 model variants "
          "(same pipeline as eval_multi_model.py)...")

    models = {}

    print("  - standard")
    models["standard"] = train_on(base_train)

    print("  - random-augmented")
    models["random"] = train_on(
        make_augmented_training_set(base_train, "random", seed=100)
    )

    print("  - trusted-augmented")
    models["trusted"] = train_on(
        make_augmented_training_set(base_train, "trusted", seed=200)
    )

    print("  - mixed-augmented")
    models["mixed"] = train_on(
        make_augmented_training_set(base_train, "mixed", seed=300)
    )

    print("  - gradient-adversarial (interleaved, feature-aware -- "
          "slower than the others, this is the new one)")
    models["gradient"] = train_gradient_adversarial(
        base_train, n_epochs=120, attack_every=2, attack_budget=8,
        rng=np.random.default_rng(400),
    )

    return models


# ---------------------------------------------------------
# EVALUATE ALL 4 + ENSEMBLE ON THE REAL GRAPH
# ---------------------------------------------------------

def evaluate_all(models, G, benign_train, benign_test, sybil_test):
    nodes = sorted(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    print("[3/3] Building trust+structural features "
          "from benign_train seeds only...")
    X, A_hat = facebook_features(G, nodes, benign_train)

    test_nodes = benign_test + sybil_test
    y = np.array([0] * len(benign_test) + [1] * len(sybil_test))
    test_idx = [node_index[n] for n in test_nodes]

    print()
    print(f"{'model':>10s}  {'AUC':>7s}  {'acc':>7s}")
    print("-" * 30)

    results = {}
    all_probs = []

    for name, model in models.items():
        probs = model.predict_proba(A_hat, X)[:, 1]
        scores = probs[test_idx]
        auc = roc_auc_score(y, scores)
        acc = accuracy_score(y, (scores >= 0.5).astype(int))
        results[name] = {"auc": auc, "acc": acc, "scores": scores}
        print(f"{name:>10s}  {auc:>7.4f}  {acc:>7.4f}")
        all_probs.append(probs)

    ensemble_probs = np.mean(np.vstack(all_probs), axis=0)
    ens_scores = ensemble_probs[test_idx]
    ens_auc = roc_auc_score(y, ens_scores)
    ens_acc = accuracy_score(y, (ens_scores >= 0.5).astype(int))
    results["ensemble"] = {"auc": ens_auc, "acc": ens_acc, "scores": ens_scores}
    print(f"{'ensemble':>10s}  {ens_auc:>7.4f}  {ens_acc:>7.4f}")

    return results, y


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("=" * 70)
    print("SYBIL DETECTION -- MULTI-MODEL FACEBOOK TRANSFER EVALUATION")
    print("=" * 70)

    print("\nLoading Facebook graph and labels...")
    G = load_graph(GRAPH_FILE)
    benign_train, sybil_train = load_labels_file(TRAIN_FILE)   # sybil_train unused
    benign_test, sybil_test = load_labels_file(TEST_FILE)

    print(f"Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"benign_train={len(benign_train)}  "
          f"benign_test={len(benign_test)}  sybil_test={len(sybil_test)}")
    print("(sybil_train is intentionally unused -- only benign_train "
          "seeds trust propagation, no test labels touch features)")

    models = train_all_models(seed=1)
    results, y = evaluate_all(models, G, benign_train, benign_test, sybil_test)

    print("\nFinished.")
    return results, y


if __name__ == "__main__":
    main()
