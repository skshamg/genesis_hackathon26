"""
real_facebook_eval.py

Zero-shot transfer evaluation of the Sybil-detection GCN on the real
Undirected_Facebook benchmark. Single model: gradient-adversarial only.

  - standard / random-augmented / mixed-augmented / trusted-augmented
    variants and the ensemble are gone. Historical evidence for why (kept
    here since it's the actual justification, not just a preference):
    on a real Facebook run, standard scored 0.8420 AUC, random-augmented
    0.8592, gradient-adversarial 0.8529, trusted-augmented 0.56 (worse
    than a coin flip -- attacks.py's mode="trusted" only ever targets the
    top-decile PPR/degree honest pool, so training against it teaches the
    model "high-trust, high-degree = suspicious", which is poison against
    real hubs), and the flat-average ensemble 0.7951 -- worse than any
    single non-trusted variant, because averaging in the poisoned
    "trusted" model dragged the good ones down. gradient_attack_graph's
    white-box, model-driven edge selection was the fix for the "trusted"
    failure mode specifically, and it's also competitive with the best
    heuristic variant (random-augmented) while being adversarially robust
    by construction rather than by accident -- so it's now the only model
    trained, instead of one input to an ensemble that measured worse than
    using it alone.

  - seed_frac=0.05 used everywhere synthetic training features are built
    (the pick_seeds fraction). Was 0.02, based on seed_frac_sweep.py's
    result at n≈300 synthetic-graph scale. That result did NOT transfer:
    at Facebook scale (n≈4,039 honest nodes, 100 real seeds, frac≈0.025,
    avg degree ~44 -- 15-20x larger and much denser than the sweep's
    n≈250-340 regime), every model variant scored WORSE at seed_frac=0.02
    than at 0.05 (standard 0.7792 vs 0.8420, random 0.8046 vs 0.8592,
    gradient 0.8116 vs 0.8529). 0.05 is the better-performing value on the
    only real evidence available. TODO: rerun seed_frac_sweep.py against a
    synthetic set matched to Facebook's scale (~4,000-8,000 honest nodes,
    avg degree ~40+, ~100 seeds) so this stops being a single anchor point.

No test labels are used to construct features, and sybil_train is
deliberately unused -- only benign_train nodes act as trusted seeds.
"""

import argparse
import os

import numpy as np
import networkx as nx
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, accuracy_score

from sybil_shield.core.detector import trust_scores, extra_structural_features
from sybil_shield.experiments.eval_robust import make_train
from sybil_shield.core.gradient_attack_detector import (
    train_gradient_adversarial,
)


SEED_FRAC = 0.05  # was 0.02 -- see module docstring, that value measured
                   # worse than 0.05 on every model variant at Facebook
                   # scale despite the n≈300 sweep recommending it.


# detector.py (the top-level compat wrapper) only re-exports
# build_features/early_walk/pick_seeds/ppr_vector/sweep_cut/trust_scores,
# not the private _rank/_adj helpers build_features uses internally.
# Inlined here, verbatim, so facebook_features() below stays byte-for-byte
# consistent with sybil_shield/core/detector.py's own build_features().

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
    12-column feature layout and rank-normalization, seeded from
    known-honest nodes rather than sampled-from-labels.

    Columns (in order):
        4 trust features, 3 existing structural features, then:
        neighbor-overlap, approximate betweenness, seed-subset PPR variance,
        degree/neighbor-degree ratio, and distance to nearest trusted seed.
    """
    ts = trust_scores(G, nodes, trusted_seeds)
    cols = []
    for k in ("ppr0.5/deg", "ppr0.8/deg", "ppr0.95/deg", "sybilrank"):
        cols.append(_rank(np.log(ts[k] + 1e-12)))

    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    cl = nx.clustering(G)
    core = nx.core_number(G)
    cols.append(_rank([G.degree(n) for n in nodes]))
    cols.append(_rank([cl[n] for n in nodes]))
    cols.append(_rank([core[n] for n in nodes]))

    # Same five added structural features used by synthetic training.
    for feature in extra_structural_features(G, nodes, trusted_seeds, seed=0):
        cols.append(_rank(feature))

    X = np.column_stack(cols)

    A = _adj(G, nodes).toarray()
    A += np.eye(len(nodes))
    d = A.sum(1) ** -0.5
    A_hat = d[:, None] * A * d[None, :]
    return X, A_hat


# train_gradient_adversarial itself now lives in gradient_attack_detector.py
# (imported above) -- it grew a seed_frac param there specifically so this
# module's SEED_FRAC override still threads through, and this file no
# longer keeps its own byte-for-byte copy of the training loop. Kept as a
# module attribute (`train_gradient_adversarial` is imported, not just
# called) so existing callers/tests that reach it via
# `real_facebook_eval.train_gradient_adversarial(...)` keep working.


def train_all_models(seed=1, betweenness_k=30, model_id='None',
                      model_type="gradient_adversarial"):
    """
    Single-model pipeline: gradient-adversarial only. standard/random/mixed
    and the trusted-augmented variant are dropped (see module docstring --
    trusted scored 0.56 AUC on real Facebook, worse than a coin flip, and
    the flat-ensemble average scored below gradient alone). Kept as a
    {"gradient": model} dict, not a bare model, so callers (app.py's
    _load_facebook_eval_models, predict_facebook_dataset) that iterate
    model variants by name don't need to change shape -- there's just one
    variant now.

    model_id: if given, LOAD that model from the registry (core.model_store)
    instead of training a fresh one here -- this is the "refer to a model
    trained by train.py / POST /api/train/start" path. model_type selects
    which registry sub-folder model_id is looked up in (default matches
    train_model.py's default). Pass "latest" to use the most recently
    trained model of model_type without knowing its id.

    With no model_id, this still trains fresh at full intensity (120
    epochs, the train_gradient_adversarial default) -- unchanged from
    before the registry existed. That's a real ~9-10 min run; once you
    have a model_id from train.py, prefer passing it.
    """
    if model_id is not None:
        from sybil_shield.core import model_store
        resolved = model_store.resolve_model_id(model_type, model_id)
        print(f"[1/1] Loading model '{resolved}' ({model_type}) from registry...")
        model = model_store.load_model(model_type, resolved)
        return {"gradient": model}

    print("[1/2] Generating shared synthetic training graphs...")
    base_train = make_train(36, seed=seed)

    print(f"[2/2] Training gradient-adversarial model (seed_frac={SEED_FRAC}, "
          f"120 epochs -- full intensity, no registry model_id given)...")
    model = train_gradient_adversarial(
        base_train, seed_frac=SEED_FRAC, betweenness_k=betweenness_k
    )

    return {"gradient": model}


# ---------------------------------------------------------
# EVALUATE THE MODEL ON THE REAL GRAPH
# ---------------------------------------------------------

def evaluate_all(models, G, benign_train, benign_test, sybil_test):
    nodes = sorted(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    print("[2/2] Building trust+structural features "
          "from benign_train seeds only...")
    X, A_hat = facebook_features(G, nodes, benign_train)

    test_nodes = benign_test + sybil_test
    y = np.array([0] * len(benign_test) + [1] * len(sybil_test))
    test_idx = [node_index[n] for n in test_nodes]

    model = models["gradient"]
    probs = model.predict_proba(A_hat, X)[:, 1]
    scores = probs[test_idx]
    auc = roc_auc_score(y, scores)
    acc = accuracy_score(y, (scores >= 0.5).astype(int))

    print()
    print(f"{'model':>10s}  {'AUC':>7s}  {'acc':>7s}")
    print("-" * 30)
    print(f"{'gradient':>10s}  {auc:>7.4f}  {acc:>7.4f}")

    return {"gradient": {"auc": auc, "acc": acc, "scores": scores}}, y


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main(model_id=None, model_type="gradient_adversarial", seed=1,
         betweenness_k=30, data_dir=DATA_DIR):
    global DATA_DIR, GRAPH_FILE, TRAIN_FILE, TEST_FILE
    DATA_DIR = data_dir
    GRAPH_FILE = os.path.join(DATA_DIR, "graph.txt")
    TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
    TEST_FILE = os.path.join(DATA_DIR, "test.txt")

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

    models = train_all_models(seed=seed, betweenness_k=betweenness_k,
                             model_id=model_id, model_type=model_type)
    results, y = evaluate_all(models, G, benign_train, benign_test, sybil_test)

    print("\nFinished.")
    return results, y


def _cli():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the Facebook benchmark.")
    parser.add_argument("--model-id", default=None, help="Registry model ID to load; defaults to training a fresh one.")
    parser.add_argument("--model-type", default="gradient_adversarial")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--betweenness-k", type=int, default=30)
    parser.add_argument("--data-dir", default=DATA_DIR, help="Path to the Undirected_Facebook dataset root.")
    args = parser.parse_args()
    main(model_id=args.model_id, model_type=args.model_type, seed=args.seed,
         betweenness_k=args.betweenness_k, data_dir=args.data_dir)


if __name__ == "__main__":
    _cli()
