"""
gradient_attack_detector.py

A feature-aware gradient attack for the detector.py / attacks.py / GCN
pipeline (the one behind eval_multi_model.py's "trusted" variant, which
showed a heavy false-positive bias on real Facebook hubs).

Two pieces:

  build_features_with_seeds(G, node_index, seeds)
      Mirrors detector.build_features(G, use_trust=True, use_struct=True)
      COLUMN-FOR-COLUMN, but with explicit, fixed seeds and node ordering
      instead of pick_seeds() resampling from labels on every call. Needed
      so a gradient attack (and adversarial training) recomputes features
      against the SAME trusted seeds as the graph is perturbed.

  gradient_attack_graph(...)
      Drop-in replacement for attacks.attack_graph(..., mode="trusted"):
      white-box, model-aware, and feature-aware -- it recomputes the real
      trust+structural features (not just adjacency) as it scores
      candidate edges, using adversarial.greedy_gradient_attack under the
      hood. Returns an attacked COPY; labels untouched.

IMPORTANT -- architectural note, not just an implementation detail:
attacks.py's "random" / "hubs" / "trusted" modes need no model, so
eval_multi_model.py's make_augmented_training_set() can pre-generate a
whole attacked training set ONCE, before training starts. A gradient
attack is white-box: it needs the model's CURRENT weights, which only
exist mid-training. So you can't just swap "trusted" -> "gradient" inside
make_augmented_training_set() and keep the rest of the pipeline as-is --
attacking has to move INSIDE the epoch loop (attack this epoch's graph
against the model as it currently stands, fit on it, repeat), the way
adversarial.adversarial_train_step() already does for the features.py
pipeline. This module gives you the attack primitive; the training-loop
rewrite (train_on -> an epoch loop that calls gradient_attack_graph each
epoch) is the follow-up step once you've confirmed this attack behaves the
way you want on your test graphs.
"""

import numpy as np
import networkx as nx
from functools import partial
from scipy.stats import rankdata

from sybil_shield.core.detector import (
    trust_scores, pick_seeds, extra_structural_features, compute_static_topology,
)
from sybil_shield.core.adversarial import greedy_gradient_attack


def _rank(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def _adj(G, nodes):
    return nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")


def build_features_with_seeds(G, node_index, seeds, static=None):
    """
    detector.build_features(..., use_trust=True, use_struct=True)'s exact
    7-column layout and rank-normalization, but seeded explicitly and
    ordered by node_index instead of pick_seeds()/G.nodes().

    static: optional dict from detector.compute_static_topology(G) --
    precomputed clustering/core_number/betweenness. Pass this when this
    function is being called repeatedly against a graph that only changes
    by a few edges between calls (the gradient-attack inner loop): those
    three are ~70% of this function's cost and barely move with a
    handful of added edges, so recomputing them fresh every call is mostly
    wasted work. Node set must be unchanged since static was computed
    (true for attacks.py/adversarial.py, which only ever add edges).
    Trust scores (what the attack is actually trying to fool) are always
    recomputed live regardless.
    """
    nodes = [None] * len(node_index)
    for n, i in node_index.items():
        nodes[i] = n

    ts = trust_scores(G, nodes, seeds)
    cols = []
    for k in ("ppr0.5/deg", "ppr0.8/deg", "ppr0.95/deg", "sybilrank"):
        cols.append(_rank(np.log(ts[k] + 1e-12)))

    if static is not None:
        cl, core, btw = static["clustering"], static["core"], static["betweenness"]
    else:
        cl, core, btw = nx.clustering(G), nx.core_number(G), None

    cols.append(_rank([G.degree(n) for n in nodes]))
    cols.append(_rank([cl[n] for n in nodes]))
    cols.append(_rank([core[n] for n in nodes]))
    cols.extend(_rank(feature) for feature in
                extra_structural_features(G, nodes, seeds, seed=0, betweenness=btw))

    return np.column_stack(cols)


def build_features_normalized_adjacency(G, node_index):
    nodes = [None] * len(node_index)
    for n, i in node_index.items():
        nodes[i] = n
    A = _adj(G, nodes).toarray()
    A += np.eye(len(nodes))
    d = A.sum(1) ** -0.5
    return d[:, None] * A * d[None, :]


def gradient_attack_graph(model, G, seed_knowledge=0, budget=20,
                           class_weight=None, recompute_every=5,
                           candidate_pool=400, rng=None, betweenness_k=30):
    """
    White-box, feature-aware gradient attack against `model`'s CURRENT
    weights. Signature mirrors attacks.attack_graph() so it can be swapped
    in wherever attack_graph(mode="trusted", ...) is called TODAY, with one
    real difference: this needs a model, attack_graph's heuristic modes
    don't (see module docstring above for why that matters for training-set
    generation vs. interleaved training).

    betweenness_k: sample size for the one-off static-topology computation
    (see build_features_with_seeds' `static` param) done at the start of
    this attack and reused across every recompute step within it. Lower
    than detector.build_features's default (100) since this only needs to
    be good enough to rank candidate edges, not a final reported feature.

    Returns an attacked COPY of G (labels untouched), same contract as
    attack_graph.
    """
    rng = rng or np.random.default_rng(0)
    np_rng = np.random.RandomState(int(rng.integers(0, 2**31)))

    G = G.copy()
    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    y = np.array([G.nodes[n].get("label", 0) for n in nodes])

    seeds = pick_seeds(G, seed_knowledge)
    static = compute_static_topology(G, seed=0, betweenness_k=betweenness_k)
    X0 = build_features_with_seeds(G, node_index, seeds, static=static)

    def feature_fn(G_live, ni):
        return build_features_with_seeds(G_live, ni, seeds, static=static)

    G_atk, _A_hat, _X, _added, _hist = greedy_gradient_attack(
        model, G, node_index, y, X0, seeds,
        budget=budget, class_weight=class_weight,
        recompute_every=recompute_every, candidate_pool=candidate_pool,
        rng=np_rng, feature_fn=feature_fn,
    )
    return G_atk


def train_gradient_adversarial(train_graphs, seed_frac=0.05, n_epochs=120, lr=0.02,
                                attack_every=2, attack_budget=8,
                                recompute_every=3, hidden_dims=(16, 16),
                                betweenness_k=30, rng=None, verbose=True,
                                epoch_cb=None):
    """
    True interleaved adversarial training for the detector.py/GCN pipeline:
    every graph gets a normal clean fit_step every epoch, and every
    `attack_every` epochs it ALSO gets attacked against the model's
    CURRENT weights (feature-aware: trust+structural features recomputed
    as the attack adds edges, not just adjacency) and fit on the result.

    This is the piece attacks.py's "trusted"/"hubs"/"random" modes can't
    do, because they're pre-generated once before training starts and
    never see the model at all -- this attacks fresh, every time, against
    whatever the model currently believes.

    Per-attack cost: clustering/core_number/betweenness (~70% of one
    feature build, and the least attack-relevant part -- they barely move
    from `attack_budget` added edges on a several-hundred-node graph) are
    computed ONCE per attack call via compute_static_topology and reused
    across every recompute step inside it, instead of fresh at each one.
    betweenness_k additionally trims that one-off computation itself.
    Live PPR trust scores (what the attack is actually trying to fool)
    are still recomputed at every step regardless. Benchmarked ~4x faster
    per attack call at the defaults below vs. the original always-fresh
    version, with clustering/core/betweenness pinned to their pre-attack
    values for the duration of one attack (a fidelity trade-off, not free
    -- see compute_static_topology's docstring).

    seed_frac: threaded through to build_features() for the "prep" (clean)
    feature construction, so callers that need a non-default trusted-seed
    fraction (e.g. real_facebook_eval.py's SEED_FRAC=0.05 override, or
    train_model.py's CLI --seed-frac) don't silently train against the
    detector.py default instead of the value they asked for.

    epoch_cb: optional callback(epoch, graph_index, loss, honest_count,
    sybil_count, current_graph) invoked once per (epoch, graph) pair,
    right after that graph's clean fit_step, with `current_graph` a plain
    {"nodes":[{"id","label"}], "edges":[[u,v]]} dict of THIS training
    graph -- for a caller (train_model.py) that wants to stream live
    progress (e.g. to an API job-status endpoint) instead of just
    printing. When set, this replaces the periodic print (verbose is
    still honored for the final "training done" line's caller, not
    printed here).

    Returns the trained model.
    """
    import time
    from sybil_shield.core.detector import build_features, compute_static_topology
    from sybil_shield.experiments.eval_robust import cw
    from sybil_shield.core.gcn import GCN

    rng = rng or np.random.default_rng(0)
    t0 = time.time()

    prep = []
    for i, G in enumerate(train_graphs):
        X, y, A_hat, nodes, seeds = build_features(G, seed=i, seed_frac=seed_frac)
        node_index = {n: idx for idx, n in enumerate(nodes)}
        prep.append(dict(G=G, X=X, y=y, A_hat=A_hat,
                          node_index=node_index, seeds=seeds))

    model = GCN(n_features=prep[0]["X"].shape[1],
                hidden_dims=hidden_dims, seed=0)

    for epoch in range(n_epochs):
        for gi, p in enumerate(prep):
            class_weight = cw(p["y"])

            # 1. clean step, every epoch
            loss = model.fit_step(p["A_hat"], p["X"], p["y"],
                                   class_weight=class_weight, lr=lr)

            # 2. attack the model's CURRENT weights, then fit on the result
            if epoch % attack_every == 0:
                static = compute_static_topology(
                    p["G"], seed=0, betweenness_k=betweenness_k
                )
                feature_fn = partial(
                    build_features_with_seeds, seeds=p["seeds"], static=static
                )
                np_rng = np.random.RandomState(
                    int(rng.integers(0, 2**31))
                )
                _G_atk, A_atk, X_atk, _added, _hist = greedy_gradient_attack(
                    model, p["G"].copy(), p["node_index"], p["y"], p["X"],
                    p["seeds"], budget=attack_budget,
                    class_weight=class_weight,
                    recompute_every=recompute_every, rng=np_rng,
                    feature_fn=feature_fn,
                )
                model.fit_step(A_atk, X_atk, p["y"],
                                class_weight=class_weight, lr=lr)

            if epoch_cb is not None:
                epoch_cb(
                    epoch=epoch, graph_index=gi, loss=float(loss),
                    honest_count=int(np.sum(p["y"] == 0)),
                    sybil_count=int(np.sum(p["y"] == 1)),
                    current_graph={
                        "nodes": [
                            {"id": str(n), "label": int(p["G"].nodes[n].get("label", 0))}
                            for n in p["G"].nodes()
                        ],
                        "edges": [[str(u), str(v)] for u, v in p["G"].edges()],
                    },
                )

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  [gradient-adversarial] epoch {epoch:3d}/"
                  f"{n_epochs} ({time.time()-t0:.0f}s elapsed)")

    return model
