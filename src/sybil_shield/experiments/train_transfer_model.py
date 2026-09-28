"""
train_transfer_model.py

A second adversarial-training path, additive next to
gradient_attack_detector.train_gradient_adversarial (that function and
every existing caller of it -- train_model.py, real_facebook_eval.py's
fresh-train fallback -- are untouched). This one exists to test three
changes together, each one motivated by a specific gap found while
transferring the existing pipeline to the real Facebook dataset:

  1. Training DISTRIBUTION: benchmark.make_transfer_train_set() instead
     of make_configurable_train_set()/make_train() -- sybil fraction
     drawn up to 0.50 (every real eval constructs a ~balanced test
     population; the CLI-default path measured mean sybil fraction
     13.5%), attack edges sampled independently of sybil population
     size instead of capped at min(300, sybil_total), and a minority of
     graphs drawn from a much larger node-count range so the model
     isn't trained exclusively at ~10x lower average degree than the
     real target graph.

  2. Trusted-SEED quality: core.seed_bootstrap.mixed_seed_strategy()
     instead of always build_features()'s internal label-based
     pick_seeds(). Half (by default) of training graphs get their
     4 PPR/trust columns + PPR-variance + nearest-seed-distance built
     from a label-free, community-bootstrapped seed set instead of a
     perfectly pure label-derived one -- so the model has actually seen
     what those columns look like when the "trusted seed" pool itself
     isn't 100% pure, which is the normal condition once you're running
     against a dataset with no benign_train file at all.

  3. Attack DIVERSITY: every attack_every epochs, the attack mode is
     drawn from {"gradient", "random", "hubs"} instead of always being
     the model's own current-weights gradient attack. Pure gradient
     self-play only ever sharpens the model against an attacker with
     white-box access to itself; mixing in attacks.py's model-agnostic
     random/hub-targeting modes tests robustness against attacker
     shapes the model didn't co-evolve with. ("trusted" mode is
     deliberately excluded -- real_facebook_eval.py's module docstring
     already found it poisons the model, scoring 0.56 AUC on real
     Facebook by teaching "high-trust, high-degree = suspicious".)

Kept as its own module (not a rewrite of train_gradient_adversarial)
specifically so the existing, already-validated training path stays
available and comparable side-by-side -- see
eval_transfer_training.py for exactly that comparison, run on small
graphs/short epochs as a sandbox check before committing to a full
(50-graph, 120-epoch, ~10 minute) run.
"""

import time
from functools import partial

import numpy as np

from sybil_shield.core.attacks import attack_graph
from sybil_shield.core.adversarial import greedy_gradient_attack
from sybil_shield.core.detector import compute_static_topology
from sybil_shield.core.gcn import GCN
from sybil_shield.core.gradient_attack_detector import (
    build_features_with_seeds, build_features_normalized_adjacency,
)
from sybil_shield.core.seed_bootstrap import mixed_seed_strategy
from sybil_shield.experiments.benchmark import make_transfer_train_set
from sybil_shield.experiments.eval_robust import cw

NON_GRADIENT_ATTACK_MODES = ("random", "hubs")  # "trusted" excluded -- see module docstring


def train_gradient_adversarial_transfer(
    train_graphs=None, num_graphs=40, seed_frac=0.05, n_epochs=120, lr=0.02,
    attack_every=2, attack_budget=8, recompute_every=3, hidden_dims=(16, 16),
    betweenness_k=30, bootstrap_seed_prob=0.5, attack_mode_weights=(0.5, 0.25, 0.25),
    rng=None, verbose=True, epoch_cb=None,
):
    """Same overall shape as gradient_attack_detector.train_gradient_adversarial
    (clean fit_step every epoch, attacked fit_step every attack_every
    epochs), with the three changes in the module docstring layered in.

    train_graphs: pass explicit graphs (e.g. for a small sandbox test);
    otherwise num_graphs are drawn from make_transfer_train_set().

    attack_mode_weights: (gradient, random, hubs) selection probabilities
    for the attack step, resampled independently per (epoch, graph).

    Returns the trained model.
    """
    rng = rng or np.random.default_rng(0)
    t0 = time.time()

    if train_graphs is None:
        train_graphs = make_transfer_train_set(
            num_graphs=num_graphs, seed=int(rng.integers(0, 2**31))
        )

    prep = []
    for i, G in enumerate(train_graphs):
        nodes = list(G.nodes())
        node_index = {n: idx for idx, n in enumerate(nodes)}
        y = np.array([G.nodes[n].get("label", 0) for n in nodes])
        seeds = mixed_seed_strategy(
            G, nodes=nodes, seed=i, seed_frac=seed_frac,
            bootstrap_prob=bootstrap_seed_prob,
        )
        X = build_features_with_seeds(G, node_index, seeds)
        A_hat = build_features_normalized_adjacency(G, node_index)
        prep.append(dict(G=G, X=X, y=y, A_hat=A_hat,
                          node_index=node_index, seeds=seeds))

    model = GCN(n_features=prep[0]["X"].shape[1], hidden_dims=hidden_dims, seed=0)
    p_gradient, p_random, p_hubs = attack_mode_weights

    for epoch in range(n_epochs):
        for gi, p in enumerate(prep):
            class_weight = cw(p["y"])

            # 1. clean step, every epoch
            loss = model.fit_step(p["A_hat"], p["X"], p["y"],
                                   class_weight=class_weight, lr=lr)

            # 2. attack the model, every attack_every epochs -- mode
            # resampled each time so no single graph always sees the
            # same attacker shape.
            if epoch % attack_every == 0:
                mode = rng.choice(["gradient", "random", "hubs"],
                                   p=[p_gradient, p_random, p_hubs])
                if mode == "gradient":
                    static = compute_static_topology(
                        p["G"], seed=0, betweenness_k=betweenness_k
                    )
                    feature_fn = partial(
                        build_features_with_seeds, seeds=p["seeds"], static=static
                    )
                    np_rng = np.random.RandomState(int(rng.integers(0, 2**31)))
                    _G_atk, A_atk, X_atk, _added, _hist = greedy_gradient_attack(
                        model, p["G"].copy(), p["node_index"], p["y"], p["X"],
                        p["seeds"], budget=attack_budget,
                        class_weight=class_weight,
                        recompute_every=recompute_every, rng=np_rng,
                        feature_fn=feature_fn,
                    )
                else:
                    np_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                    G_atk = attack_graph(p["G"], attack_budget, mode=mode, rng=np_rng)
                    X_atk = build_features_with_seeds(G_atk, p["node_index"], p["seeds"])
                    A_atk = build_features_normalized_adjacency(G_atk, p["node_index"])

                model.fit_step(A_atk, X_atk, p["y"], class_weight=class_weight, lr=lr)

            if epoch_cb is not None:
                epoch_cb(epoch=epoch, graph_index=gi, loss=float(loss))

        if verbose and epoch % 20 == 0:
            print(f"  [transfer-adversarial] epoch {epoch:3d}/{n_epochs} "
                  f"({time.time() - t0:.0f}s elapsed)")

    if verbose:
        print(f"  [transfer-adversarial] epoch {n_epochs - 1:3d}/{n_epochs} "
              f"({time.time() - t0:.0f}s elapsed)")

    return model
