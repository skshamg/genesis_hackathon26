"""
adversarial.py

White-box gradient attack on the GCN's structure: greedily adds
honest<->sybil edges that most increase the model's loss on the true
labels (i.e. most evade detection), using GCN.edge_gradients as the
saliency signal. Recomputes A_hat + gradients after each flip since
symmetric normalization means the linearized score goes stale fast.

Also provides an adversarial-training loop: interleave normal fit_step
with attack-and-retrain so the model sees camouflaged graphs during
training, not just at eval time.
"""

import numpy as np

from features import normalized_adjacency


def greedy_gradient_attack(model, G, node_index, y, X, seeds,
                            budget=20, class_weight=None, recompute_every=1,
                            candidate_pool=400, rng=None, feature_fn=None):
    """
    Adds up to `budget` honest<->sybil edges to G, chosen greedily by
    dLoss/dA_hat. Mutates and returns G (plus a per-step history of loss).

    candidate_pool: at each recompute, only score a random subsample of
    non-edges (full n_honest*n_sybil pairs is wasteful for large graphs).

    feature_fn: optional callable (G, node_index) -> X. When given, X is
    RECOMPUTED from the live (attacked-so-far) graph on the same cadence as
    A_hat/gradients (every `recompute_every` steps). Without it (default),
    X stays frozen at its initial value for the whole attack -- faithful to
    the GCN's own message-passing, but blind to any topology-derived
    features (PPR, degree, clustering, sybilrank, ...) a deployed pipeline
    recomputes after an attack. A frozen-X attack optimizes a proxy
    objective that can diverge from what actually fools the full pipeline.
    Pass e.g. functools.partial(extract_features_with_seeds, seeds=seeds)
    (features.py) or an equivalent bound to fixed trusted seeds for
    build_features (detector.py) to close that gap. Node ordering must stay
    fixed (only edges are added, never nodes), so node_index is safe to
    reuse across recomputes.
    """
    rng = rng or np.random.RandomState(0)
    honest = [n for n, d in G.nodes(data=True) if d.get("label", 0) == 0]
    sybil = [n for n, d in G.nodes(data=True) if d.get("label", 0) == 1]

    history = []
    added = []

    A_hat = normalized_adjacency(G, node_index)
    for step in range(budget):
        if step % recompute_every == 0:
            if feature_fn is not None and step > 0:
                # Refresh topology-derived features against the graph as
                # attacked so far -- not just the adjacency the GCN sees.
                X = feature_fn(G, node_index)
            probs, _ = model.forward(A_hat, X)
            loss = model.compute_loss(probs, y, class_weight)
            history.append(loss)
            dA_hat = model.edge_gradients(A_hat, X, y, class_weight)

        # sample candidate non-edges honest<->sybil
        h_sample = rng.choice(honest, size=min(candidate_pool, len(honest)), replace=True)
        s_sample = rng.choice(sybil, size=min(candidate_pool, len(sybil)), replace=True)
        best_score, best_pair = -np.inf, None
        for h, s in zip(h_sample, s_sample):
            if G.has_edge(h, s):
                continue
            i, j = node_index[h], node_index[s]
            score = dA_hat[i, j] + dA_hat[j, i]
            if score > best_score:
                best_score, best_pair = score, (h, s)

        if best_pair is None:
            break
        G.add_edge(*best_pair)
        added.append(best_pair)

        # cheap incremental A_hat update would need re-derivation of D^-1/2;
        # for correctness (not just speed) just rebuild it each flip.
        A_hat = normalized_adjacency(G, node_index)

    if feature_fn is not None:
        # Final history entry should reflect the fully-attacked graph's
        # real features, not whatever was last cached at a recompute step.
        X = feature_fn(G, node_index)
    probs, _ = model.forward(A_hat, X)
    history.append(model.compute_loss(probs, y, class_weight))
    return G, A_hat, X, added, history


def adversarial_train_step(model, G, node_index, X, y, seeds=None,
                            class_weight=None, attack_budget=5, lr=0.02,
                            rng=None, feature_fn=None):
    """
    One adversarial-training round: attack the current graph a little,
    then fit on the attacked version, so the model learns to stay robust
    to camouflage edges rather than just seeing clean graphs.

    seeds / feature_fn: forwarded to greedy_gradient_attack so the attack
    (and the fit step that follows it) can use topology-derived features
    recomputed against the live attacked graph, not a frozen snapshot.
    Returns (clean_loss, attacked_loss_after_fit).
    """
    G_attacked, A_hat_atk, X_atk, added, hist = greedy_gradient_attack(
        model, G, node_index, y, X, seeds=seeds,
        budget=attack_budget, class_weight=class_weight, rng=rng,
        feature_fn=feature_fn,
    )
    loss = model.fit_step(A_hat_atk, X_atk, y, class_weight=class_weight, lr=lr)
    return hist[0], loss, added
