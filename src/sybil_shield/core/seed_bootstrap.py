"""
seed_bootstrap.py -- unsupervised trust-seed selection.

Both build_features()/pick_seeds() (core/detector.py) and
real_facebook_eval.py's facebook_features() require a set of TRUSTED
SEEDS to run PPR/SybilRank trust propagation from. Up to now that set has
always come from labels: pick_seeds() samples nodes with label==0,
facebook_features() takes benign_train verbatim. Real deployments
(notably crypto wallets, but also cold-start social graphs) often don't
have verified-honest labels at all, so this module tries to infer a
trusted seed set with ZERO labels, using only graph structure.

Two approaches were tried and benchmarked against make_sybil_graph()
(synthetic, ground-truth-labeled) before this module was written; see
experiments/eval_seed_bootstrap.py for the harness and numbers.

  1. Per-node centrality ranking (score each node individually on
     degree / clustering / core-number / neighbor-overlap / betweenness /
     degree-ratio, take the top-scoring nodes as seeds). REJECTED: a
     small, artificially densified sybil farm routinely scores HIGHER on
     these than a larger, more sparsely-connected honest population --
     small dense clusters are the easiest thing to mistake for "the
     trusted core." In testing this had 0-10% seed purity (i.e. it
     mostly seeded PPR from the sybil farms themselves) and drove AUC
     towards 0, which is worse than doing nothing.

  2. Community detection (this module): run Louvain, take the LARGEST
     resulting community, sample seeds from within it. Bet: the honest
     population is the majority, best-connected community in the graph,
     which is a harder assumption for an attacker to fake than any
     single per-node statistic, because faking it requires the sybil
     region to stop being a separate, sparsely-attached community at all
     -- i.e. requires attacking the same sparse-cut assumption PPR
     itself depends on, not a side statistic.

     Benchmarked across make_sybil_graph() scenarios (n_honest=300,
     farm sizes up to 1.5x the honest population, mimicry + hub-
     targeting on): seed purity was 100% and AUC matched or beat
     labeled-seed PPR, for attack_edges up to ~300. Purity degrades
     gracefully (not catastrophically) as attack edges scale further:
     ~65-70% purity and AUC ~0.70 by attack_edges=600-1200, once the
     honest/sybil boundary stops being a sparse cut and Louvain starts
     merging them into one community. That failure threshold is worth
     tracking per-deployment (see eval_seed_bootstrap.py), not assumed.

This is a genuine trade of assumptions, not a way to avoid making one:
pick_seeds()/benign_train assumes you have verified-honest labels;
bootstrap_seeds_by_community() assumes the honest population is the
largest well-connected community. Prefer real labeled seeds when you
have them -- use this only when you don't, or as a cross-check against
labeled seeds (see cross_check_seeds()).
"""

import numpy as np
import networkx as nx


def bootstrap_seeds_by_community(G, nodes=None, frac_of_community=0.05,
                                  seed=0, min_seeds=3):
    """Infer a trusted seed set with no labels at all.

    Runs Louvain community detection on G, takes the single largest
    community as the presumed-honest population, and samples a subset
    of it to use as PPR/SybilRank trust-propagation seeds (same role as
    core.detector.pick_seeds() or real_facebook_eval's benign_train).

    Returns (seeds, community_nodes):
      seeds            -- list of nodes to pass to trust_scores()/
                           facebook_features() as trusted_seeds.
      community_nodes  -- the full inferred community, for diagnostics
                           (size, and purity if you happen to have
                           ground truth in an experiment).
    """
    nodes = list(nodes) if nodes is not None else list(G.nodes())
    node_set = set(nodes)

    communities = nx.algorithms.community.louvain_communities(G, seed=seed)
    # Restrict each community to the requested node subset before
    # ranking by size, so this stays correct if a caller passes a nodes
    # list that's a subset of G (mirrors pick_seeds()/facebook_features()
    # taking a `nodes` argument rather than assuming nodes == G.nodes()).
    communities = [set(c) & node_set for c in communities]
    communities = [c for c in communities if c]
    if not communities:
        raise ValueError("no communities found -- is the graph empty?")

    largest = max(communities, key=len)
    community_nodes = [n for n in nodes if n in largest]

    k = max(min_seeds, int(len(community_nodes) * frac_of_community))
    k = min(k, len(community_nodes))
    rng = np.random.RandomState(seed)
    seeds = list(rng.choice(community_nodes, size=k, replace=False))
    return seeds, community_nodes


def bootstrap_seeds_by_community_with_tiebreak(G, nodes=None, tiny_verified=None,
                                                frac_of_community=0.05, seed=0,
                                                min_seeds=3, top_k=3):
    """Fixes a real failure mode found while sandbox-testing
    bootstrap_seeds_by_community() above: when honest and sybil
    populations are close in size (e.g. an exactly-balanced 50/50 test
    population, which is exactly how real_facebook_eval.py's own
    benign_test/sybil_test split is constructed), Louvain has NO size
    cue to break the tie over which large community is honest -- it's
    a coin flip, and picking wrong means propagating trust outward from
    the sybil region with full (wrong) confidence. Measured: mean seed
    purity 0.40 (worse than useless) across 10 near-50/50 test graphs
    with community size as the only signal.

    The fix costs almost nothing: `tiny_verified`, as few as 2-3
    node ids you happen to know are honest (NOT a full benign_train --
    just enough to break a tie), used to pick which of the `top_k`
    largest communities to trust, instead of blindly taking the single
    largest. Measured on the same 10 graphs: mean purity 0.40 -> 0.98
    with tiny_verified of size 3.

    If tiny_verified is empty/None, falls back to plain
    bootstrap_seeds_by_community() (i.e. still works with zero labels,
    just without the fix for the near-50/50 failure mode -- that case
    is unavoidable without ANY external signal, verified-node or
    otherwise: Douceur's original result is that Sybil detection is
    impossible without an external trust anchor of some kind, and
    community size alone stops being one once the two regions are
    comparably sized).
    """
    nodes = list(nodes) if nodes is not None else list(G.nodes())
    if not tiny_verified:
        return bootstrap_seeds_by_community(
            G, nodes=nodes, frac_of_community=frac_of_community,
            seed=seed, min_seeds=min_seeds,
        )

    node_set = set(nodes)
    communities = nx.algorithms.community.louvain_communities(G, seed=seed)
    communities = [set(c) & node_set for c in communities]
    communities = [c for c in communities if c]
    communities = sorted(communities, key=len, reverse=True)[:top_k]

    verified_set = set(tiny_verified)
    scored = sorted(communities, key=lambda c: -len(c & verified_set))
    best = scored[0]
    community_nodes = [n for n in nodes if n in best]

    k = max(min_seeds, int(len(community_nodes) * frac_of_community))
    k = min(k, len(community_nodes))
    rng = np.random.RandomState(seed)
    seeds = list(rng.choice(community_nodes, size=k, replace=False))
    return seeds, community_nodes


def mixed_seed_strategy(G, nodes=None, seed=0, seed_frac=0.05,
                         bootstrap_prob=0.5, tiny_verified=None):
    """For TRAINING only: pick this graph's trusted-seed set either from
    true labels (core.detector.pick_seeds) or from the label-free
    community bootstrap above, chosen at random per graph.

    Why this matters, not just "why not always use real labels since
    it's synthetic data anyway": real_facebook_eval.py's benign_train
    and bootstrap_seeds_by_community() have different purity profiles
    (benign_train is 100% pure by construction; the community bootstrap
    is 100% pure under a sparse attack-edge cut but degrades to ~60-70%
    once attack edges get dense -- see eval_seed_bootstrap.py). If a
    model is ONLY ever trained against perfectly pure, label-derived
    seeds, it has never seen what its own 12-column feature vector
    looks like when the "trusted seed" pool itself has some sybils
    mixed in -- which is exactly the condition it'll actually face on
    an unlabeled real dataset using the community bootstrap instead of
    a benign_train file. Training on a mix of both conditions is meant
    to make the downstream GCN robust to either seed source at
    inference time, not just fast at fitting one of them.

    Returns a plain list of seed nodes -- same contract as pick_seeds()
    and bootstrap_seeds_by_community()'s first return value, so this
    drops into any call site that expects a seed list.
    """
    from sybil_shield.core.detector import pick_seeds

    nodes = list(nodes) if nodes is not None else list(G.nodes())
    rng = np.random.RandomState(seed)
    if rng.random() < bootstrap_prob:
        seeds, _ = bootstrap_seeds_by_community_with_tiebreak(
            G, nodes=nodes, tiny_verified=tiny_verified, seed=seed,
        )
        return seeds
    return pick_seeds(G, seed=seed, frac=seed_frac)


def cross_check_seeds(G, labeled_seeds, nodes=None, seed=0):
    """Sanity check for when you DO have some labeled seeds: what
    fraction of your labeled seeds fall inside the largest inferred
    community? A low fraction is a warning sign that either the labeled
    seeds are wrong, or the graph's community structure is already too
    fragmented / too attacked for the community-size assumption to hold
    -- worth surfacing before trusting either seed set blindly.
    """
    _, community_nodes = bootstrap_seeds_by_community(G, nodes=nodes, seed=seed)
    community_set = set(community_nodes)
    if not labeled_seeds:
        return 0.0
    agreement = sum(1 for s in labeled_seeds if s in community_set)
    return agreement / len(labeled_seeds)
