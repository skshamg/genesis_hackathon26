"""
eval_seed_bootstrap.py -- does unsupervised (label-free) trust-seed
selection hold up against labeled seeds, across attack intensity?

Compares three ways of choosing the trusted seed set that
trust_scores()/SybilRank propagates from:

  labeled       core.detector.pick_seeds() -- ground-truth benign labels,
                the existing baseline (what real_facebook_eval.py does
                with benign_train).
  community     core.seed_bootstrap.bootstrap_seeds_by_community() --
                no labels, largest Louvain community.
  naive-central core.seed_bootstrap-style per-node centrality ranking,
                inlined here rather than shipped in seed_bootstrap.py --
                kept ONLY as a regression check that we don't accidentally
                reintroduce it. It is expected to score badly (that's
                the point of the module docstring): a small, densified
                sybil farm scores higher on raw centrality than a
                larger honest population, so this seeds PPR from the
                sybils themselves.

Reports, per attack-edge level: seed purity (fraction of the chosen
seed set that's actually honest) and resulting SybilRank AUC, for both
label-free methods, next to the labeled baseline.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[3]
    SRC = ROOT / "src"
    for candidate in (str(ROOT), str(SRC)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

import numpy as np
import networkx as nx
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from sybil_shield.experiments.benchmark import (
    make_sybil_graph, count_attack_edges, HONEST_FAMILIES, ALL_FAMILIES,
)
from sybil_shield.core.detector import (
    pick_seeds, trust_scores, extra_structural_features,
)
from sybil_shield.core.seed_bootstrap import bootstrap_seeds_by_community


def _rank(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def _naive_centrality_seeds(G, nodes, frac=0.05, seed=0):
    """Rejected approach, kept only so a regression test can confirm it
    stays bad. See seed_bootstrap.py's module docstring for why."""
    cl = nx.clustering(G)
    core = nx.core_number(G)
    degree = _rank([G.degree(n) for n in nodes])
    clustering = _rank([cl[n] for n in nodes])
    corenum = _rank([core[n] for n in nodes])
    overlap, betweenness, _ppr_var, degree_ratio, _dist = \
        extra_structural_features(G, nodes, trusted_seeds=[], seed=seed)
    overlap = _rank(overlap)
    betweenness = _rank(betweenness)
    degree_ratio = _rank(degree_ratio)
    score = (corenum + clustering + overlap - betweenness
             - np.abs(degree_ratio - np.median(degree_ratio)))
    k = max(3, int(len(nodes) * frac))
    top = np.argsort(-score)[:k]
    return [nodes[i] for i in top]


def _auc_for_seeds(G, nodes, y, seeds):
    ts = trust_scores(G, nodes, seeds)
    return roc_auc_score(y, -ts["sybilrank"])


def run_level(attack_edges, n_honest=300, n_farms=3, farm_size_range=(80, 120),
              trials=5, base_seed=10_000):
    rows = {"purity_labeled": [], "auc_labeled": [],
            "purity_community": [], "auc_community": [],
            "purity_naive": [], "auc_naive": []}
    for r in range(trials):
        rng = np.random.default_rng(base_seed + attack_edges * 100 + r)
        G = make_sybil_graph(
            n_honest=n_honest,
            honest_family=str(rng.choice(HONEST_FAMILIES)),
            sybil_family=str(rng.choice(ALL_FAMILIES)),
            n_farms=n_farms, farm_size_range=farm_size_range,
            attack_edges=attack_edges, mimic=True, target_honest_hubs=True,
            seed=int(rng.integers(0, 2**31)),
        )
        nodes = sorted(G.nodes())
        y = np.array([G.nodes[n]["label"] for n in nodes])

        labeled = pick_seeds(G, seed=r)
        community_seeds, _ = bootstrap_seeds_by_community(G, nodes, seed=r)
        naive_seeds = _naive_centrality_seeds(G, nodes, seed=r)

        rows["purity_labeled"].append(np.mean([G.nodes[n]["label"] == 0 for n in labeled]))
        rows["auc_labeled"].append(_auc_for_seeds(G, nodes, y, labeled))
        rows["purity_community"].append(np.mean([G.nodes[n]["label"] == 0 for n in community_seeds]))
        rows["auc_community"].append(_auc_for_seeds(G, nodes, y, community_seeds))
        rows["purity_naive"].append(np.mean([G.nodes[n]["label"] == 0 for n in naive_seeds]))
        rows["auc_naive"].append(_auc_for_seeds(G, nodes, y, naive_seeds))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in rows.items()}


def main():
    levels = [10, 50, 150, 300, 600, 1200]
    cols = ["purity_labeled", "auc_labeled",
            "purity_community", "auc_community",
            "purity_naive", "auc_naive"]
    print(f"{'attack_edges':>12s} " + " ".join(f"{c:>18s}" for c in cols))
    for g in levels:
        res = run_level(g)
        print(f"{g:>12d} " + " ".join(f"{res[c][0]:>13.3f}±{res[c][1]:.2f}" for c in cols))


if __name__ == "__main__":
    main()
