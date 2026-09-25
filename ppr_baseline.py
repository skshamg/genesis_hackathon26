"""
ppr_baseline.py

The non-GNN baseline: personalized PageRank (PPR) seeded from trusted honest
nodes -- this is the core idea behind SybilRank. Honest nodes near the seeds
accumulate high trust; Sybil nodes, sitting behind a narrow choke point, get
starved of trust flow. Score = PPR value itself (higher = more trusted/honest).
"""

import numpy as np
from features import personalized_pagerank_from_seeds


def ppr_trust_scores(G, seeds, node_index):
    ppr = personalized_pagerank_from_seeds(G, seeds)
    scores = np.zeros(len(node_index))
    for n, i in node_index.items():
        scores[i] = ppr[n]
    return scores
