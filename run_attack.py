import numpy as np
from sklearn.metrics import roc_auc_score
from generate_graphs import make_labeled_graph
from features import extract_features, normalized_adjacency
from gcn import GCN
from adversarial import greedy_gradient_attack

# build + train
G = make_labeled_graph(n_honest=150, n_farms=2, farm_size_range=(20,30), choke_points_range=(1,2), seed=0)
X, y, node_index, seeds = extract_features(G, seed=0)
A_hat = normalized_adjacency(G, node_index)
cw = np.array([1.0, len(y)/(2*y.sum())])

model = GCN(n_features=5, hidden_dims=(16,16), n_classes=2, seed=0)
for _ in range(80):
    model.fit_step(A_hat, X, y, class_weight=cw, lr=0.02)

probs, _ = model.forward(A_hat, X)
print("clean AUC:", roc_auc_score(y, probs[:,1]))

# attack -- feature-aware: recompute X (not just A_hat) as edges are added
from functools import partial
from features import extract_features_with_seeds

feature_fn = partial(extract_features_with_seeds, seeds=seeds)
G2, A2, X2, added, hist = greedy_gradient_attack(
    model, G.copy(), node_index, y, X, seeds,
    budget=25, class_weight=cw, recompute_every=3,
    feature_fn=feature_fn,
)
probs2, _ = model.forward(A2, X2)
print("attacked AUC:", roc_auc_score(y, probs2[:,1]))
print("loss trajectory:", [round(h,4) for h in hist])