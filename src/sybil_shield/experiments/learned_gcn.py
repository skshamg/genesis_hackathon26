"""
learned_gcn.py

A GCN that builds its OWN node features instead of consuming the 12
hand-engineered columns (PPR trust, SybilRank, clustering, k-core,
betweenness, ...).

Contract (why nothing else in the repo has to change)
-----------------------------------------------------
Every consumer -- api/app.py, real_facebook_eval.py, crypto_eval.py,
train_model.evaluate_model -- does:

    X, A_hat = <build the 12-column handmade features + normalized adjacency>
    model = model_store.load_model(model_type, model_id)
    probs = model.predict_proba(A_hat, X)[:, 1]

LearnedTopologyGCN keeps that exact signature. It subclasses GCN, so
fit_step / adam_step / compute_loss / predict_proba / edge_gradients /
save_weights / load_weights / .W / .b / .n_layers all exist with the same
meaning (the API's Inspect Model tab reads .W and .b directly).

What the model actually sees
----------------------------
Only two things are read from the inputs:

  1. A_hat  -- the normalized adjacency (the raw graph).
  2. Which nodes are the trusted seeds. In the handmade X this is
     recoverable exactly: column SEED_DIST_COL (11) is "distance to
     nearest trusted seed", rank-normalized, and seeds are the nodes tied
     at its minimum (distance 0). The training path can also pass a bare
     (n, 1) seed mask instead of a full X, which is much cheaper to build.

Trusted seeds are an INPUT (the anchor every Sybil defence needs), not a
feature. Everything derived from them -- how trust flows, how far it
reaches, how structure differs between regions -- is learned by the
propagation stack below rather than hand-written.

Architecture
------------
raw node inputs R = [seed_mask, log_degree(standardized), 1]      (n x 3)

  H_0 = R
  H_l = relu( A_hat @ H_{l-1} @ W_l + b_l )         l = 1..L   (hidden = h)
  C   = concat(H_1 ... H_L)                                     (n x L*h)
  P   = softmax( C @ W_out + b_out )

Concatenating every depth (jumping knowledge) lets the classifier mix
1-hop, 2-hop, ... L-hop views, which is how the network can rediscover
PPR-like trust reach without being handed PPR. Weight list layout is
[W_1..W_L, W_out], so W[-1] is the "output_layer" for the Inspect tab.

Gradients
---------
forward/backward are exact. edge_gradients() returns dLoss/dA_hat through
every propagation layer. The degree channel of R is derived from A_hat's
diagonal each forward pass (so an attacked graph is automatically
re-featurized -- no feature_fn recompute needed), but it is treated as a
constant when differentiating w.r.t. A_hat; that is a small approximation
and is documented in the tests.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from sybil_shield.core.gcn import GCN, relu, relu_grad, softmax

# Column of the handmade 12-col X holding rank-normalized distance to the
# nearest trusted seed (see detector.extra_structural_features, 5th extra).
SEED_DIST_COL = 11
KIND = "learned_topology_gcn"
N_RAW = 3  # seed_mask, log_degree, constant


def seed_mask_features(G, node_index, seeds, static=None):
    """Cheap replacement for build_features_with_seeds() for this model:
    an (n, 1) 0/1 seed mask. Same call signature (static is ignored) so it
    can be injected as the training loop's feature builder. Skips PPR,
    betweenness, clustering, etc. entirely -- the model doesn't use them."""
    mask = np.zeros((len(node_index), 1), dtype=np.float64)
    for s in seeds:
        if s in node_index:
            mask[node_index[s], 0] = 1.0
    return mask


def decode_seed_mask(X, seed_col=SEED_DIST_COL):
    """Boolean (n,) trusted-seed mask from either a bare (n,1) mask or the
    full handmade feature matrix."""
    X = np.asarray(X)
    if X.ndim == 1:
        return X > 0.5
    if X.shape[1] == 1:
        return X[:, 0] > 0.5
    if X.shape[1] <= seed_col:
        raise ValueError(
            f"X has {X.shape[1]} columns; expected 1 (seed mask) or a handmade "
            f"feature matrix with the seed-distance column at index {seed_col}."
        )
    col = X[:, seed_col]
    return col <= col.min() + 1e-12


class LearnedTopologyGCN(GCN):
    def __init__(self, hidden_dim=32, n_prop_layers=4, n_classes=2, seed=0,
                 l2=1e-4, seed_col=SEED_DIST_COL):
        # deliberately does NOT call GCN.__init__ (different weight shapes)
        rng = np.random.RandomState(seed)
        self.hidden_dim = int(hidden_dim)
        self.n_prop_layers = int(n_prop_layers)
        self.seed_col = int(seed_col)
        dims = [N_RAW] + [self.hidden_dim] * self.n_prop_layers
        self.W, self.b = [], []
        for i in range(self.n_prop_layers):
            self.W.append(rng.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i]))
            self.b.append(np.zeros(dims[i + 1]))
        jk = self.hidden_dim * self.n_prop_layers
        self.W.append(rng.randn(jk, n_classes) * np.sqrt(2.0 / jk))
        self.b.append(np.zeros(n_classes))
        self._init_optimizer(l2)

    def _init_optimizer(self, l2):
        self.n_layers = len(self.W)
        self.l2 = l2
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(bb) for bb in self.b]
        self.vb = [np.zeros_like(bb) for bb in self.b]
        self.t = 0

    # ------------------------------------------------------------------
    def raw_inputs(self, A_hat, X):
        """(n, 3) raw node inputs. Degree comes from A_hat's diagonal:
        A_hat[i,i] = 1/(deg_i+1) after self-loop normalization."""
        seed = decode_seed_mask(X, self.seed_col).astype(np.float64)
        n = A_hat.shape[0]
        deg = 1.0 / np.maximum(np.diag(A_hat), 1e-12) - 1.0
        logd = np.log1p(np.maximum(deg, 0.0))
        std = logd.std()
        logd = (logd - logd.mean()) / (std if std > 1e-12 else 1.0)
        # Seed channel scaled so its graph-average is 1: diffusion values then
        # mean "trust density relative to average" regardless of how many
        # seeds a graph has, the same job rank-normalization does for the
        # handmade columns.
        frac = seed.mean()
        seed = seed / frac if frac > 0 else seed
        return np.column_stack([seed, logd, np.ones(n)])

    def forward(self, A_hat, X):
        R = self.raw_inputs(A_hat, X)
        H = R
        cache = {"H": [R], "Z": [], "R": R}
        Hs = []
        for l in range(self.n_prop_layers):
            AH = A_hat @ H
            Z = AH @ self.W[l] + self.b[l]
            cache["Z"].append((AH, Z))
            H = relu(Z)
            cache["H"].append(H)
            Hs.append(H)
        C = np.concatenate(Hs, axis=1)
        cache["C"] = C
        P = softmax(C @ self.W[-1] + self.b[-1])
        cache["P"] = P
        return P, cache

    def _output_grad(self, cache, y, class_weight):
        P = cache["P"]
        n, n_classes = P.shape
        onehot = np.zeros((n, n_classes))
        onehot[np.arange(n), y] = 1.0
        w = class_weight[y].reshape(-1, 1) if class_weight is not None else np.ones((n, 1))
        return (P - onehot) * w / w.sum()

    def _backprop(self, A_hat, y, cache, class_weight, want_edge=False):
        L, h = self.n_prop_layers, self.hidden_dim
        dZout = self._output_grad(cache, y, class_weight)
        gW, gb = [None] * self.n_layers, [None] * self.n_layers
        gW[-1] = cache["C"].T @ dZout + 2 * self.l2 * self.W[-1]
        gb[-1] = dZout.sum(axis=0)
        dC = dZout @ self.W[-1].T
        direct = [dC[:, i * h:(i + 1) * h] for i in range(L)]

        dA_hat = np.zeros_like(A_hat) if want_edge else None
        carry = None  # gradient flowing into H_l from layer l+1
        for l in reversed(range(L)):
            dH = direct[l] if carry is None else direct[l] + carry
            AH, Z = cache["Z"][l]
            dZ = dH * relu_grad(Z)
            gW[l] = AH.T @ dZ + 2 * self.l2 * self.W[l]
            gb[l] = dZ.sum(axis=0)
            dAH = dZ @ self.W[l].T
            if want_edge:
                dA_hat += dAH @ cache["H"][l].T
            carry = A_hat @ dAH  # A_hat symmetric; H_0 (=R) needs no grad
        return gW, gb, dA_hat

    def backward(self, A_hat, y, cache, class_weight=None):
        gW, gb, _ = self._backprop(A_hat, y, cache, class_weight)
        return gW, gb

    def edge_gradients(self, A_hat, X, y, class_weight=None):
        _, cache = self.forward(A_hat, X)
        _, _, dA_hat = self._backprop(A_hat, y, cache, class_weight, want_edge=True)
        return dA_hat

    def embedding(self, A_hat, X):
        """The learned (n, L*h) node representation the classifier reads."""
        _, cache = self.forward(A_hat, X)
        return cache["C"]

    # ------------------------------------------------------------------
    def get_architecture(self):
        return {
            "kind": KIND,
            "n_features": N_RAW,
            "input_columns": ["seed_mask", "log_degree", "const"],
            "hidden_dims": [self.hidden_dim] * self.n_prop_layers,
            "n_classes": int(self.W[-1].shape[1]),
            "l2": self.l2,
            "readout": "jumping_knowledge_concat",
            "feature_source": "learned",
        }

    def save_weights(self, path):
        arrays = {f"W{i}": w for i, w in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, kind=KIND, n_layers=self.n_layers, l2=self.l2,
                 hidden_dim=self.hidden_dim, n_prop_layers=self.n_prop_layers,
                 seed_col=self.seed_col, **arrays)

    @classmethod
    def load_weights(cls, path):
        data = np.load(path, allow_pickle=False)
        model = cls(hidden_dim=int(data["hidden_dim"]),
                    n_prop_layers=int(data["n_prop_layers"]),
                    n_classes=int(data[f"W{int(data['n_layers']) - 1}"].shape[1]),
                    seed=0, l2=float(data["l2"]), seed_col=int(data["seed_col"]))
        n = int(data["n_layers"])
        model.W = [data[f"W{i}"] for i in range(n)]
        model.b = [data[f"b{i}"] for i in range(n)]
        return model


def load_weights_any(path):
    """Load whichever model class wrote `path`. Files written before this
    module existed have no 'kind' key and load as a plain GCN, so every
    existing registry entry keeps working."""
    data = np.load(path, allow_pickle=False)
    if "kind" in data.files and str(data["kind"]) == KIND:
        return LearnedTopologyGCN.load_weights(path)
    return GCN.load_weights(path)
