"""Graph convolutional network utilities used by the detector and API."""

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class GCN:
    def __init__(self, n_features, hidden_dims=(16, 16), n_classes=2, seed=0, l2=1e-4):
        rng = np.random.RandomState(seed)
        dims = [n_features] + list(hidden_dims) + [n_classes]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.W.append(rng.randn(dims[i], dims[i + 1]) * scale)
            self.b.append(np.zeros(dims[i + 1]))
        self.n_layers = len(self.W)
        self.l2 = l2
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(bb) for bb in self.b]
        self.vb = [np.zeros_like(bb) for bb in self.b]
        self.t = 0

    def forward(self, A_hat, X):
        H = X
        cache = {"H": [H], "Z": []}
        for l in range(self.n_layers):
            AH = A_hat @ H
            Z = AH @ self.W[l] + self.b[l]
            cache["Z"].append((AH, Z))
            if l < self.n_layers - 1:
                H = relu(Z)
            else:
                H = softmax(Z)
            cache["H"].append(H)
        return H, cache

    def compute_loss(self, probs, y, class_weight=None):
        n = probs.shape[0]
        eps = 1e-12
        logp = -np.log(probs[np.arange(n), y] + eps)
        if class_weight is not None:
            w = class_weight[y]
            loss = np.sum(logp * w) / np.sum(w)
        else:
            loss = np.mean(logp)
        l2_term = self.l2 * sum(np.sum(w * w) for w in self.W)
        return loss + l2_term

    def backward(self, A_hat, y, cache, class_weight=None):
        n = cache["H"][0].shape[0]
        n_classes = cache["H"][-1].shape[1]
        onehot = np.zeros((n, n_classes))
        onehot[np.arange(n), y] = 1.0

        if class_weight is not None:
            w = class_weight[y].reshape(-1, 1)
        else:
            w = np.ones((n, 1))
        w_sum = w.sum()

        dZ = (cache["H"][-1] - onehot) * w / w_sum

        gradsW, gradsB = [None] * self.n_layers, [None] * self.n_layers
        for l in reversed(range(self.n_layers)):
            AH, Z = cache["Z"][l]
            gradsW[l] = AH.T @ dZ + 2 * self.l2 * self.W[l]
            gradsB[l] = dZ.sum(axis=0)
            if l > 0:
                dH = A_hat @ (dZ @ self.W[l].T)
                Z_prev = cache["Z"][l - 1][1]
                dZ = dH * relu_grad(Z_prev)
        return gradsW, gradsB
    def edge_gradients(self, A_hat, X, y, class_weight=None):
        """
        dLoss/dA_hat -- saliency of each adjacency entry, used by
        adversarial.greedy_gradient_attack to pick which edge to add next.
        Mirrors backward()'s layer loop but accumulates the direct
        contribution of each layer's A_hat @ H usage into dA_hat instead
        of weight/bias grads. The recursive dependency of later layers'
        H on A_hat is already captured by the normal dH backprop chain,
        exactly as in backward() -- no separate handling needed.
        """
        probs, cache = self.forward(A_hat, X)
        n = cache["H"][0].shape[0]
        n_classes = cache["H"][-1].shape[1]
        onehot = np.zeros((n, n_classes))
        onehot[np.arange(n), y] = 1.0

        if class_weight is not None:
            w = class_weight[y].reshape(-1, 1)
        else:
            w = np.ones((n, 1))
        w_sum = w.sum()

        dZ = (cache["H"][-1] - onehot) * w / w_sum

        dA_hat = np.zeros_like(A_hat)
        for l in reversed(range(self.n_layers)):
            AH, Z = cache["Z"][l]
            H_in = cache["H"][l]
            dAH = dZ @ self.W[l].T
            dA_hat += dAH @ H_in.T
            if l > 0:
                dH = A_hat @ dAH
                Z_prev = cache["Z"][l - 1][1]
                dZ = dH * relu_grad(Z_prev)
        return dA_hat
    
    def adam_step(self, gradsW, gradsB, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for l in range(self.n_layers):
            self.mW[l] = beta1 * self.mW[l] + (1 - beta1) * gradsW[l]
            self.vW[l] = beta2 * self.vW[l] + (1 - beta2) * (gradsW[l] ** 2)
            mW_hat = self.mW[l] / (1 - beta1 ** self.t)
            vW_hat = self.vW[l] / (1 - beta2 ** self.t)
            self.W[l] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            self.mb[l] = beta1 * self.mb[l] + (1 - beta1) * gradsB[l]
            self.vb[l] = beta2 * self.vb[l] + (1 - beta2) * (gradsB[l] ** 2)
            mb_hat = self.mb[l] / (1 - beta1 ** self.t)
            vb_hat = self.vb[l] / (1 - beta2 ** self.t)
            self.b[l] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def predict_proba(self, A_hat, X):
        probs, _ = self.forward(A_hat, X)
        return probs

    def fit_step(self, A_hat, X, y, class_weight=None, lr=0.01):
        probs, cache = self.forward(A_hat, X)
        loss = self.compute_loss(probs, y, class_weight)
        gradsW, gradsB = self.backward(A_hat, y, cache, class_weight)
        self.adam_step(gradsW, gradsB, lr=lr)
        return loss

    def embedding(self, A_hat, X):
        _, cache = self.forward(A_hat, X)
        return cache["H"][-2]

    def get_architecture(self):
        """{n_features, hidden_dims, n_classes, l2} -- enough to reconstruct
        an empty model of the same shape before load_weights fills it in,
        and what model_store.py writes into meta.json for the API's
        /api/model/info architecture-diagram panel."""
        dims = [self.W[0].shape[0]] + [w.shape[1] for w in self.W]
        return {
            "n_features": dims[0],
            "hidden_dims": dims[1:-1],
            "n_classes": dims[-1],
            "l2": self.l2,
        }

    def save_weights(self, path):
        """Persist architecture + trained weights (not Adam optimizer
        state -- a reloaded model is for inference/further eval, not
        exact-resume training) to a single .npz file."""
        arrays = {f"W{i}": w for i, w in enumerate(self.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, n_layers=self.n_layers, l2=self.l2, **arrays)

    @classmethod
    def load_weights(cls, path):
        """Inverse of save_weights: rebuilds a GCN with the saved
        architecture and drops the saved weights in, bypassing the
        random init in __init__."""
        data = np.load(path, allow_pickle=False)
        n_layers = int(data["n_layers"])
        l2 = float(data["l2"])
        W = [data[f"W{i}"] for i in range(n_layers)]
        b = [data[f"b{i}"] for i in range(n_layers)]
        n_features = W[0].shape[0]
        hidden_dims = tuple(w.shape[1] for w in W[:-1])
        n_classes = W[-1].shape[1]
        model = cls(n_features=n_features, hidden_dims=hidden_dims,
                     n_classes=n_classes, seed=0, l2=l2)
        model.W = W
        model.b = b
        return model
