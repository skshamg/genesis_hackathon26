#!/usr/bin/env python3
"""
SYBIL-SHIELD: frozen transfer_half_fb evaluation on Elliptic++ Actors.

This script DOES NOT TRAIN. It loads the existing 12 -> 16 -> 16 -> 2
transfer_half_fb checkpoint and evaluates it on the labeled Elliptic++
address-address graph.

The original SYBIL-SHIELD GCN uses a dense N x N adjacency matrix. That is
not possible for ~823k wallets, so this evaluator keeps the same normalized
GCN equation but stores A_hat as scipy CSR and performs sparse @ dense
multiplication.

Two topology features are estimated for scalability:
  - local clustering coefficient: sampled wedges for high-degree nodes
  - mean neighbor Jaccard overlap: sampled incident edges for high-degree nodes

Sampled betweenness is also used, matching the project's bounded-budget idea.
All trust/PPR, degree, core-number, degree-ratio, seed-distance and GCN
operations are sparse/scalable implementations.

Labels:
  1 = illicit
  2 = licit
  3 = unknown

Evaluation excludes class 3. Five percent of licit wallets are selected as
trusted benign seeds with a deterministic seed. Those seeds are excluded from
the scored evaluation set.

Run from the GENESIS repository root:
    python scripts\\elliptic_transfer_eval.py

Optional:
    python scripts\\elliptic_transfer_eval.py --data-dir C:\\Users\\saman\\OneDrive\\Desktop
    python scripts\\elliptic_transfer_eval.py --seed 1 --seed-frac 0.05
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Project/model loading
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sybil_shield.core.model_store import load_model, load_meta  # noqa: E402

MODEL_TYPE = "gradient_adversarial"
MODEL_ID = "transfer_half_fb"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rank01(x: np.ndarray) -> np.ndarray:
    """Same rank normalization as sybil_shield.core.detector._rank."""
    x = np.asarray(x, dtype=np.float64)
    return (rankdata(x, method="average") - 1.0) / max(len(x) - 1, 1)


def ppr_vector(A: sp.csr_matrix, deg: np.ndarray, s: np.ndarray,
               alpha: float, iters: int = 200) -> np.ndarray:
    """Same PPR recurrence as detector.py, using sparse A."""
    inv = 1.0 / np.maximum(deg, 1.0)
    pi = s.astype(np.float64, copy=True)
    for _ in range(iters):
        pi = alpha * (A @ (inv * pi)) + (1.0 - alpha) * s
    return pi


def early_walk(A: sp.csr_matrix, deg: np.ndarray, s: np.ndarray,
               steps: int) -> np.ndarray:
    inv = 1.0 / np.maximum(deg, 1.0)
    pi = s.astype(np.float64, copy=True)
    for _ in range(steps):
        pi = A @ (inv * pi)
    return pi


def connected_components(A: sp.csr_matrix) -> np.ndarray:
    """Component labels using scipy's sparse graph routine."""
    from scipy.sparse.csgraph import connected_components
    _, labels = connected_components(A, directed=False, return_labels=True)
    return labels


def multi_source_distance(A: sp.csr_matrix, seeds: np.ndarray) -> np.ndarray:
    """Exact unweighted distance from every node to the nearest seed."""
    n = A.shape[0]
    dist = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
    q = deque()
    for s in seeds:
        s = int(s)
        if dist[s] != 0:
            dist[s] = 0
            q.append(s)
    indptr = A.indptr
    indices = A.indices
    while q:
        u = q.popleft()
        nd = dist[u] + 1
        for p in range(indptr[u], indptr[u + 1]):
            v = int(indices[p])
            if dist[v] == np.iinfo(np.int32).max:
                dist[v] = nd
                q.append(v)
    finite = dist[dist < np.iinfo(np.int32).max]
    fallback = int(finite.max()) + 1 if finite.size else n
    dist[dist == np.iinfo(np.int32).max] = fallback
    return dist.astype(np.float64)


def core_numbers(A: sp.csr_matrix) -> np.ndarray:
    """Exact k-core numbers using a heap-based Batagelj-style peeling."""
    n = A.shape[0]
    deg = np.diff(A.indptr).astype(np.int64)
    current = deg.copy()
    core = np.zeros(n, dtype=np.int32)
    removed = np.zeros(n, dtype=np.bool_)
    heap = [(int(current[i]), i) for i in range(n)]
    heapq.heapify(heap)
    indptr, indices = A.indptr, A.indices

    while heap:
        d, u = heapq.heappop(heap)
        if removed[u] or d != int(current[u]):
            continue
        removed[u] = True
        core[u] = d
        for p in range(indptr[u], indptr[u + 1]):
            v = int(indices[p])
            if not removed[v] and current[v] > d:
                current[v] -= 1
                heapq.heappush(heap, (int(current[v]), v))
    return core.astype(np.float64)


def neighbor_degree_ratio(A: sp.csr_matrix, deg: np.ndarray) -> np.ndarray:
    """Exact degree / mean-neighbor-degree."""
    inv = 1.0 / np.maximum(deg, 1.0)
    # A @ deg gives sum of neighbor degrees.
    sum_neighbor_deg = A @ deg
    count = deg
    mean_neighbor_deg = np.divide(
        sum_neighbor_deg, np.maximum(count, 1.0),
        out=np.ones_like(sum_neighbor_deg, dtype=np.float64),
        where=count > 0,
    )
    return deg / np.maximum(mean_neighbor_deg, 1e-12)


def binary_contains(sorted_neighbors: np.ndarray, x: int) -> bool:
    return bool(np.searchsorted(sorted_neighbors, x) < len(sorted_neighbors)
                and sorted_neighbors[np.searchsorted(sorted_neighbors, x)] == x)


def exact_jaccard_rows(A: sp.csr_matrix, u: int, v: int) -> float:
    """Exact Jaccard between two sorted CSR neighbor rows."""
    au = A.indices[A.indptr[u]:A.indptr[u + 1]]
    av = A.indices[A.indptr[v]:A.indptr[v + 1]]
    i = j = inter = 0
    while i < len(au) and j < len(av):
        a, b = int(au[i]), int(av[j])
        if a == b:
            inter += 1; i += 1; j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    union = len(au) + len(av) - inter
    return inter / union if union else 0.0


def sampled_neighbor_jaccard(A: sp.csr_matrix, rng: np.random.RandomState,
                             max_edges_per_node: int = 16) -> np.ndarray:
    """Estimate mean incident-edge Jaccard, exact on sampled edges."""
    n = A.shape[0]
    out = np.zeros(n, dtype=np.float64)
    indptr = A.indptr
    for u in range(n):
        nbrs = A.indices[indptr[u]:indptr[u + 1]]
        d = len(nbrs)
        if d == 0:
            continue
        if d <= max_edges_per_node:
            chosen = nbrs
        else:
            chosen = nbrs[rng.choice(d, size=max_edges_per_node, replace=False)]
        vals = [exact_jaccard_rows(A, u, int(v)) for v in chosen]
        out[u] = float(np.mean(vals)) if vals else 0.0
    return out


def sampled_clustering(A: sp.csr_matrix, rng: np.random.RandomState,
                       max_pairs_per_node: int = 64) -> np.ndarray:
    """Estimate local clustering by sampling neighbor pairs.

    For a node with degree d, the clustering coefficient is the fraction of
    sampled unordered neighbor pairs that have an edge between them.
    """
    n = A.shape[0]
    out = np.zeros(n, dtype=np.float64)
    indptr, indices = A.indptr, A.indices
    for u in range(n):
        nbrs = indices[indptr[u]:indptr[u + 1]]
        d = len(nbrs)
        if d < 2:
            continue
        total_pairs = d * (d - 1) // 2
        k = min(max_pairs_per_node, total_pairs)
        if total_pairs <= k:
            # Exact for small-degree nodes.
            hits = 0
            possible = total_pairs
            for i in range(d - 1):
                vi = int(nbrs[i])
                row = indices[indptr[vi]:indptr[vi + 1]]
                for j in range(i + 1, d):
                    if binary_contains(row, int(nbrs[j])):
                        hits += 1
        else:
            hits = 0
            possible = k
            for _ in range(k):
                i, j = rng.choice(d, size=2, replace=False)
                vi, vj = int(nbrs[i]), int(nbrs[j])
                row = indices[indptr[vi]:indptr[vi + 1]]
                if binary_contains(row, vj):
                    hits += 1
        out[u] = hits / possible if possible else 0.0
    return out


def sampled_betweenness(A: sp.csr_matrix, rng: np.random.RandomState,
                        k: int = 30) -> np.ndarray:
    """Sampled Brandes betweenness for unweighted undirected graphs.

    This is the same bounded-source idea as NetworkX's
    betweenness_centrality(G, k=30), implemented directly over CSR so the
    full graph never becomes a NetworkX object.
    """
    n = A.shape[0]
    k = min(k, n)
    sources = rng.choice(n, size=k, replace=False)
    bc = np.zeros(n, dtype=np.float64)
    indptr, indices = A.indptr, A.indices

    for s in sources:
        s = int(s)
        stack = []
        pred = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        dist = np.full(n, -1, dtype=np.int32)
        sigma[s] = 1.0
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            nd = dist[v] + 1
            for p in range(indptr[v], indptr[v + 1]):
                w = int(indices[p])
                if dist[w] < 0:
                    dist[w] = nd
                    q.append(w)
                if dist[w] == nd:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n, dtype=np.float64)
        while stack:
            w = stack.pop()
            if sigma[w] != 0:
                coeff = (1.0 + delta[w]) / sigma[w]
                for v in pred[w]:
                    delta[v] += sigma[v] * coeff
            if w != s:
                bc[w] += delta[w]

    # NetworkX normalized undirected sampled betweenness uses a source-based
    # scaling of approximately 1 / ((n-1)(n-2)/2), adjusted for k samples.
    if n > 2:
        bc *= n / k
        bc /= ((n - 1) * (n - 2) / 2.0)
    return bc


def choose_licit_seeds(labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    licit = np.flatnonzero(labels == 2)
    if len(licit) == 0:
        raise RuntimeError("No class-2 (licit) wallets were found in wallets_classes.txt")
    k = max(3, int(len(licit) * frac))
    k = min(k, len(licit))
    return rng.choice(licit, size=k, replace=False).astype(np.int64)


def load_graph(data_dir: Path):
    labels_path = data_dir / "wallets_classes.txt"
    edges_path = data_dir / "AddrAddr_edgelist.txt"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    print(f"[1/8] Reading labels: {labels_path}")
    address_to_id: dict[str, int] = {}
    labels_list: list[int] = []
    with labels_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            addr = row[0].strip()
            if not addr:
                continue
            try:
                cls = int(row[1])
            except ValueError:
                continue
            if addr in address_to_id:
                continue
            address_to_id[addr] = len(labels_list)
            labels_list.append(cls)

    print(f"      labeled addresses: {len(labels_list):,}")
    print(f"[2/8] Reading edges: {edges_path} (~{edges_path.stat().st_size / 1e6:.1f} MB)")
    rows_u: list[int] = []
    rows_v: list[int] = []
    with edges_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        # The provided file has no meaningful header in the raw dataset, but
        # this also tolerates input_address,output_address.
        pending = [] if first is None else [first]
        for row in pending:
            if len(row) >= 2 and row[0].strip().lower() not in {"input_address", "input"}:
                a, b = row[0].strip(), row[1].strip()
                if a and b and a != b:
                    u = address_to_id.get(a)
                    if u is None:
                        u = len(labels_list); address_to_id[a] = u; labels_list.append(3)
                    v = address_to_id.get(b)
                    if v is None:
                        v = len(labels_list); address_to_id[b] = v; labels_list.append(3)
                    rows_u.append(u); rows_v.append(v)
        for row in reader:
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if not a or not b or a == b:
                continue
            u = address_to_id.get(a)
            if u is None:
                u = len(labels_list); address_to_id[a] = u; labels_list.append(3)
            v = address_to_id.get(b)
            if v is None:
                v = len(labels_list); address_to_id[b] = v; labels_list.append(3)
            rows_u.append(u); rows_v.append(v)

    n = len(labels_list)
    print(f"      raw non-self edge rows: {len(rows_u):,}")
    # The project treats the graph as undirected. COO->CSR also sums duplicate
    # entries; binarize afterward so parallel/repeated edges count once.
    u = np.asarray(rows_u, dtype=np.int32)
    v = np.asarray(rows_v, dtype=np.int32)
    data = np.ones(len(u) * 2, dtype=np.float64)
    rr = np.concatenate([u, v])
    cc = np.concatenate([v, u])
    A = sp.coo_matrix((data, (rr, cc)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.eliminate_zeros()
    A.sum_duplicates()
    A.data[:] = 1.0
    labels = np.asarray(labels_list, dtype=np.int8)
    return A, labels, address_to_id


def build_features(A: sp.csr_matrix, labels: np.ndarray, seeds: np.ndarray,
                    seed: int, betweenness_k: int, cluster_pairs: int,
                    jaccard_edges: int):
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    seed_vec = np.zeros(n, dtype=np.float64)
    seed_vec[seeds] = 1.0 / len(seeds)

    print("[3/8] Computing sparse PPR trust features...")
    cols = []
    for alpha in (0.5, 0.8, 0.95):
        pi = ppr_vector(A, deg, seed_vec, alpha, iters=200)
        cols.append(rank01(np.log(pi / np.maximum(deg, 1.0) + 1e-12)))
        print(f"      PPR alpha={alpha} done")

    steps = int(math.ceil(math.log2(max(n, 2))))
    sw = early_walk(A, deg, seed_vec, steps) / np.maximum(deg, 1.0)
    cols.append(rank01(np.log(sw + 1e-12)))
    print(f"      SybilRank-style walk done ({steps} steps)")

    print("[4/8] Computing structural features...")
    cols.append(rank01(deg))

    rng = np.random.RandomState(seed)
    clustering = sampled_clustering(A, rng, max_pairs_per_node=cluster_pairs)
    cols.append(rank01(clustering))
    print("      clustering estimate done")

    core = core_numbers(A)
    cols.append(rank01(core))
    print("      exact k-core number done")

    overlap = sampled_neighbor_jaccard(A, rng, max_edges_per_node=jaccard_edges)
    cols.append(rank01(overlap))
    print("      neighbor-Jaccard estimate done")

    btw = sampled_betweenness(A, rng, k=betweenness_k)
    cols.append(rank01(btw))
    print(f"      sampled betweenness done (k={betweenness_k})")

    print("[5/8] Computing seed-disagreement feature...")
    seed_list = np.asarray(seeds, dtype=np.int64).copy()
    rng.shuffle(seed_list)
    if len(seed_list) >= 2:
        mid = max(1, len(seed_list) // 2)
        subsets = [seed_list[:mid], seed_list[mid:]]
        subsets = [s for s in subsets if len(s)]
        pprs = []
        for subset in subsets:
            s = np.zeros(n, dtype=np.float64)
            s[subset] = 1.0 / len(subset)
            pprs.append(ppr_vector(A, deg, s, 0.85, iters=200))
        variance = np.var(np.vstack(pprs), axis=0) if len(pprs) >= 2 else np.zeros(n)
    else:
        variance = np.zeros(n, dtype=np.float64)
    cols.append(rank01(variance))

    cols.append(rank01(neighbor_degree_ratio(A, deg)))

    dist = multi_source_distance(A, seeds)
    cols.append(rank01(dist))

    X = np.column_stack(cols).astype(np.float64)
    if X.shape[1] != 12:
        raise RuntimeError(f"Expected 12 features, got {X.shape[1]}")
    return X


def sparse_gcn_predict(model, A: sp.csr_matrix, X: np.ndarray,
                       batch_size: int = 100_000) -> np.ndarray:
    """Exact GCN forward equation with sparse normalized adjacency."""
    n = A.shape[0]
    I = sp.eye(n, dtype=np.float64, format="csr")
    A_plus = A + I
    d = np.asarray(A_plus.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    # D^-1/2 A D^-1/2, implemented by scaling CSR rows then columns.
    A_hat = A_plus.multiply(inv_sqrt[:, None]).tocsr()
    A_hat = A_hat.multiply(inv_sqrt[None, :]).tocsr()

    H = X
    for l in range(model.n_layers):
        AH = A_hat @ H
        Z = AH @ model.W[l] + model.b[l]
        if l < model.n_layers - 1:
            H = np.maximum(Z, 0.0)
        else:
            Z = Z - Z.max(axis=1, keepdims=True)
            E = np.exp(Z)
            H = E / E.sum(axis=1, keepdims=True)
    return H


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(r"C:\Users\saman\OneDrive\Desktop"),
        help="Directory containing wallets_classes.txt and AddrAddr_edgelist.txt",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed-frac", type=float, default=0.05)
    parser.add_argument("--betweenness-k", type=int, default=30)
    parser.add_argument("--cluster-pairs", type=int, default=64)
    parser.add_argument("--jaccard-edges", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("elliptic_transfer_predictions.csv"))
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("SYBIL-SHIELD — Elliptic++ Actors / frozen transfer_half_fb")
    print("NO RETRAINING — sparse inference/evaluation only")
    print("=" * 72)

    A, labels, address_to_id = load_graph(args.data_dir)
    n = A.shape[0]
    m = A.nnz // 2
    print(f"      graph: {n:,} nodes, {m:,} undirected edges")
    print(f"      labels: illicit={np.sum(labels==1):,}, licit={np.sum(labels==2):,}, unknown={np.sum(labels==3):,}")

    seeds = choose_licit_seeds(labels, args.seed_frac, args.seed)
    print(f"      trusted licit seeds: {len(seeds):,} ({args.seed_frac:.1%})")

    print(f"[6/8] Loading frozen model: {MODEL_TYPE}/{MODEL_ID}")
    meta = load_meta(MODEL_TYPE, MODEL_ID)
    model = load_model(MODEL_TYPE, MODEL_ID)
    print(f"      architecture: {meta.get('architecture')}")
    print(f"      saved synthetic AUC: {meta.get('metrics', {}).get('auc')}")
    print("      IMPORTANT: saved AUC is synthetic holdout metadata; it is NOT this crypto result.")

    X = build_features(A, labels, seeds, args.seed, args.betweenness_k,
                       args.cluster_pairs, args.jaccard_edges)
    print(f"      feature matrix: {X.shape}, dtype={X.dtype}")

    print("[7/8] Running sparse GCN inference...")
    probs = sparse_gcn_predict(model, A, X)
    sybil_score = probs[:, 1]
    pred = (sybil_score >= 0.5).astype(np.int8)

    # Evaluate only known labels: class 1 = illicit, class 2 = licit.
    eval_mask = ((labels == 1) | (labels == 2))
    eval_mask[seeds] = False
    y_true = (labels[eval_mask] == 1).astype(np.int8)
    y_pred = pred[eval_mask]
    scores = sybil_score[eval_mask]

    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("[8/8] Evaluation")
    print("-" * 72)
    print(f"Evaluated known wallets (excluding seeds): {eval_mask.sum():,}")
    print(f"  illicit: {np.sum(y_true==1):,}")
    print(f"  licit:   {np.sum(y_true==0):,}")
    print()
    print(f"AUC:       {auc:.6f}")
    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1:        {f1:.6f}")
    print()
    print("Confusion matrix [rows=true licit/illicit, cols=pred licit/illicit]")
    print(cm)
    print(f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print()
    print(f"Elapsed: {(time.time()-t0)/60:.2f} minutes")

    print(f"Saving predictions: {args.output.resolve()}")
    seed_mask = np.zeros(n, dtype=np.bool_)
    seed_mask[seeds] = True
    # Stream CSV so we don't create another giant in-memory object.
    id_to_address = [None] * len(address_to_id)
    for addr, idx in address_to_id.items():
        id_to_address[idx] = addr
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["address", "label", "sybil_probability", "prediction", "is_trusted_seed"])
        for i in range(n):
            w.writerow([
                id_to_address[i],
                int(labels[i]),
                float(sybil_score[i]),
                int(pred[i]),
                int(seed_mask[i]),
            ])

    print("Done.")


if __name__ == "__main__":
    main()
