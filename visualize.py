"""
visualize.py

Produces the demo visuals:
  1. PCA scatter of GCN embeddings (honest vs sybil) for a held-out graph
     and a camouflaged graph, side by side -- the "clear visual separation"
     the brief asks for.
  2. Score-separation histograms (GCN sybil-probability vs PPR trust score)
  3. AUC comparison bar chart (GCN vs PPR) across all held-out test sets

Run after train_eval.main(); saves a single PNG to results.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from train_eval import main as run_train_eval


def plot_embedding(ax, result, title):
    emb = result["embedding"]
    y = result["y"]
    coords = PCA(n_components=2, random_state=0).fit_transform(emb)
    ax.scatter(coords[y == 0, 0], coords[y == 0, 1], s=12, alpha=0.6,
               color="#3B82F6", label="honest")
    ax.scatter(coords[y == 1, 0], coords[y == 1, 1], s=12, alpha=0.7,
               color="#EF4444", label="sybil")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, loc="best")


def plot_score_hist(ax, result, score_key, title, xlabel):
    y = result["y"]
    score = result[score_key]
    ax.hist(score[y == 0], bins=30, alpha=0.6, color="#3B82F6", label="honest")
    ax.hist(score[y == 1], bins=30, alpha=0.6, color="#EF4444", label="sybil")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.legend(fontsize=8)


def plot_auc_bars(ax, results):
    labels = [r["label"] for r in results]
    gcn_aucs = [r["gcn_auc"] for r in results]
    ppr_aucs = [r["ppr_auc"] for r in results]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, gcn_aucs, w, label="GCN", color="#10B981")
    ax.bar(x + w / 2, ppr_aucs, w, label="PPR baseline", color="#94A3B8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("GCN vs PPR baseline (AUC by test set)", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")


def main():
    model, results = run_train_eval()

    # pick one held-out-easy and one camouflaged result for the embedding plots
    easy_result = next(r for r in results if "easy" in r["label"])
    camo_result = next(r for r in results if "camouflaged" in r["label"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    plot_embedding(axes[0, 0], easy_result, "GCN embedding (PCA) — held-out easy")
    plot_embedding(axes[0, 1], camo_result, "GCN embedding (PCA) — camouflaged")
    plot_auc_bars(axes[0, 2], results)

    plot_score_hist(axes[1, 0], easy_result, "gcn_score",
                     "Score separation — easy (GCN)", "P(sybil)")
    plot_score_hist(axes[1, 1], camo_result, "gcn_score",
                     "Score separation — camouflaged (GCN)", "P(sybil)")
    plot_score_hist(axes[1, 2], camo_result, "ppr_score",
                     "Score separation — camouflaged (PPR)", "PPR trust score")

    fig.suptitle("Sybil detection: GCN vs Personalized PageRank baseline", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("/mnt/user-data/outputs/results.png", dpi=150)
    print("\nSaved visualization to results.png")


if __name__ == "__main__":
    main()
