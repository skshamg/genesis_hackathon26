"""
crypto_eval.py

Zero-shot evaluation of a trained gradient-adversarial GCN (see
core.model_store) on a real, live-fetched crypto wallet graph.

Unlike real_facebook_eval.py, there are NO ground-truth labels here at
all -- Ethereum doesn't ship a train.txt/test.txt. Two consequences:

  1. Trusted seeds for trust-propagation features come from
     seed_bootstrap.bootstrap_seeds_by_community_with_tiebreak() instead
     of a benign_train file -- see that module's docstring for the
     purity/AUC tradeoffs of that approach, and pass --tiny-verified if
     you happen to know a couple of addresses in the graph are legit
     (fixes the documented near-50/50 tie failure mode).

  2. There is no AUC/accuracy to report. "Success" here means the
     flagged wallets look structurally sane (shared funder, tight
     activity window, dense mutual edges) -- a qualitative check, not
     a metric. Don't expect a metrics table like real_facebook_eval.py's.

A real live-fetched graph can include very high-degree infrastructure
nodes (exchange hot wallets, DEX routers) that Louvain can merge the
honest community into (see seed_bootstrap.py's docstring on when the
largest-community assumption breaks). Nodes above --max-degree are
therefore excluded from build_features()'s node list entirely --
they're still part of the persisted graph, but neither scored nor
seeded from, and (by construction of facebook_features'/build_features'
adjacency-from-nodelist) edges through them stop linking otherwise-
separate regions of the graph for the GCN's message passing. That's
intentional here, not a side effect to work around: a sybil farm
connected to the honest region only via a shared exchange hop should
still look like a separate, sparsely-attached region, not get merged
into "everything's one hop from Binance."

Outputs, since there's no frontend yet:
  1. Console table -- top-N highest-risk wallets, same style as
     real_facebook_eval.py's model table.
  2. results/crypto_eval/<graph_id>__<model_id>.json -- full per-node
     scores + graph edges + seed-bootstrap diagnostics, so a frontend
     can be pointed at this file later without re-running anything.
  3. results/crypto_eval/<graph_id>__<model_id>.png -- spring-layout
     graph colored by risk score + a score histogram, matplotlib Agg,
     same convention as experiments/visualize.py.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from sybil_shield.core import graph_store, model_store
from sybil_shield.core.seed_bootstrap import (
    bootstrap_seeds_by_community_with_tiebreak,
    cross_check_seeds,
)
from sybil_shield.experiments.real_facebook_eval import facebook_features
from sybil_shield.experiments.live_crypto_graph_builder import build_live_wallet_graph

DEFAULT_MODEL_TYPE = "gradient_adversarial"


def _results_dir() -> Path:
    # this file is src/sybil_shield/experiments/crypto_eval.py -> repo
    # root is four parents up, matching model_store.py's _models_root().
    d = Path(__file__).resolve().parents[4] / "results" / "crypto_eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _scoreable_nodes(graph: nx.Graph, max_degree: int) -> tuple[List[Any], List[Any]]:
    """Split nodes into (scoreable, excluded_high_degree)."""
    scoreable, excluded = [], []
    for n in graph.nodes():
        (scoreable if graph.degree(n) <= max_degree else excluded).append(n)
    return scoreable, excluded


def evaluate_crypto_graph(
    graph: nx.Graph,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_id: str = "latest",
    max_degree: int = 500,
    tiny_verified: Optional[List[str]] = None,
    top_n: int = 20,
) -> Dict[str, Any]:
    scoreable_nodes, excluded = _scoreable_nodes(graph, max_degree)
    if excluded:
        preview = excluded[:5]
        print(
            f"Excluding {len(excluded)} high-degree (>{max_degree}) infrastructure "
            f"node(s) from scoring/seeding: {preview}{'...' if len(excluded) > 5 else ''}"
        )

    if len(scoreable_nodes) < 10:
        raise ValueError(
            f"Only {len(scoreable_nodes)} scoreable nodes after excluding high-degree "
            "nodes -- graph too small/sparse for meaningful community detection. "
            "Try a larger --max-nodes crawl or a higher --max-degree cutoff."
        )

    print(
        f"Bootstrapping trusted seeds from {len(scoreable_nodes)} scoreable nodes "
        "(no ground-truth labels available for this dataset)..."
    )
    seeds, community_nodes = bootstrap_seeds_by_community_with_tiebreak(
        graph, nodes=scoreable_nodes, tiny_verified=tiny_verified, seed=0,
    )
    print(
        f"  inferred trusted community size: {len(community_nodes)} / {len(scoreable_nodes)} "
        f"({100 * len(community_nodes) / len(scoreable_nodes):.1f}%)"
    )
    print(f"  seeds used for trust propagation: {len(seeds)}")

    purity_proxy = None
    if tiny_verified:
        purity_proxy = cross_check_seeds(graph, tiny_verified, nodes=scoreable_nodes, seed=0)
        print(f"  tiny_verified agreement with inferred community: {purity_proxy:.2f}")
    else:
        print(
            "  no --tiny-verified addresses given -- if the honest/sybil split in this "
            "graph turns out close to 50/50, seed purity can degrade badly (see "
            "seed_bootstrap.py's docstring); pass a couple of known-legit addresses "
            "to guard against that."
        )

    print("Building trust+structural features from bootstrapped seeds...")
    X, A_hat = facebook_features(graph, scoreable_nodes, seeds)

    print(f"Loading model '{model_id}' (type={model_type}) from registry...")
    resolved_id = model_store.resolve_model_id(model_type, model_id)
    model = model_store.load_model(model_type, resolved_id)
    meta = model_store.load_meta(model_type, resolved_id)

    probs = model.predict_proba(A_hat, X)[:, 1]
    scored = sorted(zip(scoreable_nodes, probs), key=lambda kv: kv[1], reverse=True)

    print(f"\n{'rank':>4s}  {'wallet':<44s}  {'risk_score':>10s}")
    print("-" * 62)
    for rank, (node, score) in enumerate(scored[:top_n], start=1):
        print(f"{rank:>4d}  {str(node):<44s}  {score:>10.4f}")

    return {
        "graph": {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "scoreable_node_count": len(scoreable_nodes),
            "excluded_high_degree_nodes": [str(n) for n in excluded],
            "edges": [[str(u), str(v)] for u, v in graph.edges()],
        },
        "seed_bootstrap": {
            "community_size": len(community_nodes),
            "seed_count": len(seeds),
            "tiny_verified": tiny_verified or [],
            "tiny_verified_purity_proxy": purity_proxy,
        },
        "model": {
            "model_type": model_type,
            "model_id": resolved_id,
            "trained_at": meta.get("trained_at"),
        },
        "scores": {str(node): float(score) for node, score in zip(scoreable_nodes, probs)},
        "top_flagged": [
            {"wallet": str(n), "risk_score": float(s)} for n, s in scored[:top_n]
        ],
    }


def save_visualization(graph: nx.Graph, results: Dict[str, Any], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = results["scores"]
    scoreable = list(scores.keys())
    node_list = [n for n in graph.nodes() if str(n) in scores]
    color = [scores[str(n)] for n in node_list]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pos = nx.spring_layout(graph, seed=0)
    nodes_drawn = nx.draw_networkx_nodes(
        graph, pos, ax=axes[0], nodelist=node_list, node_color=color,
        cmap="RdYlGn_r", node_size=40, vmin=0, vmax=1,
    )
    nx.draw_networkx_edges(graph, pos, ax=axes[0], alpha=0.15, width=0.5)
    fig.colorbar(nodes_drawn, ax=axes[0], label="P(sybil)")
    axes[0].set_title(f"Wallet graph -- colored by risk score ({len(scoreable)} scored)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].hist(list(scores.values()), bins=30, color="#EF4444", alpha=0.75)
    axes[1].set_title("Risk score distribution")
    axes[1].set_xlabel("P(sybil)")
    axes[1].set_ylabel("wallet count")

    fig.suptitle(f"Crypto Sybil scan -- model {results['model']['model_id']}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved visualization to {out_path}")


def main(
    seed_wallet: Optional[str] = None,
    graph_type: str = "ethereum_wallet",
    graph_id: Optional[str] = None,
    max_nodes: int = 500,
    max_depth: int = 2,
    max_degree: int = 500,
    model_type: str = DEFAULT_MODEL_TYPE,
    model_id: str = "latest",
    top_n: int = 20,
    tiny_verified: Optional[List[str]] = None,
    root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if graph_id:
        print(f"Loading persisted graph {graph_type}/{graph_id}...")
        graph = graph_store.load_graph(graph_type, graph_id, root_dir=root_dir)
        resolved_graph_id = graph_id
    else:
        print("Fetching live wallet graph from Etherscan...")
        graph = build_live_wallet_graph(
            seed_wallet=seed_wallet, max_nodes=max_nodes, max_depth=max_depth,
            graph_type=graph_type, persist=True, root_dir=root_dir,
        )
        resolved_graph_id = graph_store.list_graphs(graph_type, root_dir=root_dir)[0]["graph_id"]

    print(f"Graph ready: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

    t0 = time.time()
    results = evaluate_crypto_graph(
        graph, model_type=model_type, model_id=model_id,
        max_degree=max_degree, tiny_verified=tiny_verified, top_n=top_n,
    )
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    out_dir = _results_dir()
    stem = f"{resolved_graph_id}__{results['model']['model_id']}"
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {json_path}")

    save_visualization(graph, results, out_dir / f"{stem}.png")
    return results


def _cli():
    parser = argparse.ArgumentParser(
        description="Zero-shot Sybil scan of a real crypto wallet graph."
    )
    parser.add_argument("--seed-wallet", default=None,
                         help="Ethereum address to crawl from; omit for a random "
                              "recently-active wallet")
    parser.add_argument("--graph-type", default="ethereum_wallet")
    parser.add_argument("--graph-id", default=None,
                         help="Load a previously-persisted graph instead of fetching a new one")
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-degree", type=int, default=500,
                         help="Exclude nodes above this degree from scoring/seeding")
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--model-id", default="latest")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--tiny-verified", default=None,
                         help="Comma-separated addresses known to be legitimate, used to "
                              "break seed-bootstrap ties")
    parser.add_argument("--root-dir", default=None)
    args = parser.parse_args()

    tiny_verified = (
        [a.strip().lower() for a in args.tiny_verified.split(",")]
        if args.tiny_verified else None
    )

    main(
        seed_wallet=args.seed_wallet, graph_type=args.graph_type, graph_id=args.graph_id,
        max_nodes=args.max_nodes, max_depth=args.max_depth, max_degree=args.max_degree,
        model_type=args.model_type, model_id=args.model_id, top_n=args.top_n,
        tiny_verified=tiny_verified, root_dir=args.root_dir,
    )


if __name__ == "__main__":
    _cli()
