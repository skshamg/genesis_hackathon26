import random
from typing import Optional, Sequence

import networkx as nx
import numpy as np

from sybil_shield.core.graph_store import save_graph


def generate_crypto_graph(
    seed_wallet: str = "0xroot",
    n_nodes: int = 200,
    n_sybil: Optional[int] = None,
    branching: int = 3,
    max_depth: int = 4,
    seed: int = 0,
    sybil_attach_rate: float = 0.4,
    graph_type: str = "crypto_wallet",
    persist: bool = False,
    root_dir: Optional[str] = None,
) -> nx.Graph:
    """Generate a synthetic wallet graph with an honest core and a dense sybil cluster.

    The graph grows outward from a root wallet by breadth-first expansion. Honest nodes
    are labeled 0 and sybil nodes are labeled 1; edges are stored as a simple undirected
    NetworkX graph.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node("root", label=0, node_type="wallet", role="root")
    G.add_node(seed_wallet, label=0, node_type="wallet", role="seed_wallet")
    G.add_edge("root", seed_wallet)
    honest_target = max(5, int(n_nodes) - (n_sybil if n_sybil is not None else max(5, int(n_nodes * 0.2))))
    honest_target = min(honest_target, max(5, n_nodes))

    if n_sybil is None:
        n_sybil = max(5, int(n_nodes * 0.2))

    frontier = [seed_wallet]
    current_depth = {seed_wallet: 0}
    new_honest_ids = [seed_wallet]

    while G.number_of_nodes() < honest_target and frontier:
        next_frontier = []
        for parent in frontier:
            for _ in range(int(rng.integers(1, max(2, branching + 1)))):
                if G.number_of_nodes() >= honest_target:
                    break
                child = f"{parent}/wallet_{G.number_of_nodes():04d}"
                G.add_node(child, label=0, node_type="wallet", role="honest", depth=current_depth[parent] + 1)
                G.add_edge(parent, child)
                next_frontier.append(child)
                current_depth[child] = current_depth[parent] + 1
        frontier = next_frontier

    # Fill up to target honest nodes if needed by attaching extra wallet nodes to the root.
    while G.number_of_nodes() < honest_target:
        child = f"wallet_{G.number_of_nodes():04d}"
        G.add_node(child, label=0, node_type="wallet", role="honest")
        G.add_edge(seed_wallet, child)

    if "root" not in G:
        G.add_node("root", label=0, node_type="wallet", role="root")
        G.add_edge("root", seed_wallet)

    honest_nodes = [n for n, d in G.nodes(data=True) if d.get("label") == 0]
    if not honest_nodes:
        raise ValueError("No honest nodes were created")

    sybil_nodes = []
    if n_sybil > 0:
        for i in range(int(n_sybil)):
            node_id = f"sybil_{i:04d}"
            G.add_node(node_id, label=1, node_type="wallet", role="sybil")
            sybil_nodes.append(node_id)

        for i in range(1, len(sybil_nodes)):
            G.add_edge(sybil_nodes[i - 1], sybil_nodes[i])

        for sybil in sybil_nodes:
            if honest_nodes and rng.random() < sybil_attach_rate:
                attach_to = rng.choice(honest_nodes)
                G.add_edge(sybil, attach_to)
            else:
                G.add_edge(sybil, seed_wallet)

    # Ensure the root is connected to the main honest region.
    if G.number_of_nodes() > 1 and seed_wallet not in G:
        G.add_node(seed_wallet, label=0, node_type="wallet", role="root")

    if persist:
        save_graph(
            G,
            graph_type=graph_type,
            params={
                "seed_wallet": seed_wallet,
                "n_nodes": n_nodes,
                "n_sybil": n_sybil,
                "branching": branching,
                "max_depth": max_depth,
                "seed": seed,
                "sybil_attach_rate": sybil_attach_rate,
            },
            root_dir=root_dir,
        )

    return G


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate and persist a synthetic crypto wallet graph.")
    parser.add_argument("--seed-wallet", default="0xroot")
    parser.add_argument("--n-nodes", type=int, default=200)
    parser.add_argument("--n-sybil", type=int, default=None)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sybil-attach-rate", type=float, default=0.4)
    parser.add_argument("--graph-type", default="crypto_wallet")
    parser.add_argument("--root-dir", default=None, help="Optional override for the graph registry root")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    G = generate_crypto_graph(
        seed_wallet=args.seed_wallet,
        n_nodes=args.n_nodes,
        n_sybil=args.n_sybil,
        branching=args.branching,
        max_depth=args.max_depth,
        seed=args.seed,
        sybil_attach_rate=args.sybil_attach_rate,
        graph_type=args.graph_type,
        persist=not args.no_persist,
        root_dir=args.root_dir,
    )
    print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}"
          f" honest={sum(1 for _, d in G.nodes(data=True) if d.get('label') == 0)}"
          f" sybil={sum(1 for _, d in G.nodes(data=True) if d.get('label') == 1)}")


if __name__ == "__main__":
    main()
