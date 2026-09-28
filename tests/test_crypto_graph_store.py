import json
from pathlib import Path

import networkx as nx

from sybil_shield.core.graph_store import list_graphs, load_graph, save_graph
from sybil_shield.experiments.crypto_graph_builder import generate_crypto_graph


def test_generate_and_store_crypto_graph(tmp_path):
    G = generate_crypto_graph(
        seed_wallet="0xroot",
        n_nodes=30,
        n_sybil=8,
        branching=2,
        max_depth=2,
        seed=7,
    )

    assert G.number_of_nodes() >= 10
    assert G.number_of_edges() >= 1
    assert "root" in G.nodes
    assert "wallet" in nx.get_node_attributes(G, "node_type").values()

    meta = save_graph(
        G,
        graph_type="crypto_wallet",
        params={"n_nodes": 30, "n_sybil": 8},
        root_dir=str(tmp_path),
    )

    assert meta["graph_id"]
    assert (tmp_path / "crypto_wallet" / meta["graph_id"] / "graph.gpickle").exists()
    assert (tmp_path / "crypto_wallet" / meta["graph_id"] / "meta.json").exists()

    loaded = load_graph("crypto_wallet", meta["graph_id"], root_dir=str(tmp_path))
    assert loaded.number_of_nodes() == G.number_of_nodes()
    assert loaded.number_of_edges() == G.number_of_edges()

    graphs = list_graphs(graph_type="crypto_wallet", root_dir=str(tmp_path))
    assert any(item["graph_id"] == meta["graph_id"] for item in graphs)
