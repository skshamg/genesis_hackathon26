import networkx as nx

from bad_people_finder.api.app import predict_graph


def test_predict_graph_returns_four_model_outputs():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
    for node in G.nodes:
        G.nodes[node]["label"] = 0

    result = predict_graph(G)

    assert set(result.keys()) == {"standard", "random", "trusted", "mixed"}
    for model_name, scores in result.items():
        assert isinstance(scores, list)
        assert len(scores) == len(G.nodes)
        assert all(isinstance(v, float) for v in scores)
