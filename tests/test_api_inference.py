import networkx as nx
<<<<<<< HEAD

from bad_people_finder.api.app import predict_graph
=======
from fastapi.testclient import TestClient

from sybil_shield.api.app import app, get_model_names, predict_graph
>>>>>>> f345065 (Initial project commit)


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
<<<<<<< HEAD
=======


def test_api_uses_real_facebook_variant_names_without_gradient_or_trusted_legacy_variant():
    names = get_model_names()

    assert names == ["standard", "random", "mixed"]
    assert "gradient" not in names
    assert "trusted" not in names


def test_facebooktest_returns_graph_summary_and_suspicious_subgraph():
    client = TestClient(app)
    response = client.post("/facebooktest", json={"keyword": "go"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["triggered"] is True
    assert "graph" in payload
    assert isinstance(payload["graph"], dict)
    assert "nodes" in payload["graph"]
    assert "edges" in payload["graph"]
    assert "smell" in payload
    assert isinstance(payload["smell"], dict)
    assert "suspicious_nodes" in payload["smell"]
    assert isinstance(payload["smell"]["suspicious_nodes"], list)
>>>>>>> f345065 (Initial project commit)
