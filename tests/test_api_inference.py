import importlib
import queue

import networkx as nx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sybil_shield.api.app import app, _initial_train_state, get_model_names, predict_graph
from sybil_shield.experiments import real_facebook_eval as facebook_eval

app_module = importlib.import_module("sybil_shield.api.app")


@pytest.fixture(autouse=True)
def _fast_gradient_model(monkeypatch):
    """
    predict_graph()'s generic path now calls the SAME cached, zero-shot
    gradient-adversarial model as the Facebook path (that's the point of
    the consolidation -- one model, trained once, instead of 4 trained
    fresh per request). Left un-patched, the first test in the suite to
    touch predict_graph would pay the real ~9-10 min training cost. Swap
    in a few-epoch model on a tiny synthetic set and clear the lru_cache
    so tests exercise the real code path at toy scale.
    """
    from sybil_shield.experiments.eval_robust import make_train

    def fast_train_all_models(seed=1, betweenness_k=30, model_id=None, model_type=None):
        if model_id is not None:
            from sybil_shield.core import model_store
            resolved = model_store.resolve_model_id(model_type, model_id)
            return {"gradient": model_store.load_model(model_type, resolved)}
        graphs = make_train(3, seed=seed)
        model = facebook_eval.train_gradient_adversarial(
            graphs, n_epochs=4, betweenness_k=betweenness_k, verbose=False
        )
        return {"gradient": model}

    monkeypatch.setattr(facebook_eval, "train_all_models", fast_train_all_models)
    app_module._load_facebook_eval_models.cache_clear()
    yield
    app_module._load_facebook_eval_models.cache_clear()


def test_predict_graph_returns_gradient_model_output():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)])
    for node in G.nodes:
        G.nodes[node]["label"] = 0

    result = predict_graph(G)

    assert set(result.keys()) == {"gradient"}
    for model_name, scores in result.items():
        assert isinstance(scores, list)
        assert len(scores) == len(G.nodes)
        assert all(isinstance(v, float) for v in scores)


def test_api_uses_single_gradient_adversarial_variant():
    names = get_model_names()

    assert names == ["gradient"]
    assert "standard" not in names
    assert "trusted" not in names


def test_initial_train_state_has_empty_graph_payload():
    state = _initial_train_state(num_graphs=50, epochs=12)

    assert state["status"] == "running"
    assert state["current_graph_index"] == 0
    assert state["total_graphs"] == 50
    assert state["total_epochs"] == 12
    assert state["current_graph"] == {"nodes": [], "edges": []}
    assert state["loss"] is None
    assert state["honest_count"] is None
    assert state["sybil_count"] is None


def test_train_websocket_streams_progress_messages():
    job_id = "ws_test_job"
    app_module._train_jobs[job_id] = {"status": "running", "current_graph_index": 0, "total_graphs": 1, "epoch": 1, "total_epochs": 2, "loss": 0.95, "honest_count": 10, "sybil_count": 5, "current_graph": {"nodes": [], "edges": []}}
    app_module._train_events[job_id] = queue.Queue()
    app_module._train_events[job_id].put({"status": "running", "current_graph_index": 0, "total_graphs": 1, "epoch": 1, "total_epochs": 2, "loss": 0.95, "honest_count": 10, "sybil_count": 5, "current_graph": {"nodes": [], "edges": []}})
    try:
        with TestClient(app) as client:
            with client.websocket_connect(f"/ws/train/{job_id}") as websocket:
                message = websocket.receive_json()
                assert message["status"] == "running"
                assert message["epoch"] == 1
                assert message["loss"] == 0.95
    finally:
        app_module._train_jobs.pop(job_id, None)
        app_module._train_events.pop(job_id, None)


def test_facebooktest_returns_graph_summary_and_suspicious_subgraph():
    client = TestClient(app)
    response = client.post("/facebooktest", json={"keyword": "go"})

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "triggered" in payload
    assert "graph" in payload
    assert isinstance(payload["graph"], dict)
    assert "nodes" in payload["graph"]
    assert "edges" in payload["graph"]
    assert "smell" in payload
    assert isinstance(payload["smell"], dict)
    assert "suspicious_nodes" in payload["smell"]
    assert isinstance(payload["smell"]["suspicious_nodes"], list)
    assert "metrics" in payload
    assert "confusion" in payload
    assert set(payload["confusion"].keys()) == {
        "correctly_flagged_sybil",
        "incorrectly_flagged_sybil",
        "correctly_flagged_safe",
        "incorrectly_flagged_safe",
    }


def test_scan_request_accepts_api_key_and_model_id():
    from sybil_shield.api.app import ScanRequest

    req = ScanRequest(
        address="0x1234567890abcdef",
        model_type="gradient_adversarial",
        model_id="demo-model",
        api_key="etherscan-key-123",
    )

    assert req.model_id == "demo-model"
    assert req.api_key == "etherscan-key-123"


def test_model_info_includes_training_graph_catalog_when_available():
    from sybil_shield.api.app import model_info_api
    from sybil_shield.core import model_store

    meta = {
        "trained_at": "2026-01-01T00:00:00Z",
        "metrics": {},
        "training_graphs": [
            {"graph_id": "g-1", "label": "Graph 1", "nodes": 12, "edges": 20},
            {"graph_id": "g-2", "label": "Graph 2", "nodes": 14, "edges": 26},
        ],
    }
    model = type("DummyModel", (), {"W": [None], "b": [None], "n_layers": 1})()
    model.W[0] = np.array([[1.0]], dtype=float)
    model.b[0] = np.array([0.0], dtype=float)

    orig_load_meta = model_store.load_meta
    orig_load_model = model_store.load_model
    try:
        model_store.load_meta = lambda model_type, model_id: meta
        model_store.load_model = lambda model_type, model_id: model
        payload = model_info_api(model_type="gradient_adversarial", model_id="latest")
        assert payload["training_graphs"][0]["label"] == "Graph 1"
    finally:
        model_store.load_meta = orig_load_meta
        model_store.load_model = orig_load_model
