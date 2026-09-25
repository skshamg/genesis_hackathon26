from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

import networkx as nx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from bad_people_finder.core.detector import build_features
from bad_people_finder.core.gcn import GCN


class GraphRequest(BaseModel):
    nodes: Optional[List[Union[str, int]]] = None
    edges: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None
    labels: Optional[Dict[Union[str, int], int]] = None


app = FastAPI(title="Bad People Finder API", version="1.0.0")


def _safe_label_value(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _make_graph(payload: Union[nx.Graph, GraphRequest, Dict[str, Any], Sequence[Tuple[Any, Any]]]) -> nx.Graph:
    if isinstance(payload, nx.Graph):
        return payload.copy()

    if isinstance(payload, GraphRequest):
        G = nx.Graph()
        nodes = payload.nodes or []
        edges = payload.edges or []
        labels = payload.labels or {}
        if nodes:
            G.add_nodes_from(nodes)
        for u, v in edges:
            G.add_edge(u, v)
        for node, label in labels.items():
            if node in G:
                G.nodes[node]["label"] = _safe_label_value(label)
        for node in G.nodes:
            G.nodes[node].setdefault("label", 0)
        return G

    if isinstance(payload, dict):
        nodes = payload.get("nodes")
        edges = payload.get("edges", [])
        labels = payload.get("labels", {})
        return _make_graph(GraphRequest(nodes=nodes, edges=edges, labels=labels))

    G = nx.Graph()
    G.add_edges_from(payload)
    for node in G.nodes:
        G.nodes[node].setdefault("label", 0)
    return G


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2)
    counts = np.maximum(counts, 1)
    return np.median(counts) / counts


def _train_model_on_graph(graph: nx.Graph, seed: int = 0, epochs: int = 20) -> GCN:
    X, y, A_hat, nodes, _ = build_features(graph, seed=seed)
    if X.size == 0:
        raise ValueError("Graph feature matrix is empty.")
    model = GCN(n_features=X.shape[1], hidden_dims=(16, 16), n_classes=2, seed=seed)
    weights = _class_weights(y)
    for _ in range(epochs):
        model.fit_step(A_hat, X, y, class_weight=weights, lr=0.02)
    return model


def predict_graph(G: Union[nx.Graph, GraphRequest, Dict[str, Any], Sequence[Tuple[Any, Any]]]) -> Dict[str, List[float]]:
    graph = _make_graph(G)
    if len(graph.nodes) == 0:
        return {"standard": [], "random": [], "trusted": [], "mixed": []}

    models = {
        "standard": _train_model_on_graph(graph, seed=0),
        "random": _train_model_on_graph(graph, seed=101),
        "trusted": _train_model_on_graph(graph, seed=202),
        "mixed": _train_model_on_graph(graph, seed=303),
    }

    results: Dict[str, List[float]] = {}
    for name, model in models.items():
        X, y, A_hat, nodes, _ = build_features(graph, seed=0)
        probabilities = model.predict_proba(A_hat, X)[:, 1]
        results[name] = [float(value) for value in probabilities]
    return results


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_api(payload: GraphRequest) -> Dict[str, List[float]]:
    return predict_graph(payload)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Bad People Finder API is running."}
