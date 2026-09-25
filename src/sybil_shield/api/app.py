from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

import networkx as nx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sybil_shield.core.detector import build_features
from sybil_shield.core.gcn import GCN


BENCHMARK_MODEL_VARIANTS: List[str] = ["standard", "random", "mixed"]
LEGACY_MODEL_VARIANTS: List[str] = ["standard", "random", "trusted", "mixed"]


def get_model_names() -> List[str]:
    return list(BENCHMARK_MODEL_VARIANTS)


class GraphRequest(BaseModel):
    nodes: Optional[List[Union[str, int]]] = None
    edges: Optional[List[Tuple[Union[str, int], Union[str, int]]]] = None
    labels: Optional[Dict[Union[str, int], int]] = None
    dataset: Optional[str] = None
    source: Optional[str] = None
    mode: Optional[str] = None
    use_facebook: bool = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sybil_shield.api")

app = FastAPI(title="Bad People Finder API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("HTTP request received: %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("HTTP request failed: %s %s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("HTTP request complete: %s %s -> %s in %.2f ms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


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


def _resolve_facebook_dataset_dir(dataset_dir: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []

    if dataset_dir:
        candidates.append(dataset_dir)

    env_dir = os.environ.get("SYBIL_SHIELD_FACEBOOK_DIR")
    if env_dir:
        candidates.append(env_dir)

    cwd = os.getcwd()
    for base in [cwd, os.path.dirname(cwd), os.path.join(cwd, "data")]:
        if not base:
            continue
        candidates.extend(
            [
                os.path.join(base, "Undirected_Facebook"),
                os.path.join(base, "Undirected_Facebook", "Undirected_Facebook"),
                os.path.join(base, "data", "Undirected_Facebook"),
            ]
        )

    candidates.extend(
        [
            r"C:\Users\saman\OneDrive\Desktop\Undirected_Facebook\Undirected_Facebook",
            r"C:\Users\saman\OneDrive\Desktop\SYBIL-SHIELD_restructured\SYBIL-SHIELD\data\Undirected_Facebook",
        ]
    )

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        norm = os.path.abspath(os.path.normpath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        graph_file = os.path.join(norm, "graph.txt")
        if os.path.exists(graph_file):
            return norm
    return None


@lru_cache(maxsize=4)
def _load_facebook_eval_models(dataset_dir: Optional[str] = None):
    from sybil_shield.experiments import real_facebook_eval as facebook_eval

    models = facebook_eval.train_all_models(seed=1)
    return {name: models[name] for name in BENCHMARK_MODEL_VARIANTS if name in models}


def predict_facebook_dataset(dataset_dir: Optional[str] = None) -> Dict[str, List[float]]:
    from sybil_shield.experiments import real_facebook_eval as facebook_eval

    resolved_dir = _resolve_facebook_dataset_dir(dataset_dir)
    if resolved_dir is None:
        raise FileNotFoundError(
            "Facebook benchmark dataset not found. Set SYBIL_SHIELD_FACEBOOK_DIR or point dataset to the Undirected_Facebook directory."
        )

    graph_path = os.path.join(resolved_dir, "graph.txt")
    train_path = os.path.join(resolved_dir, "train.txt")

    G = facebook_eval.load_graph(graph_path)
    benign_train, _ = facebook_eval.load_labels_file(train_path)
    nodes = sorted(G.nodes())
    X, A_hat = facebook_eval.facebook_features(G, nodes, benign_train)

    models = _load_facebook_eval_models(resolved_dir)
    results: Dict[str, List[float]] = {}
    for name in BENCHMARK_MODEL_VARIANTS:
        model = models.get(name)
        if model is None:
            logger.warning("Skipping missing Facebook model variant: %s", name)
            continue
        probs = model.predict_proba(A_hat, X)[:, 1]
        results[name] = [float(v) for v in probs]

    if not results:
        raise RuntimeError("No Facebook model variants were trained.")
    return results


def _build_facebook_frontend_payload(scores: Dict[str, List[float]], graph: Optional[nx.Graph] = None, max_nodes: int = 80) -> Dict[str, Any]:
    if graph is None:
        resolved_dir = _resolve_facebook_dataset_dir()
        if resolved_dir is None:
            return {
                "scores": scores,
                "graph": {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
                "smell": {"suspicious_nodes": [], "suspicious_edges": [], "summary": "No Facebook dataset found."},
            }
        from sybil_shield.experiments import real_facebook_eval as facebook_eval
        graph = facebook_eval.load_graph(os.path.join(resolved_dir, "graph.txt"))

    node_ids = sorted(graph.nodes())
    if not node_ids:
        return {
            "scores": scores,
            "graph": {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
            "smell": {"suspicious_nodes": [], "suspicious_edges": [], "summary": "Empty graph."},
        }

    stacked = np.vstack([np.asarray(scores[name], dtype=float) for name in sorted(scores.keys())])
    mean_scores = np.mean(stacked, axis=0) if stacked.size else np.zeros(len(node_ids), dtype=float)
    node_to_score = {node: float(mean_scores[idx]) for idx, node in enumerate(node_ids)}
    ranked = sorted(node_to_score.items(), key=lambda item: item[1], reverse=True)
    suspicious = ranked[: min(max_nodes, len(ranked))]

    suspicious_nodes = [node for node, _ in suspicious]
    suspicious_set = set(suspicious_nodes)
    for node in suspicious_nodes:
        suspicious_set.update(graph.neighbors(node))

    relevant = graph.subgraph(sorted(suspicious_set)).copy()
    suspicious_payload = []
    for node, score in suspicious[:10]:
        suspicious_payload.append(
            {
                "node": int(node),
                "avg_score": float(score),
                "scores": {name: float(scores[name][node_ids.index(node)]) for name in sorted(scores.keys()) if node in node_ids},
            }
        )

    smell = {
        "suspicious_nodes": suspicious_payload,
        "suspicious_edges": [[int(u), int(v)] for u, v in relevant.edges()],
        "summary": f"Detected {len(suspicious_payload)} high-risk nodes in the suspicious cluster.",
        "threshold": float(suspicious_payload[0]["avg_score"]) if suspicious_payload else 0.0,
    }
    graph_payload = {
        "nodes": [int(n) for n in sorted(relevant.nodes())],
        "edges": [[int(u), int(v)] for u, v in relevant.edges()],
        "node_count": relevant.number_of_nodes(),
        "edge_count": relevant.number_of_edges(),
    }
    return {"scores": scores, "graph": graph_payload, "smell": smell}


def _adaptive_seed_frac(n_honest: int) -> float:
    """
    seed_frac interpolated (log n_honest) between the two scales with real
    evidence: n≈300 synthetic sweep favored ~0.06, n≈4000 Facebook showed
    0.05 beating 0.02. Only two anchor points -- stopgap until a proper
    multi-scale sweep exists, not a validated curve. Clamped to the anchors.
    """
    lo_n, lo_f = 300.0, 0.06
    hi_n, hi_f = 4000.0, 0.05
    n = max(float(n_honest), 1.0)
    if n <= lo_n:
        return lo_f
    if n >= hi_n:
        return hi_f
    t = (np.log(n) - np.log(lo_n)) / (np.log(hi_n) - np.log(lo_n))
    return lo_f + t * (hi_f - lo_f)


def _train_model_on_graph(graph: nx.Graph, seed: int = 0, epochs: int = 20,
                           seed_frac: float = 0.05) -> GCN:
    X, y, A_hat, nodes, seeds = build_features(graph, seed=seed, seed_frac=seed_frac)
    if X.size == 0:
        logger.error("Model seed=%s received empty feature matrix for graph with %d nodes", seed, len(graph.nodes))
        raise ValueError("Graph feature matrix is empty.")

    logger.info(
        "Training model seed=%s on graph with %d nodes, %d edges, %d honest nodes, seed_frac=%.4f, features=%s",
        seed,
        len(graph.nodes),
        len(graph.edges),
        sum(1 for n in graph.nodes if graph.nodes[n].get("label", 0) == 0),
        seed_frac,
        X.shape,
    )
    logger.info("Model seed=%s trusted seeds used: %s", seed, seeds[:10])

    model = GCN(n_features=X.shape[1], hidden_dims=(16, 16), n_classes=2, seed=seed)
    weights = _class_weights(y)
    for epoch in range(1, epochs + 1):
        loss = model.fit_step(A_hat, X, y, class_weight=weights, lr=0.02)
        if epoch in {1, 5, 10, 15, 20} or epoch == epochs:
            logger.info("Model seed=%s epoch %d/%d loss=%.6f", seed, epoch, epochs, float(loss))
    logger.info("Model seed=%s training complete. Final loss=%.6f", seed, float(model.compute_loss(model.predict_proba(A_hat, X), y, class_weight=weights)))
    return model


def predict_graph(G: Union[nx.Graph, GraphRequest, Dict[str, Any], Sequence[Tuple[Any, Any]]]) -> Dict[str, List[float]]:
    data = None
    if isinstance(G, GraphRequest):
        data = G.model_dump()
    elif isinstance(G, dict):
        data = G

    if data is not None:
        source = str(data.get("source") or "").strip().lower()
        dataset = data.get("dataset")
        use_facebook = bool(data.get("use_facebook"))
        mode = str(data.get("mode") or "").strip().lower()
        if use_facebook or source in {"facebook", "facebook_eval", "facebook-dataset", "fb"} or mode in {"facebook", "facebook_eval"}:
            logger.info("Using real Facebook benchmark evaluation path.")
            if isinstance(dataset, str) and dataset.strip():
                return predict_facebook_dataset(dataset)
            return predict_facebook_dataset()
        if isinstance(dataset, str) and dataset.strip() and os.path.exists(dataset):
            return predict_facebook_dataset(dataset)

    start = time.perf_counter()
    graph = _make_graph(G)
    logger.info("Graph normalized: nodes=%d edges=%d labels=%d", len(graph.nodes), len(graph.edges), sum(1 for _, d in graph.nodes(data=True) if "label" in d))
    if len(graph.nodes) == 0:
        logger.warning("Prediction skipped because graph is empty.")
        return {"standard": [], "random": [], "trusted": [], "mixed": []}

    n_honest = sum(1 for n in graph.nodes if graph.nodes[n].get("label", 0) == 0)
    n_sybil = len(graph.nodes) - n_honest
    seed_frac = _adaptive_seed_frac(n_honest)
    logger.info("Graph summary: honest=%d sybil=%d total=%d, adaptive seed_frac=%.4f", n_honest, n_sybil, len(graph.nodes), seed_frac)
    logger.info("Node labels sample: %s", list(graph.nodes(data=True))[:10])

    models = {
        "standard": _train_model_on_graph(graph, seed=0, seed_frac=seed_frac),
        "random": _train_model_on_graph(graph, seed=101, seed_frac=seed_frac),
        "trusted": _train_model_on_graph(graph, seed=202, seed_frac=seed_frac),
        "mixed": _train_model_on_graph(graph, seed=303, seed_frac=seed_frac),
    }
    # Legacy ad-hoc inference keeps the trusted variant, but the benchmark/fb path
    # intentionally excludes it to match real_facebook_eval.py.

    X, y, A_hat, nodes, _ = build_features(graph, seed=0, seed_frac=seed_frac)
    logger.info("Feature extraction: X.shape=%s, y.shape=%s, A_hat.shape=%s, node order sample=%s", X.shape, y.shape, A_hat.shape, nodes[:10])

    results: Dict[str, List[float]] = {}
    for name, model in models.items():
        probabilities = model.predict_proba(A_hat, X)[:, 1]
        score_map = {node: float(score) for node, score in zip(nodes, probabilities)}
        ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        top_nodes = ranked[:10]
        logger.info(
            "Model '%s' prediction summary: min=%.4f max=%.4f mean=%.4f top_risks=%s",
            name,
            float(np.min(probabilities)),
            float(np.max(probabilities)),
            float(np.mean(probabilities)),
            [(node, round(score, 4)) for node, score in top_nodes],
        )
        logger.info("Model '%s' node-by-node scores: %s", name, [(node, round(score, 4)) for node, score in ranked[:20]])
        results[name] = [float(value) for value in probabilities]

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Prediction pipeline complete in %.2f ms", elapsed_ms)
    return results


@app.get("/health")
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/facebook")
def facebook_dataset_info() -> Dict[str, Any]:
    dataset_dir = _resolve_facebook_dataset_dir()
    return {
        "available": dataset_dir is not None,
        "dataset_dir": dataset_dir,
        "model_variants": get_model_names(),
    }


@app.post("/predict")
@app.post("/api/predict")
def predict_api(payload: GraphRequest) -> Dict[str, List[float]]:
    logger.info(
        "Incoming prediction payload: nodes=%d edges=%d labels=%d dataset=%s source=%s mode=%s use_facebook=%s",
        len(payload.nodes or []),
        len(payload.edges or []),
        len(payload.labels or {}),
        payload.dataset,
        payload.source,
        payload.mode,
        payload.use_facebook,
    )
    return predict_graph(payload)


@app.post("/api/facebook/predict")
def facebook_predict_api(payload: GraphRequest = GraphRequest()) -> Dict[str, List[float]]:
    logger.info("Incoming Facebook benchmark prediction request.")
    return predict_graph(GraphRequest(dataset=payload.dataset or _resolve_facebook_dataset_dir() or "", source="facebook"))


@app.post("/facebooktest")
@app.post("/api/facebooktest")
def facebook_test_api(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = payload or {}
    keyword = body.get("keyword") if isinstance(body, dict) else body
    if str(keyword or "").strip().lower() != "go":
        logger.info("Facebook test not triggered. Received keyword=%r", keyword)
        return {
            "status": "waiting",
            "message": "Send {'keyword': 'go'} to run the Facebook benchmark test.",
            "triggered": False,
        }

    logger.info("Facebook benchmark trigger received. Starting real Facebook test.")
    resolved_dir = _resolve_facebook_dataset_dir()
    if resolved_dir is None:
        raise FileNotFoundError("Facebook benchmark dataset not found.")
    from sybil_shield.experiments import real_facebook_eval as facebook_eval
    graph = facebook_eval.load_graph(os.path.join(resolved_dir, "graph.txt"))
    scores = predict_facebook_dataset(resolved_dir)
    report = _build_facebook_frontend_payload(scores=scores, graph=graph)
    return {
        "status": "ok",
        "triggered": True,
        "model_variants": list(scores.keys()),
        "scores": report["scores"],
        "graph": report["graph"],
        "smell": report["smell"],
    }


@app.get("/")
@app.get("/api")
def root() -> Dict[str, str]:
    return {"message": "Bad People Finder API is running."}
