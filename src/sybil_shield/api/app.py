from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import threading
import time
import uuid
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any

import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics import accuracy_score, roc_auc_score

from sybil_shield.core.detector import build_features


BENCHMARK_MODEL_VARIANTS: List[str] = ["gradient"]
DEFAULT_MODEL_TYPE = "gradient_adversarial"


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
    # Which registered model (see core.model_store / POST /api/train/start)
    # to run inference with. None or "latest" -> most recently trained
    # model of model_type. This is the id the frontend gets back from a
    # completed training job and should pass on every /predict call.
    model_id: Optional[str] = None
    model_type: str = DEFAULT_MODEL_TYPE


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sybil_shield.api")

app = FastAPI(title="SYBIL-SHIELD API", version="1.0.0")
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


@lru_cache(maxsize=16)
def _load_facebook_eval_models(dataset_dir: Optional[str] = None,
                                model_id: Optional[str] = None,
                                model_type: str = DEFAULT_MODEL_TYPE):
    """model_id=None trains a fresh model (full intensity, see
    real_facebook_eval.train_all_models); a real id (or "latest") loads
    from the registry instead. lru_cache means each distinct
    (dataset_dir, model_id, model_type) combo is only trained/loaded
    once per process."""
    from sybil_shield.experiments import real_facebook_eval as facebook_eval

    models = facebook_eval.train_all_models(seed=1, model_id=model_id, model_type=model_type)
    return {name: models[name] for name in BENCHMARK_MODEL_VARIANTS if name in models}


def predict_facebook_dataset(dataset_dir: Optional[str] = None,
                              model_id: Optional[str] = None,
                              model_type: str = DEFAULT_MODEL_TYPE) -> Dict[str, List[float]]:
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

    models = _load_facebook_eval_models(resolved_dir, model_id, model_type)
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


def _load_facebook_truth_labels(graph: nx.Graph, dataset_dir: Optional[str] = None) -> Dict[int, int]:
    if dataset_dir is None:
        dataset_dir = _resolve_facebook_dataset_dir()
    if not dataset_dir:
        return {}

    truth: Dict[int, int] = {}
    for label_file in ["test.txt", "train.txt"]:
        path = os.path.join(dataset_dir, label_file)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            if len(lines) < 2:
                continue
            benign = [int(node) for node in lines[0].split()]
            sybil = [int(node) for node in lines[1].split()]
            for node in benign:
                truth[int(node)] = 0
            for node in sybil:
                truth[int(node)] = 1
        except OSError:
            logger.warning("Unable to read Facebook label file: %s", path)
    return truth


def _build_facebook_metrics(scores: Dict[str, List[float]], graph: nx.Graph, dataset_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    if not scores or not graph:
        return {}

    if dataset_dir is None:
        dataset_dir = _resolve_facebook_dataset_dir()
    if not dataset_dir:
        return {}

    from sybil_shield.experiments import real_facebook_eval as facebook_eval

    test_path = os.path.join(dataset_dir, "test.txt")
    if not os.path.exists(test_path):
        return {}

    benign_test, sybil_test = facebook_eval.load_labels_file(test_path)
    node_ids = sorted(graph.nodes())
    node_index = {node: idx for idx, node in enumerate(node_ids)}
    test_nodes = benign_test + sybil_test
    y_true = np.array([0] * len(benign_test) + [1] * len(sybil_test), dtype=int)
    metrics: Dict[str, Dict[str, float]] = {}
    variant_names = list(scores.keys())
    if not variant_names:
        return metrics

    for name in variant_names:
        values = np.asarray(scores[name], dtype=float)
        test_idx = [node_index[node] for node in test_nodes if node in node_index]
        test_scores = values[test_idx]
        if len(test_scores) == 0:
            continue
        metrics[name] = {
            "auc": float(roc_auc_score(y_true, test_scores)),
            "accuracy": float(accuracy_score(y_true, (test_scores >= 0.5).astype(int))),
        }

    if metrics:
        stacked = np.vstack([np.asarray(scores[name], dtype=float) for name in variant_names])
        ensemble_scores = np.mean(stacked, axis=0) if stacked.size else np.zeros(len(node_ids), dtype=float)
        ensemble_test = ensemble_scores[[node_index[node] for node in test_nodes if node in node_index]]
        metrics["ensemble"] = {
            "auc": float(roc_auc_score(y_true, ensemble_test)),
            "accuracy": float(accuracy_score(y_true, (ensemble_test >= 0.5).astype(int))),
        }

    return metrics


def _build_facebook_confusion(graph: nx.Graph, true_labels: Dict[int, int], ensemble_scores: List[float]) -> Dict[str, List[int]]:
    result = {
        "correctly_flagged_sybil": [],
        "incorrectly_flagged_sybil": [],
        "correctly_flagged_safe": [],
        "incorrectly_flagged_safe": [],
    }

    for idx, node in enumerate(sorted(graph.nodes())):
        true_label = true_labels.get(int(node), -1)
        if true_label == -1:
            continue
        predicted_sybil = float(ensemble_scores[idx]) >= 0.5
        actual_sybil = true_label == 1

        if predicted_sybil and actual_sybil:
            result["correctly_flagged_sybil"].append(int(node))
        elif predicted_sybil and not actual_sybil:
            result["incorrectly_flagged_sybil"].append(int(node))
        elif not predicted_sybil and not actual_sybil:
            result["correctly_flagged_safe"].append(int(node))
        elif not predicted_sybil and actual_sybil:
            result["incorrectly_flagged_safe"].append(int(node))

    return result


def _build_facebook_frontend_payload(scores: Dict[str, List[float]], graph: Optional[nx.Graph] = None, max_nodes: int = 80, dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    if graph is None:
        resolved_dir = dataset_dir or _resolve_facebook_dataset_dir()
        if resolved_dir is None:
            empty = {
                "scores": scores,
                "graph": {
                    "nodes": [],
                    "edges": [],
                    "node_count": 0,
                    "edge_count": 0,
                    "positions": [],
                    "true_labels": [],
                    "ensemble_scores": [],
                },
                "smell": {"suspicious_nodes": [], "suspicious_edges": [], "summary": "No Facebook dataset found."},
                "metrics": {},
                "confusion": {
                    "correctly_flagged_sybil": [],
                    "incorrectly_flagged_sybil": [],
                    "correctly_flagged_safe": [],
                    "incorrectly_flagged_safe": [],
                },
            }
            return empty
        from sybil_shield.experiments import real_facebook_eval as facebook_eval
        graph = facebook_eval.load_graph(os.path.join(resolved_dir, "graph.txt"))

    node_ids = sorted(graph.nodes())
    if not node_ids:
        empty = {
            "scores": scores,
            "graph": {
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
                "positions": [],
                "true_labels": [],
                "ensemble_scores": [],
            },
            "smell": {"suspicious_nodes": [], "suspicious_edges": [], "summary": "Empty graph."},
            "metrics": {},
            "confusion": {
                "correctly_flagged_sybil": [],
                "incorrectly_flagged_sybil": [],
                "correctly_flagged_safe": [],
                "incorrectly_flagged_safe": [],
            },
        }
        return empty

    if not scores:
        scores = {name: [0.0 for _ in node_ids] for name in BENCHMARK_MODEL_VARIANTS}

    model_names = [name for name in BENCHMARK_MODEL_VARIANTS if name in scores]
    if not model_names:
        model_names = list(scores.keys())
    stacked = np.vstack([np.asarray(scores[name], dtype=float) for name in model_names])
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
        variant_scores = {name: float(scores[name][node_ids.index(node)]) for name in model_names if node in node_ids}
        suspicious_payload.append(
            {
                "node": int(node),
                "avg_score": float(score),
                "true_label": int(_load_facebook_truth_labels(graph, dataset_dir).get(int(node), -1)),
                "scores": variant_scores,
            }
        )

    smell = {
        "suspicious_nodes": suspicious_payload,
        "suspicious_edges": [[int(u), int(v)] for u, v in relevant.edges()],
        "summary": f"Detected {len(suspicious_payload)} high-risk nodes in the suspicious cluster.",
        "threshold": float(suspicious_payload[0]["avg_score"]) if suspicious_payload else 0.0,
    }

    true_labels = _load_facebook_truth_labels(graph, dataset_dir)
    positions = []
    for idx, _ in enumerate(node_ids):
        angle = (idx / max(len(node_ids), 1)) * 2 * np.pi
        positions.append([float(np.cos(angle)), float(np.sin(angle))])

    graph_payload = {
        "nodes": [int(n) for n in node_ids],
        "edges": [[int(u), int(v)] for u, v in graph.edges()],
        "node_count": len(node_ids),
        "edge_count": graph.number_of_edges(),
        "positions": positions,
        "true_labels": [true_labels.get(int(node), -1) for node in node_ids],
        "ensemble_scores": [float(mean_scores[idx]) for idx, _ in enumerate(node_ids)],
    }

    metrics = _build_facebook_metrics(scores=scores, graph=graph, dataset_dir=dataset_dir)
    confusion = _build_facebook_confusion(graph, true_labels, list(mean_scores))
    return {"scores": scores, "graph": graph_payload, "smell": smell, "metrics": metrics, "confusion": confusion}


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


def predict_graph(G: Union[nx.Graph, GraphRequest, Dict[str, Any], Sequence[Tuple[Any, Any]]]) -> Dict[str, List[float]]:
    data = None
    if isinstance(G, GraphRequest):
        data = G.model_dump()
    elif isinstance(G, dict):
        data = G

    model_id = None
    model_type = DEFAULT_MODEL_TYPE
    if data is not None:
        model_id = data.get("model_id")
        model_type = data.get("model_type") or DEFAULT_MODEL_TYPE
        source = str(data.get("source") or "").strip().lower()
        dataset = data.get("dataset")
        use_facebook = bool(data.get("use_facebook"))
        mode = str(data.get("mode") or "").strip().lower()
        if use_facebook or source in {"facebook", "facebook_eval", "facebook-dataset", "fb"} or mode in {"facebook", "facebook_eval"}:
            logger.info("Using real Facebook benchmark evaluation path.")
            if isinstance(dataset, str) and dataset.strip():
                return predict_facebook_dataset(dataset, model_id, model_type)
            return predict_facebook_dataset(model_id=model_id, model_type=model_type)
        if isinstance(dataset, str) and dataset.strip() and os.path.exists(dataset):
            return predict_facebook_dataset(dataset, model_id, model_type)

    start = time.perf_counter()
    graph = _make_graph(G)
    logger.info("Graph normalized: nodes=%d edges=%d labels=%d", len(graph.nodes), len(graph.edges), sum(1 for _, d in graph.nodes(data=True) if "label" in d))
    if len(graph.nodes) == 0:
        logger.warning("Prediction skipped because graph is empty.")
        return {name: [] for name in BENCHMARK_MODEL_VARIANTS}

    n_honest = sum(1 for n in graph.nodes if graph.nodes[n].get("label", 0) == 0)
    n_sybil = len(graph.nodes) - n_honest
    seed_frac = _adaptive_seed_frac(n_honest)
    logger.info("Graph summary: honest=%d sybil=%d total=%d, adaptive seed_frac=%.4f", n_honest, n_sybil, len(graph.nodes), seed_frac)
    logger.info("Node labels sample: %s", list(graph.nodes(data=True))[:10])

    # Zero-shot inference with the cached gradient-adversarial model (same
    # one _load_facebook_eval_models trains/loads for the Facebook path) --
    # trained/loaded ONCE and reused across every request via lru_cache,
    # instead of the old per-request path that trained 4 fresh models
    # (standard/random/trusted/mixed) from scratch on every call. For a
    # live crypto scan this is the real latency fix: no training in the
    # request path at all after the first (cold-start) call -- and with a
    # model_id from the registry, not even that first call trains.
    models = _load_facebook_eval_models(model_id=model_id, model_type=model_type)
    model = models["gradient"]

    X, y, A_hat, nodes, _ = build_features(graph, seed=0, seed_frac=seed_frac)
    logger.info("Feature extraction: X.shape=%s, y.shape=%s, A_hat.shape=%s, node order sample=%s", X.shape, y.shape, A_hat.shape, nodes[:10])

    results: Dict[str, List[float]] = {}
    probabilities = model.predict_proba(A_hat, X)[:, 1]
    score_map = {node: float(score) for node, score in zip(nodes, probabilities)}
    ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    top_nodes = ranked[:10]
    logger.info(
        "Model 'gradient' prediction summary: min=%.4f max=%.4f mean=%.4f top_risks=%s",
        float(np.min(probabilities)),
        float(np.max(probabilities)),
        float(np.mean(probabilities)),
        [(node, round(score, 4)) for node, score in top_nodes],
    )
    logger.info("Model 'gradient' node-by-node scores: %s", [(node, round(score, 4)) for node, score in ranked[:20]])
    results["gradient"] = [float(value) for value in probabilities]

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
    return predict_graph(GraphRequest(
        dataset=payload.dataset or _resolve_facebook_dataset_dir() or "",
        source="facebook", model_id=payload.model_id, model_type=payload.model_type,
    ))


@app.post("/facebooktest")
@app.post("/api/facebooktest")
def facebook_test_api(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = payload or {}
    keyword = body.get("keyword") if isinstance(body, dict) else body
    # Which registered model to test. Without one, predict_facebook_dataset
    # would train a fresh ~10 min model, so default to the latest saved one.
    model_id = (body.get("model_id") if isinstance(body, dict) else None) or "latest"
    model_type = (body.get("model_type") if isinstance(body, dict) else None) or DEFAULT_MODEL_TYPE
    _validate_model_id(model_id)
    empty_graph = {
        "nodes": [],
        "edges": [],
        "node_count": 0,
        "edge_count": 0,
        "positions": [],
        "true_labels": [],
        "ensemble_scores": [],
    }
    empty_confusion = {
        "correctly_flagged_sybil": [],
        "incorrectly_flagged_sybil": [],
        "correctly_flagged_safe": [],
        "incorrectly_flagged_safe": [],
    }

    if str(keyword or "").strip().lower() != "go":
        logger.info("Facebook test not triggered. Received keyword=%r", keyword)
        return {
            "status": "waiting",
            "message": "Send {'keyword': 'go'} to run the Facebook benchmark test.",
            "triggered": False,
            "model_variants": [],
            "scores": {},
            "graph": empty_graph,
            "smell": {"suspicious_nodes": [], "suspicious_edges": [], "summary": "Facebook benchmark is waiting to be triggered."},
            "metrics": {},
            "confusion": empty_confusion,
        }

    logger.info("Facebook benchmark trigger received. Starting real Facebook test.")
    resolved_dir = _resolve_facebook_dataset_dir()
    if resolved_dir is None:
        logger.warning("Facebook benchmark dataset not found.")
        return {
            "status": "missing_dataset",
            "triggered": False,
            "message": "Facebook benchmark dataset not found. Set SYBIL_SHIELD_FACEBOOK_DIR or point dataset to the Undirected_Facebook directory.",
            "model_variants": [],
            "scores": {},
            "graph": empty_graph,
            "smell": {"suspicious_nodes": [], "suspicious_edges": [], "summary": "Dataset missing; benchmark could not run."},
            "metrics": {},
            "confusion": empty_confusion,
        }

    try:
        from sybil_shield.experiments import real_facebook_eval as facebook_eval
        graph = facebook_eval.load_graph(os.path.join(resolved_dir, "graph.txt"))
        scores = predict_facebook_dataset(resolved_dir, model_id, model_type)
        report = _build_facebook_frontend_payload(scores=scores, graph=graph, dataset_dir=resolved_dir)
        return {
            "status": "ok",
            "triggered": True,
            "message": report["smell"].get("summary", "Facebook benchmark completed."),
            "model_id": model_id,
            "model_variants": list(scores.keys()),
            "scores": report["scores"],
            "graph": report["graph"],
            "smell": report["smell"],
            "metrics": report["metrics"],
            "confusion": report["confusion"],
        }
    except Exception as exc:
        logger.exception("Facebook benchmark evaluation failed.")
        return {
            "status": "error",
            "triggered": False,
            "message": f"Real test failed for model '{model_id}': {exc}",
            "model_variants": [],
            "scores": {},
            "graph": empty_graph,
            "smell": {"suspicious_nodes": [], "suspicious_edges": [], "summary": "Benchmark evaluation failed."},
            "metrics": {},
            "confusion": empty_confusion,
        }



_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _validate_model_id(model_id: Optional[str]) -> None:
    """model_id becomes a folder name under models/<type>/, so keep it to
    a safe charset (no slashes / '..'). \"latest\" and None are allowed."""
    if model_id in (None, "", "latest"):
        return
    if not _MODEL_ID_RE.match(model_id):
        raise HTTPException(
            status_code=422,
            detail="model_id must be 1-64 chars: letters, digits, '.', '_' or '-' (no spaces or slashes).",
        )


# =====================================================================
# MODEL REGISTRY -- browse what train.py / POST /api/train/start saved
# =====================================================================

@app.get("/api/models")
def list_models_api(model_type: Optional[str] = None) -> Dict[str, Any]:
    from sybil_shield.core import model_store
    return {"models": model_store.list_models(model_type)}


@app.get("/api/model/info")
def model_info_api(model_type: str = DEFAULT_MODEL_TYPE, model_id: str = "latest") -> Dict[str, Any]:
    """Inspect Model tab: architecture diagram, per-layer weight
    heatmaps, metrics, confusion matrix, ROC curve -- all read straight
    from what train_model.py computed and model_store.save_model wrote,
    no recomputation here."""
    from sybil_shield.core import model_store

    _validate_model_id(model_id)
    try:
        resolved = model_store.resolve_model_id(model_type, model_id)
        meta = model_store.load_meta(model_type, resolved)
        model = model_store.load_model(model_type, resolved)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    layers, weights = [], []
    for i, (w, b) in enumerate(zip(model.W, model.b)):
        name = f"gcn_layer_{i + 1}" if i < model.n_layers - 1 else "output_layer"
        layers.append({"name": name, "input_dim": int(w.shape[0]), "output_dim": int(w.shape[1])})
        weights.append({"layer_name": name, "matrix": w.tolist()})

    m = meta.get("metrics", {})
    training_graphs = meta.get("training_graphs") or []
    if not training_graphs:
        try:
            training_graphs = model_store.load_training_graphs(model_type, resolved)
        except FileNotFoundError:
            training_graphs = []
    return {
        "model_id": resolved,
        "model_type": model_type,
        "trained_at": meta.get("trained_at"),
        "params": meta.get("params", {}),
        "architecture": {"layers": layers},
        "weights": weights,
        "metrics": {
            "auc": m.get("auc"), "accuracy": m.get("accuracy"),
            "precision": m.get("precision"), "recall": m.get("recall"),
        },
        "confusion_matrix": m.get("confusion_matrix", {}),
        "roc_curve": m.get("roc_curve", []),
        "training_graphs": training_graphs,
    }


# =====================================================================
# TRAINING -- POST starts a background job running train_model.py's
# train_model(); GET polls its status. No WebSocket yet (the frontend's
# spec falls back to this same polling shape on WS failure, so this
# alone is enough to drive the Train tab; add a WS push later if the
# 1s poll interval turns out to feel laggy).
# =====================================================================

class TrainRequest(BaseModel):
    model_type: str = DEFAULT_MODEL_TYPE
    num_graphs: int = 50
    min_nodes: int = 100
    max_nodes: int = 500
    sybil_fraction_min: float = 0.05
    sybil_fraction_max: float = 0.30
    epochs: int = 120
    learning_rate: float = 0.02
    attack_every: int = 2
    attack_budget: int = 8
    recompute_every: int = 3
    hidden_dims: List[int] = Field(default_factory=lambda: [16, 16])
    betweenness_k: int = 30
    seed_frac: float = 0.05
    seed: int = 1
    model_id: Optional[str] = None  # custom id; default is timestamp-based

    @field_validator("model_id")
    @classmethod
    def _clean_model_id(cls, v: Optional[str]) -> Optional[str]:
        v = (v or "").strip()
        if not v:
            return None
        if v == "latest" or not _MODEL_ID_RE.match(v):
            raise ValueError(
                "model_id must be 1-64 chars: letters, digits, '.', '_' or '-' (and not 'latest')."
            )
        return v


_train_jobs: Dict[str, Dict[str, Any]] = {}
_train_events: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}
_train_jobs_lock = threading.Lock()


def _initial_train_state(num_graphs: int, epochs: int) -> Dict[str, Any]:
    return {
        "status": "running",
        "current_graph_index": 0,
        "total_graphs": num_graphs,
        "epoch": 0,
        "total_epochs": epochs,
        "loss": None,
        "honest_count": None,
        "sybil_count": None,
        "current_graph": {"nodes": [], "edges": []},
    }


def _run_training_job(job_id: str, req: TrainRequest) -> None:
    from sybil_shield.experiments.train_model import train_model

    evq = _train_events.get(job_id, queue.Queue())

    def _on_progress(msg: Dict[str, Any]) -> None:
        with _train_jobs_lock:
            _train_jobs[job_id].update(msg)
        evq.put(msg)

    try:
        train_model(
            model_type=req.model_type, num_graphs=req.num_graphs,
            min_nodes=req.min_nodes, max_nodes=req.max_nodes,
            sybil_fraction_min=req.sybil_fraction_min,
            sybil_fraction_max=req.sybil_fraction_max,
            epochs=req.epochs, learning_rate=req.learning_rate,
            attack_every=req.attack_every, attack_budget=req.attack_budget,
            recompute_every=req.recompute_every, hidden_dims=req.hidden_dims,
            betweenness_k=req.betweenness_k, seed_frac=req.seed_frac,
            seed=req.seed, model_id=req.model_id, progress_cb=_on_progress,
        )
        # A newly trained model may now be "latest" -- drop any cached
        # zero-shot model so the next inference request picks it up.
        _load_facebook_eval_models.cache_clear()
    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        err = {"status": "error", "message": str(exc)}
        with _train_jobs_lock:
            _train_jobs[job_id].update(err)
        evq.put(err)


@app.post("/api/train/start")
def train_start(payload: TrainRequest) -> Dict[str, str]:
    if payload.model_id:
        from sybil_shield.core import model_store
        if (model_store.model_dir(payload.model_type, payload.model_id) / "meta.json").exists():
            raise HTTPException(
                status_code=409,
                detail=f"Model id '{payload.model_id}' already exists. Pick a different id.",
            )
    job_id = uuid.uuid4().hex
    with _train_jobs_lock:
        _train_jobs[job_id] = _initial_train_state(payload.num_graphs, payload.epochs)
        _train_events[job_id] = queue.Queue()
    thread = threading.Thread(target=_run_training_job, args=(job_id, payload), daemon=True)
    thread.start()
    logger.info("Training job %s started: %s", job_id, payload.model_dump())
    return {"job_id": job_id}


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str) -> Dict[str, Any]:
    with _train_jobs_lock:
        state = _train_jobs.get(job_id)
    if state is None:
        return {"status": "error", "message": f"Unknown job_id '{job_id}'"}
    return dict(state)


# =====================================================================
# SCAN -- POST starts a background job that crawls a live wallet graph
# (live_crypto_graph_builder.build_live_wallet_graph) and scores it
# (crypto_eval.evaluate_crypto_graph). Node/edge/progress events from
# the crawl are pushed onto a per-job queue as they happen and streamed
# out over /ws/scan/{job_id} verbatim -- their shape already matches
# the frontend's ScanEvent type (sybil-types.ts), so no reshaping here.
# GET /api/scan/results/{job_id} is what the frontend calls once it
# sees the "done" event, for the full graph + risk-cluster payload.
# =====================================================================

class ScanRequest(BaseModel):
    address: Optional[str] = None
    max_nodes: int = 300
    max_depth: int = 2
    model_type: str = DEFAULT_MODEL_TYPE
    model_id: str = "latest"
    api_key: Optional[str] = None
    tiny_verified: Optional[List[str]] = None


_scan_jobs: Dict[str, Dict[str, Any]] = {}
_scan_events: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}
_scan_jobs_lock = threading.Lock()


def _derive_clusters(
    graph: nx.Graph, scores: Dict[str, float], threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Group high-risk wallets by shared funder for the frontend's
    cluster cards (funder, member wallets, activity time window, mean
    risk). Not the model's own decision boundary -- purely a display
    grouping over evaluate_crypto_graph's per-node scores, using the
    same "shared funder + tight time window" signature crypto_eval.py's
    docstring names as the qualitative check for a real sybil farm."""
    flagged = [n for n, s in scores.items() if s >= threshold]
    if not flagged:
        return []

    funder_of: Dict[str, str] = {}
    for n in flagged:
        neighbors = [nb for nb in graph.neighbors(n) if nb not in scores or scores.get(nb, 0) < threshold]
        if not neighbors:
            neighbors = list(graph.neighbors(n))
        if not neighbors:
            continue
        funder_of[n] = max(neighbors, key=lambda nb: graph.degree(nb))

    by_funder: Dict[str, List[str]] = {}
    for n, funder in funder_of.items():
        by_funder.setdefault(funder, []).append(n)

    clusters = []
    for funder, wallets in by_funder.items():
        timestamps = []
        for w in wallets:
            if graph.has_edge(funder, w):
                ts = graph.edges[funder, w].get("timestamp")
                if ts:
                    timestamps.append(int(ts))
        window = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0
        clusters.append({
            "funder": str(funder),
            "wallets": [str(w) for w in wallets],
            "time_window_seconds": window,
            "risk_score": sum(scores[w] for w in wallets) / len(wallets),
        })
    clusters.sort(key=lambda c: c["risk_score"], reverse=True)
    return clusters


def _run_scan_job(job_id: str, req: ScanRequest) -> None:
    from sybil_shield.experiments.live_crypto_graph_builder import build_live_wallet_graph
    from sybil_shield.experiments.crypto_eval import evaluate_crypto_graph

    evq = _scan_events[job_id]

    def _on_event(evt: Dict[str, Any]) -> None:
        evq.put(evt)

    try:
        graph = build_live_wallet_graph(
            seed_wallet=req.address, max_nodes=req.max_nodes, max_depth=req.max_depth,
            persist=True, event_cb=_on_event, api_key=req.api_key,
        )
        tiny_verified = [a.lower() for a in req.tiny_verified] if req.tiny_verified else None
        results = evaluate_crypto_graph(
            graph, model_type=req.model_type, model_id=req.model_id,
            tiny_verified=tiny_verified,
        )
        scores = {k: float(v) for k, v in results["scores"].items()}
        graph_payload = {
            "nodes": [
                {
                    "id": str(n),
                    "node_type": graph.nodes[n].get("node_type", "wallet"),
                    "score": scores.get(str(n)),
                    "label": 1 if scores.get(str(n), 0.0) >= 0.5 else 0,
                }
                for n in graph.nodes()
            ],
            "edges": [[str(u), str(v)] for u, v in graph.edges()],
            "model_id": req.model_id,
            "model_type": req.model_type,
        }
        with _scan_jobs_lock:
            _scan_jobs[job_id]["status"] = "done"
            _scan_jobs[job_id]["results"] = {
                "graph": graph_payload,
                "clusters": _derive_clusters(graph, scores),
            }
        evq.put({"type": "done"})
    except Exception as exc:
        logger.exception("Scan job %s failed", job_id)
        with _scan_jobs_lock:
            _scan_jobs[job_id]["status"] = "error"
            _scan_jobs[job_id]["message"] = str(exc)
        evq.put({"type": "done"})


@app.post("/api/scan/start")
def scan_start(payload: ScanRequest) -> Dict[str, str]:
    job_id = uuid.uuid4().hex
    with _scan_jobs_lock:
        _scan_jobs[job_id] = {"status": "running", "results": None}
        _scan_events[job_id] = queue.Queue()
    thread = threading.Thread(target=_run_scan_job, args=(job_id, payload), daemon=True)
    thread.start()
    logger.info("Scan job %s started: %s", job_id, payload.model_dump())
    return {"job_id": job_id}


@app.websocket("/ws/train/{job_id}")
async def train_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    with _train_jobs_lock:
        evq = _train_events.get(job_id)
    if evq is None:
        await websocket.close(code=4404)
        return
    try:
        while True:
            try:
                evt = evq.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            await websocket.send_json(evt)
            if evt.get("status") in {"completed", "error"}:
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed by the client


@app.get("/api/scan/status/{job_id}")
def scan_status(job_id: str) -> Dict[str, Any]:
    with _scan_jobs_lock:
        state = _scan_jobs.get(job_id)
    if state is None:
        return {"status": "error", "message": f"Unknown job_id '{job_id}'"}
    return {"status": state["status"]}


@app.get("/api/scan/results/{job_id}")
def scan_results(job_id: str) -> Dict[str, Any]:
    with _scan_jobs_lock:
        state = _scan_jobs.get(job_id)
    if state is None or state.get("results") is None:
        return {"graph": {"nodes": [], "edges": []}, "clusters": []}
    return state["results"]


@app.websocket("/ws/scan/{job_id}")
async def scan_ws(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    with _scan_jobs_lock:
        evq = _scan_events.get(job_id)
    if evq is None:
        await websocket.close(code=4404)
        return
    try:
        while True:
            try:
                evt = evq.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            await websocket.send_json(evt)
            if evt.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed by the client


@app.get("/")
@app.get("/api")
def root() -> Dict[str, str]:
    return {"message": "SYBIL-SHIELD API is running."}