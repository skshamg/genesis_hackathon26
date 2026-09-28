import json
import os
import pickle
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx


def _graphs_root(root_dir: Optional[str] = None) -> Path:
    if root_dir is not None:
        return Path(root_dir)
    env = os.environ.get("SYBIL_SHIELD_GRAPHS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "crypto_graphs"


def _new_graph_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def graph_type_dir(graph_type: str, root_dir: Optional[str] = None) -> Path:
    d = _graphs_root(root_dir) / graph_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def graph_dir(graph_type: str, graph_id: str, root_dir: Optional[str] = None) -> Path:
    return graph_type_dir(graph_type, root_dir) / graph_id


def save_graph(
    graph: nx.Graph,
    graph_type: str,
    params: Optional[Dict[str, Any]] = None,
    root_dir: Optional[str] = None,
    graph_id: Optional[str] = None,
) -> Dict[str, Any]:
    graph_id = graph_id or _new_graph_id()
    d = graph_dir(graph_type, graph_id, root_dir)
    d.mkdir(parents=True, exist_ok=True)

    with open(d / "graph.gpickle", "wb") as f:
        pickle.dump(graph, f)

    meta = {
        "graph_id": graph_id,
        "graph_type": graph_type,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "num_honest": sum(1 for _, d in graph.nodes(data=True) if d.get("label") == 0),
        "num_sybil": sum(1 for _, d in graph.nodes(data=True) if d.get("label") == 1),
        "params": params or {},
    }
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_meta(graph_type: str, graph_id: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    path = graph_dir(graph_type, graph_id, root_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No metadata for {graph_type}/{graph_id} at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graph(graph_type: str, graph_id: str, root_dir: Optional[str] = None) -> nx.Graph:
    path = graph_dir(graph_type, graph_id, root_dir) / "graph.gpickle"
    if not path.exists():
        raise FileNotFoundError(f"No graph for {graph_type}/{graph_id} at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def list_graphs(graph_type: Optional[str] = None, root_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    root = _graphs_root(root_dir)
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    types = [graph_type] if graph_type else [p.name for p in root.iterdir() if p.is_dir()]
    for gt in types:
        td = root / gt
        if not td.exists():
            continue
        for d in td.iterdir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def resolve_graph_id(graph_type: str, graph_id: Optional[str] = None, root_dir: Optional[str] = None) -> str:
    if graph_id and graph_id != "latest":
        return graph_id
    candidates = list_graphs(graph_type, root_dir)
    if not candidates:
        raise FileNotFoundError(f"No saved graphs of type '{graph_type}' in {graph_type_dir(graph_type, root_dir)}")
    return candidates[0]["graph_id"]
