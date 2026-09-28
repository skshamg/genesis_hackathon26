"""
model_store.py

Filesystem-backed model registry. Layout:

    models/<model_type>/<model_id>/weights.npz   -- GCN.save_weights() output
    models/<model_type>/<model_id>/meta.json     -- params, metrics, timestamp

<model_type> is a free-form string ("gradient_adversarial" today, an
"unsupervised" variant later). <model_id> is generated at save time
(UTC timestamp + short random suffix) unless the caller supplies one.

This registry is meant to be the single source of truth: train_model.py
writes to it, and every consumer (the API's zero-shot crypto-scan path,
real_facebook_eval.py, future scripts) should LOAD from it by model_id
rather than training its own throwaway copy.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sybil_shield.core.gcn import GCN


def _models_root() -> Path:
    env = os.environ.get("SYBIL_SHIELD_MODELS_DIR")
    if env:
        return Path(env)
    # this file is src/sybil_shield/core/model_store.py -> repo root is
    # three parents up (core -> sybil_shield -> src -> repo_root)
    return Path(__file__).resolve().parents[3] / "models"


def _new_model_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def model_type_dir(model_type: str) -> Path:
    d = _models_root() / model_type
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir(model_type: str, model_id: str) -> Path:
    return model_type_dir(model_type) / model_id


def save_model(model: GCN, model_type: str, params: Dict[str, Any],
               metrics: Optional[Dict[str, Any]] = None,
               model_id: Optional[str] = None,
               training_graphs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    model_id = model_id or _new_model_id()
    d = model_dir(model_type, model_id)
    d.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(d / "weights.npz"))

    meta = {
        "model_id": model_id,
        "model_type": model_type,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": model.get_architecture(),
        "params": params,
        "metrics": metrics or {},
        "training_graphs": training_graphs or [],
    }
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    if training_graphs:
        with open(d / "training_graphs.json", "w") as f:
            json.dump(training_graphs, f, indent=2)
    return meta


def load_meta(model_type: str, model_id: str) -> Dict[str, Any]:
    path = model_dir(model_type, model_id) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"No metadata for {model_type}/{model_id} at {path}")
    with open(path) as f:
        return json.load(f)


def load_training_graphs(model_type: str, model_id: str) -> List[Dict[str, Any]]:
    path = model_dir(model_type, model_id) / "training_graphs.json"
    if not path.exists():
        raise FileNotFoundError(f"No training graph catalog for {model_type}/{model_id} at {path}")
    with open(path) as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def load_model(model_type: str, model_id: str) -> GCN:
    path = model_dir(model_type, model_id) / "weights.npz"
    if not path.exists():
        raise FileNotFoundError(f"No weights for {model_type}/{model_id} at {path}")
    return GCN.load_weights(str(path))


def list_models(model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Newest first. Reads every meta.json under the registry (or under
    one model_type if given) -- fine at the scale this registry runs at
    (dozens/hundreds of trained models, not millions)."""
    root = _models_root()
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    types = [model_type] if model_type else [p.name for p in root.iterdir() if p.is_dir()]
    for mt in types:
        td = root / mt
        if not td.exists():
            continue
        for d in td.iterdir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    out.append(json.load(f))
    out.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return out


def resolve_model_id(model_type: str, model_id: Optional[str] = None) -> str:
    """None or "latest" -> most recently trained model of this type.
    Anything else is returned as-is (existence is checked on load)."""
    if model_id and model_id != "latest":
        return model_id
    candidates = list_models(model_type)
    if not candidates:
        raise FileNotFoundError(
            f"No trained models of type '{model_type}' in {model_type_dir(model_type)}. "
            "Run train.py first."
        )
    return candidates[0]["model_id"]


def load_latest(model_type: str) -> GCN:
    return load_model(model_type, resolve_model_id(model_type))
