from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from sybil_shield.experiments.train_model import train_model


def run_lightweight_test() -> dict:
    """Run a very small smoke test for the training pipeline.

    This is intentionally tiny so it is fast enough to use as a quick
    validation step while developing or debugging. It still exercises the
    real training and evaluation path, just with a compact configuration.
    """
    print("[lightweight-test] starting smoke test...")

    meta = train_model(
        model_type="gradient_adversarial",
        training_mode="baseline",
        num_graphs=2,
        min_nodes=80,
        max_nodes=120,
        sybil_fraction_min=0.10,
        sybil_fraction_max=0.35,
        epochs=1,
        learning_rate=0.01,
        attack_every=1,
        attack_budget=5,
        recompute_every=1,
        hidden_dims=(8, 8),
        betweenness_k=10,
        seed_frac=0.05,
        seed=7,
    )

    metrics = meta.get("metrics", {})
    print("[lightweight-test] completed")
    print(f"  model_id: {meta.get('model_id')}")
    print(f"  auc: {metrics.get('auc')}")
    print(f"  accuracy: {metrics.get('accuracy')}")

    if "auc" not in metrics:
        raise RuntimeError("lightweight test did not return evaluation metrics")

    return meta


if __name__ == "__main__":
    run_lightweight_test()
