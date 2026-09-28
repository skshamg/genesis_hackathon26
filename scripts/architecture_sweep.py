from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(ROOT), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from sybil_shield.experiments.train_model import train_model


DEFAULT_CANDIDATES = [
    (8,),
    (8, 8),
    (16,),
    (16, 16),
    (32,),
    (32, 32),
    (64,),
    (64, 64),
    (16, 32, 16),
    (32, 32, 32),
    (8, 16, 8),
    (16, 16, 16),
]


def parse_candidates(raw: str | None) -> list[tuple[int, ...]]:
    if raw:
        candidates: list[tuple[int, ...]] = []
        for group in raw.split("|"):
            cleaned = [part.strip() for part in group.split(",") if part.strip()]
            if not cleaned:
                continue
            candidates.append(tuple(int(x) for x in cleaned))
        if candidates:
            return candidates
    return [tuple(c) for c in DEFAULT_CANDIDATES]


def run_sweep(
    candidates: Sequence[Sequence[int]],
    *,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    sybil_fraction_min: float,
    sybil_fraction_max: float,
    epochs: int,
    learning_rate: float,
    attack_every: int,
    attack_budget: int,
    recompute_every: int,
    seed: int,
    seed_frac: float,
) -> list[dict]:
    results: list[dict] = []

    for hidden_dims in candidates:
        print(f"\n=== testing hidden_dims={hidden_dims} ===")
        meta = train_model(
            model_type="gradient_adversarial",
            training_mode="baseline",
            num_graphs=num_graphs,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            sybil_fraction_min=sybil_fraction_min,
            sybil_fraction_max=sybil_fraction_max,
            epochs=epochs,
            learning_rate=learning_rate,
            attack_every=attack_every,
            attack_budget=attack_budget,
            recompute_every=recompute_every,
            hidden_dims=tuple(hidden_dims),
            betweenness_k=10,
            seed_frac=seed_frac,
            seed=seed,
        )

        metrics = meta.get("metrics", {})
        entry = {
            "hidden_dims": tuple(hidden_dims),
            "auc": float(metrics.get("auc", -1.0)),
            "accuracy": float(metrics.get("accuracy", -1.0)),
            "model_id": meta.get("model_id"),
            "train_seconds": float(metrics.get("train_seconds", 0.0)),
        }
        results.append(entry)

        print(f"auc={entry['auc']:.4f}  accuracy={entry['accuracy']:.4f}  model_id={entry['model_id']}")

    ranked = sorted(results, key=lambda item: item["auc"], reverse=True)
    print("\n=== best by AUC ===")
    for i, item in enumerate(ranked, start=1):
        print(f"{i}. hidden_dims={item['hidden_dims']}  auc={item['auc']:.4f}  accuracy={item['accuracy']:.4f}  model_id={item['model_id']}")

    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep a few hidden-layer architectures and keep the best one.")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Optional custom candidate list, e.g. '8,8|16,16|32,32,32|16,32,16'")
    parser.add_argument("--num-graphs", type=int, default=2)
    parser.add_argument("--min-nodes", type=int, default=80)
    parser.add_argument("--max-nodes", type=int, default=120)
    parser.add_argument("--sybil-frac-min", type=float, default=0.10)
    parser.add_argument("--sybil-frac-max", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--attack-every", type=int, default=1)
    parser.add_argument("--attack-budget", type=int, default=5)
    parser.add_argument("--recompute-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-frac", type=float, default=0.05)
    args = parser.parse_args()

    candidates = parse_candidates(args.candidates)
    run_sweep(
        candidates=candidates,
        num_graphs=args.num_graphs,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        sybil_fraction_min=args.sybil_frac_min,
        sybil_fraction_max=args.sybil_frac_max,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        attack_every=args.attack_every,
        attack_budget=args.attack_budget,
        recompute_every=args.recompute_every,
        seed=args.seed,
        seed_frac=args.seed_frac,
    )


if __name__ == "__main__":
    main()
