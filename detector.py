"""Compatibility entry point for the organized package layout."""

from bad_people_finder.core.detector import (
    build_features,
    early_walk,
    pick_seeds,
    ppr_vector,
    sweep_cut,
    trust_scores,
)

__all__ = [
    "build_features",
    "early_walk",
    "pick_seeds",
    "ppr_vector",
    "sweep_cut",
    "trust_scores",
]

