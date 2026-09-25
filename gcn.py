"""Compatibility entry point for the organized package layout."""

from bad_people_finder.core.gcn import GCN, relu, relu_grad, softmax

__all__ = ["GCN", "relu", "relu_grad", "softmax"]
