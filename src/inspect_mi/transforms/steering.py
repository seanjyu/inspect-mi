"""Steering and ablation mutators for activation space interventions."""

from collections.abc import Callable
from typing import Any

import torch

MutatorFn = Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]


def additive_steering_mutator(
    vector: torch.Tensor,
    coefficient: float = 1.0,
) -> MutatorFn:
    """Creates a mutator function that applies additive steering: h' = h + c * v.

    Args:
        vector: Direction or concept vector v (shape: [d_model]).
        coefficient: Scaling multiplier c controlling intervention strength.
    """

    def mutator(h: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        # Device & dtype auto-alignment
        v_aligned = vector.to(device=h.device, dtype=h.dtype)

        # Additive steering: h' = h + c * v
        return h + coefficient * v_aligned

    return mutator


def directional_ablation_mutator(
    direction: torch.Tensor,
    eps: float = 1e-8,
) -> MutatorFn:
    """Creates a mutator function that projects out a vector component from h.

    Formula: h' = h - ((h · v) / ||v||^2) * v

    Args:
        direction: Target vector v to ablate/project out (shape: [d_model]).
        eps: Small constant for numerical stability.
    """

    def mutator(h: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        # Device & dtype auto-alignment
        v = direction.to(device=h.device, dtype=h.dtype)
        v_norm_sq = torch.sum(v * v) + eps

        # Project h onto v across the d_model dimension (dim=-1)
        # Keeps dimensions broadcastable: [batch, seq_len, 1]
        proj_scalar = torch.sum(h * v, dim=-1, keepdim=True) / v_norm_sq
        h_parallel = proj_scalar * v

        # Subtract orthogonal component to perform directional ablation
        return h - h_parallel

    return mutator
