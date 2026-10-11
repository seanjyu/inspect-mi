"""Observer functions for linear probing and activation harvesting."""

from collections.abc import Callable
from typing import Any

import torch

from inspect_mi.core.context import get_current_context

ObserverFn = Callable[[torch.Tensor, dict[str, Any]], None]


def probe_observer(
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    buffer_key: str = "probe_outputs",
) -> ObserverFn:
    """Creates an observer function for on-the-fly probe evaluation: p = sigmoid(W^T h + b).

    Evaluates predictions at each step and appends CPU probabilities to `context.buffer`.

    Args:
        weight: Linear probe weights tensor W (shape: [d_model] or [d_model, num_classes]).
        bias: Optional bias vector b.
        buffer_key: Key in `context.buffer` to store step-wise probe results.
    """

    def observer(h: torch.Tensor, metadata: dict[str, Any]) -> None:
        ctx = get_current_context()
        if ctx is None:
            return

        w = weight.to(device=h.device, dtype=h.dtype)
        b = bias.to(device=h.device, dtype=h.dtype) if bias is not None else None

        # Compute probe logits: h @ W + b
        logits = torch.matmul(h, w)
        if b is not None:
            logits = logits + b

        # Step-wise probability calculation
        probs = torch.sigmoid(logits)

        # VRAM Guard: Detach from compute graph and transfer to host CPU memory
        probs_cpu = probs.detach().cpu()

        if buffer_key not in ctx.buffer:
            ctx.buffer[buffer_key] = []
        ctx.buffer[buffer_key].append(probs_cpu)

    return observer


def activation_harvesting_observer(
    buffer_key: str = "harvested_activations",
) -> ObserverFn:
    """Creates an observer function that captures and buffers raw activations.

    Enforces a strict VRAM Guard (.detach().cpu()) to prevent CUDA graph retention
    and memory leaks during multi-sample evaluation loops.

    Args:
        buffer_key: Key in `context.buffer` to store activation tensors.
    """

    def observer(h: torch.Tensor, metadata: dict[str, Any]) -> None:
        ctx = get_current_context()
        if ctx is None:
            return

        # VRAM Guard: Sever autograd graph & copy tensor to system RAM
        h_cpu = h.detach().cpu()

        if buffer_key not in ctx.buffer:
            ctx.buffer[buffer_key] = []
        ctx.buffer[buffer_key].append(h_cpu)

    return observer
