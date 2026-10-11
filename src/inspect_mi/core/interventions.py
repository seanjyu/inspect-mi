from collections.abc import Callable
from typing import Any

import torch

from inspect_mi.core.schema import TargetSpec

MutatorFn = Callable[[torch.Tensor, Any], torch.Tensor]
ObserverFn = Callable[[torch.Tensor, Any], None]


def _get_token_slice(target_spec: TargetSpec) -> slice:
    """Helper to convert TokenPolicy into a rank-preserving slice."""
    policy = target_spec.token_policy

    if policy == "all":
        return slice(None)
    elif policy == "last_prompt_token":
        return slice(-1, None)
    elif policy == "first_token":
        return slice(0, 1)
    elif policy == "index":
        idx = target_spec.token_index
        if idx is None:
            raise ValueError("token_index must be set when policy is 'index'")
        return slice(idx, None if idx == -1 else idx + 1)
    elif policy == "slice":
        if target_spec.token_slice is None:
            raise ValueError("token_slice must be set when policy is 'slice'")
        start, stop = target_spec.token_slice
        return slice(start, stop)
    else:
        raise ValueError(f"Unsupported token_policy: {policy}")


def create_steering_mutator(
    steering_vector: torch.Tensor, coeff: float = 1.0
) -> MutatorFn:
    """Factory creating a raw steering vector addition mutator."""

    def raw_steering_mutator(
        tensor: torch.Tensor, spec_or_meta: Any = None
    ) -> torch.Tensor:
        return tensor + (coeff * steering_vector.to(tensor.device, dtype=tensor.dtype))

    return raw_steering_mutator


def wrap_mutator_with_policy(
    mutator_fn: MutatorFn,
    target_spec: TargetSpec,
) -> MutatorFn:
    """Wraps a user MutatorFn to handle token policy slicing automatically."""
    # Use the target_spec captured during wrapper registration
    token_slice = _get_token_slice(target_spec)

    def policy_wrapped_mutator(
        tensor: torch.Tensor, runtime_context: Any = None
    ) -> torch.Tensor:
        # Forward either the explicit target_spec or any runtime context passed by dispatcher
        ctx_to_pass = runtime_context if runtime_context is not None else target_spec

        if tensor.ndim == 3:
            activation_slice = tensor[:, token_slice, :]
            mutated_slice = mutator_fn(activation_slice, ctx_to_pass)
            tensor[:, token_slice, :] = mutated_slice
        elif tensor.ndim == 2:
            activation_slice = tensor[token_slice, :]
            mutated_slice = mutator_fn(activation_slice, ctx_to_pass)
            tensor[token_slice, :] = mutated_slice
        else:
            raise ValueError(
                f"Expected 2D or 3D activation tensor, got shape {tensor.shape}"
            )

        return tensor

    return policy_wrapped_mutator


def wrap_observer_with_policy(
    observer_fn: ObserverFn,
    target_spec: TargetSpec,
) -> ObserverFn:
    """Wraps a user ObserverFn to slice activations by TokenPolicy prior to observation."""
    token_slice = _get_token_slice(target_spec)

    def policy_wrapped_observer(
        tensor: torch.Tensor, runtime_context: Any = None
    ) -> None:
        ctx_to_pass = runtime_context if runtime_context is not None else target_spec

        if tensor.ndim == 3:
            activation_slice = tensor[:, token_slice, :]
        elif tensor.ndim == 2:
            activation_slice = tensor[token_slice, :]
        else:
            raise ValueError(
                f"Expected 2D or 3D activation tensor, got shape {tensor.shape}"
            )

        observer_fn(activation_slice, ctx_to_pass)

    return policy_wrapped_observer
