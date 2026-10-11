"""Hook dispatcher bridging MIContext with nnterp tracing and PyTorch models."""

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from inspect_mi.core.context import MIContext, active_mi_context, get_current_context
from inspect_mi.core.interventions import (
    wrap_mutator_with_policy,
    wrap_observer_with_policy,
)
from inspect_mi.core.schema import TargetSpec

logger = logging.getLogger("inspect_mi")


def _extract_activation_tensor(output: Any) -> tuple[torch.Tensor, str]:
    """Helper to extract the primary activation tensor and its container type."""
    if isinstance(output, torch.Tensor):
        return output, "tensor"
    if isinstance(output, tuple) and len(output) > 0:
        return output[0], "tuple"
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state, "model_output"
    raise TypeError(f"Unsupported module output type for hook: {type(output)}")


@contextmanager
def bind_context(ctx: MIContext) -> Generator[MIContext, None, None]:
    """Binds an MIContext instance to the active thread/coroutine context variable."""
    token = active_mi_context.set(ctx)
    try:
        yield ctx
    finally:
        active_mi_context.reset(token)


# TODO add verbose
@contextmanager
def hook_pytorch_model(
    model: nn.Module, ctx: MIContext | None = None
) -> Generator[nn.Module, None, None]:
    """Context manager for running standard PyTorch models with MIContext hooks.

    Attaches standard PyTorch forward hooks based on active mutators and observers
    in the context. Enables fast offline unit testing without nnterp tracing.
    """
    if ctx is None:
        # ctx = active_mi_context.get()
        ctx = get_current_context()
    if ctx is None:
        logger.debug("No active MIContext found within hook_pytorch_model")
        yield model
        return

    handles = []
    all_target_paths = set(ctx.mutators.keys()) | set(ctx.observers.keys())
    print(f"[DEBUG HOOK] Targets to hook: {all_target_paths}")
    logger.debug(f"Registering PyTorch hooks for targets: {all_target_paths}")

    for target_path in all_target_paths:
        target_spec: TargetSpec | None = ctx.targets.get(target_path)
        if target_spec is None:
            # print(f"[DEBUG HOOK] ❌ target_spec is None for '{target_path}' (ctx.targets keys: {list(ctx.targets.keys())})")
            logger.warning(
                f"TargetSpec missing for path '{target_path}'. Skipping hook."
            )
            continue

        mutator = ctx.mutators.get(target_path)
        observer = ctx.observers.get(target_path)

        wrapped_mutator = (
            wrap_mutator_with_policy(mutator, target_spec) if mutator else None
        )
        wrapped_observer = (
            wrap_observer_with_policy(observer, target_spec) if observer else None
        )

        try:
            submodule = _resolve_module_path(model, target_path)
            # print(f"[DEBUG HOOK] ✅ Module resolved for '{target_path}': {type(submodule)}")
            logger.debug(
                f"Resolved module for '{target_path}': {type(submodule).__name__}"
            )
        except (AttributeError, KeyError, IndexError) as e:
            # print(f"[DEBUG HOOK] ❌ Module resolution failed for '{target_path}'")
            logger.error(f"Failed to resolve module path '{target_path}': {e}")
            continue

        def make_hook(w_mut, w_obs, spec):
            def forward_hook(module: nn.Module, inputs: Any, output: Any) -> Any:
                # print(f"[DEBUG HOOK] 🔥 FORWARD HOOK FIRED on {target_path}!")
                # logger.debug(f"Forward hook fired on target: '{path}'")
                is_tuple = isinstance(output, tuple)
                act_tensor = output[0] if is_tuple else output

                mutated_act = act_tensor
                if w_mut is not None:
                    mutated_act = w_mut(act_tensor, spec)

                if w_obs is not None:
                    w_obs(mutated_act, spec)

                if is_tuple:
                    return (mutated_act, *output[1:])
                return mutated_act

            return forward_hook

        hook_fn = make_hook(wrapped_mutator, wrapped_observer, target_spec)
        handle = submodule.register_forward_hook(hook_fn)
        handles.append(handle)
        # print(f"[DEBUG HOOK] 🚀 Successfully registered hook on '{target_path}'")
        logger.debug(f"Successfully attached forward hook to '{target_path}'")

    try:
        yield model
    finally:
        for handle in handles:
            handle.remove()
        logger.debug(f"Detached {len(handles)} PyTorch forward hooks.")


import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _resolve_module_path(model: Any, path: str) -> Any:
    """Resolves canonical nnterp paths ('layers.0.mlp', 'layers.0.attention')

    directly against an nnterp StandardizedTransformer wrapper.
    """
    curr = model
    for part in path.split("."):
        if part.isdigit():
            curr = curr[int(part)]
        else:
            if not hasattr(curr, part):
                raise AttributeError(
                    f"[inspect_mi] Submodule '{part}' not found on '{type(curr).__name__}'. "
                    f"Ensure target paths follow nnterp canonical format (e.g., 'layers.0.mlp')."
                )
            curr = getattr(curr, part)
    return curr


def apply_nnterp_hooks(model: Any, ctx: MIContext) -> None:
    if ctx is None:
        return

    def _get_execution_order(path: str) -> tuple[int, int]:
        spec = ctx.targets.get(path)
        layer_idx = 0
        if spec and getattr(spec, "layer", None) is not None:
            layer_idx = spec.layer
        else:
            match = re.search(r"\d+", path)
            layer_idx = int(match.group()) if match else 0

        # Submodule priority within the same layer (Attention runs before MLP)
        path_lower = path.lower()
        if any(k in path_lower for k in ["attn", "attention"]):
            sub_priority = 0
        elif any(k in path_lower for k in ["mlp", "feed_forward", "ffn"]):
            sub_priority = 1
        else:
            sub_priority = 2

        return (layer_idx, sub_priority)

    sorted_target_paths = sorted(
        set(ctx.mutators.keys()) | set(ctx.observers.keys()), key=_get_execution_order
    )

    for target_path in sorted_target_paths:
        target_spec = ctx.targets.get(target_path)
        submodule = _resolve_module_path(model, target_path)

        if not hasattr(type(submodule), "output") and not hasattr(submodule, "_output"):
            raise AttributeError(
                f"[inspect_mi] Submodule '{target_path}' is not a valid nnsight Envoy. "
                "Ensure apply_nnterp_hooks is called inside 'with model.trace(...):'."
            )

        out_proxy = submodule.output

        # Strict tuple check for nnsight proxies
        is_tuple_output = (
            isinstance(out_proxy, tuple)
            and not hasattr(out_proxy, "shape")
            and not hasattr(out_proxy, "dtype")
        )

        act_tensor = out_proxy[0] if is_tuple_output else out_proxy

        # 1. Pre-Observers
        pre_observer = getattr(ctx, "pre_observers", {}).get(target_path)
        if pre_observer is not None:
            wrapped_pre = wrap_observer_with_policy(pre_observer, target_spec)
            wrapped_pre(act_tensor.save(), target_spec)

        # 2. Mutators
        mutator = ctx.mutators.get(target_path)
        if mutator is not None:
            wrapped_mutator = wrap_mutator_with_policy(mutator, target_spec)
            mutated_act = wrapped_mutator(act_tensor, target_spec)

            if is_tuple_output:
                submodule.output[0] = mutated_act
            else:
                submodule.output = mutated_act

            act_tensor = mutated_act

        # 3. Post-Observers
        observer = ctx.observers.get(target_path)
        if observer is not None:
            wrapped_observer = wrap_observer_with_policy(observer, target_spec)
            wrapped_observer(act_tensor.save(), target_spec)
