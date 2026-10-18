import contextvars
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType  # TODO do we need this?
from typing import Any, Optional

import torch

from inspect_mi.core.resolver import resolve_path
from inspect_mi.core.schema import TargetSpec, TokenPolicy

MutatorFn = Callable[[torch.Tensor, TargetSpec], torch.Tensor]
ObserverFn = Callable[[torch.Tensor, TargetSpec], None]


@dataclass
class MIContext:
    """Sample-isolated context holding active interventions, observers, and telemetry data."""

    sample_id: str | None = None
    mutators: dict[str, MutatorFn] = field(default_factory=dict)
    observers: dict[str, ObserverFn] = field(default_factory=dict)
    buffer: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, TargetSpec] = field(default_factory=dict)

    def get_mutator(self, target_path: str) -> MutatorFn | None:
        """Retrieves a mutator registered for a specific nnterp path."""
        return self.mutators.get(target_path)

    def get_observer(self, target_path: str) -> ObserverFn | None:
        """Retrieves an observer registered for a specific nnterp path."""
        return self.observers.get(target_path)

    def store(self, key: str, value: Any) -> None:
        """Helper to thread-safely record activation captures or probe outputs."""
        self.buffer[key] = value

    def clear_buffer(self) -> None:
        """Empties the sample buffer store."""
        self.buffer.clear()


active_mi_context: contextvars.ContextVar[Optional["MIContext"]] = (
    contextvars.ContextVar("active_mi_context", default=None)
)


class BoundContextManager:
    """Context manager supporting both synchronous (`with`) and asynchronous (`async with`) bindings."""

    def __init__(self, context: "MIContext"):
        self.context = context
        self._tokens: list[contextvars.Token] = []

    def __enter__(self) -> "MIContext":
        token = active_mi_context.set(self.context)
        self._tokens.append(token)
        return self.context

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._tokens:
            token = self._tokens.pop()
            active_mi_context.reset(token)

    async def __aenter__(self) -> "MIContext":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


def bind_context(context: "MIContext") -> BoundContextManager:
    """Factory helper to bind an MIContext via `with bind_context(ctx):` or `async with bind_context(ctx):`."""
    return BoundContextManager(context)


def get_current_context() -> MIContext | None:
    """Retrieves the active sample context for the current thread/task."""
    return active_mi_context.get()


# Type alias for cleaner function signatures
TargetPoint = tuple[int, str]  # (layer, location) e.g., (0, "mlp_out")


def register_multi_point_intervention(
    ctx: MIContext,
    model_type: str,
    locations: list[TargetPoint],  # Strictly expects [(0, "mlp_out"), (1, "attn_out")]
    token_policy: TokenPolicy = "last_prompt_token",
    mutator: MutatorFn | None = None,
    observer: ObserverFn | None = None,
    **token_policy_kwargs: Any,
) -> list[str]:
    """Registers a single mutator/observer across multiple target points on a model."""
    resolved_paths = []

    for layer, location in locations:
        spec = TargetSpec(
            layer=layer,
            location=location,
            token_policy=token_policy,
            **token_policy_kwargs,
        )
        target_path = resolve_path(spec, model_type=model_type)
        spec.module_path = target_path

        # Bind to active context
        ctx.targets[target_path] = spec
        if mutator is not None:
            ctx.mutators[target_path] = mutator
        if observer is not None:
            ctx.observers[target_path] = observer

        resolved_paths.append(target_path)

    return resolved_paths
