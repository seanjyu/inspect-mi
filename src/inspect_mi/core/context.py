import contextvars
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType  # TODO do we need this?
from typing import Any, Optional

import torch

# from inspect_mi.core.context import MIContext
from inspect_mi.core.resolver import resolve_path
from inspect_mi.core.schema import TargetSpec, TokenPolicy

# from inspect_mi.core.interventions import MutatorFn, ObserverFn


# # Type signatures for interventions and monitoring
# # Mutator: (activation_tensor, metadata) -> modified_activation_tensor
# MutatorFn = Callable[[torch.Tensor, Dict[str, Any]], torch.Tensor]
# # Observer: (activation_tensor, metadata) -> side-effect (e.g., store in buffer)
# ObserverFn = Callable[[torch.Tensor, Dict[str, Any]], None]

# This one has TargetSpec
# Type aliases for intervention functions
# MutatorFn: takes (tensor, target_spec) -> mutated_tensor
MutatorFn = Callable[[torch.Tensor, TargetSpec], torch.Tensor]
# ObserverFn: takes (tensor, target_spec) -> None (side-effects like writing to ctx.buffer)
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


# # Global async/thread-isolated context variable
# active_mi_context: contextvars.ContextVar[Optional[MIContext]] = contextvars.ContextVar(
#     "active_mi_context", default=None
# )

# class BoundContextManager:
#     """Context manager that safely binds an MIContext to the current asyncio task lifecycle."""
#     def __init__(self, context: MIContext):
#         self.context = context
#         self._token: Optional[contextvars.Token] = None

#     def __enter__(self) -> MIContext:
#         self._token = active_mi_context.set(self.context)
#         return self.context

#     def __exit__(self, exc_type, exc_val, exc_tb) -> None:
#         if self._token is not None:
#             active_mi_context.reset(self._token)
#             self._token = None


# def bind_context(context: MIContext) -> BoundContextManager:
#     """Factory helper to bind an MIContext via `with bind_context(ctx):` blocks."""
#     return BoundContextManager(context)


# def get_current_context() -> Optional[MIContext]:
#     """Retrieves the active sample context if executed within an active intervention task."""
#     return active_mi_context.get()

# class BoundContextManager:
#     """Context manager supporting both synchronous (`with`) and asynchronous (`async with`) bindings."""

#     def __init__(self, context: MIContext):
#         self.context = context
#         self._token: Optional[contextvars.Token] = None

#     def __enter__(self) -> MIContext:
#         self._token = active_mi_context.set(self.context)
#         return self.context

#     def __exit__(
#         self,
#         exc_type: Optional[type],
#         exc_val: Optional[BaseException],
#         exc_tb: Optional[TracebackType],
#     ) -> None:
#         if self._token is not None:
#             active_mi_context.reset(self._token)
#             self._token = None

#     async def __aenter__(self) -> MIContext:
#         return self.__enter__()

#     async def __aexit__(
#         self,
#         exc_type: Optional[type],
#         exc_val: Optional[BaseException],
#         exc_tb: Optional[TracebackType],
#     ) -> None:
#         self.__exit__(exc_type, exc_val, exc_tb)


# def bind_context(context: MIContext) -> BoundContextManager:
#     """Factory helper to bind an MIContext via `with bind_context(ctx):` or `async with bind_context(ctx):`."""
#     return BoundContextManager(context)

# Guarantee default=None so .get() never raises LookupError
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


# # added later
# def register_multi_point_intervention(
#     ctx: MIContext,
#     model_type: str,
#     locations: List[HookLocation],
#     token_policy: str = "last_prompt_token",
#     mutator: Optional[MutatorFn] = None,
#     observer: Optional[ObserverFn] = None,
#     **token_policy_kwargs: Any,
# ) -> List[str]:
#     """Registers a single mutator and/or observer across multiple target points on a model."""
#     resolved_paths = []

#     for loc in locations:
#         spec = TargetSpec(
#             hook_location=loc,
#             token_policy=token_policy,
#             **token_policy_kwargs,
#         )
#         target_path = resolve_path(spec, model_type=model_type)

#         ctx.targets[target_path] = spec
#         if mutator is not None:
#             ctx.mutators[target_path] = mutator
#         if observer is not None:
#             ctx.observers[target_path] = observer

#         resolved_paths.append(target_path)

#     return resolved_paths

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
