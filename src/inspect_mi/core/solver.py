from collections.abc import Callable
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver

from inspect_mi.core.context import MIContext, bind_context
from inspect_mi.core.resolver import resolve_path
from inspect_mi.core.schema import InterventionConfig


@solver
def intervene(
    config: InterventionConfig,
    mutator_factory: Callable[[dict[str, Any]], Any] | None = None,
    observer_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> Solver:
    """Inspect AI Solver using native nnterp model provider integration."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        ctx = MIContext(sample_id=str(state.sample_id))

        for target in config.targets:
            # Resolve canonical point to nnterp target path string
            target_path = target_path = resolve_path(target)
            ctx.targets[target_path] = target
            ctx.buffer[f"spec:{target_path}"] = target

            # if config.method_type == "steering" and mutator_factory:
            #     ctx.mutators[target_path] = mutator_factory(target.metadata)
            # elif config.method_type == "probing" and observer_factory:
            #     ctx.observers[target_path] = observer_factory(target.metadata)

            if mutator_factory is not None:
                ctx.mutators[target_path] = mutator_factory(target)  #

            if observer_factory is not None:
                ctx.observers[target_path] = observer_factory(target)  #

        # Bind context; inspect's nnterp provider executes hooks during generation
        with bind_context(ctx):
            state = await generate(state)

        # Export captured telemetry from MIContext buffer to state.store
        for key, value in ctx.buffer.items():
            state.store.set(f"mi:{key}", value)

        return state

    return solve
