from collections.abc import Callable
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver

from inspect_mi.core.context import MIContext, bind_context
from inspect_mi.core.resolver import resolve_path
from inspect_mi.core.schema import InterventionConfig, TargetSpec


@solver
def intervene(
    config: InterventionConfig,
    mutator_factory: Callable[[TargetSpec], Any] | None = None,
    observer_factory: Callable[[TargetSpec], Any] | None = None,
) -> Solver:
    """Inspect AI Solver using native nnterp model provider integration.

    Note: Requires the nnterp model provider; other providers will not apply hooks.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        ctx = MIContext(sample_id=str(state.sample_id))

        for target in config.targets:
            # 1. Fixed duplicate assignment (Item 2)
            target_path = resolve_path(target)
            ctx.targets[target_path] = target

            # 2. Export JSON-serializable dict instead of raw Pydantic model (Item 3)
            ctx.buffer[f"spec:{target_path}"] = target.model_dump()

            if mutator_factory is not None:
                ctx.mutators[target_path] = mutator_factory(target)

            if observer_factory is not None:
                ctx.observers[target_path] = observer_factory(target)

        # Bind context; inspect's nnterp provider executes hooks during generation
        with bind_context(ctx):
            state = await generate(state)

        # 3. Fail loudly if mutators/observers were configured but zero hooks fired (Item 4)
        has_interventions = bool(ctx.mutators or ctx.observers)
        hooks_fired = ctx.buffer.get("hooks_applied", False)

        if has_interventions and not hooks_fired:
            raise RuntimeError(
                f"[inspect_mi] Intervention configured for targets {list(ctx.targets.keys())}, "
                "but no activation hooks fired during generation. "
                "Ensure you are using the 'nnterp' model provider and that module paths match the model architecture."
            )

        # Export captured telemetry from MIContext buffer to state.store
        for key, value in ctx.buffer.items():
            state.store.set(f"mi:{key}", value)

        return state

    return solve
