import asyncio

import torch
import torch.nn as nn
from inspect_ai.solver import TaskState

from inspect_mi.core.context import get_current_context
from inspect_mi.core.hooks import hook_pytorch_model  # Dispatcher context manager
from inspect_mi.core.interventions import (
    create_steering_mutator,
    wrap_mutator_with_policy,
    wrap_observer_with_policy,
)
from inspect_mi.core.resolver import resolve_path
from inspect_mi.core.schema import InterventionConfig, TargetSpec
from inspect_mi.core.solver import intervene
from inspect_mi.transforms.probing import activation_harvesting_observer


class DummyGPT2Model(nn.Module):
    """Minimal PyTorch dummy model with HF-style module structure."""

    def __init__(self, d_model=64):
        super().__init__()
        self.transformer = nn.ModuleDict(
            {"h": nn.ModuleList([nn.ModuleDict({"mlp": nn.Linear(d_model, d_model)})])}
        )

    def forward(self, x):
        return self.transformer["h"][0]["mlp"](x)


def run_solver_diagnostic():
    print("\n================ STARTING SOLVER DIAGNOSTIC ================")

    # 1. Target & Model Setup
    model = DummyGPT2Model()
    target_spec = TargetSpec(
        layer=0,
        location="mlp_out",
        token_policy="all",
    )
    config = InterventionConfig(targets=[target_spec])

    # Diagnostic Step A: Path Resolution Matching
    target_path = resolve_path(target_spec, model_type="gpt2")
    model_modules = dict(model.named_modules())

    print("\n[STEP 1] Module Path Matching:")
    print(f"  • Resolved Target Path: '{target_path}'")
    print(f"  • Available Model Modules: {list(model_modules.keys())}")

    if target_path in model_modules:
        print("  --> ✅ MATCH: Target path exists in PyTorch model.")
    else:
        print("  --> ❌ ERROR: Path mismatch! PyTorch hook will NEVER trigger.")
        return

    # 2. Factories with Verbose Logging Wrappers
    def mutator_factory(spec: TargetSpec):
        raw = create_steering_mutator(torch.ones(64), coeff=1.0)
        wrapped = wrap_mutator_with_policy(raw, spec)

        def logging_mutator(tensor, spec_arg):
            print("  --> 🔥 [HOOK EVENT] Mutator executed!")
            return wrapped(tensor, spec_arg)

        return logging_mutator

    def observer_factory(spec: TargetSpec):
        raw = activation_harvesting_observer(buffer_key="captured_acts")
        wrapped = wrap_observer_with_policy(raw, spec)

        def logging_observer(tensor, spec_arg):
            print("  --> 👁️  [HOOK EVENT] Observer executed!")
            return wrapped(tensor, spec_arg)

        return logging_observer

    # 3. TaskState & Mock Generate
    state = TaskState(
        sample_id="debug_sample_123",
        model="gpt2",
        input="debug prompt",
        epoch=1,
        messages=[],
    )

    async def mock_generate(state_arg: TaskState) -> TaskState:
        ctx = get_current_context()
        print("\n[STEP 2] Inside mock_generate Generation Loop:")
        print(f"  • Active Context Sample ID: {ctx.sample_id if ctx else None}")
        print(f"  • Registered Mutators: {list(ctx.mutators.keys())}")
        print(f"  • Registered Observers: {list(ctx.observers.keys())}")

        inputs = torch.randn(1, 4, 64)
        print("\n[STEP 3] Executing Model Forward Pass...")
        with hook_pytorch_model(model, ctx):
            _ = model(inputs)

        print("\n[STEP 4] Forward Pass Complete:")
        print(f"  • Keys stored in ctx.buffer: {list(ctx.buffer.keys())}")
        return state_arg

    # 4. Instantiate & Run Solver
    print("\n[STEP 5] Running Solver...")
    solver_fn = intervene(
        config=config,
        mutator_factory=mutator_factory,
        observer_factory=observer_factory,
    )

    # Run the async solver directly using asyncio
    final_state = asyncio.run(solver_fn(state, mock_generate))

    # 5. Final Assertions Verification
    print("\n[STEP 6] Final State Inspection:")
    captured = final_state.store.get("mi:captured_acts")

    if captured is not None:
        print(
            "  --> ✅ SUCCESS: Harvested tensor found in final_state.store['mi:captured_acts']!"
        )
        print(f"      Tensor type/shape: {type(captured)}")
    else:
        print("  --> ❌ FAILURE: final_state.store['mi:captured_acts'] is None!")

    unbound_ctx = get_current_context()
    if unbound_ctx is None:
        print("  --> ✅ SUCCESS: Context correctly unbound after completion.")
    else:
        print(f"  --> ❌ WARNING: Context remains bound: {unbound_ctx}")

    print("\n================ DIAGNOSTIC COMPLETE ================\n")


if __name__ == "__main__":
    run_solver_diagnostic()
