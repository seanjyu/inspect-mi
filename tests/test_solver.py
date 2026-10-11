from unittest.mock import MagicMock

import pytest
import torch
from inspect_ai.solver import TaskState

from inspect_mi.core.context import get_current_context
from inspect_mi.core.hooks import hook_pytorch_model
from inspect_mi.core.interventions import (
    create_steering_mutator,
    wrap_mutator_with_policy,
    wrap_observer_with_policy,
)
from inspect_mi.core.schema import TargetSpec
from inspect_mi.core.solver import intervene
from inspect_mi.transforms.probing import activation_harvesting_observer


class DummyConfig:
    """Mock config container with target specs."""

    def __init__(self, targets):
        self.targets = targets
        self.method_type = "steering"


@pytest.mark.asyncio
async def test_mi_intervene_solver_full_trace(dummy_model):
    # 1. Spec & Config Setup
    target_spec = TargetSpec(
        layer=0, location="mlp_out", token_policy="last_prompt_token"
    )
    config = DummyConfig(targets=[target_spec])

    # 2. Mutator & Observer Factories
    steering_vector = torch.randn(64)

    def mutator_factory(spec: TargetSpec):
        raw = create_steering_mutator(steering_vector, coeff=2.0)
        return wrap_mutator_with_policy(raw, spec)

    def observer_factory(spec: TargetSpec):
        raw = activation_harvesting_observer(buffer_key="captured_acts")
        return wrap_observer_with_policy(raw, spec)

    # 3. Mock Inspect AI TaskState & Model
    mock_model = MagicMock()
    mock_model.name = "gpt2"

    state = TaskState(
        sample_id="test_solver_sample",
        model="gpt2",
        epoch=1,
        input="test input",
        messages=[],
    )

    # 4. Mock Inspect AI Generate Function
    async def mock_generate(state: TaskState) -> TaskState:
        # Verify solver bound the active context during generate call
        ctx = get_current_context()
        assert ctx is not None
        assert ctx.sample_id == "test_solver_sample"
        assert len(ctx.mutators) == 1
        assert len(ctx.observers) == 1

        # Simulate model forward pass execution inside Inspect's generation loop
        inputs = torch.tensor([[1, 5, 12, 42]])
        with hook_pytorch_model(dummy_model, ctx):
            _ = dummy_model(inputs)

        return state

    # 5. Instantiate and Execute Solver
    solver_fn = intervene(
        config=config,
        mutator_factory=mutator_factory,
        observer_factory=observer_factory,
    )

    final_state = await solver_fn(state, mock_generate)

    # 6. Assertions
    # Check that harvested activations were exported to state.store
    captured_acts = final_state.store.get("mi:captured_acts")
    assert captured_acts is not None
    assert len(captured_acts) > 0

    # Verify context variable was cleanly unbound after completion
    assert get_current_context() is None
