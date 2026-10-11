import torch

from inspect_mi.core.context import MIContext, bind_context
from inspect_mi.core.hooks import hook_pytorch_model  # Dispatcher context manager
from inspect_mi.core.interventions import (
    create_steering_mutator,
    wrap_mutator_with_policy,
    wrap_observer_with_policy,
)
from inspect_mi.core.resolver import resolve_path
from inspect_mi.core.schema import TargetSpec

# from inspect_mi.core.interventions import (
#     wrap_mutator_with_policy,
#     wrap_observer_with_policy,
# )
# from inspecst_mi.core.schema import CanonicalLocation, CanonicalPoint, TargetSpec, TokenPolicy
from inspect_mi.transforms.probing import activation_harvesting_observer

# def test_dummy_model_steering_and_harvesting(dummy_model):
#     target_path = "transformer.h.0.mlp"
#     target_spec = TargetSpec(
#         point=CanonicalPoint(layer=0, location=CanonicalLocation.MLP_OUT),
#         token_policy=TokenPolicy.LAST_PROMPT_TOKEN,
#     )

#     ctx = MIContext(sample_id="test_sample")
#     ctx.buffer[f"spec:{target_path}"] = target_spec

#     steering_vec = torch.randn(64)
#     raw_mutator = additive_steering_mutator(steering_vec, coefficient=2.0)
#     raw_observer = activation_harvesting_observer(buffer_key="captured_acts")

#     ctx.mutators[target_path] = raw_mutator
#     ctx.observers[target_path] = raw_observer

#     # 1. Resolve the submodule on the dummy model
#     submodule = dummy_model.transformer.h[0].mlp

#     # 2. Wrap functions with token-policy indexing
#     wrapped_mutator = wrap_mutator_with_policy(raw_mutator, target_spec)
#     wrapped_observer = wrap_observer_with_policy(raw_observer, target_spec)

#     # 3. Define the PyTorch forward hook callback
#     def forward_hook(module, inputs, output):
#         # Handle tuple outputs (hidden_states, ...) if applicable
#         act = output[0] if isinstance(output, tuple) else output

#         # Apply mutator followed by observer
#         mutated_act = wrapped_mutator(act, {"target_path": target_path})
#         wrapped_observer(mutated_act, {"target_path": target_path})

#         if isinstance(output, tuple):
#             return (mutated_act, *output[1:])
#         return mutated_act

#     # 4. Register hook, execute forward pass, and clean up handle
#     handle = submodule.register_forward_hook(forward_hook)
#     try:
#         inputs = torch.tensor([[1, 5, 12, 42]])
#         with bind_context(ctx):
#             outputs = dummy_model(inputs)
#     finally:
#         handle.remove()

#     # 5. Assertions
#     assert outputs.logits.shape == (1, 4, 100)
#     assert "captured_acts" in ctx.buffer
#     assert len(ctx.buffer["captured_acts"]) > 0
#     assert ctx.buffer["captured_acts"][0].device.type == "cpu"


# def test_dummy_model_full_framework_trace(dummy_model):
#     # 1. Spec setup
#     # point = CanonicalPoint(layer=0, location=CanonicalLocation.MLP_OUT)
#     # target_spec = TargetSpec(point=point, token_policy=TokenPolicy.LAST_PROMPT_TOKEN)
#     # NEW
#     hook_location = HookLocation(layer=0, location="mlp_out")
#     target_spec = TargetSpec(
#         hook_location=hook_location, token_policy="last_prompt_token"
#     )

#     # 2. Test actual resolver module
#     # target_path = resolve_canonical_path(point, model_type="gpt2")  # -> "transformer.h.0.mlp"
#     # NEW
#     target_path = resolve_path(
#         target_spec, model_type="gpt2"
#     )

#     # 3. Context setup
#     ctx = MIContext(sample_id="test_sample")
#     ctx.targets[target_path] = target_spec
#     ctx.mutators[target_path] = additive_steering_mutator(torch.randn(64), coefficient=2.0)
#     ctx.observers[target_path] = activation_harvesting_observer(buffer_key="captured_acts")

#     # # 4. Forward pass using actual framework hooks dispatcher (no manual register_forward_hook!)
#     # inputs = torch.tensor([[1, 5, 12, 42]])
#     # with bind_context(ctx):
#     #     with hook_pytorch_model(dummy_model, ctx):
#     #         outputs = dummy_model(inputs)

#     # Updated setup prior to step 4
#     hook_location = HookLocation(layer=0, location="mlp_out")
#     target_spec = TargetSpec(hook_location=hook_location, token_policy="last_prompt_token")
#     target_path = resolve_path(target_spec, model_type="gpt2")
#     # Create the raw mutator
#     raw_mutator = create_steering_mutator(steering_vector, coeff=2.0)

#     ctx = MIContext(sample_id="test_0")
#     ctx.targets[target_path] = target_spec
#     ctx.mutators[target_path] = wrap_mutator_with_policy(raw_mutator, target_spec)

#     # 5. Assertions
#     assert outputs.logits.shape == (1, 4, 100)
#     assert "captured_acts" in ctx.buffer
#     assert len(ctx.buffer["captured_acts"]) > 0


def test_dummy_model_full_framework_trace(dummy_model):
    # 1. Spec & Path Resolution Setup
    # hook_location = HookLocation(layer=0, location="mlp_out")
    target_spec = TargetSpec(
        layer=0, location="mlp_out", token_policy="last_prompt_token"
    )
    target_path = resolve_path(target_spec, model_type="gpt2")

    # 2. Interventions & Observer Setup
    steering_vector = torch.randn(64)
    raw_mutator = create_steering_mutator(steering_vector, coeff=2.0)
    wrapped_mutator = wrap_mutator_with_policy(raw_mutator, target_spec)

    raw_observer = activation_harvesting_observer(buffer_key="captured_acts")
    wrapped_observer = wrap_observer_with_policy(raw_observer, target_spec)

    # 3. Context Setup (Single Instance)
    ctx = MIContext(sample_id="test_sample")
    ctx.targets[target_path] = target_spec
    ctx.mutators[target_path] = wrapped_mutator
    ctx.observers[target_path] = wrapped_observer

    # 4. Forward Pass Execution with Hook Context Managers
    inputs = torch.tensor([[1, 5, 12, 42]])
    with bind_context(ctx):
        with hook_pytorch_model(dummy_model, ctx):
            outputs = dummy_model(inputs)

    # 5. Assertions
    assert outputs.logits.shape == (1, 4, 100)
    assert "captured_acts" in ctx.buffer
    assert len(ctx.buffer["captured_acts"]) > 0
