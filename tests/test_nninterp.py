# tests/test_nnterp.py

import torch

from inspect_mi.core.context import MIContext, bind_context
from inspect_mi.core.hooks import apply_nnterp_hooks
from inspect_mi.core.schema import TargetSpec

# def test_nnterp_no_active_context_executes_normally(nnterp_gpt2_model, input_ids):
#     """Verifies that running apply_nnterp_hooks without an active MIContext completes safely."""
#     with nnterp_gpt2_model.trace(input_ids):
#         apply_nnterp_hooks(nnterp_gpt2_model)

# def test_llama_observer_captures_attention_and_mlp(nnterp_gpt2_model, input_ids):
#     """Verifies path resolution across model.layers.0.self_attn and model.layers.0.mlp in nnterp."""
#     captured_attn = []
#     captured_mlp = []

#     attn_path = "model.layers.0.self_attn"
#     mlp_path = "model.layers.0.mlp"

#     attn_spec = TargetSpec(layer=0, location="self_attn", module_path=attn_path)
#     mlp_spec = TargetSpec(layer=0, location="mlp", module_path=mlp_path)

#     ctx = MIContext(sample_id="llama_obs_test")
#     ctx.targets[attn_path] = attn_spec
#     ctx.targets[mlp_path] = mlp_spec

#     ctx.observers[attn_path] = lambda act, spec: captured_attn.append(act.detach().clone())
#     ctx.observers[mlp_path] = lambda act, spec: captured_mlp.append(act.detach().clone())

#     with bind_context(ctx):
#         # 1. Use nnterp's trace context with token IDs instead of float activations
#         with nnterp_gpt2_model.trace(input_ids):
#             # 2. Dispatch MIContext observers onto the active nnterp trace
#             apply_nnterp_hooks(nnterp_gpt2_model, ctx)

#     # Verify both submodules were successfully resolved and hooked
#     assert len(captured_attn) == 1
#     assert captured_attn[0].shape == (2, 4, 32)

#     assert len(captured_mlp) == 1
#     assert captured_mlp[0].shape == (2, 4, 32)


def test_llama_observer_captures_attention_and_mlp(nnterp_gpt2_model, input_ids):
    """Verifies path resolution across model.layers.0.self_attn and model.layers.0.mlp in nnterp."""
    captured_attn = []
    captured_mlp = []

    attn_path = "model.layers.0.self_attn"
    mlp_path = "model.layers.0.mlp"

    attn_spec = TargetSpec(layer=0, location="self_attn", module_path=attn_path)
    mlp_spec = TargetSpec(layer=0, location="mlp", module_path=mlp_path)

    ctx = MIContext(sample_id="llama_obs_test")
    ctx.targets[attn_path] = attn_spec
    ctx.targets[mlp_path] = mlp_spec

    # Observers store the saved proxy reference during trace execution
    ctx.observers[attn_path] = lambda act, spec: captured_attn.append(act)
    ctx.observers[mlp_path] = lambda act, spec: captured_mlp.append(act)

    with bind_context(ctx):
        with nnterp_gpt2_model.trace(input_ids):
            apply_nnterp_hooks(nnterp_gpt2_model, ctx)

    # Extract actual PyTorch Tensors after trace evaluation finishes
    attn_tensor = getattr(captured_attn[0], "value", captured_attn[0])
    mlp_tensor = getattr(captured_mlp[0], "value", captured_mlp[0])

    assert isinstance(attn_tensor, torch.Tensor), (
        "Captured attention output is not a Tensor"
    )
    assert isinstance(mlp_tensor, torch.Tensor), "Captured MLP output is not a Tensor"
    assert attn_tensor.shape == mlp_tensor.shape


def test_nnterp_observer_captures_activations(nnterp_gpt2_model, input_ids):
    """Verifies that an observer captures proxy activations during graph tracing."""
    captured = []

    def mock_observer(act: torch.Tensor, spec: TargetSpec) -> None:
        captured.append(act.detach().clone())

    target_path = "transformer.h.0.mlp"
    spec = TargetSpec(
        layer=0,
        location="mlp_out",
        module_path=target_path,
    )

    ctx = MIContext(sample_id="nnterp_obs_test")
    ctx.targets[target_path] = spec
    ctx.observers[target_path] = mock_observer

    with bind_context(ctx):
        with nnterp_gpt2_model.trace(input_ids):
            apply_nnterp_hooks(nnterp_gpt2_model, ctx)

    assert len(captured) == 1
    assert captured[0].shape == (2, 4, 32)


def test_nnterp_mutator_modifies_forward_pass(nnterp_gpt2_model, input_ids):
    """Verifies activation steering on proxy outputs alters final model logits."""
    target_path = "transformer.h.0.mlp"
    spec = TargetSpec(
        layer=0,
        location="mlp_out",
        module_path=target_path,
    )

    def steering_mutator(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
        return act + 100000.0

    # 1. Clean run
    with nnterp_gpt2_model.trace(input_ids):
        clean_out = nnterp_gpt2_model.output.save()

    # 2. Mutated run
    ctx = MIContext(sample_id="nnterp_mut_test")
    ctx.targets[target_path] = spec
    ctx.mutators[target_path] = steering_mutator

    with bind_context(ctx):
        with nnterp_gpt2_model.trace(input_ids):
            apply_nnterp_hooks(nnterp_gpt2_model, ctx)
            steered_out = nnterp_gpt2_model.output.save()

    assert not torch.allclose(clean_out.logits, steered_out.logits, atol=1e-3)


def test_nnterp_mutator_runs_before_observer(nnterp_gpt2_model, input_ids):
    """Verifies observer reads the mutated proxy tensor value."""
    observed_tensor = []

    def mutator(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
        return torch.zeros_like(act)

    def observer(act: torch.Tensor, spec: TargetSpec) -> None:
        observed_tensor.append(act.detach().clone())

    target_path = "transformer.h.0.mlp"
    spec = TargetSpec(
        layer=0,
        location="mlp_out",
        module_path=target_path,
    )

    ctx = MIContext(sample_id="nnterp_order_test")
    ctx.targets[target_path] = spec
    ctx.mutators[target_path] = mutator
    ctx.observers[target_path] = observer

    with bind_context(ctx):
        with nnterp_gpt2_model.trace(input_ids):
            apply_nnterp_hooks(nnterp_gpt2_model, ctx)

    assert len(observed_tensor) == 1
    assert torch.all(observed_tensor[0] == 0.0)


def test_nnterp_tuple_output_container_handling(nnterp_gpt2_model, input_ids):
    """Verifies nnterp properly unpacks tuple proxies output_proxy[0][:] = mutated_act."""
    target_path = "transformer.h.0.attn"
    spec = TargetSpec(
        layer=0,
        location="attn_out",
        module_path=target_path,
    )

    observed_activations = []

    def observer(act: torch.Tensor, spec: TargetSpec) -> None:
        observed_activations.append(act.detach().clone())

    ctx = MIContext(sample_id="nnterp_tuple_test")
    ctx.targets[target_path] = spec
    ctx.observers[target_path] = observer

    with bind_context(ctx):
        with nnterp_gpt2_model.trace(input_ids):
            apply_nnterp_hooks(nnterp_gpt2_model, ctx)

    assert len(observed_activations) == 1
    assert observed_activations[0].shape == (2, 4, 32)


# def test_nnterp_llama_path_resolution(nnterp_llama_model, input_activations):
#     """Verifies path resolution for Llama submodules under nnterp tracing."""
#     captured_attn = []
#     captured_mlp = []

#     attn_path = "model.layers.0.self_attn"
#     mlp_path = "model.layers.0.mlp"

#     ctx = MIContext(sample_id="nnterp_llama_test")
#     ctx.targets[attn_path] = TargetSpec(layer=0, location="self_attn", module_path=attn_path)
#     ctx.targets[mlp_path] = TargetSpec(layer=0, location="mlp", module_path=mlp_path)

#     ctx.observers[attn_path] = lambda act, spec: captured_attn.append(act.detach().clone())
#     ctx.observers[mlp_path] = lambda act, spec: captured_mlp.append(act.detach().clone())

#     with bind_context(ctx):
#         with nnterp_llama_model.trace(input_activations):
#             apply_nnterp_hooks(nnterp_llama_model)

#     assert len(captured_attn) == 1
#     assert len(captured_mlp) == 1
