import torch

from inspect_mi.core.context import MIContext, bind_context
from inspect_mi.core.hooks import hook_pytorch_model
from inspect_mi.core.schema import TargetSpec


def test_no_active_context_executes_normally(gpt2_model, input_tokens):
    """Verifies that running without an active MIContext leaves model execution untouched."""
    with hook_pytorch_model(gpt2_model):
        output = gpt2_model(input_tokens)

    assert isinstance(output, torch.Tensor)


def test_observer_captures_activations(gpt2_model, input_tokens):
    """Verifies that an observer receives activation tensors during a forward pass."""
    captured = []

    def mock_observer(act: torch.Tensor, spec: TargetSpec) -> None:
        captured.append(act.detach().clone())

    target_path = "transformer.h.0.mlp"
    spec = TargetSpec(
        layer=0,
        location="mlp_out",
        module_path=target_path,
    )

    ctx = MIContext(sample_id="obs_test")
    ctx.targets[target_path] = spec
    ctx.observers[target_path] = mock_observer

    with bind_context(ctx):
        with hook_pytorch_model(gpt2_model):
            _ = gpt2_model(input_tokens)

    assert len(captured) == 1
    # Check tensor shape matches expected [batch=2, seq=4, d_model=16]
    assert captured[0].shape == (2, 4, 16)


def test_mutator_modifies_forward_pass(gpt2_model, input_tokens):
    """Verifies that a mutator alters internal activations and changes the final output."""
    target_path = "transformer.h.0.mlp"
    spec = TargetSpec(
        layer=0,
        location="mlp_out",
        module_path=target_path,
    )

    def steering_mutator(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
        return act + 100.0  # Massive injection to ensure logit drift

    # 1. Clean forward pass
    clean_logits = gpt2_model(input_tokens)

    # 2. Mutated forward pass
    ctx = MIContext(sample_id="mut_test")
    ctx.targets[target_path] = spec
    ctx.mutators[target_path] = steering_mutator

    with bind_context(ctx):
        with hook_pytorch_model(gpt2_model, ctx):
            steered_logits = gpt2_model(input_tokens)

    # Output should differ significantly
    assert not torch.allclose(clean_logits, steered_logits, atol=1e-3)


def test_mutator_runs_before_observer(gpt2_model, input_tokens):
    """Verifies that when both are attached, Observer reads the mutated activation."""
    observed_tensor = []

    def mutator(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
        return torch.zeros_like(act)  # Zero out activations

    def observer(act: torch.Tensor, spec: TargetSpec) -> None:
        observed_tensor.append(act.detach().clone())

    target_path = "transformer.h.0.mlp"
    spec = TargetSpec(
        layer=0,
        location="mlp_out",
        module_path=target_path,
    )

    ctx = MIContext(sample_id="order_test")
    ctx.targets[target_path] = spec
    ctx.mutators[target_path] = mutator
    ctx.observers[target_path] = observer

    with bind_context(ctx):
        with hook_pytorch_model(gpt2_model):
            _ = gpt2_model(input_tokens)

    assert len(observed_tensor) == 1
    # Observer must observe the zeroed tensor returned by the mutator
    assert torch.all(observed_tensor[0] == 0.0)


def test_tuple_output_container_handling(gpt2_model, input_tokens):
    """Verifies submodules returning tuples (hidden_states, extra_tensor) are preserved."""
    target_path = "transformer.h.0.attn"

    # Flat TargetSpec instantiation
    spec = TargetSpec(
        layer=0,
        location="attn_out",
        module_path=target_path,
    )

    observed_activations = []

    def observer(act: torch.Tensor, spec: TargetSpec) -> None:
        observed_activations.append(act.detach().clone())

    ctx = MIContext(sample_id="tuple_test")
    ctx.targets[target_path] = spec
    ctx.observers[target_path] = observer

    with bind_context(ctx):
        with hook_pytorch_model(gpt2_model, ctx):
            _ = gpt2_model(input_tokens)

    # Observer receives the first element of the tuple [batch=2, seq=4, d_model=16]
    assert len(observed_activations) == 1
    assert observed_activations[0].shape == (2, 4, 16)


def test_hook_cleanup_on_exit(gpt2_model, input_tokens):
    """Verifies forward hook handles are removed after exiting hook_pytorch_model."""
    target_path = "transformer.h.0.mlp"
    spec = TargetSpec(
        layer=0,
        location="mlp_out",
        module_path=target_path,
    )

    submodule = gpt2_model.transformer.h[0].mlp

    ctx = MIContext(sample_id="cleanup_test")
    ctx.targets[target_path] = spec
    ctx.observers[target_path] = lambda act, spec: None

    # Before block: no hooks registered
    assert len(submodule._forward_hooks) == 0

    with bind_context(ctx):
        with hook_pytorch_model(gpt2_model):
            # Inside block: 1 hook registered
            assert len(submodule._forward_hooks) == 1

    # After exiting block: hooks cleanly removed
    assert len(submodule._forward_hooks) == 0


def test_missing_target_spec_handled_gracefully(gpt2_model, input_tokens):
    """Verifies target paths without matching TargetSpec in ctx.targets skip cleanly."""
    ctx = MIContext(sample_id="missing_spec")
    # Add a mutator without putting a TargetSpec in ctx.targets
    ctx.mutators["transformer.h.0.mlp"] = lambda act, spec: act

    with bind_context(ctx):
        with hook_pytorch_model(gpt2_model):
            # Should proceed without raising KeyError or AttributeError
            out = gpt2_model(input_tokens)

    assert isinstance(out, torch.Tensor)


def test_llama_observer_captures_attention_and_mlp(llama_model, input_tokens):
    """Verifies path resolution across model.layers.0.self_attn and model.layers.0.mlp."""
    captured_attn = []
    captured_mlp = []

    attn_path = "model.layers.0.self_attn"
    mlp_path = "model.layers.0.mlp"

    attn_spec = TargetSpec(layer=0, location="self_attn", module_path=attn_path)
    mlp_spec = TargetSpec(layer=0, location="mlp", module_path=mlp_path)

    ctx = MIContext(sample_id="llama_obs_test")
    ctx.targets[attn_path] = attn_spec
    ctx.targets[mlp_path] = mlp_spec

    ctx.observers[attn_path] = lambda act, spec: captured_attn.append(
        act.detach().clone()
    )
    ctx.observers[mlp_path] = lambda act, spec: captured_mlp.append(
        act.detach().clone()
    )

    with bind_context(ctx):
        with hook_pytorch_model(llama_model):
            _ = llama_model(input_tokens)

    # Verify both submodules were successfully resolved and hooked
    assert len(captured_attn) == 1
    assert captured_attn[0].shape == (2, 4, 16)

    assert len(captured_mlp) == 1
    assert captured_mlp[0].shape == (2, 4, 16)


def test_llama_mutator_steering(llama_model, input_tokens):
    """Verifies activation steering on Llama self-attention submodules."""
    attn_path = "model.layers.0.self_attn"
    spec = TargetSpec(layer=0, location="self_attn", module_path=attn_path)

    clean_out = llama_model(input_tokens)

    ctx = MIContext(sample_id="llama_mut_test")
    ctx.targets[attn_path] = spec
    ctx.mutators[attn_path] = lambda act, spec: act + 50.0

    with bind_context(ctx):
        with hook_pytorch_model(llama_model):
            steered_out = llama_model(input_tokens)

    assert not torch.allclose(clean_out, steered_out, atol=1e-3)
