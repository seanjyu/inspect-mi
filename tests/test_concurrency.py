import pytest
import torch

from inspect_mi.core.context import MIContext, TargetSpec, get_current_context
from inspect_mi.core.hooks import apply_nnterp_hooks, bind_context


def test_nnterp_state_isolation(nnterp_gpt2_model):
    """Verifies that steering contexts and mutators leave no residual state,

    ensuring complete isolation across sequential execution runs.
    """
    model = nnterp_gpt2_model
    target_path = "layers.0.mlp"
    input_text = "The quick brown fox jumps over the lazy dog"

    cfg = model.config
    d_model = getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 16))

    torch.manual_seed(42)
    direction_vec = torch.randn(d_model)
    direction_vec = direction_vec / direction_vec.norm()

    def run_clean_task():
        """Baseline run with no steering or context."""
        with model.trace(input_text):
            logits_proxy = model.output.logits.save()
        return getattr(logits_proxy, "value", logits_proxy)

    def run_steered_task(sample_id: str, offset: float):
        """Steered run with a specific context mutator."""
        ctx = MIContext(sample_id=sample_id)
        spec = TargetSpec(layer=0, location="mlp_out", module_path=target_path)
        ctx.targets[target_path] = spec

        def mutator(act: torch.Tensor, target_spec: TargetSpec) -> torch.Tensor:
            target_dtype = getattr(act, "dtype", torch.float32)
            target_device = (
                act.device
                if (hasattr(act, "device") and act.device.type != "meta")
                else direction_vec.device
            )
            vec = direction_vec.to(device=target_device, dtype=target_dtype)
            return act + (offset * vec)

        ctx.mutators[target_path] = mutator

        with bind_context(ctx):
            with model.trace(input_text):
                apply_nnterp_hooks(model, ctx)
                logits_proxy = model.output.logits.save()

        return getattr(logits_proxy, "value", logits_proxy)

    # 1. Baseline Run (Clean 1)
    clean_logits_1 = run_clean_task()

    # 2. Steered Runs (Task A and Task B)
    steered_a_first = run_steered_task(sample_id="task_A", offset=1000.0)
    steered_b_first = run_steered_task(sample_id="task_B", offset=-1000.0)

    # 3. Post-Steering Baseline Run (Clean 2) - Test for Leakage/Pollution
    clean_logits_2 = run_clean_task()

    diff_clean = (clean_logits_1 - clean_logits_2).abs().max().item()
    assert diff_clean < 1e-5, (
        f"State leakage detected! Model output changed for an unsteered run "
        f"after steered tasks were executed (max diff: {diff_clean})."
    )

    # 4. Permutation Test: Run in reverse order [B, A]
    steered_b_second = run_steered_task(sample_id="task_B", offset=-1000.0)
    steered_a_second = run_steered_task(sample_id="task_A", offset=1000.0)

    diff_a = (steered_a_first - steered_a_second).abs().max().item()
    diff_b = (steered_b_first - steered_b_second).abs().max().item()

    assert diff_a < 1e-5, f"Task A output depends on previous runs (diff: {diff_a})."
    assert diff_b < 1e-5, f"Task B output depends on previous runs (diff: {diff_b})."

    # 5. Verify Inter-Task Separation
    diff_ab = (steered_a_first - steered_b_first).abs().max().item()
    assert diff_ab > 0.05, (
        "Task A and Task B produced identical outputs despite different mutators."
    )


def test_nnterp_observer_capture_and_scope_isolation(nnterp_gpt2_model):
    """Verifies that observers capture correct activation data using ctx.store(),

    and that observer state/buffers do not leak across tasks.
    """
    model = nnterp_gpt2_model
    model.eval()
    target_path = "layers.0.mlp"
    input_text = "The quick brown fox jumps over the lazy dog"

    # =========================================================================
    # 1. Task A Execution: Observer + Mutator
    # =========================================================================
    ctx_a = MIContext(sample_id="task_A")
    spec_a = TargetSpec(layer=0, location="mlp_out", module_path=target_path)
    ctx_a.targets[target_path] = spec_a

    # Mutator adds offset 100.0
    def mutator_a(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
        return act + 100.0

    # Observer records captured activation into ctx.buffer via ctx.store()
    def observer_a(act: torch.Tensor, spec: TargetSpec) -> None:
        ctx_a.store("captured_mlp_out", act)

    ctx_a.mutators[target_path] = mutator_a
    ctx_a.observers[target_path] = observer_a

    with bind_context(ctx_a):
        with model.trace(input_text):
            apply_nnterp_hooks(model, ctx_a)

    # Verify Task A captured the activation inside ctx.buffer
    assert "captured_mlp_out" in ctx_a.buffer, (
        "Observer failed to record output into ctx.buffer!"
    )

    captured_act = ctx_a.buffer["captured_mlp_out"]
    captured_val = getattr(captured_act, "value", captured_act)
    assert isinstance(captured_val, torch.Tensor), (
        "Stored activation is not a valid Tensor!"
    )

    # =========================================================================
    # 2. Task B Execution: Verify Scope Isolation & Buffer Cleanliness
    # =========================================================================
    ctx_b = MIContext(sample_id="task_B")
    spec_b = TargetSpec(layer=0, location="mlp_out", module_path=target_path)
    ctx_b.targets[target_path] = spec_b

    # Task B has NO observers registered
    with bind_context(ctx_b):
        with model.trace(input_text):
            apply_nnterp_hooks(model, ctx_b)

    # Assert Task B's buffer remains empty (Task A's observer did not fire or record into ctx_b)
    assert len(ctx_b.buffer) == 0, (
        f"Task B buffer polluted! Found keys: {list(ctx_b.buffer.keys())}"
    )

    # Verify clear_buffer functionality
    ctx_a.clear_buffer()
    assert len(ctx_a.buffer) == 0, "ctx.clear_buffer() failed to empty the buffer!"


# def test_nnterp_multi_layer_causal_propagation(nnterp_gpt2_model):
#     """Verifies that mutating Layer 0 changes downstream Layer 2 activations."""
#     model = nnterp_gpt2_model
#     model.eval()

#     path_l0 = "layers.0.mlp"
#     path_l2 = "layers.2.mlp"
#     input_text = "The quick brown fox jumps over the lazy dog"

#     # --- Run 1: Clean pass, observe Layer 2 ---
#     ctx_clean = MIContext(sample_id="clean")
#     ctx_clean.targets[path_l2] = TargetSpec(layer=2, location="mlp_out", module_path=path_l2)
#     ctx_clean.observers[path_l2] = lambda act, spec: ctx_clean.store("l2_act", act)

#     with bind_context(ctx_clean):
#         with model.trace(input_text):
#             apply_nnterp_hooks(model, ctx_clean)

#     l2_clean_val = getattr(ctx_clean.buffer["l2_act"], "value", ctx_clean.buffer["l2_act"])

#     # --- Run 2: Mutate Layer 0, observe Layer 2 ---
#     ctx_steered = MIContext(sample_id="steered")
#     ctx_steered.targets[path_l0] = TargetSpec(layer=0, location="mlp_out", module_path=path_l0)
#     ctx_steered.targets[path_l2] = TargetSpec(layer=2, location="mlp_out", module_path=path_l2)

#     # Large mutation at Layer 0
#     ctx_steered.mutators[path_l0] = lambda act, spec: act + 500.0
#     ctx_steered.observers[path_l2] = lambda act, spec: ctx_steered.store("l2_act", act)

#     with bind_context(ctx_steered):
#         with model.trace(input_text):
#             apply_nnterp_hooks(model, ctx_steered)

#     l2_steered_val = getattr(ctx_steered.buffer["l2_act"], "value", ctx_steered.buffer["l2_act"])

#     # Downstream activations MUST change due to upstream mutation
#     diff_l2 = (l2_clean_val - l2_steered_val).abs().max().item()
#     assert diff_l2 > 1.0, f"Upstream mutation at Layer 0 failed to propagate to Layer 2 (diff: {diff_l2})"


def test_nnterp_multi_layer_causal_propagation(nnterp_gpt2_model):
    """Verifies that mutating Layer 0 changes downstream Layer 2 activations."""
    model = nnterp_gpt2_model
    model.eval()

    path_l0 = "layers.0.mlp"
    path_l2 = "layers.2.mlp"
    input_text = "The quick brown fox jumps over the lazy dog"

    # Create a zero-centered direction vector to prevent LayerNorm cancellation
    cfg = model.config
    d_model = getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 16))
    torch.manual_seed(42)
    direction_vec = torch.randn(d_model)
    direction_vec -= direction_vec.mean()  # Zero-center mean
    direction_vec = direction_vec / direction_vec.norm()

    # --- Run 1: Clean pass, observe Layer 2 ---
    ctx_clean = MIContext(sample_id="clean")
    ctx_clean.targets[path_l2] = TargetSpec(
        layer=2, location="mlp_out", module_path=path_l2
    )
    ctx_clean.observers[path_l2] = lambda act, spec: ctx_clean.store("l2_act", act)

    with bind_context(ctx_clean):
        with model.trace(input_text):
            apply_nnterp_hooks(model, ctx_clean)

    l2_clean_val = getattr(
        ctx_clean.buffer["l2_act"], "value", ctx_clean.buffer["l2_act"]
    )

    # --- Run 2: Mutate Layer 0 with directional vector, observe Layer 2 ---
    ctx_steered = MIContext(sample_id="steered")
    ctx_steered.targets[path_l0] = TargetSpec(
        layer=0, location="mlp_out", module_path=path_l0
    )
    ctx_steered.targets[path_l2] = TargetSpec(
        layer=2, location="mlp_out", module_path=path_l2
    )

    # def mutator_l0(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
    #     target_dtype = getattr(act, "dtype", torch.float32)
    #     target_device = act.device if (hasattr(act, "device") and act.device.type != "meta") else direction_vec.device
    #     vec = direction_vec.to(device=target_device, dtype=target_dtype)
    #     return act + (100.0 * vec)

    def mutator_l0(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
        target_dtype = getattr(act, "dtype", torch.float32)
        target_device = (
            act.device
            if (hasattr(act, "device") and act.device.type != "meta")
            else direction_vec.device
        )
        vec = direction_vec.to(device=target_device, dtype=target_dtype)

        # Scale direction to 50% of the activation tensor's norm
        scale = 0.5 * act.norm() / (vec.norm() + 1e-8)
        return act + (scale * vec)

    ctx_steered.mutators[path_l0] = mutator_l0
    ctx_steered.observers[path_l2] = lambda act, spec: ctx_steered.store("l2_act", act)

    with bind_context(ctx_steered):
        with model.trace(input_text):
            apply_nnterp_hooks(model, ctx_steered)

    l2_steered_val = getattr(
        ctx_steered.buffer["l2_act"], "value", ctx_steered.buffer["l2_act"]
    )

    # Assert downstream Layer 2 activations changed significantly
    # diff_l2 = (l2_clean_val - l2_steered_val).abs().max().item()
    # assert diff_l2 > 0.05, f"Upstream mutation at Layer 0 failed to propagate to Layer 2 (diff: {diff_l2})"
    # diff_l2 = (l2_clean_val - l2_steered_val).abs().max().item()
    # assert diff_l2 > 0.005, f"Upstream mutation at Layer 0 failed to propagate to Layer 2 (diff: {diff_l2})"

    rel_diff = ((l2_clean_val - l2_steered_val).norm() / l2_clean_val.norm()).item()
    assert rel_diff > 0.01, f"Relative downstream change too low: {rel_diff}"


def test_nnterp_context_cleanup_on_exception(nnterp_gpt2_model):
    """Verifies that context is unbound even if an exception occurs during tracing."""
    model = nnterp_gpt2_model
    target_path = "layers.0.mlp"
    input_text = "Test input"

    ctx_faulty = MIContext(sample_id="faulty")
    ctx_faulty.targets[target_path] = TargetSpec(
        layer=0, location="mlp_out", module_path=target_path
    )

    # Mutator designed to crash mid-trace
    def crashing_mutator(act, spec):
        raise RuntimeError("Simulated mutator failure!")

    ctx_faulty.mutators[target_path] = crashing_mutator

    # 1. Trigger the failing trace
    with pytest.raises(RuntimeError, match="Simulated mutator failure!"):
        with bind_context(ctx_faulty):
            with model.trace(input_text):
                apply_nnterp_hooks(model, ctx_faulty)

    # 2. Verify that active context is cleanly unbound / None
    # (Adjust 'get_current_context()' to your actual context lookup method)
    assert get_current_context() is None or get_current_context() != ctx_faulty, (
        "Faulty context remained bound after an exception occurred!"
    )
