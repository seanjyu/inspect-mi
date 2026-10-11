import asyncio

import pytest
import torch
import torch.nn.functional as F
from inspect_ai.solver import TaskState

from inspect_mi.core.context import MIContext, TargetSpec, get_current_context
from inspect_mi.core.hooks import apply_nnterp_hooks, bind_context
from inspect_mi.core.schema import InterventionConfig
from inspect_mi.core.solver import intervene


def test_nnterp_e2e_logit_probability_shift(nnterp_gpt2_model):
    """E2E Test: Verifies that internal activation steering produces a predictable,
    measurable probability shift on the next-token prediction logits.
    """
    model = nnterp_gpt2_model
    model.eval()

    # 1. Dynamically select a valid middle layer for the test fixture
    cfg = model.config
    num_layers = getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 2))
    target_layer = max(0, num_layers // 2)
    target_path = f"layers.{target_layer}.mlp"

    tokenizer = getattr(model, "tokenizer", None)
    input_text = "The quick brown fox jumps over the lazy dog"

    # 2. Baseline Pass: Capture clean next-token logits
    with model.trace(input_text):
        clean_logits_proxy = model.output.logits[0, -1, :].save()

    clean_logits = getattr(clean_logits_proxy, "value", clean_logits_proxy)
    clean_probs = F.softmax(clean_logits, dim=-1)

    top_token_id = clean_logits.argmax(dim=-1).item()
    clean_top_prob = clean_probs[top_token_id].item()

    # 3. Derive Zero-Centered Steering Vector from Target Token Embedding
    wte_weight = getattr(model, "transformer", model).wte.weight
    direction_vec = wte_weight[top_token_id].detach().clone()
    direction_vec -= direction_vec.mean()  # Prevent LN cancellation
    direction_vec = direction_vec / direction_vec.norm()

    # 4. Steered Pass: Anti-steer at selected target layer
    ctx = MIContext(sample_id="e2e_anti_steer")
    spec = TargetSpec(layer=target_layer, location="mlp_out", module_path=target_path)
    ctx.targets[target_path] = spec

    steering_scale = 50.0

    def mutator(act: torch.Tensor, target_spec: TargetSpec) -> torch.Tensor:
        target_dtype = getattr(act, "dtype", torch.float32)
        target_device = (
            act.device
            if (hasattr(act, "device") and act.device.type != "meta")
            else direction_vec.device
        )
        vec = direction_vec.to(device=target_device, dtype=target_dtype)
        return act - (steering_scale * vec)

    ctx.mutators[target_path] = mutator

    with bind_context(ctx):
        with model.trace(input_text):
            apply_nnterp_hooks(model, ctx)
            steered_logits_proxy = model.output.logits[0, -1, :].save()

    steered_logits = getattr(steered_logits_proxy, "value", steered_logits_proxy)
    steered_probs = F.softmax(steered_logits, dim=-1)
    steered_top_prob = steered_probs[top_token_id].item()

    # 5. Behavioral Assertions
    logit_diff = (clean_logits - steered_logits).abs().max().item()

    # Raw logit drop specifically on the targeted top token
    target_logit_drop = (
        clean_logits[top_token_id] - steered_logits[top_token_id]
    ).item()

    # Relative probability reduction
    rel_prob_drop = (clean_top_prob - steered_top_prob) / (clean_top_prob + 1e-8)

    # A. Global logit perturbation check
    assert logit_diff > 0.5, (
        f"Steering failed to alter logits (max delta: {logit_diff})"
    )

    # B. Target token logit drop check (invariant to softmax dilution)
    assert target_logit_drop > 0.5, (
        f"Anti-steering failed to suppress target logit! "
        f"Clean logit: {clean_logits[top_token_id].item():.4f}, "
        f"Steered logit: {steered_logits[top_token_id].item():.4f} (drop: {target_logit_drop:.4f})"
    )

    # C. Relative probability reduction check
    assert rel_prob_drop > 0.30, (
        f"Relative probability drop too small! "
        f"Baseline prob: {clean_top_prob:.6f}, Steered prob: {steered_top_prob:.6f} "
        f"(relative drop: {rel_prob_drop * 100:.1f}%)"
    )


# Import your framework modules
# from inspect_mi.core.context import MIContext, bind_context, get_current_context
# from inspect_mi.solvers import intervene, InterventionConfig, TargetSpec


@pytest.mark.asyncio
async def test_nnterp_inspect_solver_e2e_integration():
    """E2E Test: Verifies that the intervene solver constructs MIContext,
    maintains bound context across async generate calls, and exports telemetry.
    """
    target_path = "layers.0.mlp"
    target_spec = TargetSpec(layer=0, location="mlp_out", module_path=target_path)
    config = InterventionConfig(targets=[target_spec])

    # 1. Define factories
    def mutator_factory(spec):
        return lambda act, s: act + 10.0

    def observer_factory(spec):
        def observer(act, s):
            ctx = get_current_context()
            if ctx is not None:
                ctx.store("observed_mean", 42.0)

        return observer

    # 2. Initialize TaskState
    state = TaskState(
        model="gpt2",
        sample_id="sample_001",
        epoch=1,
        input="The quick brown fox",
        messages=[],
    )

    context_active_during_generate = False

    # 3. Dummy Generate function to simulate model execution inside inspect_ai
    async def mock_generate(s: TaskState) -> TaskState:
        nonlocal context_active_during_generate

        # Retrieve context bound by 'with bind_context(ctx)' inside the solver
        active_ctx = get_current_context()
        assert active_ctx is not None, "Context was not bound during generate() call!"
        assert active_ctx.sample_id == "sample_001", (
            f"Sample ID mismatch: {active_ctx.sample_id}"
        )
        assert target_path in active_ctx.mutators, (
            "Mutator factory failed to register on context"
        )
        assert target_path in active_ctx.observers, (
            "Observer factory failed to register on context"
        )

        # Simulate observer execution writing to buffer during generation
        active_ctx.observers[target_path](torch.tensor([1.0]), target_spec)

        context_active_during_generate = True
        return s

    # 4. Instantiate solver and run solve()
    solver_fn = intervene(
        config=config,
        mutator_factory=mutator_factory,
        observer_factory=observer_factory,
    )

    final_state = await solver_fn(state, mock_generate)

    # 5. Assertions
    assert context_active_during_generate, (
        "mock_generate was not executed within bound context!"
    )

    # Verify TargetSpec export
    assert final_state.store.get(f"mi:spec:{target_path}") == target_spec, (
        "TargetSpec was not exported to state.store under 'mi:spec:' prefix!"
    )

    # Verify telemetry export from MIContext buffer
    assert final_state.store.get("mi:observed_mean") == 42.0, (
        "Telemetry in MIContext.buffer was not exported to state.store under 'mi:' prefix!"
    )


@pytest.mark.asyncio
async def test_nnterp_concurrent_context_isolation():
    """E2E Concurrency Test: Ensures that when multiple samples run in parallel,

    Context A's mutators/observers/telemetry never leak into Context B.
    """
    num_concurrent_samples = 10
    target_path = "layers.0.mlp"
    target_spec = TargetSpec(layer=0, location="mlp_out", module_path=target_path)
    config = InterventionConfig(targets=[target_spec])

    async def run_single_sample(sample_idx: int):
        sample_id = f"sample_{sample_idx:03d}"
        expected_scalar = float(sample_idx * 100)

        # Mutator unique to this sample
        def mutator_factory(spec):
            def mutator(act: torch.Tensor, s: TargetSpec) -> torch.Tensor:
                ctx = get_current_context()
                assert ctx is not None, (
                    f"No context bound during mutator execution in {sample_id}"
                )
                assert ctx.sample_id == sample_id, (
                    f"Cross-contamination detected! Mutator running in {sample_id} saw {ctx.sample_id}"
                )
                return act + expected_scalar

            return mutator

        # Observer unique to this sample
        def observer_factory(spec):
            def observer(act: torch.Tensor, s: TargetSpec):
                ctx = get_current_context()
                assert ctx is not None, (
                    f"No context bound during observer execution in {sample_id}"
                )
                assert ctx.sample_id == sample_id, (
                    f"Cross-contamination detected! Observer running in {sample_id} saw {ctx.sample_id}"
                )
                ctx.store("sample_scalar", expected_scalar)

            return observer

        state = TaskState(
            model="gpt2",
            sample_id=sample_id,
            epoch=1,
            input=f"Prompt for {sample_id}",
            messages=[],
        )

        async def mock_generate_with_context_switch(s: TaskState) -> TaskState:
            # 1. Check context before context switch
            ctx_before = get_current_context()
            assert ctx_before is not None
            assert ctx_before.sample_id == sample_id

            # 2. Simulate mutator and observer call
            dummy_act = torch.tensor([1.0])
            ctx_before.mutators[target_path](dummy_act, target_spec)
            ctx_before.observers[target_path](dummy_act, target_spec)

            # 3. Force async context switch to yield control to other concurrent tasks
            await asyncio.sleep(0.01)

            # 4. Check context after resuming from async sleep
            ctx_after = get_current_context()
            assert ctx_after is not None, (
                f"Context was lost after async sleep in {sample_id}"
            )
            assert ctx_after.sample_id == sample_id, (
                f"Context leaked during sleep! Expected {sample_id}, found {ctx_after.sample_id}"
            )

            return s

        solver_fn = intervene(
            config=config,
            mutator_factory=mutator_factory,
            observer_factory=observer_factory,
        )

        final_state = await solver_fn(state, mock_generate_with_context_switch)
        return sample_id, expected_scalar, final_state

    # Execute all samples concurrently
    tasks = [run_single_sample(i) for i in range(num_concurrent_samples)]
    results = await asyncio.gather(*tasks)

    # Validate that every sample retained its own unique telemetry without pollution
    for sample_id, expected_scalar, final_state in results:
        stored_scalar = final_state.store.get("mi:sample_scalar")
        assert stored_scalar == expected_scalar, (
            f"Telemetry pollution in {sample_id}! Expected {expected_scalar}, found {stored_scalar}"
        )


import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from inspect_mi.core.context import (
    bind_context,
)


def test_nnterp_multithreaded_context_isolation():
    """Verifies that MIContext state and active context resolution remain strictly

    isolated across concurrent OS worker threads running in parallel.
    """
    num_threads = 8
    target_path = "layers.0.mlp"
    target_spec = TargetSpec(layer=0, location="mlp_out", module_path=target_path)

    def worker_task(thread_idx: int):
        sample_id = f"thread_sample_{thread_idx:02d}"
        expected_val = float((thread_idx + 1) * 42)

        # 1. Thread-specific context construction
        ctx = MIContext(sample_id=sample_id)
        ctx.targets[target_path] = target_spec

        def observer(act: torch.Tensor, spec: TargetSpec):
            active_ctx = get_current_context()
            assert active_ctx is not None, f"No context bound in {sample_id}"
            assert active_ctx.sample_id == sample_id, (
                f"Thread context cross-contamination! Worker {sample_id} saw {active_ctx.sample_id}"
            )
            active_ctx.store("thread_value", expected_val)

        ctx.observers[target_path] = observer

        # 2. Execute under bound context with thread sleep to force OS context switching
        with bind_context(ctx):
            # Assert context is bound before sleep
            current = get_current_context()
            assert current is not None and current.sample_id == sample_id

            # Simulate computation and force OS thread context switch
            dummy_tensor = torch.tensor([1.0, 2.0, 3.0])
            ctx.observers[target_path](dummy_tensor, target_spec)
            time.sleep(0.02)  # Real OS thread sleep

            # Assert context remains untouched after waking up
            current_after = get_current_context()
            assert current_after is not None, f"Context lost after sleep in {sample_id}"
            assert current_after.sample_id == sample_id, (
                f"Context mutated during thread sleep! Expected {sample_id}, got {current_after.sample_id}"
            )

        # 3. Assert context is cleanly unbound after exiting context manager in this thread
        post_exit_ctx = get_current_context()
        assert post_exit_ctx is None or post_exit_ctx.sample_id != sample_id, (
            f"Context remained bound in thread {sample_id} after exiting 'with bind_context'!"
        )

        return sample_id, expected_val, ctx.buffer.get("thread_value")

    # Run tasks across real OS threads
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_task, i) for i in range(num_threads)]
        for future in as_completed(futures):
            results.append(future.result())

    # 4. Verify thread buffer integrity
    assert len(results) == num_threads
    for sample_id, expected_val, captured_val in results:
        assert captured_val == expected_val, (
            f"Data corruption in worker {sample_id}! Expected {expected_val}, captured {captured_val}"
        )
