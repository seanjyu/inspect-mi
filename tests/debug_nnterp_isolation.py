from nnterp import StandardizedTransformer

from inspect_mi.core.context import MIContext, TargetSpec
from inspect_mi.core.hooks import apply_nnterp_hooks, bind_context


def debug_nnterp_isolation():
    # model = nnterp_gpt2_model
    model = StandardizedTransformer(
        "hf-internal-testing/tiny-random-GPT2LMHeadModel",
        device_map="auto",
        dispatch=True,
    )
    model.eval()

    input_text = "The quick brown fox jumps over the lazy dog"
    target_path = "layers.0.mlp"

    print("\n" + "=" * 60)
    print("STARTING DIAGNOSTIC SUITE FOR NNTERP ISOLATION")
    print("=" * 60)

    # -------------------------------------------------------------
    # DIAGNOSTIC 1: Base Model Determinism (Control Test)
    # -------------------------------------------------------------
    print("\n--- [1] Checking Base Model Determinism (No Hooks Ever Applied) ---")

    # Force dropout off on the underlying HuggingFace model if wrapped
    hf_model = getattr(model, "_model", getattr(model, "model", model))
    hf_model.eval()

    with model.trace(input_text):
        res1 = model.output.logits.save()
    with model.trace(input_text):
        res2 = model.output.logits.save()

    diff_control = (res1 - res2).abs().max().item()
    print(f"Base model consecutive difference: {diff_control:.8f}")
    if diff_control > 1e-5:
        print("❌ FAIL: Base model itself is non-deterministic!")
        print(
            "   -> Cause: Inner model dropout/dropout_p is active, or KV-caching is retaining state."
        )
        return
    else:
        print("✅ PASS: Base model is 100% deterministic.")

    # -------------------------------------------------------------
    # DIAGNOSTIC 2: PyTorch Module Forward Hook Registry Hygiene
    # -------------------------------------------------------------
    print("\n--- [2] Checking PyTorch Hook Leakage ---")

    def count_total_hooks(model):
        hooks_count = 0
        for name, submodule in model.named_modules():
            hooks_count += len(getattr(submodule, "_forward_hooks", {}))
            hooks_count += len(getattr(submodule, "_forward_pre_hooks", {}))
        return hooks_count

    initial_hooks = count_total_hooks(model)
    print(f"Initial total model hooks: {initial_hooks}")

    # Build and run a steered task
    ctx = MIContext(sample_id="debug_task_1")
    spec = TargetSpec(layer=0, location="mlp_out", module_path=target_path)
    ctx.targets[target_path] = spec

    execution_log = []

    def logging_mutator(act, spec):
        execution_log.append("MUTATOR_EXECUTED")
        return act + 100.0

    ctx.mutators[target_path] = logging_mutator

    with bind_context(ctx):
        with model.trace(input_text):
            apply_nnterp_hooks(model, ctx)
            steered_res = model.output.logits.save()

    post_steered_hooks = count_total_hooks(model)
    print(f"Hooks after steered run: {post_steered_hooks}")

    if post_steered_hooks > initial_hooks:
        print(
            f"❌ FAIL: {post_steered_hooks - initial_hooks} hooks LEAKED onto model modules!"
        )
        print(
            "   -> Cause: apply_nnterp_hooks or nnsight is calling `register_forward_hook` without removing them upon exit."
        )
        return
    else:
        print("✅ PASS: Zero forward hooks leaked onto model modules.")

    # -------------------------------------------------------------
    # DIAGNOSTIC 3: Mutator Execution Leakage
    # -------------------------------------------------------------
    print("\n--- [3] Checking for Ghost Mutator Execution on Clean Run ---")
    execution_log.clear()

    # Run an unsteered clean pass after steering
    with model.trace(input_text):
        clean_post_steer = model.output.logits.save()

    if len(execution_log) > 0:
        print("❌ FAIL: Mutator function was executed during an UNSTEERED run!")
        print(
            "   -> Cause: Context state or active mutators leaked outside `bind_context`."
        )
        return
    else:
        print("✅ PASS: Mutator was NOT invoked during clean run.")

    # -------------------------------------------------------------
    # DIAGNOSTIC 4: Output Logit Isolation Check
    # -------------------------------------------------------------
    print("\n--- [4] Comparing Pre-Steer Clean vs Post-Steer Clean Logits ---")
    diff_clean_leak = (res1 - clean_post_steer).abs().max().item()
    print(f"Difference between Clean 1 and Clean 2: {diff_clean_leak:.8f}")

    if diff_clean_leak > 1e-5:
        print("❌ FAIL: Clean logits changed after running a steered task!")
    else:
        print("✅ PASS: Clean logits are identical before and after steering!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    debug_nnterp_isolation()
