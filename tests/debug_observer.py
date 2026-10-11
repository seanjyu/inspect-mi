import torch
from nnterp import StandardizedTransformer

# Adjust these imports to match your project's module paths
from inspect_mi.core.context import MIContext, bind_context
from inspect_mi.core.hooks import apply_nnterp_hooks
from inspect_mi.core.schema import TargetSpec


def run_debug():
    print("=" * 60)
    print("1. LOADING MODEL AND CREATING INPUTS")
    print("=" * 60)
    model = StandardizedTransformer("hf-internal-testing/tiny-random-GPT2LMHeadModel")
    input_ids = torch.randint(low=0, high=100, size=(2, 4), dtype=torch.long)
    print(f"Model loaded: {type(model).__name__}")
    print(f"input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")

    print("\n" + "=" * 60)
    print("2. CONFIGURING MICONTEXT & TARGET SPECS")
    print("=" * 60)
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

    print(f"Target paths registered: {list(ctx.targets.keys())}")
    print(f"Observers registered:    {list(ctx.observers.keys())}")

    print("\n" + "=" * 60)
    print("3. EXECUTING TRACE CONTEXT & APPLYING HOOKS")
    print("=" * 60)

    with bind_context(ctx):
        with model.trace(input_ids):
            apply_nnterp_hooks(model, ctx=ctx)

    print("\n" + "=" * 60)
    print("4. INSPECTING CAPTURED RESULTS")
    print("=" * 60)

    # Attention Assertion
    print(f"captured_attn count: {len(captured_attn)}")
    if len(captured_attn) == 1:
        print(f"✅ captured_attn[0] shape: {captured_attn[0].shape}")
    else:
        print(f"❌ EXPECTED 1 TENSOR IN captured_attn, GOT {len(captured_attn)}")

    # MLP Assertion
    print(f"captured_mlp count:  {len(captured_mlp)}")
    if len(captured_mlp) == 1:
        print(f"✅ captured_mlp[0] shape:  {captured_mlp[0].shape}")
    else:
        print(f"❌ EXPECTED 1 TENSOR IN captured_mlp, GOT {len(captured_mlp)}")


if __name__ == "__main__":
    run_debug()
