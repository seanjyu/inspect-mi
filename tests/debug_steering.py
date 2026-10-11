# import torch
# from nnterp import StandardizedTransformer
# from inspect_mi.core.context import MIContext, TargetSpec
# from inspect_mi.core.hooks import apply_nnterp_hooks

# # 1. Setup model & input
# model = StandardizedTransformer("hf-internal-testing/tiny-random-GPT2LMHeadModel").eval()
# input_ids = torch.randint(0, 500, (2, 4))

# # 2. Baseline clean run
# with model.trace(input_ids):
#     clean_logits = model.output.logits.save()

# # 3. Steered run (+1000.0 steering vector)
# target_path = "transformer.h.0.mlp"
# spec = TargetSpec(layer=0, location="mlp", module_path=target_path)

# ctx = MIContext(sample_id="debug_steering")
# ctx.targets[target_path] = spec
# ctx.mutators[target_path] = lambda act, spec: act + 1000.0

# with model.trace(input_ids):
#     apply_nnterp_hooks(model, ctx)
#     steered_logits = model.output.logits.save()

# # 4. Check logit diff
# max_diff = (steered_logits.value - clean_logits.value).abs().max().item()
# print(f"\n---> Maximum absolute logit difference: {max_diff:.4f}")

# if max_diff < 1.0:
#     print("❌ FAIL: Steering signal was NOT propagated down the computation graph.")
# else:
#     print("✅ SUCCESS: Steering successfully altered model outputs!")

import torch
from nnsight import LanguageModel

from inspect_mi.core.context import MIContext, TargetSpec
from inspect_mi.core.hooks import apply_nnterp_hooks

# 1. Initialize via LanguageModel
model = LanguageModel(
    "hf-internal-testing/tiny-random-GPT2LMHeadModel", device_map="auto", dispatch=True
)
input_ids = torch.randint(0, 500, (2, 4)).to(model.device)

# 2. Baseline clean run
with model.trace(input_ids):
    clean_logits = model.output.logits.save()

# 3. Steered run (+1000.0 steering vector)
target_path = "layers.1.mlp"
spec = TargetSpec(layer=0, location="mlp", module_path=target_path)

ctx = MIContext(sample_id="debug_steering")
ctx.targets[target_path] = spec
ctx.mutators[target_path] = lambda act, spec: act * 0.0

with model.trace(input_ids):
    apply_nnterp_hooks(model, ctx)
    steered_logits = model.output.logits.save()

# 4. Check logit difference
# max_diff = (steered_logits.value - clean_logits.value).abs().max().item()
max_diff = (steered_logits - clean_logits).abs().max().item()
print(f"\n---> Maximum absolute logit difference: {max_diff:.4f}")
