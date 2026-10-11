# debug_nnterp.py
import torch
from nnterp import StandardizedTransformer

print("Loading model...")
model = StandardizedTransformer("hf-internal-testing/tiny-random-GPT2LMHeadModel")
input_tokens = torch.randint(0, 100, (1, 4), dtype=torch.long)

captured_activations = []

print("\n--- ENTERING TRACE CONTEXT ---")
with model.trace(input_tokens) as tracer:
    # Standardized layer output returns a live torch.Tensor
    act_tensor = model.layers_output[0]

    print(f"1. Object type inside trace: {type(act_tensor)}")
    print(f"2. Is it a torch.Tensor? {isinstance(act_tensor, torch.Tensor)}")

    # OBSERVER: Capture tensor by cloning and detaching
    captured_activations.append(act_tensor.detach().clone())
    print("3. Observer captured tensor successfully.")

    # MUTATOR: Mutate tensor in-place
    act_tensor[:] = act_tensor * 2.0
    print("4. Mutator applied in-place.")

print("--- EXITED TRACE CONTEXT ---\n")

# POST-TRACE: Activations are already standard PyTorch Tensors
print(f"5. Captured tensor type: {type(captured_activations[0])}")
print(f"6. Captured shape: {captured_activations[0].shape}")
