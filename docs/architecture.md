# Architecture 
This document outlines the overarching architecture of the inspect_mi framework, detailing its integration bridge between inspect_ai and mechanistic interpretability (MI) primitives. 

# Lifecycle
1. Configuration & Target Specification
    - User defines an `inspect_mi.core.schema.InterventionConfig` containing one or more `inspect_mi.core.schema.TargetSpec` definitions. This specifies the specific location of the model to be intervened or monitored. 
    - Pydantic is used to ensure that the target locations are valid for the specified model.
    - Token indexing policies (TokenPolicy: all, last_prompt_token, first_token, index, slice) are configured in the `TargetSpec` to govern activation slicing.
2. Architecture Path Resolution
    - High-level abstract site locations (mlp_out, attn_out, resid_pre, resid_post, head_out) and absolute layer indices are translated into concrete module path strings (layers.0.mlp) using the model architecture registry (see src/core/model_registry.py, src/core/resolver.py).
3. Sample Isolated Context Initialization
    - During Inspect AI task execution, the intervene solver initializes a thread-safe and async-safe MIContext instance (see src/core/context.py) bound uniquely to the current sample ID.
    - The context container aggregates active mutators, observers, targets, and a local telemetry buffer.
4. Async Context Binding
    - Using bind_context(ctx), the framework binds the sample's MIContext to a Python contextvars.ContextVar (active_mi_context).
    - This guarantees absolute state isolation across concurrent asyncio evaluation tasks and multi-threaded worker pools, preventing state cross-contamination or telemetry leaks between samples.
5. Hook Dispatch, Policy Wrapping and Tensor Mutation
    - Depending on the execution mode, hooks are dispatched either via PyTorch native forward hooks (hooks.py) or in-flight nnsight / nnterp graph tracing (apply_nnterp_hooks).
    - Token Policy Wrapping (interventions.py): Intercepted activation tensors are sliced dynamically according to the target's TokenPolicy (e.g., isolating the last prompt token for steering) before passing them to user-defined mutator or observer functions.
6. Inspect AI Solver Integration and Telemetry Harvesting
    - The high-order intervene solver wrapper (solver.py) intercepts Inspect AI's execution flow.
    - It wraps sample generation within the active context manager, allowing models to execute evaluations with live activation steering and observation.
    - Upon completing execution, captured telemetry and target specifications from the MIContext buffer are automatically flushed into Inspect AI's TaskState.store under the structured mi:* namespace.


