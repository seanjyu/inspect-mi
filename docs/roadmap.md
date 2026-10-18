# Roadmap

## Current Implementation
- Sample-Isolated Context Management (`src/core/context.py`)
    - ContextVar Binding: Implemented thread-safe and async-safe state isolation using Python's contextvars.ContextVar (active_mi_context).
    - MIContext Container: Binds a unique execution context per sample ID, housing local mutators, observers, targets, and a dedicated telemetry buffer to prevent cross-sample contamination during concurrent evaluation loops.

- Declarative Targeting & Slicing (`src/core/schema.py` & `src/core/interventions.py`)
    - `InterventionConfig` & `TargetSpec`: Pydantic-backed configuration schemas with `@model_validator` decorators that gracefully coerce flexible inputs (such as legacy dictionaries or tuples) into normalized layer indices and hook locations.
    - Token Slicing Policies (`TokenPolicy`): Built-in support for dynamic tensor slicing (`all`, `last_prompt_token`, `first_token`, `index`, `slice`) to isolate specific activation tokens before applying interventions.

- Heterogeneous Architecture Path Resolution (`model_registry.py` & `resolver.py`)
    - Canonical Path Translation: Translates abstract functional locations (`mlp_out`, `attn_out`, `resid_pre`, `resid_post`, `head_out`) and absolute layer indices into concrete PyTorch module path strings (e.g., `layers.0.mlp`).
    - Multi-Family Support: Handles heterogeneous transformer architectures out-of-the-box (GPT-2, LLaMA, Mistral, Qwen, Gemma, Pythia/GPT-NeoX) via extensible string-matching templates.

- Tensor Mutation & Numerical Guardrails (`hooks.py`)
    - Activation Interception: Dispatches hooks via PyTorch native forward hooks or in-flight graph tracing (`nnsight`/`nnterp`).
    - LayerNorm Guardrails: Enforces zero-mean orthogonal direction vectors during mutations to prevent downstream LayerNorm mean-centering cancellation and gradient/NaN traps.

- Inspect AI Solver Integration & Telemetry Export (`solver.py`)
    - High-Order Solver Wrapper: Implements the core `@solver` entry point that intercepts evaluation flow, manages the active context lifecycle, and supports explicit white-box verification runs.
    - Telemetry Harvesting: Automatically flushes captured tensor metrics, logit shifts, and target metadata from the local `MIContext` buffer directly into Inspect AI's `TaskState.store` under the structured `mi:*` namespace.


## Current Challenges
- Need to remake the intervene method since there is a fundamental mismatch in Inspect AI's execution engine. The solution will involve explicitly requiring an nnterp/nnsight-backed white-box model architecture, thereby leveraging native graph tracing and autoregressive generation blocks within the solver lifecycle.
- Hooks are registered on a single shared model instance; if one sample's hooks remain attached while another sample traces, mutators fire on the wrong activations. Therefore the current iteration will only work on serialized runs.
- ContextVar isolates per-sample state (mutators, observers, telemetry) across async tasks and threads, but the model is one shared resource. Note isolation of metadata does not isolate execution.
- A possible solution could be to serialise traces with an async lock (short-term), then a single context-dispatching hook per module, then batched inference with per-row interventions.