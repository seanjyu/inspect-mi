# inspect-mi
A Mechanistic Interpretability (MI) execution framework and context manager for Inspect AI that bridges low-level neural activation hooks (nnterp / PyTorch) with high-level evaluation loops. 

## Key Features
- Surgical Activation Steering & Mutation: Perform in-flight hidden state interventions ($act \to act - \alpha \cdot v$) during model forward passes (model.trace()), enabling real-time causal steering, ablation, and representation editing while measuring downstream logit and probability shifts.
- Thread & Async-Safe Context Isolation: Leverage ContextVar-backed state containers (MIContext) to guarantee strict, zero-leak isolation of mutators, observers, and telemetry buffers across concurrent asyncio evaluation loops and multi-threaded (ThreadPoolExecutor) worker pools.
- Native Inspect AI Solver Adapter: Provide high-order solver integrations (intervene()) that transparently wrap model generation calls (generate()), manage lifecycle hook bindings, and handle automatic context teardown on completion or error.
- Declarative Target & Schema Specification: Standardize layer targeting (TargetSpec, InterventionConfig) across heterogeneous LLM architectures (e.g., GPT-2, LLaMA, Qwen, Mistral), mapping unified layer specs (location="mlp_out") to concrete PyTorch module paths.
- JSON-Safe Telemetry & Store Export: Automatically convert internal activation tensors, vector norms, and scalar metrics into JSON-serializable primitives, flushing captured telemetry directly into Inspect AI’s TaskState.store under the structured mi:* namespace for downstream log analysis.

## System Requirements
- Python 3.12 or higher
## Install
1. Clone this repo 
``` bash
git clone https://github.com/seanjyu/inspect-mi
```
OPTIONAL - Create and activate python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install the package and dependencies
```bash
pip install .
```
Alternatively use developer mode (if you want to modify the code).
```bash
pip install -e ".[dev]"
```


## Quickstart

The following example demonstrates how to set up an activation steering and telemetry pipeline using `inspect_mi` inside an Inspect AI evaluation. 
<!-- A more in depth example can be found here -->

### 1. Define Target Layers and Intervention Config

Specify where in the model architecture you want to intercept hidden states using `TargetSpec` and `InterventionConfig`.

```python
from inspect_mi import InterventionConfig, TargetSpec

# Target the MLP output at layer 0
config = InterventionConfig(
    targets=[TargetSpec(layer=0, location="mlp_out", module_path="layers.0.mlp")]
)
```

### 2. Define Mutator or Observer Factories

Choose whether you want to **mutate** activations (e.g., active vector steering or ablation) or **observe** them (e.g., passive linear probing or feature logging).

```python
import torch
from inspect_mi import TargetSpec


# Mutator: Subtracts a target concept vector during model forward passes
def steering_mutator_factory(scale: float = 2.0):
    steering_vec = torch.ones(768)  # Replace with your concept/direction vector

    def mutator(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
        return act - (scale * steering_vec.to(act.device))

    return mutator


# Observer: Passive tracking of hidden state norm into the MI telemetry buffer
def norm_observer_factory():
    def observer(act: torch.Tensor, spec: TargetSpec, ctx):
        ctx.store(f"act_norm_layer_{spec.layer}", act.norm().item())

    return observer
```

### 3. Build the Inspect AI Task with the `intervene` Solver

Wrap your generation solver pipeline using `inspect_mi`’s `intervene()` solver function inside your Inspect AI task definition.

```python
from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate
from inspect_mi import intervene


@task
def mi_steering_eval() -> Task:
    # Instantiate the MI solver wrapper
    mi_solver = intervene(
        config=config,
        mutator_factory=steering_mutator_factory(scale=2.0),
        observer_factory=norm_observer_factory(),
    )

    dataset = MemoryDataset(
        [Sample(input="Explain the concept of neural network interpretability.")]
    )

    return Task(
        dataset=dataset,
        plan=[
            mi_solver,  # Binds thread/async MIContext and activates hooks
            generate(),  # Performs generation with active interventions
        ],
    )


if __name__ == "__main__":
    inspect_eval(mi_steering_eval(), model="hf/gpt2")
```

### 4. Execute the Evaluation

Execute the evaluation script via Python CLI or using the native Inspect AI CLI:

```bash
python your_mi_task.py

```

Or using Inspect CLI:

```bash
inspect eval your_mi_task.py --model hf/gpt2

```

### 5. Inspect Telemetry & Activation Metrics

All metrics and telemetry logged during execution are automatically flushed to `TaskState.store` under the `mi:*` key namespace. You can inspect these in the Inspect AI log viewer (`inspect view`) or read them directly from the evaluation log files:

```json
{
  "store": {
    "mi:act_norm_layer_0": 14.8231,
    "mi:intervention_status": "success"
  }
}

```
<!-- ## Documentation and Testing
- Further details regarding the architecture can be found [here](docs/architecture.md)
- Testing for this framework was done through pytest, more details can be found [here](tests/testing.md)
- A roadmap of future features and considerations can be found [here](docs/roadmap.md) -->
