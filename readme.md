# inspect-mi
A Mechanistic Interpretability (MI) execution framework and context manager for Inspect AI that bridges low-level neural activation hooks (nnterp / PyTorch) with high-level evaluation loops. 

## Key Features
- Activation Steering & Mutation: Perform in-flight hidden state interventions ($act \to act - \alpha \cdot v$) during model forward passes (model.trace()), enabling causal steering, ablation, and representation editing while measuring downstream logit and probability shifts.
- Thread & Async-Safe Context Isolation: Leverage ContextVar-backed state containers (MIContext) to isolate per-sample state (mutators, observers, telemetry), concurrent model execution is serialised (see [roadmap](docs/roadmap.md)).
- Native Inspect AI Solver Adapter: Provide high-order solver integrations (intervene()) that wrap model generation calls (generate()), manage lifecycle hook bindings, and handle automatic context teardown on completion or error.
- Declarative Target & Schema Specification: Standardize layer targeting (TargetSpec, InterventionConfig) across nnterp supported architectures, mapping unified layer specs (location="mlp_out") to concrete PyTorch module paths.
- JSON-Safe Telemetry & Store Export: Automatically convert internal activation tensors, vector norms, and scalar metrics into JSON-serializable primitives, flushing captured telemetry directly into Inspect AI’s TaskState.store under the structured mi:* namespace for downstream log analysis.

## Work In Progress Notice
This framework is still work in progress, more details regarding the current progress and technical challenges can be found [here](docs/roadmap.md#current-challenges)

## Documentation
Further details regarding the software architecture can be found [here](docs/architecture.md).


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

### 1. Define Target Layers & Config 
You can declare target layers and configuration specs using TargetSpec and InterventionConfig:

```python
from inspect_mi import InterventionConfig, TargetSpec

config = InterventionConfig(
    targets=[TargetSpec(layer=0, location="mlp_out", module_path="layers.0.mlp")]
)
```

### 2. Define Mutator or Observer Factories
Choose whether you want to mutate activations (e.g., active vector steering) or observe them (e.g., passive tracking of hidden state norms):
```python
from inspect_mi import TargetSpec


# Mutator factory: returns a function that computes act - scale * v
def steering_mutator_factory(scale: float = 50.0, vector: torch.Tensor = None):
    def factory(spec: TargetSpec):
        def mutator(act: torch.Tensor, s: TargetSpec) -> torch.Tensor:
            return act - scale * vector.to(act.device, act.dtype)

        return mutator

    return factory
```

### 3. Build the Inspect AI Task with the intervene Solver
Wrap your evaluation pipeline using inspect_mi’s intervene() solver. (Note: intervene handles execution internally, so separate generation wrappers are not required).

```Python
from inspect_ai import Task, task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_mi import intervene


@task
def mi_steering_eval() -> Task:
    mi_solver = intervene(
        config=config,
        mutator_factory=steering_mutator_factory(scale=50.0, vector=my_vector),
    )

    dataset = MemoryDataset(
        [Sample(input="The quick brown fox jumps over the lazy dog")]
    )

    return Task(
        dataset=dataset,
        solver=mi_solver,  # Binds per-sample state and executes interventions
    )


if __name__ == "__main__":
    inspect_eval(mi_steering_eval(), model="NNTERP_MODEL")
```

### 4. Inspect Telemetry & Logit Shift Metrics 
All telemetry and activation metrics logged during execution are automatically flushed to TaskState.store under the mi:* key namespace. You can inspect these in the Inspect AI log viewer (inspect view) or read them directly from evaluation logs.

## Runnable Example
This project is still very much work in progress but a runnable example can be found [here](docs/example/smoke_test.py). It uses a custom solver (rather than inspect_mi's intervene method).


<!-- ## Documentation and Testing
- Further details regarding the architecture can be found [here](docs/architecture.md)
- Testing for this framework was done through pytest, more details can be found [here](tests/testing.md)
- A roadmap of future features and considerations can be found [here](docs/roadmap.md) -->
