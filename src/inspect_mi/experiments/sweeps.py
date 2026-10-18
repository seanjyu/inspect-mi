from collections.abc import Callable, Sequence

from inspect_ai import Task

from inspect_mi.core.schema import TargetSpec


def sweep_layers(
    task_factory: Callable[[TargetSpec], Task],
    layers: Sequence[int],
    location: str = "resid_pre",
    name_prefix: str = "sweep",
) -> list[Task]:
    """Generates a list of Tasks configured across a range of layer target locations.

    Args:
        task_factory: Function taking a TargetSpec and returning an inspect_ai Task.
        layers: Iterable of layer indices to evaluate (e.g. range(0, 12)).
        location: Target site name (e.g. "resid_pre", "attn_out", "mlp_out").
        name_prefix: Prefix for task naming in inspect_ai logs.

    Returns:
        List of parameterized Task objects ready to pass directly to `inspect_ai.eval()`.
    """
    tasks = []
    for layer in layers:
        target = TargetSpec(layer=layer, location=location)
        task = task_factory(target)
        # Unique task name allows inspect view to group and compare them cleanly
        task.name = f"{name_prefix}_L{layer}_{location}"
        tasks.append(task)
    return tasks


def sweep_coefficients(
    task_factory: Callable[[float], Task],
    coefficients: Sequence[float],
    name_prefix: str = "steering_sweep",
) -> list[Task]:
    """Generates a list of Tasks configured across a range of steering coefficients."""
    tasks = []
    for coeff in coefficients:
        task = task_factory(coeff)
        task.name = f"{name_prefix}_coeff_{coeff:+.2f}"
        tasks.append(task)
    return tasks


# src/inspect_mi/core/target.py


# TODO need to collapse layerlocation into spec
# def sweep_models(
#     task_factory: Callable[[str, TargetSpec], Task],
#     model_configs: Dict[str, int],  # e.g., {"gpt2": 12, "gpt2-medium": 24, "hf/meta-llama/Llama-2-7b-hf": 32}
#     relative_depth: float = 0.5,
#     location: str = "resid_pre",
# ) -> List[Task]:
#     """Generates tasks across multiple models targeting equivalent relative network depths.

#     Args:
#         task_factory: Function taking (model_name, TargetSpec) and returning a Task.
#         model_configs: Dict mapping model identifiers to their total layer counts.
#         relative_depth: Normalized depth in range [0.0, 1.0] (0.5 = mid-network).
#         location: Target hook site (e.g. "resid_pre", "attn_out").

#     Returns:
#         List of Tasks ready for inspect_ai evaluation.
#     """
#     tasks = []
#     for model_name, num_layers in model_configs.items():
#         # loc = HookLocation(relative_depth=relative_depth, location=location)
#         target = TargetSpec(layer = relative_depth, location = location)

#         # Resolve exact layer index for task metadata/naming
#         resolved_layer = loc.resolve_layer(num_layers)

#         task = task_factory(model_name, target)
#         clean_model_id = model_name.split("/")[-1]
#         task.name = f"sweep_{clean_model_id}_depth_{int(relative_depth*100)}pct_L{resolved_layer}"
#         tasks.append(task)

#     return tasks
