
from inspect_mi.core.model_registry import get_hook_templates
from inspect_mi.core.schema import TargetSpec


class LayerResolutionError(Exception):
    """Raised when a location cannot be resolved to a submodule path."""

    pass

def resolve_path(target: TargetSpec, model_type: str = "gpt2") -> str:
    """Translates a TargetSpec into an nnterp-accessible submodule path string."""
    if target.module_path:
        return target.module_path

    if target.location is None or target.layer is None:
        raise LayerResolutionError(
            "TargetSpec must specify both 'location' and 'layer' when 'module_path' is omitted."
        )

    try:
        templates = get_hook_templates(model_type)
    except KeyError as e:
        raise LayerResolutionError(str(e)) from e

    if target.location not in templates:
        raise LayerResolutionError(
            f"Unsupported hook location '{target.location}' for model type '{model_type}'"
        )

    block_path = templates["block"].format(layer=target.layer)
    template_str = templates[target.location]

    # Render template (supports formatting with {block} or direct {layer})
    return template_str.format(block=block_path, layer=target.layer)
