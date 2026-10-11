# # from typing import Dict, Optional
# # from inspect_mi.core.schema import CanonicalLocation, CanonicalPoint, TargetSpec


# # class LayerResolutionError(Exception):
# #     """Raised when a canonical point cannot be translated to an nnterp path."""
# #     pass


# # # Standardized submodule suffix mappings for nnterp / NNsight models
# # LOCATION_SUFFIX_MAP: Dict[CanonicalLocation, str] = {
# #     CanonicalLocation.RESID_PRE: "",            # Resolves directly to the layer block
# #     CanonicalLocation.ATTN_OUT: ".self_attn",   # or .attn depending on nnterp binding
# #     CanonicalLocation.MLP_OUT: ".mlp",
# #     CanonicalLocation.RESID_POST: "",           # Captured via post-hook on block
# # }


# # class NNTerpLayerResolver:
# #     """Translates CanonicalPoints into nnterp-compatible module target strings."""

# #     @classmethod
# #     def resolve_target_path(cls, target: TargetSpec) -> str:
# #         """Returns the submodule path string for nnterp tracing."""
# #         # Tier 1: Explicit module path override
# #         if target.module_path:
# #             return target.module_path

# #         # Tier 2: Canonical point translation
# #         point = target.point
# #         suffix = LOCATION_SUFFIX_MAP.get(point.location)
# #         if suffix is None:
# #             raise LayerResolutionError(f"Unsupported canonical location: {point.location}")

# #         # Standard nnterp block path indexing (e.g., 'model.layers.16.mlp')
# #         base_path = f"model.layers.{point.layer}"
# #         return f"{base_path}{suffix}"

# from typing import Dict, Optional, Union
# from inspect_mi.core.schema import CanonicalLocation, CanonicalPoint, TargetSpec


# class LayerResolutionError(Exception):
#     """Raised when a canonical point cannot be translated to an nnterp/PyTorch path."""
#     pass


# # Standardized submodule suffix mappings
# LOCATION_SUFFIX_MAP: Dict[CanonicalLocation, str] = {
#     CanonicalLocation.RESID_PRE: "",          # Resolves directly to the layer block
#     CanonicalLocation.ATTN_OUT: ".self_attn", # Modified dynamically for GPT-2 (.attn)
#     CanonicalLocation.MLP_OUT: ".mlp",
#     CanonicalLocation.RESID_POST: "",         # Captured via post-hook on block
# }


# class NNTerpLayerResolver:
#     """Translates CanonicalPoints or TargetSpecs into module target strings."""

#     @classmethod
#     def resolve_target_path(
#         cls, target: Union[TargetSpec, CanonicalPoint], model_type: str = "llama"
#     ) -> str:
#         """Returns the submodule path string for tracing or hook registration.

#         Args:
#             target: A TargetSpec or CanonicalPoint to resolve.
#             model_type: Architecture type (e.g. 'gpt2', 'llama', 'gemma', 'mistral').
#         """
#         # Tier 1: Explicit module path override on TargetSpec
#         if isinstance(target, TargetSpec):
#             if getattr(target, "module_path", None):
#                 return target.module_path
#             point = target.point
#         else:
#             point = target

#         model_type_lower = model_type.lower()

#         # Architecture prefix resolution
#         if "gpt2" in model_type_lower:
#             base_path = f"transformer.h.{point.layer}"
#             attn_suffix = ".attn"
#         else:
#             base_path = f"model.layers.{point.layer}"
#             attn_suffix = ".self_attn"

#         # Location suffix resolution
#         if point.location == CanonicalLocation.ATTN_OUT:
#             suffix = attn_suffix
#         else:
#             suffix = LOCATION_SUFFIX_MAP.get(point.location)

#         if suffix is None:
#             raise LayerResolutionError(f"Unsupported canonical location: {point.location}")

#         return f"{base_path}{suffix}"


# def resolve_canonical_path(
#     target: Union[TargetSpec, CanonicalPoint], model_type: str = "gpt2"
# ) -> str:
#     """Top-level functional interface for resolving canonical paths."""
#     return NNTerpLayerResolver.resolve_target_path(target, model_type=model_type)

from inspect_mi.core.model_registry import get_hook_templates
from inspect_mi.core.schema import TargetSpec


class LayerResolutionError(Exception):
    """Raised when a location cannot be resolved to a submodule path."""

    pass


# def resolve_path(target: TargetSpec, model_type: str = "gpt2") -> str:
#     #TODO Do we need model_type? is it really different for gpt2 vs other models?
#     """Translates a TargetSpec into an nnterp-accessible submodule path string."""
#     if target.module_path:
#         return target.module_path

#     loc = target.hook_location
#     model_type_lower = model_type.lower()

#     # Determine base block path and attention submodule name
#     if "gpt2" in model_type_lower:
#         block = f"transformer.h.{loc.layer}"
#         attn_submodule = "attn"
#     else:  # Llama, Gemma, Mistral, Qwen
#         block = f"model.layers.{loc.layer}"
#         attn_submodule = "self_attn"

#     # Explicit mapping for all site locations
#     mapping = {
#         "resid_pre": block,
#         "attn_out": f"{block}.{attn_submodule}",
#         "mlp_out": f"{block}.mlp",
#         "resid_post": block,
#         "head_out": f"{block}.{attn_submodule}",
#     }

#     if loc.location not in mapping:
#         raise LayerResolutionError(f"Unsupported hook location: {loc.location}")

#     return mapping[loc.location]

# def resolve_path(target: TargetSpec, model_type: str = "gpt2") -> str:
#     """Translates a TargetSpec into an nnterp-accessible submodule path string."""
#     if target.module_path:
#         return target.module_path

#     loc = target.hook_location

#     try:
#         templates = get_hook_templates(model_type)
#     except KeyError as e:
#         raise LayerResolutionError(str(e)) from e

#     if loc.location not in templates:
#         raise LayerResolutionError(
#             f"Unsupported hook location '{loc.location}' for model type '{model_type}'"
#         )

#     block_path = templates["block"].format(layer=loc.layer)
#     template_str = templates[loc.location]

#     # Render template (supports formatting with {block} or direct {layer})
#     return template_str.format(block=block_path, layer=loc.layer)


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
