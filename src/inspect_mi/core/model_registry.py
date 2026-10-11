"""Registry mapping model architecture families to submodule path templates."""


HOOK_TEMPLATES: dict[str, dict[str, str]] = {
    "gpt2": {
        "block": "transformer.h.{layer}",
        "resid_pre": "{block}",
        "attn_out": "{block}.attn",
        "mlp_out": "{block}.mlp",
        "resid_post": "{block}",
        "head_out": "{block}.attn",
    },
    "llama": {  # Covers Llama 2/3, Gemma, Mistral, Qwen
        "block": "model.layers.{layer}",
        "resid_pre": "{block}",
        "attn_out": "{block}.self_attn",
        "mlp_out": "{block}.mlp",
        "resid_post": "{block}",
        "head_out": "{block}.self_attn",
    },
    "gpt_neox": {  # Covers Pythia, GPT-NeoX
        "block": "gpt_neox.layers.{layer}",
        "resid_pre": "{block}",
        "attn_out": "{block}.attention",
        "mlp_out": "{block}.mlp",
        "resid_post": "{block}",
        "head_out": "{block}.attention",
    },
}

# String match priority lookup table
ARCH_ALIASES: dict[str, str] = {
    "gpt2": "gpt2",
    "llama": "llama",
    "mistral": "llama",
    "qwen": "llama",
    "gemma": "llama",
    "pythia": "gpt_neox",
    "gpt_neox": "gpt_neox",
}


def register_architecture(
    arch_name: str,
    templates: dict[str, str],
    aliases: list[str] | None = None,
) -> None:
    """Registers or updates a model architecture hook specification at runtime.

    Args:
        arch_name: Unique canonical name for the architecture family.
        templates: Dictionary mapping site locations ("block", "resid_pre", etc.) to string templates.
        aliases: Optional list of model name substrings that should map to this architecture.
    """
    HOOK_TEMPLATES[arch_name] = templates
    ARCH_ALIASES[arch_name] = arch_name
    if aliases:
        for alias in aliases:
            ARCH_ALIASES[alias.lower()] = arch_name


def get_hook_templates(model_type: str) -> dict[str, str]:
    """Resolves a model_type or model checkpoint string to its architecture templates."""
    model_type_lower = model_type.lower()

    # Match registered alias substrings
    arch_key = None
    for alias, canonical_key in ARCH_ALIASES.items():
        if alias in model_type_lower:
            arch_key = canonical_key
            break

    if not arch_key or arch_key not in HOOK_TEMPLATES:
        raise KeyError(
            f"Unsupported or unregistered model architecture type: '{model_type}'. "
            f"Use `register_architecture()` to add support."
        )

    return HOOK_TEMPLATES[arch_key]
