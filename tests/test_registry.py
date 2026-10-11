# tests/test_registry.py

import pytest

from inspect_mi.core.model_registry import (
    HOOK_TEMPLATES,
    get_hook_templates,
    register_architecture,
)


def test_canonical_architecture_lookups():
    """Verifies direct canonical architecture key lookups."""
    assert get_hook_templates("gpt2")["block"] == "transformer.h.{layer}"
    assert get_hook_templates("llama")["block"] == "model.layers.{layer}"
    assert get_hook_templates("gpt_neox")["block"] == "gpt_neox.layers.{layer}"


def test_hf_repo_id_alias_matching():
    """Verifies string substring matching for Hugging Face repo identifiers."""
    # Llama variants
    assert (
        get_hook_templates("meta-llama/Meta-Llama-3-8B-Instruct")
        == HOOK_TEMPLATES["llama"]
    )
    assert get_hook_templates("mistralai/Mistral-7B-v0.1") == HOOK_TEMPLATES["llama"]
    assert get_hook_templates("Qwen/Qwen2.5-7B-Instruct") == HOOK_TEMPLATES["llama"]
    assert get_hook_templates("google/gemma-2-9b") == HOOK_TEMPLATES["llama"]

    # Pythia / GPT-NeoX
    assert get_hook_templates("EleutherAI/pythia-160m") == HOOK_TEMPLATES["gpt_neox"]


def test_case_insensitive_matching():
    """Verifies case-insensitive resolution across upper and lower case inputs."""
    assert get_hook_templates("GPT2") == HOOK_TEMPLATES["gpt2"]
    assert get_hook_templates("LLaMA-2-7b") == HOOK_TEMPLATES["llama"]


def test_register_architecture_runtime():
    """Verifies registering a custom architecture family and alias at runtime."""
    phi_templates = {
        "block": "model.layers.{layer}",
        "resid_pre": "{block}",
        "attn_out": "{block}.self_attn",
        "mlp_out": "{block}.mlp",
    }

    register_architecture("phi", phi_templates, aliases=["microsoft/phi-2"])

    # Query via canonical name and alias
    assert get_hook_templates("phi") == phi_templates
    assert get_hook_templates("microsoft/phi-2") == phi_templates


def test_unsupported_architecture_raises_keyerror():
    """Verifies that unmapped architecture strings raise a informative KeyError."""
    with pytest.raises(
        KeyError, match="Unsupported or unregistered model architecture"
    ):
        get_hook_templates("unknown_architecture_v1")
