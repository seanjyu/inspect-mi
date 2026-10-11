import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dummy_model() -> GPT2LMHeadModel:
    """Provides a tiny, randomly-initialized GPT-2 model on CPU for testing."""
    config = GPT2Config(
        vocab_size=100,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


"""Shared pytest fixtures for inspect_mi test suite."""

import pytest
import torch
from nnterp import StandardizedTransformer

from inspect_mi.core.context import MIContext
from inspect_mi.core.schema import TargetSpec
from tests.utils import DummyGPT2Model, DummyLlamaModel, make_nnterp_model


@pytest.fixture
def dummy_tokens():
    """Batch of 2 sequences with sequence length of 4."""
    return torch.tensor([[1, 5, 12, 8], [3, 9, 21, 4]], dtype=torch.long)


# @pytest.fixture
# def dummy_gpt2():
#     """Instantiates a 2-layer GPT-2 model on CPU."""
#     model = DummyGPT2Model(num_layers=2, d_model=16, vocab_size=100)
#     model.eval()
#     return model


# @pytest.fixture
# def dummy_llama():
#     """Instantiates a 2-layer Llama model on CPU."""
#     model = DummyLlamaModel(num_layers=2, d_model=16, vocab_size=100)
#     model.eval()
#     return model

# @pytest.fixture
# def input_tokens():
#     """Dummy token batch [batch_size=2, seq_len=4]."""
#     return torch.tensor([[1, 5, 2, 8], [3, 9, 4, 7]], dtype=torch.long)
#     # return torch.randn(2, 4, 16)
# @pytest.fixture
# def input_tokens():
#     """Float activation batch [batch_size=2, seq_len=4, d_model=16]."""
#     return torch.randn(2, 4, 16)


@pytest.fixture
def input_tokens():
    """Discrete token IDs batch [batch_size=2, seq_len=4]."""
    torch.manual_seed(42)
    return torch.randint(0, 100, (2, 4), dtype=torch.long)


# @pytest.fixture
# def input_tokens():
#     """Discrete token IDs batch [batch_size=2, seq_len=4]."""
#     return torch.randint(0, 100, (2, 4), dtype=torch.long)

# @pytest.fixture
# def input_activations(input_tokens):
#     """Alias for input_tokens to support both parameter names."""
#     return input_tokens


@pytest.fixture
def gpt2_model():
    torch.manual_seed(42)
    model = DummyGPT2Model(d_model=16)
    model.eval()
    return model


@pytest.fixture
def llama_model():
    torch.manual_seed(42)
    model = DummyLlamaModel(d_model=16)
    model.eval()
    return model


# @pytest.fixture
# def nnterp_gpt2_model(gpt2_model):
#     """nnterp-wrapped DummyGPT2Model with .trace() support."""
#     return make_nnterp_model(gpt2_model)


@pytest.fixture
def nnterp_llama_model(llama_model):
    """nnterp-wrapped DummyLlamaModel with .trace() support."""
    return make_nnterp_model(llama_model)


@pytest.fixture
def sample_context():
    """Initializes a basic MIContext with target resolution configured."""
    ctx = MIContext(sample_id="test_sample_0")

    # Layer 0 residual stream pre
    target_l0 = TargetSpec(layer=0, location="resid_pre", module_path="transformer.h.0")

    ctx.targets[target_l0.module_path] = target_l0
    return ctx


@pytest.fixture
def nnterp_gpt2_model():
    """Loads a minimal HF GPT-2 model compatible with nnterp tracing."""
    model = StandardizedTransformer("hf-internal-testing/tiny-random-GPT2LMHeadModel")
    model.eval()
    return model


@pytest.fixture
def input_ids(nnterp_gpt2_model):
    """Discrete integer token IDs derived from the model's vocab_size."""
    vocab_size = getattr(nnterp_gpt2_model.config, "vocab_size", 100)
    # [batch_size=2, seq_len=4] with values in [0, vocab_size)
    return torch.randint(0, vocab_size, (2, 4), dtype=torch.long)


@pytest.fixture
def input_activations(nnterp_gpt2_model):
    """Float activation embeddings matching the model's actual hidden_size."""
    d_model = getattr(
        nnterp_gpt2_model.config,
        "hidden_size",
        getattr(nnterp_gpt2_model.config, "n_embd", 32),
    )
    # [batch_size=2, seq_len=4, d_model]
    return torch.randn(2, 4, d_model, dtype=torch.float32)
