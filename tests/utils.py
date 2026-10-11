import nnterp
import torch
import torch.nn as nn
from transformers import GPT2Config, LlamaConfig


class DummyAttnModule(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(x), torch.ones_like(x)


def make_nnterp_model(model: nn.Module):
    """Wraps a PyTorch nn.Module instance into an nnterp traceable model."""
    return nnterp.StandardizedTransformer(model)


import torch.nn as nn


class DummyLlamaModel(nn.Module):
    def __init__(self, d_model: int = 16):
        super().__init__()
        self.config = LlamaConfig(
            hidden_size=d_model,
            intermediate_size=d_model,  # Prevents LlamaConfig from defaulting to 32
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=100,
        )

        self.model = nn.ModuleDict(
            {
                "embed_tokens": nn.Embedding(100, d_model),
                "layers": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "self_attn": nn.Linear(d_model, d_model),
                                "mlp": nn.Linear(d_model, d_model),
                            }
                        )
                    ]
                ),
                "norm": nn.Identity(),
            }
        )
        self.lm_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.long:
            h = self.model["embed_tokens"](x)
        else:
            h = x

        h = self.model["layers"][0]["self_attn"](h)
        h = self.model["layers"][0]["mlp"](h)
        h = self.model["norm"](h)
        return self.lm_head(h)


class DummyGPT2Model(nn.Module):
    def __init__(self, d_model: int = 16):
        super().__init__()
        self.config = GPT2Config(
            n_embd=d_model,
            n_inner=d_model,  # Prevents n_inner from defaulting to 4 * d_model (64)
            n_layer=1,
            n_head=2,
            vocab_size=100,
        )

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(100, d_model),
                "h": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "attn": nn.Linear(d_model, d_model),
                                "mlp": nn.Linear(d_model, d_model),
                            }
                        )
                    ]
                ),
                # StandardizedTransformer looks for transformer.ln_f to rename to ln_final
                "ln_f": nn.Identity(),
            }
        )
        self.lm_head = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.long:
            h = self.transformer["wte"](x)
        else:
            h = x

        h = self.transformer["h"][0]["attn"](h)
        h = self.transformer["h"][0]["mlp"](h)
        h = self.transformer["ln_f"](h)
        return self.lm_head(h)
