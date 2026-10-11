from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Location = Literal["resid_pre", "attn_out", "mlp_out", "resid_post", "head_out"]
TokenPolicy = Literal["all", "last_prompt_token", "first_token", "index", "slice"]


class TargetSpec(BaseModel):
    """Complete specification of an intervention/observation target."""

    layer: int = Field(..., description="Target model layer index")
    location: str = Field(
        ...,
        description="Target site location (e.g., 'mlp_out', 'attn_out', 'resid_pre')",
    )
    token_policy: TokenPolicy = Field(
        default="all", description="Token indexing policy"
    )
    token_index: int | None = Field(
        default=None, description="Explicit index when policy is 'index'"
    )
    token_slice: tuple[int | None, int | None] | None = Field(
        default=None,
        description="Explicit (start, stop) slice range when policy is 'slice'",
    )
    module_path: str | None = Field(
        default=None,
        description="Optional manual module string override (bypasses resolver)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Experiment metadata"
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_hook_inputs(cls, data: Any) -> Any:
        """Coerces flexible inputs into flat layer and location fields."""
        if isinstance(data, dict):
            # 1. Unpack if old hook_location key or object is passed in
            if "hook_location" in data:
                hl = data.pop("hook_location")
                if hasattr(hl, "layer") and hasattr(hl, "location"):
                    data["layer"], data["location"] = hl.layer, hl.location
                elif isinstance(hl, (tuple, list)) and len(hl) == 2:
                    data["layer"], data["location"] = hl[0], hl[1]
                elif isinstance(hl, dict):
                    data["layer"], data["location"] = hl["layer"], hl["location"]

            # 2. Allow passing a tuple directly via 'target' or positional key
            if (
                "target" in data
                and isinstance(data["target"], (tuple, list))
                and len(data["target"]) == 2
            ):
                target_tuple = data.pop("target")
                data["layer"], data["location"] = target_tuple[0], target_tuple[1]

        return data

    @model_validator(mode="after")
    def validate_token_indexing(self) -> "TargetSpec":
        if self.token_policy == "index" and self.token_index is None:
            raise ValueError("token_index must be set when token_policy is 'index'")
        if self.token_policy == "slice" and self.token_slice is None:
            raise ValueError("token_slice must be set when token_policy is 'slice'")
        return self


# @dataclass
# class InterventionConfig:
#     """Configures a homogeneous evaluation run (e.g. steering or probing)."""

#     method_type: str
#     targets: List[TargetSpec]
#     params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionConfig:
    targets: list[TargetSpec]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerLocation:
    layer: int | None = None
    relative_depth: float | None = None  # e.g., 0.5 for 50% depth (mid-network)
    location: str = "resid_pre"

    def resolve_layer(self, total_layers: int) -> int:
        """Resolves absolute layer index given total model layers."""
        if self.layer is not None:
            return self.layer
        if self.relative_depth is not None:
            # Map 0.0 -> layer 0, 1.0 -> layer (N-1)
            return min(int(self.relative_depth * total_layers), total_layers - 1)
        raise ValueError("Either layer or relative_depth must be specified.")
