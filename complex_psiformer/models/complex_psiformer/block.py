"""Complex residual attention block."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..kerr_mlp import ComplexMLP
from .attention import ComplexMultiHeadAttention


class ComplexBlock(nn.Module):
    """Apply an attention residual followed by a residual complex MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_attn: int,
        d_val: int,
        *,
        gate_init: float = 1.0,
        gate_trainable: bool = False,
        score_mode: str = "abs_dot",
        score_scale_mult: float = 1.0,
        gate_parameterization: str = "positive_softplus",
        gate_max: float = 1.0,
        residual_norm: str = "rms_cap",
        residual_norm_eps: float = 1e-6,
        residual_norm_cap_ratio: float = 1.0,
        activation: str = "kerr",
    ) -> None:
        super().__init__()
        if activation != "kerr":
            raise ValueError("Complex PsiFormer requires activation='kerr'")
        self.attn = ComplexMultiHeadAttention(
            d_model,
            n_heads,
            d_attn,
            d_val,
            gate_init=gate_init,
            gate_trainable=gate_trainable,
            score_mode=score_mode,
            score_scale_mult=score_scale_mult,
            gate_parameterization=gate_parameterization,
            gate_max=gate_max,
            residual_norm=residual_norm,
            residual_norm_eps=residual_norm_eps,
            residual_norm_cap_ratio=residual_norm_cap_ratio,
        )
        self.mlp = ComplexMLP(d_model, activation=activation, linear_kind="free")

    def forward(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Apply dense all-pairs attention; ``r`` is kept for API parity."""
        del r
        return self.mlp(h + self.attn(h))
