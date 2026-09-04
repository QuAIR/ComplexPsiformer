"""Complex self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kerr_mlp import FreeComplexLinear

_HEAD_SCORE_MODES = frozenset(
    {
        "abs",
        "abs_dot",
        "real_dot",
        "imag_dot",
        "abs_phase_transport",
        "coherent_kernel",
    }
)
_MULTIHEAD_SCORE_MODES = _HEAD_SCORE_MODES | {"quadrature_dot"}
_PHASE_EPS = 1e-6


def _zero_cross(layer: FreeComplexLinear) -> None:
    """Start a free complex map without real/imaginary cross coupling."""
    with torch.no_grad():
        layer.W_AB.zero_()
        layer.W_BA.zero_()


def gate_warmup_multiplier(step: int, warmup_steps: int = 500) -> float:
    """Return the non-trainable linear gate multiplier for one step."""
    if warmup_steps <= 0:
        return 1.0
    return min(max(float(step) / float(warmup_steps), 0.0), 1.0)


class ComplexAttentionHead(nn.Module):
    """One dense all-pairs Hermitian-magnitude attention head."""

    def __init__(
        self,
        d_model: int,
        d_attn: int,
        d_val: int,
        *,
        score_mode: str = "abs_dot",
        score_scale_mult: float = 1.0,
    ) -> None:
        super().__init__()
        if score_mode not in _HEAD_SCORE_MODES:
            modes = ", ".join(repr(mode) for mode in sorted(_HEAD_SCORE_MODES))
            raise ValueError(f"score_mode must be one of {modes}, got {score_mode!r}")
        if d_model <= 0 or d_attn <= 0 or d_val <= 0:
            raise ValueError("d_model, d_attn, and d_val must be positive")
        if score_scale_mult <= 0.0:
            raise ValueError("score_scale_mult must be positive")

        self.d_attn = int(d_attn)
        self.d_val = int(d_val)
        self.score_mode = score_mode
        self.score_scale_mult = float(score_scale_mult)
        self.Wq = FreeComplexLinear(d_model, d_attn, bias=False)
        self.Wk = FreeComplexLinear(d_model, d_attn, bias=False)
        self.Wv = FreeComplexLinear(d_model, d_val, bias=False)
        _zero_cross(self.Wq)
        _zero_cross(self.Wk)
        _zero_cross(self.Wv)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Aggregate complex values with real Complex PsiFormer weights."""
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)

        raw = q.real @ k.real.transpose(-1, -2) + q.imag @ k.imag.transpose(-1, -2)
        raw_cross = q.real @ k.imag.transpose(-1, -2) - q.imag @ k.real.transpose(-1, -2)
        scale = self.score_scale_mult * (self.d_attn**-0.5)
        if self.score_mode == "abs":
            raw_abs = torch.sqrt(raw.square() + raw_cross.square() + 1e-12)
            eps = 1e-6
            q_norm = torch.sqrt((q.real.square() + q.imag.square()).sum(-1) + eps)
            k_norm = torch.sqrt((k.real.square() + k.imag.square()).sum(-1) + eps)
            score = scale * raw_abs / (q_norm.unsqueeze(-1) * k_norm.unsqueeze(-2) + eps)
        elif self.score_mode == "abs_dot":
            score = scale * torch.abs(torch.complex(raw, raw_cross))
        elif self.score_mode == "real_dot":
            score = scale * raw
        elif self.score_mode == "imag_dot":
            score = scale * raw_cross
        else:
            overlap = torch.complex(raw, raw_cross)
            overlap_abs = torch.abs(overlap)
            if self.score_mode == "abs_phase_transport":
                score = scale * overlap_abs
                alpha = F.softmax(score, dim=-1).to(overlap.dtype)
                transport = overlap.conj() / torch.sqrt(overlap_abs.square() + _PHASE_EPS**2)
                return (alpha * transport) @ v
            denominator = overlap_abs.sum(dim=-1, keepdim=True) + _PHASE_EPS
            return (overlap.conj() / denominator) @ v
        alpha = F.softmax(score, dim=-1)
        return torch.complex(alpha @ v.real, alpha @ v.imag)


class ComplexMultiHeadAttention(nn.Module):
    """Multi-head Complex PsiFormer with one-sided RMS control and a positive gate."""

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
    ) -> None:
        super().__init__()
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if score_mode not in _MULTIHEAD_SCORE_MODES:
            modes = ", ".join(repr(mode) for mode in sorted(_MULTIHEAD_SCORE_MODES))
            raise ValueError(f"score_mode must be one of {modes}, got {score_mode!r}")
        if score_mode == "quadrature_dot" and n_heads % 2 != 0:
            raise ValueError("quadrature_dot requires an even number of heads")
        if gate_parameterization != "positive_softplus":
            raise ValueError("Complex PsiFormer requires gate_parameterization='positive_softplus'")
        if residual_norm != "rms_cap":
            raise ValueError("Complex PsiFormer requires residual_norm='rms_cap'")
        if gate_init <= 0.0:
            raise ValueError("gate_init must be positive")
        if residual_norm_eps <= 0.0 or residual_norm_cap_ratio <= 0.0:
            raise ValueError("RMS epsilon and cap ratio must be positive")

        self.gate_parameterization = gate_parameterization
        self.gate_max = float(gate_max)
        self.residual_norm = residual_norm
        self.residual_norm_eps = float(residual_norm_eps)
        self.residual_norm_cap_ratio = float(residual_norm_cap_ratio)
        head_score_modes = (
            ["real_dot" if index % 2 == 0 else "imag_dot" for index in range(n_heads)]
            if score_mode == "quadrature_dot"
            else [score_mode] * n_heads
        )
        self.heads = nn.ModuleList(
            [
                ComplexAttentionHead(
                    d_model,
                    d_attn,
                    d_val,
                    score_mode=head_score_mode,
                    score_scale_mult=score_scale_mult,
                )
                for head_score_mode in head_score_modes
            ]
        )
        self.Wo = FreeComplexLinear(n_heads * d_val, d_model, bias=False)
        _zero_cross(self.Wo)

        gate_tensor = torch.tensor(math.log(math.expm1(max(float(gate_init), 1e-12))))
        if gate_trainable:
            self.gate_raw = nn.Parameter(gate_tensor)
        else:
            self.register_buffer("gate_raw", gate_tensor)
        self.register_buffer("gate_multiplier", torch.tensor(1.0))

        # Non-trainable state buffers used by the residual block.
        self.register_buffer("residual_norm_blend", torch.tensor(1.0))
        self.register_buffer("residual_norm_amp_cap", torch.tensor(1.0))

    def effective_gate(self) -> torch.Tensor:
        """Return the positive gate (fixed in the published configuration)."""
        return F.softplus(self.gate_raw)

    def set_gate_multiplier(self, value: float) -> None:
        """Set the non-trainable warmup multiplier."""
        self.gate_multiplier.fill_(float(value))

    def apply_rms_cap(self, out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Match oversized branch RMS to the input without amplification."""
        eps = self.residual_norm_eps
        out_rms = torch.sqrt(out.abs().square().mean(dim=(-2, -1), keepdim=True) + eps)
        h_rms = torch.sqrt(h.abs().square().mean(dim=(-2, -1), keepdim=True) + eps)
        scale_match = h_rms / (out_rms + eps)
        scale = torch.clamp(self.residual_norm_cap_ratio * scale_match, max=1.0)
        return out * scale

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply all heads, output projection, RMS cap, and positive gate."""
        head_out = torch.cat([head(h) for head in self.heads], dim=-1)
        out = self.apply_rms_cap(self.Wo(head_out), h)
        gate = self.gate_multiplier * self.effective_gate()
        return gate.to(out.dtype) * out
