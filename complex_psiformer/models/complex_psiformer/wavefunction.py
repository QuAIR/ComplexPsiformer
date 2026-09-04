"""Complex PsiFormer wavefunction."""

from __future__ import annotations

import torch
import torch.nn as nn

from ...features.magnetic import TorusMagneticSections
from ...features.periodic import ComplexPeriodicFeatures
from ...systems.supercell import Supercell
from ..kerr_mlp import FreeComplexLinear, KerrActivation
from .block import ComplexBlock


class ComplexPsiFormer(nn.Module):
    """Complex multi-determinant wavefunction with Complex PsiFormer residual blocks."""

    def __init__(
        self,
        cell: Supercell,
        d_model: int = 32,
        n_layers: int = 3,
        n_heads: int = 2,
        d_attn: int = 8,
        d_val: int = 8,
        n_det: int = 4,
        *,
        trainable_determinant_coefficients: bool = False,
        embedding_kind: str = "free",
        per_block_kind: str = "free",
        activation: str = "kerr",
        gate_init: float = 1.0,
        gate_trainable: bool = False,
        attention_kind: str = "all_pairs",
        attention_score_mode: str = "abs_dot",
        attention_score_scale: float = 1.0,
        gate_parameterization: str = "positive_softplus",
        gate_max: float = 1.0,
        attention_residual_norm: str = "rms_cap",
        attention_residual_norm_eps: float = 1e-6,
        attention_residual_norm_cap_ratio: float = 1.0,
        magnetic_B: float | None = None,
        magnetic_flux_quanta: int | float | None = None,
        magnetic_origin: torch.Tensor | tuple[float, float] | None = None,
        enforce_magnetic_boundary: bool = False,
    ) -> None:
        super().__init__()
        required = {
            "embedding_kind": (embedding_kind, "free"),
            "per_block_kind": (per_block_kind, "free"),
            "activation": (activation, "kerr"),
            "attention_kind": (attention_kind, "all_pairs"),
            "gate_parameterization": (gate_parameterization, "positive_softplus"),
            "attention_residual_norm": (attention_residual_norm, "rms_cap"),
        }
        for name, (actual, expected) in required.items():
            if actual != expected:
                raise ValueError(f"Complex PsiFormer requires {name}={expected!r}, got {actual!r}")
        attention_score_modes = {
            "abs",
            "abs_dot",
            "real_dot",
            "imag_dot",
            "abs_phase_transport",
            "quadrature_dot",
            "coherent_kernel",
        }
        if attention_score_mode not in attention_score_modes:
            modes = ", ".join(repr(mode) for mode in sorted(attention_score_modes))
            raise ValueError(
                f"attention_score_mode must be one of {modes}, got {attention_score_mode!r}"
            )
        if n_layers <= 0 or n_det <= 0:
            raise ValueError("n_layers and n_det must be positive")

        self.cell = cell
        self.n_electrons = cell.n_electrons
        self.n_det = int(n_det)
        self.trainable_determinant_coefficients = bool(trainable_determinant_coefficients)
        self.embedding_kind = embedding_kind
        self.per_block_kind = per_block_kind
        self.activation_kind = activation
        self.gate_init = float(gate_init)
        self.gate_trainable = bool(gate_trainable)
        self.attention_kind = attention_kind
        self.attention_score_mode = attention_score_mode
        self.attention_score_scale = float(attention_score_scale)
        self.gate_parameterization = gate_parameterization
        self.gate_max = float(gate_max)
        self.attention_residual_norm = attention_residual_norm
        self.attention_residual_norm_eps = float(attention_residual_norm_eps)
        self.attention_residual_norm_cap_ratio = float(attention_residual_norm_cap_ratio)
        self.enforce_magnetic_boundary = bool(enforce_magnetic_boundary)
        self.magnetic_B = None if magnetic_B is None else float(magnetic_B)
        self.magnetic_flux_quanta = magnetic_flux_quanta

        self.features = ComplexPeriodicFeatures(cell)
        self.embed = FreeComplexLinear(self.features.out_dim, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [
                ComplexBlock(
                    d_model,
                    n_heads,
                    d_attn,
                    d_val,
                    gate_init=gate_init,
                    gate_trainable=gate_trainable,
                    score_mode=attention_score_mode,
                    score_scale_mult=attention_score_scale,
                    gate_parameterization=gate_parameterization,
                    gate_max=gate_max,
                    residual_norm=attention_residual_norm,
                    residual_norm_eps=attention_residual_norm_eps,
                    residual_norm_cap_ratio=attention_residual_norm_cap_ratio,
                    activation=activation,
                )
                for _ in range(n_layers)
            ]
        )

        scale = d_model**-0.5
        self.w_re = nn.Parameter(torch.randn(n_det, cell.n_electrons, d_model) * scale)
        self.w_im = nn.Parameter(torch.randn(n_det, cell.n_electrons, d_model) * scale)
        if self.trainable_determinant_coefficients:
            self.c = nn.Parameter(torch.ones(n_det) / n_det)
        else:
            self.register_parameter("c", None)

        if self.enforce_magnetic_boundary:
            if magnetic_B is None or magnetic_flux_quanta is None or magnetic_origin is None:
                raise ValueError(
                    "magnetic_B, magnetic_flux_quanta, and magnetic_origin are required "
                    "when enforce_magnetic_boundary=True"
                )
            origin = torch.as_tensor(
                magnetic_origin,
                device=cell.L.device,
                dtype=cell.L.dtype,
            ).clone()
            self.register_buffer("magnetic_origin", origin, persistent=False)
            self.magnetic_sections: TorusMagneticSections | None = TorusMagneticSections(
                cell,
                magnetic_B=magnetic_B,
                magnetic_flux_quanta=magnetic_flux_quanta,
                origin=origin,
                n_sections=cell.n_electrons,
            )
        else:
            self.register_buffer("magnetic_origin", None, persistent=False)
            self.magnetic_sections = None

    def set_gate_multiplier(self, value: float) -> None:
        """Set the warmup multiplier on every Complex PsiFormer block."""
        for block in self.blocks:
            block.attn.set_gate_multiplier(value)

    def imag_params(self) -> list[nn.Parameter]:
        """Return the historical imaginary-channel regularization tensors."""
        tensors: list[nn.Parameter] = [self.embed.W_AB, self.embed.W_BA]
        for block in self.blocks:
            mlp_linear = block.mlp.linear
            tensors.extend([mlp_linear.W_AB, mlp_linear.W_BA])
            if mlp_linear.b_im is not None:
                tensors.append(mlp_linear.b_im)
            if isinstance(block.mlp.activation, KerrActivation):
                tensors.append(block.mlp.activation.r0)
            for head in block.attn.heads:
                for layer in (head.Wq, head.Wk, head.Wv):
                    tensors.extend([layer.W_AB, layer.W_BA])
            tensors.extend([block.attn.Wo.W_AB, block.attn.Wo.W_BA])
        return tensors

    def _compute_orbitals(self, r: torch.Tensor) -> torch.Tensor:
        features = self.features(r)
        h = self.embed(features)
        for block in self.blocks:
            h = block(h, r)
        w_conj = torch.complex(self.w_re, -self.w_im)
        orbitals = torch.einsum("bid,mjd->bmij", h, w_conj)
        if self.magnetic_sections is not None:
            orbitals = orbitals * self.magnetic_sections(r).unsqueeze(1)
        return orbitals

    def log_psi(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stable ``log|Psi|`` and the complex unit phase."""
        phi = self._compute_orbitals(r)
        signs, logdets = torch.linalg.slogdet(phi)
        max_logdet = logdets.max(dim=-1, keepdim=True).values
        det_vals = signs * torch.exp(logdets - max_logdet)
        if self.c is None:
            psi = det_vals.sum(dim=-1)
        else:
            psi = (self.c.to(det_vals.dtype) * det_vals).sum(dim=-1)
        abs_psi = torch.abs(psi)
        log_abs = torch.log(abs_psi) + max_logdet.squeeze(-1)
        phase = psi / (abs_psi + 1e-30)
        return log_abs, phase

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return ``log|Psi|`` for Metropolis sampling."""
        return self.log_psi(r)[0]
