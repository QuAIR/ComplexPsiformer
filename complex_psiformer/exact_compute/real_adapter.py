# Adapted from qm-perf9828 commit 9828f69871ed47baecf07fbc3bfba88969db0cd5.
"""Exact forward-Laplacian adapter for the registered magnetic Real Psiformer."""

from __future__ import annotations

import torch
from torch import nn

from complex_psiformer.features.magnetic import TorusMagneticSections
from complex_psiformer.features.periodic import RealPeriodicFeatures
from .primitives import (
    ComplexVGL,
    add,
    coordinate_seed,
    matmul,
    product,
    real_linear,
    real_softmax,
)
from .policy import MatmulBackend
from .complex_adapter import (
    ComplexLogPsiVGL,
    _cat,
    _scale,
    _complex,
    _cos,
    _logpsi_from_orbitals,
    _magnetic_sections_vgl,
    _reshape,
    _sin,
    _split,
    _stack,
    _tanh,
    _transpose,
    _unsqueeze,
)
from complex_psiformer.models.real_psiformer import RealPsiFormer
from complex_psiformer.models.real_psiformer import RealAttentionBlock, RealMultiHeadAttention


def _validate_real_support(model: RealPsiFormer) -> None:
    """Reject any unregistered primitive before seeding coordinate work."""
    if type(model) is not RealPsiFormer:
        raise TypeError("magnetic_real_log_psi_vgl requires RealPsiFormer")
    if not isinstance(model.features, RealPeriodicFeatures):
        raise TypeError("Real forward Laplacian requires RealPeriodicFeatures")
    if not isinstance(model.embed, nn.Linear) or model.embed.bias is not None:
        raise TypeError("Real forward Laplacian requires the registered bias-free embedding")
    for block in model.blocks:
        if not isinstance(block, RealAttentionBlock) or not isinstance(block.attn, RealMultiHeadAttention):
            raise TypeError("Real forward Laplacian requires registered attention blocks")
        if not isinstance(block.attn.Wqkv, nn.Linear) or block.attn.Wqkv.bias is not None:
            raise TypeError("Real forward Laplacian requires the registered fused bias-free QKV map")
        if not isinstance(block.attn.Wo, nn.Linear) or block.attn.Wo.bias is not None:
            raise TypeError("Real forward Laplacian requires the registered bias-free output map")
        if any(not isinstance(layer, nn.Linear) for layer in block.mlp_layers):
            raise TypeError("Real forward Laplacian requires linear tanh residual layers")


def _validate_coordinates(model: RealPsiFormer, r: torch.Tensor) -> None:
    if not isinstance(r, torch.Tensor) or r.ndim != 3 or r.shape[-1] != 2:
        shape = None if not isinstance(r, torch.Tensor) else tuple(r.shape)
        raise ValueError(f"r must have shape [B,N,2], got {shape}")
    if r.shape[0] <= 0 or r.shape[1] != model.n_electrons:
        raise ValueError(f"r must have a nonempty batch and {model.n_electrons} electrons")
    if not torch.is_floating_point(r):
        raise TypeError("r must be real floating-point coordinates")


def _real_features_from_seed(model: RealPsiFormer, coordinates: ComplexVGL) -> ComplexVGL:
    phases = real_linear(coordinates, model.features.G.to(coordinates.value.dtype))
    return _cat([_sin(phases), _cos(phases)], dim=-1)


def _real_attention_vgl(
    attention: RealMultiHeadAttention,
    h: ComplexVGL,
    *,
    matmul_backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    if attention.attention_normalization not in ("paper", "standard_scaled"):
        raise ValueError("unsupported real attention normalization")
    batch, n_electrons, _ = h.value.shape
    qkv = real_linear(h, attention.Wqkv.weight)
    q, k, v = _split(
        qkv,
        (
            attention.n_heads * attention.d_attn,
            attention.n_heads * attention.d_attn,
            attention.n_heads * attention.d_val,
        ),
        dim=-1,
    )
    q = _transpose(_reshape(q, (batch, n_electrons, attention.n_heads, attention.d_attn)), 1, 2)
    k = _transpose(_reshape(k, (batch, n_electrons, attention.n_heads, attention.d_attn)), 1, 2)
    v = _transpose(_reshape(v, (batch, n_electrons, attention.n_heads, attention.d_val)), 1, 2)
    score = product(
        matmul(
            q,
            _transpose(k, -1, -2),
            backend=matmul_backend,
        ),
        ComplexVGL(
            h.value.new_tensor(1.0 if attention.attention_normalization == "paper" else attention.d_attn**-0.5),
            h.value.new_zeros(h.coordinate_shape),
            h.value.new_zeros(()),
            h.coordinate_shape,
        ),
    )
    alpha = real_softmax(score, dim=-1)
    out = matmul(
        alpha,
        v,
        backend=matmul_backend,
    )
    if attention.attention_normalization == "paper":
        out = _scale(out, attention.d_val**-0.5)
    out = _reshape(_transpose(out, 1, 2), (batch, n_electrons, attention.n_heads * attention.d_val))
    return real_linear(out, attention.Wo.weight)


def _real_streams_from_seed(
    model: RealPsiFormer,
    coordinates: ComplexVGL,
    *,
    matmul_backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    h = real_linear(_real_features_from_seed(model, coordinates), model.embed.weight)
    for block in model.blocks:
        h = add(
            h,
            _real_attention_vgl(
                block.attn,
                h,
                matmul_backend=matmul_backend,
            ),
        )
        for layer in block.mlp_layers:
            h = add(h, _tanh(real_linear(h, layer.weight, layer.bias)))
    return h


def _real_orbitals_from_seed(
    model: RealPsiFormer,
    coordinates: ComplexVGL,
    *,
    matmul_backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    h = _real_streams_from_seed(
        model,
        coordinates,
        matmul_backend=matmul_backend,
    )
    determinant_orbitals = []
    for determinant in range(model.n_det):
        determinant_orbitals.append(
            _complex(
                real_linear(h, model.w_re[determinant]),
                real_linear(h, model.w_im[determinant]),
            )
        )
    orbitals = _stack(determinant_orbitals, dim=1)
    return orbitals
