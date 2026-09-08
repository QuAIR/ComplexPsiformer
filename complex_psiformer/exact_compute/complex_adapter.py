# Adapted from qm-perf9828 commit 9828f69871ed47baecf07fbc3bfba88969db0cd5.
"""Exact forward-Laplacian adapter for the registered magnetic ComplexPsiFormer model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul

import torch

from complex_psiformer.features.magnetic import TorusMagneticSections
from complex_psiformer.features.periodic import ComplexPeriodicFeatures
from .primitives import (
    ComplexVGL,
    add,
    complex_slogdet,
    coordinate_seed,
    matmul,
    positive_sqrt,
    product,
    real_linear,
    real_softmax,
    unary,
    unrestricted_complex_linear,
)
from .policy import MatmulBackend, SlogdetBackend
from complex_psiformer.models.kerr_mlp import ComplexMLP, FreeComplexLinear, KerrActivation
from complex_psiformer.models.complex_psiformer.attention import ComplexAttentionHead, ComplexMultiHeadAttention
from complex_psiformer.models.complex_psiformer.block import ComplexBlock
from complex_psiformer.models.complex_psiformer.wavefunction import ComplexPsiFormer


@dataclass(frozen=True, slots=True)
class ComplexLogPsiVGL:
    """Complex log-wavefunction value, coordinate gradients, and Laplacians."""

    log_abs: torch.Tensor
    phase: torch.Tensor
    grad_log_abs: torch.Tensor
    grad_phase_angle: torch.Tensor
    lap_log_abs: torch.Tensor
    lap_phase_angle: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.log_abs,
            self.phase,
            self.grad_log_abs,
            self.grad_phase_angle,
            self.lap_log_abs,
            self.lap_phase_angle,
        )
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("ComplexLogPsiVGL fields must be torch tensors")
        if self.log_abs.ndim != 1:
            raise ValueError("ComplexLogPsiVGL log_abs must have shape [B]")
        batch = int(self.log_abs.shape[0])
        if self.phase.shape != (batch,) or not torch.is_complex(self.phase):
            raise ValueError("ComplexLogPsiVGL phase must be complex with shape [B]")
        if self.grad_log_abs.ndim != 3 or self.grad_log_abs.shape[0] != batch or self.grad_log_abs.shape[-1] != 2:
            raise ValueError("ComplexLogPsiVGL grad_log_abs must have shape [B,N,2]")
        if self.grad_phase_angle.shape != self.grad_log_abs.shape:
            raise ValueError("ComplexLogPsiVGL coordinate gradients must have identical shapes")
        if self.lap_log_abs.shape != (batch,) or self.lap_phase_angle.shape != (batch,):
            raise ValueError("ComplexLogPsiVGL Laplacians must have shape [B]")
        real_fields = (
            self.log_abs,
            self.grad_log_abs,
            self.grad_phase_angle,
            self.lap_log_abs,
            self.lap_phase_angle,
        )
        if any(torch.is_complex(tensor) or not torch.is_floating_point(tensor) for tensor in real_fields):
            raise TypeError("ComplexLogPsiVGL log and derivative fields must be real floating-point")
        if any(tensor.device != self.log_abs.device for tensor in tensors):
            raise ValueError("ComplexLogPsiVGL fields must share one device")
        if any(tensor.dtype != self.log_abs.dtype for tensor in real_fields[1:]):
            raise ValueError("ComplexLogPsiVGL real fields must share one dtype")
        if self.phase.real.dtype != self.log_abs.dtype:
            raise ValueError("ComplexLogPsiVGL phase precision must match log_abs")


def _validate_complex_support(model: ComplexPsiFormer) -> None:
    """Reject any unregistered primitive before seeding coordinate work."""
    if type(model) is not ComplexPsiFormer:
        raise TypeError("hmga_log_psi_vgl requires ComplexPsiFormer")
    required_values = {
        "embedding_kind": "free",
        "per_block_kind": "free",
        "activation_kind": "kerr",
        "attention_kind": "all_pairs",
        "attention_score_mode": "abs_dot",
        "gate_parameterization": "positive_softplus",
        "attention_residual_norm": "rms_cap",
    }
    for name, expected in required_values.items():
        actual = getattr(model, name, None)
        if actual != expected:
            raise ValueError(f"ComplexPsiFormer forward Laplacian requires {name}={expected!r}, got {actual!r}")
    if not isinstance(model.features, ComplexPeriodicFeatures):
        raise TypeError("ComplexPsiFormer forward Laplacian requires ComplexPeriodicFeatures")
    if not isinstance(model.embed, FreeComplexLinear):
        raise TypeError("ComplexPsiFormer forward Laplacian requires a FreeComplexLinear embedding")
    if model.magnetic_sections is not None and type(model.magnetic_sections) is not TorusMagneticSections:
        raise TypeError("unsupported magnetic sections")
    for block in model.blocks:
        if not isinstance(block, ComplexBlock) or not isinstance(block.attn, ComplexMultiHeadAttention):
            raise TypeError("ComplexPsiFormer forward Laplacian requires registered ComplexBlock attention")
        if not isinstance(block.mlp, ComplexMLP):
            raise TypeError("ComplexPsiFormer forward Laplacian requires registered ComplexMLP blocks")
        if not isinstance(block.mlp.linear, FreeComplexLinear) or not isinstance(block.mlp.activation, KerrActivation):
            raise ValueError("ComplexPsiFormer forward Laplacian requires free-linear Kerr activation blocks")
        for head in block.attn.heads:
            if not isinstance(head, ComplexAttentionHead):
                raise TypeError("ComplexPsiFormer forward Laplacian requires registered ComplexPsiFormer attention heads")
            if any(not isinstance(layer, FreeComplexLinear) for layer in (head.Wq, head.Wk, head.Wv)):
                raise TypeError("ComplexPsiFormer forward Laplacian requires free-linear Q/K/V maps")
        if not isinstance(block.attn.Wo, FreeComplexLinear):
            raise TypeError("ComplexPsiFormer forward Laplacian requires a free-linear attention output map")


def _validate_coordinates(model: ComplexPsiFormer, r: torch.Tensor) -> None:
    if not isinstance(r, torch.Tensor) or r.ndim != 3 or r.shape[-1] != 2:
        shape = None if not isinstance(r, torch.Tensor) else tuple(r.shape)
        raise ValueError(f"r must have shape [B,N,2], got {shape}")
    if r.shape[0] <= 0 or r.shape[1] != model.n_electrons:
        raise ValueError(f"r must have a nonempty batch and {model.n_electrons} electrons")
    if not torch.is_floating_point(r):
        raise TypeError("r must be real floating-point coordinates")


def _constant(value: torch.Tensor, coordinate_shape: tuple[int, int]) -> ComplexVGL:
    return ComplexVGL(
        value=value,
        gradient=torch.zeros(
            value.shape + coordinate_shape,
            dtype=value.dtype,
            device=value.device,
        ),
        laplacian=torch.zeros_like(value),
        coordinate_shape=coordinate_shape,
    )


def _as_constant(
    value: torch.Tensor | float,
    reference: ComplexVGL,
    *,
    dtype: torch.dtype | None = None,
) -> ComplexVGL:
    tensor = torch.as_tensor(
        value,
        dtype=reference.value.dtype if dtype is None else dtype,
        device=reference.value.device,
    )
    return _constant(tensor, reference.coordinate_shape)


def _real_constant(value: torch.Tensor | float, reference: ComplexVGL) -> ComplexVGL:
    tensor = torch.as_tensor(
        value,
        dtype=reference.value.real.dtype,
        device=reference.value.device,
    )
    return _constant(tensor, reference.coordinate_shape)


def _scale(operand: ComplexVGL, value: torch.Tensor | float) -> ComplexVGL:
    return product(operand, _as_constant(value, operand))


def _subtract(left: ComplexVGL, right: ComplexVGL) -> ComplexVGL:
    return add(left, _scale(right, -1.0))


def _real(operand: ComplexVGL) -> ComplexVGL:
    return ComplexVGL(
        operand.value.real,
        operand.gradient.real,
        operand.laplacian.real,
        operand.coordinate_shape,
    )


def _imag(operand: ComplexVGL) -> ComplexVGL:
    return ComplexVGL(
        operand.value.imag,
        operand.gradient.imag,
        operand.laplacian.imag,
        operand.coordinate_shape,
    )


def _complex(real: ComplexVGL, imag: ComplexVGL) -> ComplexVGL:
    if real.coordinate_shape != imag.coordinate_shape:
        raise ValueError("real and imaginary carriers must share coordinate_shape")
    return ComplexVGL(
        torch.complex(real.value, imag.value),
        torch.complex(real.gradient, imag.gradient),
        torch.complex(real.laplacian, imag.laplacian),
        real.coordinate_shape,
    )


def _normalize_dim(dim: int, ndim: int, *, allow_end: bool = False) -> int:
    upper = ndim + (1 if allow_end else 0)
    normalized = dim + upper if dim < 0 else dim
    if normalized < 0 or normalized >= upper:
        raise IndexError(f"dimension {dim} is out of range for ndim={ndim}")
    return normalized


def _reshape(operand: ComplexVGL, shape: tuple[int, ...]) -> ComplexVGL:
    if reduce(mul, shape, 1) != operand.value.numel():
        raise ValueError(f"cannot reshape {tuple(operand.value.shape)} to {shape}")
    return ComplexVGL(
        operand.value.reshape(shape),
        operand.gradient.reshape(shape + operand.coordinate_shape),
        operand.laplacian.reshape(shape),
        operand.coordinate_shape,
    )


def _transpose(operand: ComplexVGL, dim0: int, dim1: int) -> ComplexVGL:
    first = _normalize_dim(dim0, operand.value.ndim)
    second = _normalize_dim(dim1, operand.value.ndim)
    return ComplexVGL(
        operand.value.transpose(first, second),
        operand.gradient.transpose(first, second),
        operand.laplacian.transpose(first, second),
        operand.coordinate_shape,
    )


def _unsqueeze(operand: ComplexVGL, dim: int) -> ComplexVGL:
    normalized = dim
    if normalized < 0:
        normalized += operand.value.ndim + 1
    if normalized < 0 or normalized > operand.value.ndim:
        raise IndexError(f"dimension {dim} is out of range for unsqueeze")
    return ComplexVGL(
        operand.value.unsqueeze(normalized),
        operand.gradient.unsqueeze(normalized),
        operand.laplacian.unsqueeze(normalized),
        operand.coordinate_shape,
    )


def _squeeze(operand: ComplexVGL, dim: int) -> ComplexVGL:
    normalized = _normalize_dim(dim, operand.value.ndim)
    if operand.value.shape[normalized] != 1:
        raise ValueError("only singleton value dimensions can be squeezed")
    return ComplexVGL(
        operand.value.squeeze(normalized),
        operand.gradient.squeeze(normalized),
        operand.laplacian.squeeze(normalized),
        operand.coordinate_shape,
    )


def _narrow(operand: ComplexVGL, dim: int, start: int, length: int) -> ComplexVGL:
    normalized = _normalize_dim(dim, operand.value.ndim)
    return ComplexVGL(
        operand.value.narrow(normalized, start, length),
        operand.gradient.narrow(normalized, start, length),
        operand.laplacian.narrow(normalized, start, length),
        operand.coordinate_shape,
    )


def _split(operand: ComplexVGL, sizes: tuple[int, ...], dim: int = -1) -> tuple[ComplexVGL, ...]:
    normalized = _normalize_dim(dim, operand.value.ndim)
    if sum(sizes) != operand.value.shape[normalized]:
        raise ValueError("split sizes must cover the selected value dimension")
    result = []
    start = 0
    for size in sizes:
        result.append(_narrow(operand, normalized, start, size))
        start += size
    return tuple(result)


def _cat(operands: list[ComplexVGL], dim: int = -1) -> ComplexVGL:
    if not operands:
        raise ValueError("cannot concatenate an empty carrier list")
    normalized = _normalize_dim(dim, operands[0].value.ndim)
    coordinate_shape = operands[0].coordinate_shape
    if any(operand.coordinate_shape != coordinate_shape for operand in operands):
        raise ValueError("concatenated carriers must share coordinate_shape")
    return ComplexVGL(
        torch.cat([operand.value for operand in operands], dim=normalized),
        torch.cat([operand.gradient for operand in operands], dim=normalized),
        torch.cat([operand.laplacian for operand in operands], dim=normalized),
        coordinate_shape,
    )


def _stack(operands: list[ComplexVGL], dim: int = 0) -> ComplexVGL:
    if not operands:
        raise ValueError("cannot stack an empty carrier list")
    normalized = dim
    if normalized < 0:
        normalized += operands[0].value.ndim + 1
    coordinate_shape = operands[0].coordinate_shape
    if any(operand.coordinate_shape != coordinate_shape for operand in operands):
        raise ValueError("stacked carriers must share coordinate_shape")
    return ComplexVGL(
        torch.stack([operand.value for operand in operands], dim=normalized),
        torch.stack([operand.gradient for operand in operands], dim=normalized),
        torch.stack([operand.laplacian for operand in operands], dim=normalized),
        coordinate_shape,
    )


def _sum(operand: ComplexVGL, dim: int | tuple[int, ...], *, keepdim: bool = False) -> ComplexVGL:
    dims = (dim,) if isinstance(dim, int) else dim
    normalized = tuple(_normalize_dim(value, operand.value.ndim) for value in dims)
    return ComplexVGL(
        operand.value.sum(dim=normalized, keepdim=keepdim),
        operand.gradient.sum(dim=normalized, keepdim=keepdim),
        operand.laplacian.sum(dim=normalized, keepdim=keepdim),
        operand.coordinate_shape,
    )


def _mean(operand: ComplexVGL, dim: int | tuple[int, ...], *, keepdim: bool = False) -> ComplexVGL:
    dims = (dim,) if isinstance(dim, int) else dim
    normalized = tuple(_normalize_dim(value, operand.value.ndim) for value in dims)
    count = reduce(mul, (operand.value.shape[value] for value in normalized), 1)
    return _scale(_sum(operand, normalized, keepdim=keepdim), 1.0 / count)


def _square(operand: ComplexVGL) -> ComplexVGL:
    return product(operand, operand)


def _abs_square(operand: ComplexVGL) -> ComplexVGL:
    return add(_square(_real(operand)), _square(_imag(operand)))


def _reciprocal(operand: ComplexVGL) -> ComplexVGL:
    if bool(torch.any(operand.value == 0).item()):
        raise ValueError("forward Laplacian reciprocal encountered zero")
    return unary(
        operand,
        value_fn=torch.reciprocal,
        first_derivative_fn=lambda value: -value.reciprocal().square(),
        second_derivative_fn=lambda value: 2.0 * value.reciprocal().pow(3),
    )


def _tanh(operand: ComplexVGL) -> ComplexVGL:
    return unary(
        operand,
        value_fn=torch.tanh,
        first_derivative_fn=lambda value: 1.0 - torch.tanh(value).square(),
        second_derivative_fn=lambda value: -2.0 * torch.tanh(value) * (1.0 - torch.tanh(value).square()),
    )


def _sin(operand: ComplexVGL) -> ComplexVGL:
    return unary(
        operand,
        value_fn=torch.sin,
        first_derivative_fn=torch.cos,
        second_derivative_fn=lambda value: -torch.sin(value),
    )


def _cos(operand: ComplexVGL) -> ComplexVGL:
    return unary(
        operand,
        value_fn=torch.cos,
        first_derivative_fn=lambda value: -torch.sin(value),
        second_derivative_fn=lambda value: -torch.cos(value),
    )


def _exp(operand: ComplexVGL) -> ComplexVGL:
    return unary(
        operand,
        value_fn=torch.exp,
        first_derivative_fn=torch.exp,
        second_derivative_fn=torch.exp,
    )


def _exp_i(operand: ComplexVGL) -> ComplexVGL:
    return unary(
        operand,
        value_fn=lambda value: torch.exp(1j * value),
        first_derivative_fn=lambda value: 1j * torch.exp(1j * value),
        second_derivative_fn=lambda value: -torch.exp(1j * value),
    )


def _log(operand: ComplexVGL) -> ComplexVGL:
    if bool(torch.any(operand.value == 0).item()):
        raise ValueError("forward Laplacian logarithm encountered zero")
    return unary(
        operand,
        value_fn=torch.log,
        first_derivative_fn=torch.reciprocal,
        second_derivative_fn=lambda value: -value.reciprocal().square(),
    )


def _clamp_max(operand: ComplexVGL, maximum: float) -> ComplexVGL:
    mask = operand.value <= maximum
    derivative_mask = mask.unsqueeze(-1).unsqueeze(-1)
    return ComplexVGL(
        torch.clamp(operand.value, max=maximum),
        torch.where(derivative_mask, operand.gradient, torch.zeros_like(operand.gradient)),
        torch.where(mask, operand.laplacian, torch.zeros_like(operand.laplacian)),
        operand.coordinate_shape,
    )


def _max_last(operand: ComplexVGL, *, keepdim: bool) -> ComplexVGL:
    indices = operand.value.argmax(dim=-1, keepdim=True)
    value = torch.gather(operand.value, -1, indices)
    laplacian = torch.gather(operand.laplacian, -1, indices)
    gradient_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(indices.shape + operand.coordinate_shape)
    gradient = torch.gather(operand.gradient, -3, gradient_indices)
    result = ComplexVGL(value, gradient, laplacian, operand.coordinate_shape)
    return result if keepdim else _squeeze(result, -1)


def _free_linear_vgl(layer: FreeComplexLinear, operand: ComplexVGL) -> ComplexVGL:
    return unrestricted_complex_linear(
        operand,
        layer.W_AA,
        layer.W_AB,
        layer.W_BA,
        layer.W_BB,
        layer.b_re,
        layer.b_im,
    )


def _magnetic_sections_vgl(sections: TorusMagneticSections, coordinates: ComplexVGL) -> ComplexVGL:
    """Mirror the registered finite torus-section formula with exact VGL rules."""
    if not isinstance(sections, TorusMagneticSections):
        raise TypeError("forward Laplacian requires TorusMagneticSections")
    r = coordinates.value
    lattice = sections.lattice.to(device=r.device, dtype=r.dtype)
    inverse_lattice = sections.inverse_lattice.to(device=r.device, dtype=r.dtype)
    origin = sections.origin.to(device=r.device, dtype=r.dtype)
    axis_x = sections.axis_x.to(device=r.device, dtype=r.dtype)
    axis_y = sections.axis_y.to(device=r.device, dtype=r.dtype)
    length_x = sections.length_x.to(device=r.device, dtype=r.dtype)
    shear = sections.shear.to(device=r.device, dtype=r.dtype)
    length_y = sections.length_y.to(device=r.device, dtype=r.dtype)
    field = sections.magnetic_B.to(device=r.device, dtype=r.dtype)
    images = sections.images.to(device=r.device, dtype=r.dtype)
    levels = sections.landau_levels.to(device=r.device)
    labels = sections.flux_labels.to(device=r.device, dtype=r.dtype)

    reduced = real_linear(coordinates, inverse_lattice.T)
    windings = _constant(torch.floor(reduced.value), coordinates.coordinate_shape)
    wrapped = real_linear(_subtract(reduced, windings), lattice.T)
    relative = _subtract(wrapped, _as_constant(origin, wrapped))
    X = _squeeze(real_linear(relative, axis_x.unsqueeze(0)), -1)
    Y = _squeeze(real_linear(relative, axis_y.unsqueeze(0)), -1)
    base_momenta = 2.0 * math.pi * labels / length_x
    momenta = base_momenta[:, None] + field * length_y * images[None, :]
    oscillator_x = _scale(
        add(
            _unsqueeze(_unsqueeze(Y, -1), -1),
            _as_constant(momenta / field, Y),
        ),
        torch.sqrt(field),
    )

    hermite_values = [_scale(_exp(_scale(_square(oscillator_x), -0.5)), math.pi ** (-0.25))]
    if sections.max_landau_level >= 1:
        hermite_values.append(_scale(product(oscillator_x, hermite_values[0]), math.sqrt(2.0)))
    for level in range(1, sections.max_landau_level):
        hermite_values.append(
            _subtract(
                _scale(product(oscillator_x, hermite_values[level]), math.sqrt(2.0 / (level + 1))),
                _scale(hermite_values[level - 1], math.sqrt(level / (level + 1))),
            )
        )
    oscillator: ComplexVGL | None = None
    for level, value in enumerate(hermite_values):
        mask = (levels == level).to(dtype=r.dtype).view(1, 1, -1, 1)
        selected = product(value, _as_constant(mask, value))
        oscillator = selected if oscillator is None else add(oscillator, selected)
    if oscillator is None:
        raise RuntimeError("magnetic Hermite inventory is empty")

    image_angle = (
        images[None, :] * base_momenta[:, None] * shear + 0.5 * field * length_y * shear * images[None, :].square()
    )
    plane_angle = product(
        _unsqueeze(_unsqueeze(X, -1), -1),
        _as_constant(momenta, X),
    )
    image_terms = product(
        product(oscillator, _exp_i(plane_angle)),
        _as_constant(
            torch.exp(1j * image_angle),
            plane_angle,
            dtype=torch.complex128 if r.dtype == torch.float64 else torch.complex64,
        ),
    )
    image_sum = _sum(image_terms, dim=-1)
    gauge_transform = _exp_i(_scale(product(X, Y), 0.5 * field))
    canonical = _scale(
        product(_unsqueeze(gauge_transform, -1), image_sum),
        field.pow(0.25) / torch.sqrt(length_x),
    )

    n1, n2 = (_squeeze(part, -1) for part in _split(windings, (1, 1), dim=-1))
    L1, L2 = lattice
    chi_L1 = _scale(
        _squeeze(real_linear(relative, torch.stack((-L1[1], L1[0])).unsqueeze(0)), -1),
        0.5 * field,
    )
    after_L1 = add(
        wrapped,
        product(_unsqueeze(n1, -1), _as_constant(L1, n1)),
    )
    after_L1_relative = _subtract(after_L1, _as_constant(origin, after_L1))
    chi_L2 = _scale(
        _squeeze(real_linear(after_L1_relative, torch.stack((-L2[1], L2[0])).unsqueeze(0)), -1),
        0.5 * field,
    )
    winding_phase = _exp_i(add(product(n1, chi_L1), product(n2, chi_L2)))
    return product(_unsqueeze(winding_phase, -1), canonical)


def _complex_features_from_seed(model: ComplexPsiFormer, coordinates: ComplexVGL) -> ComplexVGL:
    phases = real_linear(coordinates, model.features.G.to(coordinates.value.dtype))
    return _exp_i(phases)


def _complex_attention_head_vgl(
    head: ComplexAttentionHead,
    h: ComplexVGL,
    *,
    matmul_backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    q = _free_linear_vgl(head.Wq, h)
    k = _free_linear_vgl(head.Wk, h)
    v = _free_linear_vgl(head.Wv, h)
    q_real, q_imag = _real(q), _imag(q)
    k_real, k_imag = _real(k), _imag(k)
    raw = add(
        matmul(
            q_real,
            _transpose(k_real, -1, -2),
            backend=matmul_backend,
        ),
        matmul(
            q_imag,
            _transpose(k_imag, -1, -2),
            backend=matmul_backend,
        ),
    )
    raw_cross = _subtract(
        matmul(
            q_real,
            _transpose(k_imag, -1, -2),
            backend=matmul_backend,
        ),
        matmul(
            q_imag,
            _transpose(k_real, -1, -2),
            backend=matmul_backend,
        ),
    )
    # Published abs_dot: no norm denominator and no smoothing epsilon.
    if head.score_mode != "abs_dot":
        raise ValueError("only published abs_dot complex attention is supported")
    raw_abs = positive_sqrt(add(_square(raw), _square(raw_cross)))
    score = _scale(raw_abs, head.score_scale_mult * (head.d_attn**-0.5))
    alpha = real_softmax(score, dim=-1)
    return _complex(
        matmul(
            alpha,
            _real(v),
            backend=matmul_backend,
        ),
        matmul(
            alpha,
            _imag(v),
            backend=matmul_backend,
        ),
    )


def _complex_attention_vgl(
    attention: ComplexMultiHeadAttention,
    h: ComplexVGL,
    *,
    matmul_backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    heads = [
        _complex_attention_head_vgl(
            head,
            h,
            matmul_backend=matmul_backend,
        )
        for head in attention.heads
    ]
    out = _free_linear_vgl(attention.Wo, _cat(heads, dim=-1))
    out_rms = positive_sqrt(
        add(
            _mean(_abs_square(out), dim=(-2, -1), keepdim=True),
            _real_constant(attention.residual_norm_eps, out),
        )
    )
    h_rms = positive_sqrt(
        add(
            _mean(_abs_square(h), dim=(-2, -1), keepdim=True),
            _real_constant(attention.residual_norm_eps, h),
        )
    )
    scale_match = product(h_rms, _reciprocal(add(out_rms, _as_constant(attention.residual_norm_eps, out_rms))))
    cap = _clamp_max(_scale(scale_match, attention.residual_norm_cap_ratio), 1.0)
    capped = product(out, cap)
    gate = attention.gate_multiplier * attention.effective_gate()
    return _scale(capped, gate)


def _kerr_vgl(activation: KerrActivation, operand: ComplexVGL) -> ComplexVGL:
    modulus = positive_sqrt(_abs_square(operand))
    shifted = _subtract(modulus, _as_constant(activation.r0, modulus))
    unit_phase = product(operand, _reciprocal(add(modulus, _as_constant(1e-8, modulus))))
    return product(_tanh(shifted), unit_phase)


def _complex_streams_from_seed(
    model: ComplexPsiFormer,
    coordinates: ComplexVGL,
    *,
    matmul_backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    h = _free_linear_vgl(model.embed, _complex_features_from_seed(model, coordinates))
    for block in model.blocks:
        attention_residual = add(
            h,
            _complex_attention_vgl(
                block.attn,
                h,
                matmul_backend=matmul_backend,
            ),
        )
        h = add(
            attention_residual, _kerr_vgl(block.mlp.activation, _free_linear_vgl(block.mlp.linear, attention_residual))
        )
    return h


def _complex_orbitals_from_seed(
    model: ComplexPsiFormer,
    coordinates: ComplexVGL,
    *,
    matmul_backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    h = _complex_streams_from_seed(
        model,
        coordinates,
        matmul_backend=matmul_backend,
    )
    determinant_orbitals = []
    for determinant in range(model.n_det):
        w_re = model.w_re[determinant]
        w_im = model.w_im[determinant]
        determinant_orbitals.append(unrestricted_complex_linear(h, w_re, w_im, -w_im, w_re))
    orbitals = _stack(determinant_orbitals, dim=1)
    if model.magnetic_sections is not None:
        sections = _magnetic_sections_vgl(model.magnetic_sections, coordinates)
        orbitals = product(orbitals, _unsqueeze(sections, 1))
    return orbitals


def _logpsi_from_orbitals(
    orbitals: ComplexVGL,
    coefficients: torch.Tensor | None,
    *,
    slogdet_backend: SlogdetBackend = "inverse_loop",
) -> ComplexLogPsiVGL:
    log_determinants = complex_slogdet(
        orbitals,
        backend=slogdet_backend,
    )
    if coefficients is None:
        coefficients = orbitals.value.real.new_ones(orbitals.value.shape[1])
    max_logdet = _max_last(_real(log_determinants), keepdim=True)
    centered = _subtract(log_determinants, max_logdet)
    determinant_values = _exp(centered)
    psi = _sum(
        product(determinant_values, _as_constant(coefficients.to(determinant_values.value.dtype), determinant_values)),
        dim=-1,
    )
    absolute_psi = positive_sqrt(_abs_square(psi))
    log_abs_vgl = add(_log(absolute_psi), _squeeze(max_logdet, -1))
    complex_log_psi = _log(psi)
    phase_angle_vgl = _imag(complex_log_psi)
    phase = psi.value / (absolute_psi.value + 1e-30)
    return ComplexLogPsiVGL(
        log_abs=log_abs_vgl.value,
        phase=phase,
        grad_log_abs=log_abs_vgl.gradient,
        grad_phase_angle=phase_angle_vgl.gradient,
        lap_log_abs=log_abs_vgl.laplacian,
        lap_phase_angle=phase_angle_vgl.laplacian,
    )
