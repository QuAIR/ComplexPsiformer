# Adapted from qm-perf9828 commit 9828f69871ed47baecf07fbc3bfba88969db0cd5.
"""Exact value/coordinate-gradient/Laplacian propagation primitives.

This module carries only the diagonal Hessian trace needed by kinetic-energy
evaluation.  The rules are analytic: there is deliberately no Hessian,
stochastic-trace, or model-specific fallback hidden behind this interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .policy import MatmulBackend, SlogdetBackend

CoordinateShape = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ComplexVGL:
    """A value together with derivatives with respect to one ``(N, 2)`` input.

    ``gradient`` has shape ``value.shape + coordinate_shape`` and
    ``laplacian`` has shape ``value.shape``.  Batched values are supported;
    each batch member carries derivatives with respect to its own coordinate
    array rather than cross-batch derivatives.
    """

    value: torch.Tensor
    gradient: torch.Tensor
    laplacian: torch.Tensor
    coordinate_shape: CoordinateShape

    def __post_init__(self) -> None:
        if not all(isinstance(tensor, torch.Tensor) for tensor in (self.value, self.gradient, self.laplacian)):
            raise TypeError("ComplexVGL value, gradient, and laplacian must be torch tensors")
        coordinate_shape = tuple(self.coordinate_shape)
        if (
            len(coordinate_shape) != 2
            or any(not isinstance(size, int) for size in coordinate_shape)
            or coordinate_shape[0] <= 0
            or coordinate_shape[1] != 2
        ):
            raise ValueError(f"coordinate_shape must be (N, 2) with N > 0, got {self.coordinate_shape!r}")
        object.__setattr__(self, "coordinate_shape", coordinate_shape)
        expected_gradient_shape = self.value.shape + coordinate_shape
        if self.gradient.shape != expected_gradient_shape:
            raise ValueError(
                "ComplexVGL gradient must have shape value.shape + coordinate_shape, "
                f"got value={tuple(self.value.shape)}, gradient={tuple(self.gradient.shape)}, "
                f"coordinate_shape={coordinate_shape}"
            )
        if self.laplacian.shape != self.value.shape:
            raise ValueError(
                "ComplexVGL laplacian must have shape value.shape, "
                f"got value={tuple(self.value.shape)} and laplacian={tuple(self.laplacian.shape)}"
            )
        if self.gradient.device != self.value.device or self.laplacian.device != self.value.device:
            raise ValueError("ComplexVGL value, gradient, and laplacian must be on the same device")
        if self.gradient.dtype != self.value.dtype or self.laplacian.dtype != self.value.dtype:
            raise ValueError("ComplexVGL value, gradient, and laplacian must have the same dtype")


def coordinate_seed(coordinates: torch.Tensor) -> ComplexVGL:
    """Seed exact derivatives for real coordinates with trailing shape ``(N, 2)``."""

    if not isinstance(coordinates, torch.Tensor):
        raise TypeError("coordinates must be a torch tensor")
    if coordinates.ndim < 2 or coordinates.shape[-1] != 2 or coordinates.shape[-2] <= 0:
        raise ValueError(
            f"coordinates must have trailing shape (N, 2); last dimension must be 2, got {coordinates.shape}"
        )
    if not torch.is_floating_point(coordinates):
        raise TypeError("coordinates must be real floating-point values")
    coordinate_shape = (coordinates.shape[-2], coordinates.shape[-1])
    coordinate_size = coordinate_shape[0] * coordinate_shape[1]
    identity = torch.eye(coordinate_size, dtype=coordinates.dtype, device=coordinates.device).reshape(
        coordinate_shape + coordinate_shape
    )
    batch_shape = coordinates.shape[:-2]
    gradient = identity.expand(batch_shape + identity.shape)
    return ComplexVGL(
        value=coordinates,
        gradient=gradient,
        laplacian=torch.zeros_like(coordinates),
        coordinate_shape=coordinate_shape,
    )


def _require_same_coordinates(left: ComplexVGL, right: ComplexVGL) -> CoordinateShape:
    if left.coordinate_shape != right.coordinate_shape:
        raise ValueError(
            "ComplexVGL operands must use the same coordinate_shape, "
            f"got {left.coordinate_shape} and {right.coordinate_shape}"
        )
    return left.coordinate_shape


def _with_coordinate_axes(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(-1).unsqueeze(-1)


def add(left: ComplexVGL, right: ComplexVGL) -> ComplexVGL:
    """Add two carriers, including ordinary PyTorch value broadcasting."""

    coordinate_shape = _require_same_coordinates(left, right)
    return ComplexVGL(
        value=left.value + right.value,
        gradient=left.gradient + right.gradient,
        laplacian=left.laplacian + right.laplacian,
        coordinate_shape=coordinate_shape,
    )


def product(left: ComplexVGL, right: ComplexVGL) -> ComplexVGL:
    """Pointwise product using the exact unconjugated complex product rule."""

    coordinate_shape = _require_same_coordinates(left, right)
    left_value = _with_coordinate_axes(left.value)
    right_value = _with_coordinate_axes(right.value)
    return ComplexVGL(
        value=left.value * right.value,
        gradient=left.gradient * right_value + left_value * right.gradient,
        laplacian=(
            left.laplacian * right.value
            + 2.0 * (left.gradient * right.gradient).sum(dim=(-2, -1))
            + left.value * right.laplacian
        ),
        coordinate_shape=coordinate_shape,
    )


def unary(
    operand: ComplexVGL,
    *,
    value_fn: Callable[[torch.Tensor], torch.Tensor],
    first_derivative_fn: Callable[[torch.Tensor], torch.Tensor],
    second_derivative_fn: Callable[[torch.Tensor], torch.Tensor],
) -> ComplexVGL:
    """Compose an elementwise twice-differentiable scalar function."""

    value = value_fn(operand.value)
    first = first_derivative_fn(operand.value)
    second = second_derivative_fn(operand.value)
    return ComplexVGL(
        value=value,
        gradient=_with_coordinate_axes(first) * operand.gradient,
        laplacian=first * operand.laplacian + second * operand.gradient.square().sum(dim=(-2, -1)),
        coordinate_shape=operand.coordinate_shape,
    )


def _require_real_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if torch.is_complex(tensor):
        raise TypeError(f"{name} must be real-valued")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating-point")


def real_linear(
    operand: ComplexVGL,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> ComplexVGL:
    """Apply a real-coefficient affine map to the final value axis."""

    if operand.value.ndim < 1:
        raise ValueError("real_linear requires an input feature axis")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("real_linear weight must have shape (out_features, in_features)")
    _require_real_tensor(weight, name="real_linear weight")
    if weight.shape[1] != operand.value.shape[-1]:
        raise ValueError(
            f"real_linear weight input size {weight.shape[1]} does not match operand size {operand.value.shape[-1]}"
        )
    if weight.device != operand.value.device:
        raise ValueError("real_linear weight and operand must be on the same device")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.shape != (weight.shape[0],):
            raise ValueError(f"real_linear bias must have shape ({weight.shape[0]},)")
        _require_real_tensor(bias, name="real_linear bias")
        if bias.device != operand.value.device:
            raise ValueError("real_linear bias and operand must be on the same device")

    value_weight = weight.to(dtype=operand.value.dtype)
    value = torch.einsum("...i,oi->...o", operand.value, value_weight)
    if bias is not None:
        value = value + bias.to(dtype=value.dtype)
    return ComplexVGL(
        value=value,
        gradient=torch.einsum("...icd,oi->...ocd", operand.gradient, value_weight),
        laplacian=torch.einsum("...i,oi->...o", operand.laplacian, value_weight),
        coordinate_shape=operand.coordinate_shape,
    )


def _validate_unrestricted_weights(
    operand: ComplexVGL,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[int, int]:
    if not torch.is_complex(operand.value):
        raise TypeError("unrestricted_complex_linear requires a complex-valued operand")
    if operand.value.ndim < 1:
        raise ValueError("unrestricted_complex_linear requires an input feature axis")
    if any(not isinstance(weight, torch.Tensor) or weight.ndim != 2 for weight in weights):
        raise ValueError("unrestricted complex weights must all have shape (out_features, in_features)")
    shape = weights[0].shape
    if any(weight.shape != shape for weight in weights):
        raise ValueError("AA, AB, BA, and BB weights must have identical shapes")
    if shape[1] != operand.value.shape[-1]:
        raise ValueError(
            f"unrestricted complex weight input size {shape[1]} does not match operand size {operand.value.shape[-1]}"
        )
    for name, weight in zip(("AA", "AB", "BA", "BB"), weights):
        _require_real_tensor(weight, name=f"unrestricted complex {name} weight")
        if weight.device != operand.value.device:
            raise ValueError("unrestricted complex weights and operand must be on the same device")
    return shape


def unrestricted_complex_linear(
    operand: ComplexVGL,
    weight_aa: torch.Tensor,
    weight_ab: torch.Tensor,
    weight_ba: torch.Tensor,
    weight_bb: torch.Tensor,
    bias_real: torch.Tensor | None = None,
    bias_imag: torch.Tensor | None = None,
) -> ComplexVGL:
    """Apply the four-independent-real-block map used by ``FreeComplexLinear``."""

    weights = (weight_aa, weight_ab, weight_ba, weight_bb)
    out_features, _ = _validate_unrestricted_weights(operand, weights)
    if (bias_real is None) != (bias_imag is None):
        raise ValueError("unrestricted complex bias_real and bias_imag must be both present or both absent")
    if bias_real is not None and bias_imag is not None:
        for name, bias in (("bias_real", bias_real), ("bias_imag", bias_imag)):
            if not isinstance(bias, torch.Tensor) or bias.shape != (out_features,):
                raise ValueError(f"unrestricted complex {name} must have shape ({out_features},)")
            _require_real_tensor(bias, name=f"unrestricted complex {name}")
            if bias.device != operand.value.device:
                raise ValueError("unrestricted complex biases and operand must be on the same device")

    real_dtype = operand.value.real.dtype
    aa, ab, ba, bb = (weight.to(dtype=real_dtype) for weight in weights)

    def map_values(tensor: torch.Tensor) -> torch.Tensor:
        real = torch.einsum("...i,oi->...o", tensor.real, aa) + torch.einsum("...i,oi->...o", tensor.imag, ab)
        imag = torch.einsum("...i,oi->...o", tensor.real, ba) + torch.einsum("...i,oi->...o", tensor.imag, bb)
        return torch.complex(real, imag)

    def map_gradients(tensor: torch.Tensor) -> torch.Tensor:
        real = torch.einsum("...icd,oi->...ocd", tensor.real, aa) + torch.einsum("...icd,oi->...ocd", tensor.imag, ab)
        imag = torch.einsum("...icd,oi->...ocd", tensor.real, ba) + torch.einsum("...icd,oi->...ocd", tensor.imag, bb)
        return torch.complex(real, imag)

    value = map_values(operand.value)
    if bias_real is not None and bias_imag is not None:
        value = value + torch.complex(bias_real.to(dtype=real_dtype), bias_imag.to(dtype=real_dtype))
    return ComplexVGL(
        value=value,
        gradient=map_gradients(operand.gradient),
        laplacian=map_values(operand.laplacian),
        coordinate_shape=operand.coordinate_shape,
    )


def matmul(
    left: ComplexVGL,
    right: ComplexVGL,
    *,
    backend: MatmulBackend = "vectorized",
) -> ComplexVGL:
    """Matrix-multiply two carriers with exact first and trace-second rules."""

    coordinate_shape = _require_same_coordinates(left, right)
    if left.value.ndim < 2 or right.value.ndim < 2:
        raise ValueError("matmul supports tensors whose final two value axes are matrices")
    if left.value.shape[-1] != right.value.shape[-2]:
        raise ValueError(f"matmul inner dimensions must agree, got {left.value.shape[-1]} and {right.value.shape[-2]}")
    value = torch.matmul(left.value, right.value)
    coordinate_size = coordinate_shape[0] * coordinate_shape[1]
    left_gradient = left.gradient.reshape((*left.value.shape, coordinate_size))
    right_gradient = right.gradient.reshape((*right.value.shape, coordinate_size))
    if backend == "coordinate_loop":
        gradient_terms = []
        cross_terms = []
        for coordinate in range(coordinate_size):
            left_derivative = left_gradient[..., coordinate]
            right_derivative = right_gradient[..., coordinate]
            gradient_terms.append(
                torch.matmul(left_derivative, right.value)
                + torch.matmul(left.value, right_derivative)
            )
            cross_terms.append(torch.matmul(left_derivative, right_derivative))
        gradient = torch.stack(gradient_terms, dim=-1).reshape(value.shape + coordinate_shape)
        cross_laplacian = torch.stack(cross_terms, dim=-1).sum(dim=-1)
    elif backend == "vectorized":
        left_derivatives = left_gradient.movedim(-1, -3)
        right_derivatives = right_gradient.movedim(-1, -3)
        gradient_by_direction = torch.matmul(
            left_derivatives,
            right.value.unsqueeze(-3),
        ) + torch.matmul(
            left.value.unsqueeze(-3),
            right_derivatives,
        )
        gradient = gradient_by_direction.movedim(-3, -1).reshape(
            value.shape + coordinate_shape
        )
        cross_laplacian = torch.matmul(
            left_derivatives,
            right_derivatives,
        ).sum(dim=-3)
    else:
        raise ValueError(f"matmul backend is not registered: {backend!r}")
    laplacian = (
        torch.matmul(left.laplacian, right.value) + 2.0 * cross_laplacian + torch.matmul(left.value, right.laplacian)
    )
    return ComplexVGL(value, gradient, laplacian, coordinate_shape)


def real_softmax(operand: ComplexVGL, dim: int = -1) -> ComplexVGL:
    """Apply real softmax, retaining all normalization-induced couplings."""

    _require_real_tensor(operand.value, name="real_softmax operand")
    if torch.is_complex(operand.gradient) or torch.is_complex(operand.laplacian):
        raise TypeError("real_softmax operand derivatives must be real-valued")
    if operand.value.ndim == 0:
        raise ValueError("real_softmax requires a non-scalar value")
    normalized_dim = dim if dim >= 0 else operand.value.ndim + dim
    if normalized_dim < 0 or normalized_dim >= operand.value.ndim:
        raise IndexError(f"real_softmax dim {dim} is out of range for a {operand.value.ndim}-D value")

    value = torch.softmax(operand.value, dim=normalized_dim)
    weighted_gradient = _with_coordinate_axes(value) * operand.gradient
    mean_gradient = weighted_gradient.sum(dim=normalized_dim, keepdim=True)
    centered_gradient = operand.gradient - mean_gradient
    gradient = _with_coordinate_axes(value) * centered_gradient

    mean_laplacian = (value * operand.laplacian).sum(dim=normalized_dim, keepdim=True)
    mean_gradient_square = (_with_coordinate_axes(value) * operand.gradient.square()).sum(
        dim=normalized_dim, keepdim=True
    )
    gradient_variance = mean_gradient_square - mean_gradient.square()
    laplacian_correction = (centered_gradient.square() - gradient_variance).sum(dim=(-2, -1))
    laplacian = value * (operand.laplacian - mean_laplacian + laplacian_correction)
    return ComplexVGL(value, gradient, laplacian, operand.coordinate_shape)


def positive_sqrt(operand: ComplexVGL) -> ComplexVGL:
    """Apply the real square root on its smooth, strictly positive domain."""

    _require_real_tensor(operand.value, name="positive_sqrt operand")
    if torch.is_complex(operand.gradient) or torch.is_complex(operand.laplacian):
        raise TypeError("positive_sqrt operand derivatives must be real-valued")
    if not bool(torch.all(operand.value > 0).item()):
        raise ValueError("positive_sqrt requires strictly positive real values")
    return unary(
        operand,
        value_fn=torch.sqrt,
        first_derivative_fn=lambda value: 0.5 / torch.sqrt(value),
        second_derivative_fn=lambda value: -0.25 / value.pow(1.5),
    )


def _trace(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def complex_slogdet(
    matrix: ComplexVGL,
    *,
    backend: SlogdetBackend = "inverse_loop",
) -> ComplexVGL:
    """Return the local complex log determinant using a stable ``slogdet`` primal.

    The value is ``log(abs(det(A))) + i arg(det(A))``.  Derivatives follow
    ``d logdet(A) = tr(A^-1 dA)`` and include the exact quadratic trace in
    the Laplacian.  Singular matrices and real matrices are intentionally
    unsupported by this complex primitive.
    """

    if not torch.is_complex(matrix.value):
        raise TypeError("complex_slogdet requires a complex-valued matrix")
    if matrix.value.ndim < 2 or matrix.value.shape[-2] != matrix.value.shape[-1]:
        raise ValueError("complex_slogdet requires square matrices on the final two value axes")
    coordinate_shape = matrix.coordinate_shape
    coordinate_size = coordinate_shape[0] * coordinate_shape[1]
    matrix_gradient = matrix.gradient.reshape((*matrix.value.shape, coordinate_size))
    if backend == "inverse_loop":
        sign, log_abs_determinant = torch.linalg.slogdet(matrix.value)
        if bool(torch.any(sign == 0).item()):
            raise ValueError("complex_slogdet does not support singular matrices")
        value = torch.complex(log_abs_determinant, torch.angle(sign))
        inverse = torch.linalg.inv(matrix.value)
        gradient_terms = []
        quadratic_terms = []
        for coordinate in range(coordinate_size):
            derivative = matrix_gradient[..., coordinate]
            inverse_derivative = torch.matmul(inverse, derivative)
            gradient_terms.append(_trace(inverse_derivative))
            quadratic_terms.append(
                _trace(torch.matmul(inverse_derivative, inverse_derivative))
            )
        gradient = torch.stack(gradient_terms, dim=-1).reshape(
            value.shape + coordinate_shape
        )
        laplacian = _trace(torch.matmul(inverse, matrix.laplacian)) - torch.stack(
            quadratic_terms,
            dim=-1,
        ).sum(dim=-1)
    else:
        raise ValueError(f"complex_slogdet backend is not registered: {backend!r}")
    return ComplexVGL(value, gradient, laplacian, coordinate_shape)


__all__ = [
    "ComplexVGL",
    "add",
    "complex_slogdet",
    "coordinate_seed",
    "matmul",
    "positive_sqrt",
    "product",
    "real_linear",
    "real_softmax",
    "unary",
    "unrestricted_complex_linear",
]
