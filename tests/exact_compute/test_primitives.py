# Analytic oracles adapted from pinned qm-perf9828 primitive tests.
"""Analytic tests for the exact forward value/gradient/Laplacian calculus."""

from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from complex_psiformer.exact_compute.primitives import (
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


@pytest.fixture(autouse=True)
def _use_float64_by_default() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _component_derivatives(
    scalar: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = torch.autograd.grad(
        scalar,
        coordinates,
        create_graph=True,
        retain_graph=True,
    )[0]
    laplacian = torch.zeros((), dtype=coordinates.dtype, device=coordinates.device)
    for index, first in enumerate(gradient.reshape(-1)):
        if first.requires_grad:
            second = torch.autograd.grad(
                first,
                coordinates,
                create_graph=True,
                retain_graph=True,
            )[0]
            laplacian = laplacian + second.reshape(-1)[index]
    return gradient, laplacian


def _autograd_vgl(
    function: Callable[[torch.Tensor], torch.Tensor],
    coordinates: torch.Tensor,
) -> ComplexVGL:
    coordinates = coordinates.detach().clone().requires_grad_(True)
    value = function(coordinates)
    gradients: list[torch.Tensor] = []
    laplacians: list[torch.Tensor] = []
    for scalar in value.reshape(-1):
        real_gradient, real_laplacian = _component_derivatives(scalar.real, coordinates)
        if torch.is_complex(value):
            imag_gradient, imag_laplacian = _component_derivatives(scalar.imag, coordinates)
            gradients.append(torch.complex(real_gradient, imag_gradient))
            laplacians.append(torch.complex(real_laplacian, imag_laplacian))
        else:
            gradients.append(real_gradient)
            laplacians.append(real_laplacian)
    return ComplexVGL(
        value=value,
        gradient=torch.stack(gradients).reshape(value.shape + coordinates.shape),
        laplacian=torch.stack(laplacians).reshape(value.shape),
        coordinate_shape=tuple(coordinates.shape),
    )


def _assert_vgl_close(actual: ComplexVGL, expected: ComplexVGL) -> None:
    torch.testing.assert_close(actual.value, expected.value, atol=1.0e-11, rtol=1.0e-11)
    torch.testing.assert_close(actual.gradient, expected.gradient, atol=2.0e-10, rtol=2.0e-10)
    torch.testing.assert_close(actual.laplacian, expected.laplacian, atol=5.0e-10, rtol=5.0e-10)
    assert actual.coordinate_shape == expected.coordinate_shape


def test_coordinate_seed_has_frozen_validated_shape_contract() -> None:
    coordinates = torch.tensor([[0.2, -0.3], [0.4, 0.7]])
    seeded = coordinate_seed(coordinates)

    assert seeded.value is coordinates
    assert seeded.gradient.shape == (2, 2, 2, 2)
    torch.testing.assert_close(seeded.gradient.reshape(4, 4), torch.eye(4))
    torch.testing.assert_close(seeded.laplacian, torch.zeros_like(coordinates))
    assert seeded.coordinate_shape == (2, 2)

    with pytest.raises(FrozenInstanceError):
        seeded.coordinate_shape = (1, 2)  # type: ignore[misc]
    with pytest.raises(ValueError, match="gradient must have shape"):
        ComplexVGL(coordinates, torch.zeros(2, 2, 4), coordinates, (2, 2))
    with pytest.raises(ValueError, match="last dimension must be 2"):
        coordinate_seed(torch.zeros(2, 3))


def test_add_and_product_match_polynomial_closed_form() -> None:
    coordinates = torch.tensor([[0.25, -0.4]])
    seeded = coordinate_seed(coordinates)
    polynomial = add(product(seeded, seeded), product(seeded, seeded))

    expected = ComplexVGL(
        value=2.0 * coordinates.square(),
        gradient=(4.0 * coordinates).unsqueeze(-1).unsqueeze(-1) * torch.eye(2).reshape(1, 2, 1, 2),
        laplacian=torch.full_like(coordinates, 4.0),
        coordinate_shape=(1, 2),
    )
    _assert_vgl_close(polynomial, expected)


def test_unary_and_real_linear_match_gaussian_closed_form_and_keep_parameter_gradients() -> None:
    coordinates = torch.tensor([[0.3, -0.45]])
    weight = torch.nn.Parameter(torch.tensor([[-1.2, -0.7]]))
    bias = torch.nn.Parameter(torch.tensor([-0.4]))
    seeded = coordinate_seed(coordinates)
    squared = product(seeded, seeded)
    exponent = real_linear(squared, weight, bias)
    gaussian = unary(
        exponent,
        value_fn=torch.exp,
        first_derivative_fn=torch.exp,
        second_derivative_fn=torch.exp,
    )

    expected = _autograd_vgl(
        lambda x: torch.exp(x.square() @ weight.detach().T + bias.detach()),
        coordinates,
    )
    _assert_vgl_close(gaussian, expected)

    (gaussian.value.square().sum() + gaussian.gradient.square().sum() + gaussian.laplacian.square().sum()).backward()
    assert weight.grad is not None and torch.count_nonzero(weight.grad) == weight.numel()
    assert bias.grad is not None and torch.count_nonzero(bias.grad) == bias.numel()


def test_unary_matches_complex_plane_wave_closed_form() -> None:
    coordinates = torch.tensor([[0.2, -0.35]])
    wavevector = torch.tensor([[1.3, -0.8]])
    phase = real_linear(coordinate_seed(coordinates), wavevector)
    plane_wave = unary(
        phase,
        value_fn=lambda x: torch.exp(1j * x),
        first_derivative_fn=lambda x: 1j * torch.exp(1j * x),
        second_derivative_fn=lambda x: -torch.exp(1j * x),
    )

    value = torch.exp(1j * (coordinates @ wavevector.T))
    expected_gradient = (1j * value[..., None, None]) * wavevector.reshape(1, 1, 1, 2)
    expected = ComplexVGL(
        value=value,
        gradient=expected_gradient,
        laplacian=-(wavevector.square().sum() * value),
        coordinate_shape=(1, 2),
    )
    _assert_vgl_close(plane_wave, expected)


def test_unrestricted_complex_linear_uses_all_four_blocks_and_is_not_wz() -> None:
    coordinates = torch.tensor([[0.35, -0.2]])
    seeded = coordinate_seed(coordinates)
    complex_features = unary(
        seeded,
        value_fn=lambda x: torch.complex(x, x.square() + 0.1),
        first_derivative_fn=lambda x: torch.complex(torch.ones_like(x), 2.0 * x),
        second_derivative_fn=lambda x: torch.complex(torch.zeros_like(x), 2.0 * torch.ones_like(x)),
    )
    aa = torch.nn.Parameter(torch.tensor([[1.1, -0.3], [0.2, 0.7]]))
    ab = torch.nn.Parameter(torch.tensor([[0.4, 0.9], [-0.6, 0.5]]))
    ba = torch.nn.Parameter(torch.tensor([[-0.8, 0.1], [0.3, 1.2]]))
    bb = torch.nn.Parameter(torch.tensor([[0.6, -0.2], [0.75, -0.4]]))
    bias_real = torch.nn.Parameter(torch.tensor([0.15, -0.25]))
    bias_imag = torch.nn.Parameter(torch.tensor([-0.35, 0.45]))

    actual = unrestricted_complex_linear(complex_features, aa, ab, ba, bb, bias_real, bias_imag)

    def expected_function(x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x, x.square() + 0.1)
        real = z.real @ aa.detach().T + z.imag @ ab.detach().T + bias_real.detach()
        imag = z.real @ ba.detach().T + z.imag @ bb.detach().T + bias_imag.detach()
        return torch.complex(real, imag)

    expected = _autograd_vgl(expected_function, coordinates)
    _assert_vgl_close(actual, expected)

    one_matrix = complex_features.value @ torch.complex(aa.detach(), ba.detach()).T + torch.complex(
        bias_real.detach(), bias_imag.detach()
    )
    assert not torch.allclose(one_matrix, expected.value)

    loss = (
        actual.value.abs().square().sum() + actual.gradient.abs().square().sum() + actual.laplacian.abs().square().sum()
    )
    loss.backward()
    for parameter in (aa, ab, ba, bb, bias_real, bias_imag):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == parameter.numel()


def test_matmul_matches_autograd_with_both_factors_coordinate_dependent() -> None:
    coordinates = torch.tensor([[0.2, -0.4]])

    def left_function(x: torch.Tensor) -> torch.Tensor:
        u, v = x.reshape(-1)
        return torch.stack((torch.stack((u.square() + 1.0, v)), torch.stack((u * v, 2.0 - u))))

    def right_function(x: torch.Tensor) -> torch.Tensor:
        u, v = x.reshape(-1)
        return torch.stack((torch.stack((1.0 + v, u)), torch.stack((v.square(), u - v))))

    actual = matmul(_autograd_vgl(left_function, coordinates), _autograd_vgl(right_function, coordinates))
    expected = _autograd_vgl(lambda x: left_function(x) @ right_function(x), coordinates)
    _assert_vgl_close(actual, expected)


def _noncontiguous_complex(shape: tuple[int, ...]) -> torch.Tensor:
    source = torch.randn(shape + (2,), dtype=torch.float64)
    return torch.complex(source[..., 0], source[..., 1])


def test_vectorized_matmul_keeps_every_direction_for_broadcast_noncontiguous_complex_carriers() -> None:
    """Dropping one vectorized direction or a broadcast axis changes this literal oracle."""
    torch.manual_seed(811)
    coordinate_shape = (12, 2)
    left_value = _noncontiguous_complex((2, 1, 4, 3)).transpose(-1, -2)
    right_value = _noncontiguous_complex((1, 5, 2, 4)).transpose(-1, -2)
    left_gradient = _noncontiguous_complex(left_value.shape + (12, 4))[..., ::2]
    right_gradient = _noncontiguous_complex(right_value.shape + (12, 4))[..., ::2]
    left_laplacian = _noncontiguous_complex((2, 1, 4, 3)).transpose(-1, -2)
    right_laplacian = _noncontiguous_complex((1, 5, 2, 4)).transpose(-1, -2)
    assert not left_value.is_contiguous()
    assert not right_value.is_contiguous()
    assert not left_gradient.is_contiguous()
    assert not right_gradient.is_contiguous()
    left = ComplexVGL(left_value, left_gradient, left_laplacian, coordinate_shape)
    right = ComplexVGL(right_value, right_gradient, right_laplacian, coordinate_shape)

    actual = matmul(left, right, backend="vectorized")

    left_d = left_gradient.reshape(left_value.shape + (24,))
    right_d = right_gradient.reshape(right_value.shape + (24,))
    expected_gradient_d = torch.stack(
        [
            left_d[..., direction] @ right_value
            + left_value @ right_d[..., direction]
            for direction in range(24)
        ],
        dim=-1,
    )
    expected_cross = sum(
        left_d[..., direction] @ right_d[..., direction]
        for direction in range(24)
    )
    expected = ComplexVGL(
        left_value @ right_value,
        expected_gradient_d.reshape((2, 5, 3, 2, 12, 2)),
        left_laplacian @ right_value + 2.0 * expected_cross + left_value @ right_laplacian,
        coordinate_shape,
    )
    _assert_vgl_close(actual, expected)


def test_vectorized_matmul_trace_second_keeps_the_exact_factor_two() -> None:
    """Removing the quadratic cross-term factor two halves this hand-derived Laplacian."""
    coordinate_shape = (12, 2)
    directions = torch.arange(1, 25, dtype=torch.float64)
    reverse = torch.arange(24, 0, -1, dtype=torch.float64)
    left = ComplexVGL(
        torch.tensor([[0.0]]),
        directions.reshape(1, 1, 12, 2),
        torch.zeros(1, 1),
        coordinate_shape,
    )
    right = ComplexVGL(
        torch.tensor([[0.0]]),
        reverse.reshape(1, 1, 12, 2),
        torch.zeros(1, 1),
        coordinate_shape,
    )

    actual = matmul(left, right, backend="vectorized")

    expected_cross = torch.tensor(2600.0)
    torch.testing.assert_close(actual.laplacian, 2.0 * expected_cross.reshape(1, 1))


def test_real_softmax_includes_cross_feature_and_cross_coordinate_coupling() -> None:
    coordinates = torch.tensor([[0.25, -0.35]])

    def logits(x: torch.Tensor) -> torch.Tensor:
        u, v = x.reshape(-1)
        return torch.stack((u.square() + v, u - 0.5 * v.square(), u * v)).reshape(1, 3)

    actual = real_softmax(_autograd_vgl(logits, coordinates), dim=-1)
    expected = _autograd_vgl(lambda x: torch.softmax(logits(x), dim=-1), coordinates)
    _assert_vgl_close(actual, expected)
    assert torch.count_nonzero(actual.gradient[..., 0, 1]) == 3


def test_positive_sqrt_matches_autograd_and_rejects_unsupported_values() -> None:
    coordinates = torch.tensor([[0.3, -0.5]])
    squared = product(coordinate_seed(coordinates), coordinate_seed(coordinates))
    positive = real_linear(squared, torch.tensor([[0.8, 1.1]]), torch.tensor([1.7]))

    actual = positive_sqrt(positive)
    expected = _autograd_vgl(
        lambda x: torch.sqrt(0.8 * x[..., :1].square() + 1.1 * x[..., 1:].square() + 1.7), coordinates
    )
    _assert_vgl_close(actual, expected)

    with pytest.raises(ValueError, match="strictly positive real"):
        positive_sqrt(ComplexVGL(torch.tensor([0.0]), torch.zeros(1, 1, 2), torch.zeros(1), (1, 2)))
    with pytest.raises(TypeError, match="real-valued"):
        real_softmax(
            ComplexVGL(
                torch.ones(1, dtype=torch.complex128),
                torch.zeros(1, 1, 2, dtype=torch.complex128),
                torch.zeros(1, dtype=torch.complex128),
                (1, 2),
            )
        )


def test_complex_slogdet_matches_2x2_autograd_oracle_with_quadratic_term() -> None:
    coordinates = torch.tensor([[0.18, -0.27]])

    def matrix(x: torch.Tensor) -> torch.Tensor:
        u, v = x.reshape(-1)
        return torch.stack(
            (
                torch.stack((2.0 + u + 0.2j * v, 0.3 + 0.1j * u)),
                torch.stack((0.2 * v - 0.15j + 0.0j * u, 1.5 - 0.25j * u + 0.1 * v.square())),
            )
        )

    actual = complex_slogdet(_autograd_vgl(matrix, coordinates))
    expected = _autograd_vgl(lambda x: torch.logdet(matrix(x)), coordinates)
    _assert_vgl_close(actual, expected)
    assert actual.value.dtype == torch.complex128
    assert actual.value.imag.abs() > 0.0

    nonsquare = ComplexVGL(
        torch.ones(2, 3, dtype=torch.complex128),
        torch.zeros(2, 3, 1, 2, dtype=torch.complex128),
        torch.zeros(2, 3, dtype=torch.complex128),
        (1, 2),
    )
    with pytest.raises(ValueError, match="square matrices"):
        complex_slogdet(nonsquare)
