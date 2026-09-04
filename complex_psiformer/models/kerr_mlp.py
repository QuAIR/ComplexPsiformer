"""Complex-valued MLP primitives.

Contains the three building blocks used throughout the complex ansatz:

* :class:`KerrActivation` — U(1)-equivariant nonlinearity applied to the
  modulus only.
* :class:`ComplexLinear` — complex dense layer stored as two real
  parameter tensors for gradient/optimizer compatibility.
* :class:`ComplexMLP` — one residual MLP layer composed of the two above.

The published complex models use the four-real-matrix ``FreeComplexLinear``.
"""

import torch
import torch.nn as nn


class KerrActivation(nn.Module):
    """Radial nonlinearity ``σ(z) = tanh(|z| − r₀) z / (|z| + 1e-8)``.

    The learnable per-neuron threshold can make the real radial factor
    negative, reversing phase by pi; the map remains U(1)-equivariant.

    Attributes
    ----------
    r0 : nn.Parameter, shape (dim,)
        Learnable modulus threshold per complex neuron.
    """

    def __init__(self, dim: int) -> None:
        """Allocate the learnable per-neuron threshold.

        Parameters
        ----------
        dim : int
            Number of complex neurons sharing this activation.
        """
        super().__init__()
        self.r0 = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the Kerr nonlinearity elementwise.

        Parameters
        ----------
        z : torch.Tensor, shape (..., dim), complex
            Pre-activation values.

        Returns
        -------
        torch.Tensor, shape (..., dim), complex
            Bounded output with a signed real radial multiplier.
        """
        modulus = torch.abs(z)
        # The positive denominator regularizes z=0 and slightly reduces modulus.
        phase = z / (modulus + 1e-8)
        return torch.tanh(modulus - self.r0) * phase


class SplitTanhActivation(nn.Module):
    """Split nonlinearity ``tanh(Re z) + i tanh(Im z)``."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.tanh(z.real), torch.tanh(z.imag))


class ComplexLinear(nn.Module):
    """Complex dense layer ``z → W z + b`` with separate real/imag parameters.

    The weight is stored as two real :class:`nn.Parameter` tensors so that
    standard PyTorch optimizers (Adam, KFAC-for-real-params) and the complex
    chain rule behave predictably. Initialisation follows a complex Glorot
    scheme scaled by ``(2 · in_dim)**-0.5``.

    Attributes
    ----------
    W_re, W_im : nn.Parameter, shape (out_dim, in_dim)
        Real and imaginary parts of the weight matrix.
    b_re, b_im : nn.Parameter or None
        Real and imaginary parts of the bias vector (``None`` if ``bias=False``).
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        """Allocate and initialise the real and imaginary parameter tensors.

        Parameters
        ----------
        in_dim : int
            Input feature dimension (complex).
        out_dim : int
            Output feature dimension (complex).
        bias : bool, default True
            If ``False``, no bias vectors are allocated.
        """
        super().__init__()
        scale = (2.0 * in_dim) ** -0.5
        self.W_re = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
        self.W_im = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
        if bias:
            self.b_re = nn.Parameter(torch.zeros(out_dim))
            self.b_im = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("b_re", None)
            self.register_parameter("b_im", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the complex linear transformation.

        Parameters
        ----------
        z : torch.Tensor, shape (..., in_dim), complex

        Returns
        -------
        torch.Tensor, shape (..., out_dim), complex
        """
        z_re, z_im = z.real, z.imag
        out_re = z_re @ self.W_re.T - z_im @ self.W_im.T
        out_im = z_re @ self.W_im.T + z_im @ self.W_re.T
        if self.b_re is not None:
            out_re = out_re + self.b_re
            out_im = out_im + self.b_im
        return torch.complex(out_re, out_im)


class FreeComplexLinear(nn.Module):
    """Unconstrained complex-output linear map with four real matrices."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        scale = (2.0 * in_dim) ** -0.5
        self.W_AA = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
        self.W_AB = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
        self.W_BA = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
        self.W_BB = nn.Parameter(torch.randn(out_dim, in_dim) * scale)
        if bias:
            self.b_re = nn.Parameter(torch.zeros(out_dim))
            self.b_im = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("b_re", None)
            self.register_parameter("b_im", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        re, im = z.real, z.imag
        out_re = re @ self.W_AA.T + im @ self.W_AB.T
        out_im = re @ self.W_BA.T + im @ self.W_BB.T
        if self.b_re is not None:
            out_re = out_re + self.b_re
            out_im = out_im + self.b_im
        return torch.complex(out_re, out_im)


class ComplexMLP(nn.Module):
    """Residual complex MLP layer.

    Computes ``h_out = h_in + σ(W h_in + b)`` where ``σ`` is
    :class:`KerrActivation`. Preserves the complex dtype and the residual
    skip is used in the attention-block composition.
    """

    def __init__(
        self,
        dim: int,
        activation: str = "kerr",
        linear_kind: str = "complex",
    ) -> None:
        """Construct one :class:`ComplexLinear` + :class:`KerrActivation` layer.

        Parameters
        ----------
        dim : int
            Feature dimension (in == out for residual compatibility).
        activation : {"kerr", "split_tanh"}, default "kerr"
            Nonlinear activation.  The default preserves old behavior.
        linear_kind : {"complex", "free"}, default "complex"
            Linear map kind.  The default preserves old behavior.
        """
        super().__init__()
        if linear_kind == "complex":
            self.linear: nn.Module = ComplexLinear(dim, dim)
        elif linear_kind == "free":
            self.linear = FreeComplexLinear(dim, dim, bias=True)
        else:
            raise ValueError(f"linear_kind must be 'complex' or 'free', got {linear_kind!r}.")
        if activation == "kerr":
            self.activation: nn.Module = KerrActivation(dim)
        elif activation == "split_tanh":
            self.activation = SplitTanhActivation()
        else:
            raise ValueError(f"activation must be 'kerr' or 'split_tanh', got {activation!r}.")

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Apply the residual complex MLP layer.

        Parameters
        ----------
        f : torch.Tensor, shape (batch, N, dim), complex

        Returns
        -------
        torch.Tensor, shape (batch, N, dim), complex
        """
        return f + self.activation(self.linear(f))
