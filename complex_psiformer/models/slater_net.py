"""Real SlaterNet with one Slater determinant."""

import torch
import torch.nn as nn

from ..features.periodic import RealPeriodicFeatures
from ..systems.supercell import Supercell


class RealSlaterNet(nn.Module):
    """Single-determinant wavefunction with a residual electron-wise MLP."""

    def __init__(
        self,
        cell: Supercell,
        d_model: int = 64,
        n_layers: int = 3,
        n_det: int = 1,
        paired_orbital_readout: bool = False,
        embed_bias: bool = False,
        embed_activation: str = "none",
        trainable_determinant_coefficients: bool = False,
    ) -> None:
        """Construct the single-determinant model from the stated parameters."""
        super().__init__()
        if isinstance(n_det, bool) or n_det != 1:
            raise ValueError("SlaterNet supports n_det=1 only in this package")
        if trainable_determinant_coefficients:
            raise ValueError("The single determinant has a fixed unit coefficient")
        self.N = cell.n_electrons
        self.n_det = n_det
        self.d_model = d_model
        self.paired_orbital_readout = bool(paired_orbital_readout)
        self.embed_activation_kind = embed_activation
        self.trainable_determinant_coefficients = bool(
            trainable_determinant_coefficients
        )
        if self.paired_orbital_readout and d_model % 2 != 0:
            raise ValueError("paired_orbital_readout requires an even d_model")
        if embed_activation not in {"tanh", "none"}:
            raise ValueError("embed_activation must be 'tanh' or 'none'")

        self.features = RealPeriodicFeatures(cell)
        layers: list[nn.Module] = [
            nn.Linear(self.features.out_dim, d_model, bias=embed_bias)
        ]
        for _ in range(n_layers):
            layers.append(nn.Linear(d_model, d_model))
        self.net = nn.ModuleList(layers)

        orbital_width = d_model if self.paired_orbital_readout else 2 * d_model
        self.w = nn.Parameter(
            torch.randn(n_det, self.N, orbital_width) * d_model ** -0.5
        )
        determinant_coefficients = torch.ones(n_det) / n_det
        if self.trainable_determinant_coefficients:
            self.c = nn.Parameter(determinant_coefficients)
        else:
            self.register_buffer("c", determinant_coefficients)

    def _orbitals(self, r: torch.Tensor) -> torch.Tensor:
        """Compute per-determinant complex orbital matrix.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        torch.Tensor, shape (batch, n_det, N, N), complex
        """
        h = self.net[0](self.features(r))
        if self.embed_activation_kind == "tanh":
            h = torch.tanh(h)
        for layer in self.net[1:]:
            h = h + torch.tanh(layer(h))

        if self.paired_orbital_readout:
            half = self.d_model // 2
            h_re, h_im = h.split(half, dim=-1)
            w_re, w_im = self.w.split(half, dim=-1)
            phi_re = torch.einsum("bid,mjd->bmij", h_re, w_re)
            phi_re = phi_re + torch.einsum("bid,mjd->bmij", h_im, w_im)
            phi_im = torch.einsum("bid,mjd->bmij", h_im, w_re)
            phi_im = phi_im - torch.einsum("bid,mjd->bmij", h_re, w_im)
            return torch.complex(phi_re, phi_im)

        w_re = self.w[:, :, : self.d_model]
        w_im = self.w[:, :, self.d_model :]
        # Same index convention as ComplexPsiformer: electron i is the row,
        # orbital j is the column of the Slater matrix.
        phi_re = torch.einsum("bid,mjd->bmij", h, w_re)
        phi_im = torch.einsum("bid,mjd->bmij", h, w_im)
        return torch.complex(phi_re, phi_im)

    def log_psi(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``log|Ψ|`` and the complex phase of ``Ψ``.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        log_psi_abs : torch.Tensor, shape (batch,), real
        phase_psi : torch.Tensor, shape (batch,), complex, unit modulus
        """
        phi = self._orbitals(r)
        signs, logdets = torch.linalg.slogdet(phi)

        max_logdet = logdets.max(dim=-1, keepdim=True).values
        det_vals = signs * torch.exp(logdets - max_logdet)
        c = self.c.to(det_vals.dtype)
        psi = (c * det_vals).sum(dim=-1)

        abs_psi = torch.abs(psi)
        log_psi_abs = torch.log(abs_psi) + max_logdet.squeeze(-1)
        phase_psi = psi / (abs_psi + 1e-30)
        return log_psi_abs, phase_psi

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return ``log|Ψ|`` only.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        torch.Tensor, shape (batch,), real
        """
        log_psi_abs, _ = self.log_psi(r)
        return log_psi_abs
