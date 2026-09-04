"""Shared orbital-level magnetic-boundary carrier for native wavefunctions."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from ..features.magnetic import TorusMagneticSections
from ..systems.supercell import Supercell


class MagneticBoundaryAdapter(nn.Module):
    """Apply one analytic magnetic section to every native orbital row.

    The wrapped model keeps its own architecture and trainable parameters.
    The adapter only replaces the final determinant assembly so all native
    models obey the same symmetric-gauge torus transition law.
    """

    def __init__(
        self,
        base_model: nn.Module,
        cell: Supercell,
        magnetic_B: float,
        magnetic_flux_quanta: int | float,
        magnetic_origin: torch.Tensor | tuple[float, float],
    ) -> None:
        super().__init__()
        if bool(getattr(base_model, "enforce_magnetic_boundary", False)) or getattr(
            base_model,
            "magnetic_sections",
            None,
        ) is not None:
            raise ValueError("base model already applies a magnetic boundary carrier")
        if not callable(getattr(base_model, "_compute_orbitals", None)) and not callable(
            getattr(base_model, "_orbitals", None)
        ):
            raise TypeError(
                f"{type(base_model).__name__} has no _compute_orbitals or _orbitals interface"
            )
        n_det = getattr(base_model, "n_det", None)
        if isinstance(n_det, bool) or not isinstance(n_det, int) or n_det <= 0:
            raise TypeError("base model must expose a positive integer n_det")
        coefficients = getattr(base_model, "c", None)
        if coefficients is not None and (
            not isinstance(coefficients, torch.Tensor)
            or coefficients.ndim != 1
            or coefficients.numel() != n_det
        ):
            raise TypeError(
                "base model determinant coefficients c must be absent or have shape (n_det,)"
            )

        flux_value = float(magnetic_flux_quanta)
        flux_quanta = round(flux_value)
        if flux_quanta <= 0 or not math.isclose(
            flux_value,
            flux_quanta,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("magnetic_flux_quanta must be a positive integer")

        origin = torch.as_tensor(
            magnetic_origin,
            device=cell.L.device,
            dtype=cell.L.dtype,
        ).clone()
        if origin.shape != (2,):
            raise ValueError("magnetic_origin must have shape (2,)")

        self.base_model = base_model
        self.cell = cell
        self.n_det = n_det
        self.magnetic_B = float(magnetic_B)
        self.magnetic_flux_quanta = int(flux_quanta)
        self.register_buffer("magnetic_origin", origin, persistent=False)
        self.sections = TorusMagneticSections(
            cell,
            magnetic_B=magnetic_B,
            magnetic_flux_quanta=flux_quanta,
            origin=origin,
            n_sections=cell.n_electrons,
        )

    @property
    def c(self) -> torch.Tensor | None:
        """Return trainable coefficients, or ``None`` for a fixed equal sum."""
        return self.base_model.c

    def _native_orbitals(self, r: torch.Tensor) -> torch.Tensor:
        compute: Callable[[torch.Tensor], torch.Tensor] | None = getattr(
            self.base_model,
            "_compute_orbitals",
            None,
        )
        if callable(compute):
            return compute(r)
        orbitals: Callable[[torch.Tensor], torch.Tensor] | None = getattr(
            self.base_model,
            "_orbitals",
            None,
        )
        if callable(orbitals):
            return orbitals(r)
        raise TypeError(f"{type(self.base_model).__name__} has no orbital interface")

    def _magnetic_orbitals(self, r: torch.Tensor) -> torch.Tensor:
        orbitals = self._native_orbitals(r)
        expected_shape = (
            r.shape[0],
            self.n_det,
            self.cell.n_electrons,
            self.cell.n_electrons,
        )
        if orbitals.shape != expected_shape:
            raise ValueError(
                f"native orbitals must have shape {expected_shape}, got {tuple(orbitals.shape)}"
            )
        return orbitals * self.sections(r).unsqueeze(1)

    def log_psi(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stable ``log|Psi|`` and complex unit phase."""
        orbitals = self._magnetic_orbitals(r)
        signs, logdets = torch.linalg.slogdet(orbitals)
        offset = logdets.max(dim=-1, keepdim=True).values
        determinant_values = signs * torch.exp(logdets - offset)
        if self.c is None:
            psi = determinant_values.sum(dim=-1)
        else:
            psi = (self.c.to(determinant_values.dtype) * determinant_values).sum(dim=-1)
        modulus = torch.abs(psi)
        return torch.log(modulus) + offset.squeeze(-1), psi / (modulus + 1e-30)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return ``log|Psi|`` for Metropolis sampling."""
        return self.log_psi(r)[0]

    def set_gate_multiplier(self, value: float) -> None:
        """Forward the attention gate warmup while remaining a no-op for references."""
        setter = getattr(self.base_model, "set_gate_multiplier", None)
        if callable(setter):
            setter(value)
