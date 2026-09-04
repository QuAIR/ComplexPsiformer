"""Interaction scaling used by the Hamiltonian."""
import math
import torch
from torch import nn
from .hamiltonian import EwaldCoulomb

class ScaledEwald(nn.Module):
    """Expose the Ewald API while scaling only interaction energy."""

    def __init__(self, ewald: EwaldCoulomb, scale: float) -> None:
        super().__init__()
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("lambda_C must be finite and nonnegative")
        self.ewald = ewald
        self.scale = float(scale)

    def _zeros(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(value.shape[:-2], dtype=value.dtype, device=value.device)

    def electron_electron(self, r: torch.Tensor) -> torch.Tensor:
        if self.scale == 0.0:
            return self._zeros(r)
        return self.scale * self.ewald.electron_electron(r)

    def electron_electron_from_pair_geometry(self, pair_cart: torch.Tensor) -> torch.Tensor:
        if self.scale == 0.0:
            return torch.zeros(
                pair_cart.shape[:-3],
                dtype=pair_cart.dtype,
                device=pair_cart.device,
            )
        return self.scale * self.ewald.electron_electron_from_pair_geometry(pair_cart)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.electron_electron(r)
