"""Fixed torus Landau sections with symmetric-gauge boundary phases."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..systems.supercell import Supercell


def symmetric_gauge_transition_phase(
    r: torch.Tensor,
    lattice_vector: torch.Tensor,
    magnetic_B: float | torch.Tensor,
    origin: torch.Tensor,
) -> torch.Tensor:
    """Return ``exp(i B (L_x (y-o_y) - L_y (x-o_x)) / 2)``."""
    lattice = torch.as_tensor(lattice_vector, device=r.device, dtype=r.dtype)
    gauge_origin = torch.as_tensor(origin, device=r.device, dtype=r.dtype)
    field = torch.as_tensor(magnetic_B, device=r.device, dtype=r.dtype)
    relative = r - gauge_origin
    angle = 0.5 * field * (
        lattice[0] * relative[..., 1] - lattice[1] * relative[..., 0]
    )
    return torch.exp(1j * angle)


def _streamed_landau_image_sums(
    oscillator_x: torch.Tensor,
    phase: torch.Tensor,
    *,
    n_levels: int,
) -> torch.Tensor:
    """Evaluate all level-major image sums from flux-label coordinates."""
    if oscillator_x.ndim != 4 or oscillator_x.shape != phase.shape:
        raise ValueError(
            "oscillator_x and phase must share shape "
            "(batch, electrons, flux_labels, images)"
        )
    if n_levels <= 0:
        raise ValueError("n_levels must be positive")

    previous = math.pi ** (-0.25) * torch.exp(-0.5 * oscillator_x.square())
    image_sums = [torch.sum(previous * phase, dim=-1)]
    if n_levels >= 2:
        current = math.sqrt(2.0) * oscillator_x * previous
        image_sums.append(torch.sum(current * phase, dim=-1))
        for level in range(1, n_levels - 1):
            following = (
                math.sqrt(2.0 / (level + 1)) * oscillator_x * current
                - math.sqrt(level / (level + 1)) * previous
            )
            image_sums.append(torch.sum(following * phase, dim=-1))
            previous, current = current, following

    stacked = torch.stack(image_sums, dim=2)
    return stacked.reshape(*stacked.shape[:2], -1)


class TorusMagneticSections(nn.Module):
    """Oblique-cell Landau sections sharing one symmetric-gauge transition law."""

    def __init__(
        self,
        cell: Supercell,
        magnetic_B: float,
        magnetic_flux_quanta: int | float,
        origin: torch.Tensor,
        n_sections: int,
        image_radius: int = 8,
    ) -> None:
        super().__init__()
        lattice = cell.L.detach().clone()
        if not lattice.is_floating_point():
            lattice = lattice.to(torch.get_default_dtype())
        if not math.isfinite(float(magnetic_B)) or magnetic_B <= 0.0:
            raise ValueError("magnetic_B must be finite and positive")
        flux_value = float(magnetic_flux_quanta)
        flux_quanta = round(flux_value)
        if flux_quanta <= 0 or not math.isclose(
            flux_value, flux_quanta, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("magnetic_flux_quanta must be a positive integer")
        expected_flux = 2.0 * math.pi * flux_quanta
        actual_flux = float(magnetic_B) * cell.area
        if not math.isclose(actual_flux, expected_flux, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(
                "magnetic flux mismatch: magnetic_B * cell.area must equal "
                "2*pi*magnetic_flux_quanta"
            )
        if isinstance(n_sections, bool) or int(n_sections) != n_sections or n_sections <= 0:
            raise ValueError("n_sections must be a positive integer")
        if isinstance(image_radius, bool) or int(image_radius) != image_radius or image_radius < 1:
            raise ValueError("image_radius must be a positive integer")

        L1, L2 = lattice
        length_x = torch.linalg.vector_norm(L1)
        if float(length_x) == 0.0:
            raise ValueError("the first cell vector must be nonzero")
        axis_x = L1 / length_x
        axis_y = torch.stack((-axis_x[1], axis_x[0]))
        shear = torch.dot(L2, axis_x)
        length_y = torch.dot(L2, axis_y)
        if abs(float(length_y)) <= torch.finfo(lattice.dtype).eps:
            raise ValueError("cell vectors must be linearly independent")

        flux_labels = torch.arange(flux_quanta, device=lattice.device)
        images = torch.arange(
            -int(image_radius),
            int(image_radius) + 1,
            device=lattice.device,
            dtype=lattice.dtype,
        )

        self.n_sections = int(n_sections)
        self.image_radius = int(image_radius)
        self.magnetic_flux_quanta = int(flux_quanta)
        self.max_landau_level = (self.n_sections - 1) // self.magnetic_flux_quanta
        self.register_buffer("lattice", lattice, persistent=False)
        self.register_buffer(
            "inverse_lattice",
            torch.linalg.inv(lattice),
            persistent=False,
        )
        self.register_buffer(
            "magnetic_B",
            torch.as_tensor(magnetic_B, device=lattice.device, dtype=lattice.dtype),
            persistent=False,
        )
        self.register_buffer(
            "origin",
            torch.as_tensor(origin, device=lattice.device, dtype=lattice.dtype).clone(),
            persistent=False,
        )
        self.register_buffer("axis_x", axis_x, persistent=False)
        self.register_buffer("axis_y", axis_y, persistent=False)
        self.register_buffer("length_x", length_x, persistent=False)
        self.register_buffer("shear", shear, persistent=False)
        self.register_buffer("length_y", length_y, persistent=False)
        self.register_buffer("flux_labels", flux_labels, persistent=False)
        self.register_buffer("images", images, persistent=False)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate sections with shape ``(batch, electrons, n_sections)``."""
        if r.ndim != 3 or r.shape[-1] != 2:
            raise ValueError(f"r must have shape (batch, electrons, 2), got {tuple(r.shape)}")
        lattice = self.lattice.to(device=r.device, dtype=r.dtype)
        inverse_lattice = self.inverse_lattice.to(device=r.device, dtype=r.dtype)
        origin = self.origin.to(device=r.device, dtype=r.dtype)
        axis_x = self.axis_x.to(device=r.device, dtype=r.dtype)
        axis_y = self.axis_y.to(device=r.device, dtype=r.dtype)
        length_x = self.length_x.to(device=r.device, dtype=r.dtype)
        shear = self.shear.to(device=r.device, dtype=r.dtype)
        length_y = self.length_y.to(device=r.device, dtype=r.dtype)
        field = self.magnetic_B.to(device=r.device, dtype=r.dtype)
        images = self.images.to(device=r.device, dtype=r.dtype)
        labels = self.flux_labels.to(device=r.device, dtype=r.dtype)

        reduced = r @ inverse_lattice
        windings = torch.floor(reduced)
        wrapped = (reduced - windings) @ lattice
        relative = wrapped - origin
        X = relative @ axis_x
        Y = relative @ axis_y
        base_momenta = 2.0 * math.pi * labels / length_x
        momenta = base_momenta[:, None] + field * length_y * images[None, :]
        oscillator_x = torch.sqrt(field) * (
            Y[:, :, None, None] + momenta[None, None, :, :] / field
        )
        image_angle = (
            images[None, :] * base_momenta[:, None] * shear
            + 0.5 * field * length_y * shear * images[None, :].square()
        )
        plane_angle = X[:, :, None, None] * momenta[None, None, :, :]
        image_sum = _streamed_landau_image_sums(
            oscillator_x,
            torch.exp(1j * plane_angle)
            * torch.exp(1j * image_angle)[None, None, :, :],
            n_levels=self.max_landau_level + 1,
        )[..., : self.n_sections]
        normalization = field.pow(0.25) / torch.sqrt(length_x)
        gauge_transform = torch.exp(0.5j * field * X * Y)
        canonical_sections = normalization * gauge_transform[:, :, None] * image_sum

        n1 = windings[:, :, 0]
        n2 = windings[:, :, 1]
        L1, L2 = lattice
        chi_L1 = 0.5 * field * (
            L1[0] * relative[:, :, 1] - L1[1] * relative[:, :, 0]
        )
        after_L1 = wrapped + n1[:, :, None] * L1
        after_L1_relative = after_L1 - origin
        chi_L2_after_L1 = 0.5 * field * (
            L2[0] * after_L1_relative[:, :, 1]
            - L2[1] * after_L1_relative[:, :, 0]
        )
        winding_phase = torch.exp(1j * (n1 * chi_L1 + n2 * chi_L2_after_L1))
        return winding_phase[:, :, None] * canonical_sections
