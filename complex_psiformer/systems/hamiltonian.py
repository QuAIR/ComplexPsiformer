"""Hamiltonian terms for the 2D periodic electron gas.

Contains the 2D Ewald-summed Coulomb interaction (Fraser et al. 1996, adapted
for 2D as in the target paper's Appendix A) and the triangular moiré
one-body potential.
"""

import math

import torch
import torch.nn as nn

from .supercell import Supercell


class EwaldCoulomb(nn.Module):
    """2D Ewald summation for the periodic electron-electron Coulomb energy.

    Splits the periodic Coulomb potential
    ``V(r) = Σ_L 1/|r + L|`` into a short-range real-space piece
    (weighted by ``erfc``) and a long-range reciprocal-space piece
    (weighted by ``erfc`` of the dimensionless reciprocal magnitude),
    plus a ``G = 0`` finite correction and — for the self-image
    case — the divergent self-interaction subtraction.

    With ``α = 1/(2η)`` the Ewald splitting parameter, the three parts are

    * real-space: ``Σ_{L} erfc(α|r+L|)/|r+L|``,
    * reciprocal: ``(2π/A) Σ_{G≠0} cos(G·r) erfc(|G|/(2α))/|G|``,
    * ``G=0`` finite piece (constant): ``−(2π/A)·(2η/√π)``.

    For the Madelung (self-image) constant one additionally subtracts the
    singular on-site term ``1/(η√π)``.

    The default ``η = √(A/π)`` balances the two series so each converges
    within a handful of images in a primitive cell. The plan's original
    ``η = √(π/A)`` is dimensionally inconsistent (``η`` must be a
    length) and is not used.

    Inherits from :class:`torch.nn.Module` so that the precomputed lattice
    buffers move with ``.to(device)`` alongside the wavefunction model.
    """

    def __init__(
        self,
        cell: Supercell,
        n_images: int = 5,
        eta: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Precompute lattice sums and the Madelung constant.

        Parameters
        ----------
        cell : Supercell
            The simulation supercell.
        n_images : int, default 5
            Range of image indices ``(n1, n2)`` included in both the real
            and reciprocal lattice sums (so ``(2n+1)²`` vectors each).
        eta : float, optional
            Ewald splitting length in the same units as ``cell.L``. If
            ``None``, defaults to ``sqrt(cell.area / π)``.
        device : torch.device or str, optional
            Device on which to allocate the lattice buffers. ``None``
            inherits ``cell.L.device`` (typically CPU). Equivalent to
            constructing on CPU and then calling ``.to(device)``.
        """
        super().__init__()
        self.cell = cell
        self.n_images = int(n_images)
        self.eta = float(math.sqrt(cell.area / math.pi)) if eta is None else float(eta)

        real_vecs, recip_vecs, recip_norms = self._build_lattices()
        self.register_buffer("real_vecs", real_vecs)
        self.register_buffer("recip_vecs", recip_vecs)
        self.register_buffer("recip_norms", recip_norms)
        # Madelung is a Python float (geometry-only scalar); safe to mix
        # with GPU tensors via broadcasting.
        self.madelung: float = self._compute_madelung()
        if device is not None:
            self.to(device)

    def _build_lattices(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ``(real_vecs, recip_vecs, recip_norms)`` on ``cell.L.device``.

        Returned tensors are ready to be registered as buffers — the
        ``register_buffer`` call in ``__init__`` makes them move with
        subsequent ``.to(device)``.
        """
        dtype = self.cell.L.dtype
        device = self.cell.L.device
        ns = torch.arange(-self.n_images, self.n_images + 1, dtype=dtype, device=device)
        n1, n2 = torch.meshgrid(ns, ns, indexing="ij")

        real_vecs = (
            n1.reshape(-1, 1) * self.cell.L[0]
            + n2.reshape(-1, 1) * self.cell.L[1]
        )

        G_vecs = (
            n1.reshape(-1, 1) * self.cell.G[0]
            + n2.reshape(-1, 1) * self.cell.G[1]
        )
        norms = torch.linalg.vector_norm(G_vecs, dim=-1)
        mask = norms > 1e-10
        recip_vecs = G_vecs[mask]
        recip_norms = norms[mask]
        return real_vecs, recip_vecs, recip_norms

    def _compute_madelung(self) -> float:
        """Compute the Madelung self-image energy ``ξ_M``.

        ``ξ_M = Σ_{L≠0} erfc(|L|/(2η))/|L|
                + (2π/A) Σ_{G≠0} erfc(η|G|)/|G|
                − (2π/A)(2η/√π)
                − 1/(η√π)``

        Returns
        -------
        float
            Madelung constant in the same units as ``1/|L|``. Depends
            only on supercell geometry.
        """
        eta = self.eta
        sqrt_pi = math.sqrt(math.pi)
        area = self.cell.area

        L_norms = torch.linalg.vector_norm(self.real_vecs, dim=-1)
        mask = L_norms > 1e-10
        L_nz = L_norms[mask]
        xi_sr = torch.sum(torch.erfc(L_nz / (2.0 * eta)) / L_nz)

        xi_lr = (2.0 * math.pi / area) * torch.sum(
            torch.erfc(eta * self.recip_norms) / self.recip_norms
        )

        xi_q0 = (2.0 * math.pi / area) * (2.0 * eta / sqrt_pi)
        xi_self = 1.0 / (eta * sqrt_pi)

        return float(xi_sr.item() + xi_lr.item() - xi_q0 - xi_self)

    def electron_electron(self, r: torch.Tensor) -> torch.Tensor:
        """Total electron-electron Coulomb energy for one or many configurations.

        ``V_ee = Σ_{i<j} V_pair(r_i − r_j) + (N/2) · ξ_M``

        with ``V_pair(r) = Σ_L erfc(|r+L|/(2η))/|r+L|
              + (2π/A) Σ_{G≠0} cos(G·r) erfc(η|G|)/|G|
              − (2π/A)(2η/√π)``.

        Accepts either a single walker ``(N, 2)`` or a batch of walkers
        ``(batch, N, 2)``. In the batched case all walkers are processed
        in one vectorised call — every ``(walker, pair, image)`` triple
        appears along its own tensor axis with no Python-level iteration.

        Parameters
        ----------
        r : torch.Tensor, shape ``(N, 2)`` or ``(batch, N, 2)``

        Returns
        -------
        torch.Tensor, shape ``()`` if input was ``(N, 2)``, else ``(batch,)``.
        """
        if r.ndim == 2:
            return self._electron_electron_batched(r.unsqueeze(0))[0]
        if r.ndim == 3:
            return self._electron_electron_batched(r)
        raise ValueError(
            f"electron_electron expects r of shape (N, 2) or (batch, N, 2); "
            f"got {tuple(r.shape)}."
        )

    def electron_electron_from_pair_geometry(self, pair_cart: torch.Tensor) -> torch.Tensor:
        """Total electron-electron energy from a dense minimum-image pair table.

        Parameters
        ----------
        pair_cart : torch.Tensor
            Dense pair displacements with shape ``(N, N, 2)`` or
            ``(batch, N, N, 2)``. The displacements may already be wrapped into
            the minimum image; the periodic Ewald lattice sums remain exact
            because shifting ``r_ij`` by a lattice vector only reindexes the
            image summation.
        """

        if pair_cart.ndim == 3:
            return self._electron_electron_from_pair_geometry_batched(pair_cart.unsqueeze(0))[0]
        if pair_cart.ndim == 4:
            return self._electron_electron_from_pair_geometry_batched(pair_cart)
        raise ValueError(
            "electron_electron_from_pair_geometry expects pair_cart of shape "
            f"(N, N, 2) or (batch, N, N, 2); got {tuple(pair_cart.shape)}."
        )

    def _electron_electron_batched(self, r: torch.Tensor) -> torch.Tensor:
        """Vectorised implementation of :meth:`electron_electron`.

        Parameters
        ----------
        r : torch.Tensor, shape ``(batch, N, 2)``

        Returns
        -------
        torch.Tensor, shape ``(batch,)``
        """
        _, N, _ = r.shape
        # All pair differences: (B, N, N, 2), then keep only i<j via a mask.
        diff = r.unsqueeze(2) - r.unsqueeze(1)              # (B, N, N, 2)
        return self._electron_electron_from_pair_geometry_batched(diff)

    def _electron_electron_from_pair_geometry_batched(self, pair_cart: torch.Tensor) -> torch.Tensor:
        """Vectorised implementation from dense pair displacements."""

        _, N, _, _ = pair_cart.shape
        eta = self.eta
        sqrt_pi = math.sqrt(math.pi)
        q0_per_pair = (2.0 * math.pi / self.cell.area) * (2.0 * eta / sqrt_pi)
        iu = torch.triu_indices(N, N, offset=1, device=pair_cart.device)
        rij = pair_cart[:, iu[0], iu[1], :]                  # (B, P, 2), P = N(N-1)/2

        # Real-space sum over image lattice: disp[B, P, L, 2], d[B, P, L].
        disp = rij.unsqueeze(2) + self.real_vecs              # (B, P, L, 2)
        d = torch.linalg.vector_norm(disp, dim=-1)            # (B, P, L)
        V_sr = torch.sum(torch.erfc(d / (2.0 * eta)) / d, dim=-1)  # (B, P)

        # Reciprocal-space sum: phase[B, P, G] = recip·rij.
        phase = torch.einsum("gd,bpd->bpg", self.recip_vecs, rij)
        recip_weight = torch.erfc(eta * self.recip_norms) / self.recip_norms  # (G,)
        V_lr = (2.0 * math.pi / self.cell.area) * torch.einsum(
            "bpg,g->bp", torch.cos(phase), recip_weight
        )

        pair_energy = (V_sr + V_lr - q0_per_pair).sum(dim=-1)  # (B,)
        return pair_energy + 0.5 * N * self.madelung


class MoirePotential(nn.Module):
    """Three-star triangular moiré one-body potential.

    ``V(r) = −2 V₀ Σ_{j=1}^{3} cos(g_j · r + φ)`` where the three
    ``g_j`` are the first-shell moiré reciprocal vectors, related by
    120° rotations. They are derived from the primitive reciprocal basis
    ``{b_1, b_2}`` as ``{b_1, b_2, −(b_1 + b_2)}``; for a rectangular
    ``n_1 × n_2`` supercell this means ``b_1 = n_1 · G_1`` and
    ``b_2 = n_2 · G_2``.

    Inherits :class:`torch.nn.Module` so the ``g`` buffer migrates with
    ``.to(device)``.
    """

    def __init__(
        self,
        cell: Supercell,
        V0: float,
        phi: float,
        device: torch.device | str | None = None,
    ) -> None:
        """Store potential parameters and build the three ``g`` vectors.

        Parameters
        ----------
        cell : Supercell
            Must have been built via :meth:`Supercell.triangular` so that the
            primitive moiré reciprocal basis is known.
        V0 : float
            Potential amplitude (effective Hartree). The prefactor is
            ``-2 V0`` per star vector.
        phi : float
            Phase offset in radians.
        device : torch.device or str, optional
            Device for the ``g`` buffer. ``None`` keeps ``cell.L.device``.
        """
        super().__init__()
        if cell.n_cell_sides is None:
            raise ValueError(
                "MoirePotential requires a triangular Supercell "
                "constructed via `Supercell.triangular(...)` so that the "
                "primitive cell tiling is known."
            )
        self.cell = cell
        self.V0 = float(V0)
        self.phi = float(phi)

        b1, b2 = cell.primitive_reciprocal_vectors()
        # Three first-shell star vectors, mutually 120°-related; exactly
        # reciprocal-lattice vectors of the primitive moiré cell, hence
        # periodic under r -> r + a_i and therefore under r -> r + L_i.
        g = torch.stack([b1, b2, -(b1 + b2)]).to(cell.L.dtype)
        self.register_buffer("g", g)
        if device is not None:
            self.to(device)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate ``V(r)`` elementwise.

        Parameters
        ----------
        r : torch.Tensor, shape (..., 2)
            Real-space coordinates.

        Returns
        -------
        torch.Tensor, shape (...)
            Potential values at each coordinate.
        """
        phases = r @ self.g.T + self.phi  # (..., 3)
        return -2.0 * self.V0 * torch.cos(phases).sum(dim=-1)
