"""2D periodic supercell geometry.

Provides a single ``Supercell`` dataclass that all downstream code consumes so
that lattice vectors, reciprocal vectors, and cell area are defined in exactly
one place.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass
class Supercell:
    """2D periodic supercell in effective-Bohr units.

    All VMC code operates in the dimensionless effective-atomic-unit system
    (length in ``a*``, energy in ``Ha*``). Instances are typically built via
    the :meth:`triangular` classmethod rather than directly.

    Attributes
    ----------
    L : torch.Tensor, shape (2, 2)
        Rows are the two lattice vectors ``L1``, ``L2``.
    n_electrons : int
        Number of electrons in the simulation cell.
    G : torch.Tensor, shape (2, 2)
        Reciprocal lattice vectors; rows satisfy ``G_i . L_j = 2π δ_ij``.
        Computed in :meth:`__post_init__`.
    area : float
        Absolute area ``|det(L)|`` of the supercell. Computed in
        :meth:`__post_init__`.
    """

    L: torch.Tensor
    n_electrons: int
    G: torch.Tensor = field(init=False)
    area: float = field(init=False)
    # Populated by :meth:`triangular`; ``None`` otherwise. Needed by
    # downstream code (e.g. :class:`MoirePotential`) that references the
    # moiré primitive cell rather than the simulation supercell.
    a_M: float | None = field(default=None)
    n_cell_sides: tuple[int, int] | None = field(default=None)
    n_side: int | None = field(default=None)

    def __post_init__(self) -> None:
        """Derive reciprocal vectors and cell area from ``L``.

        Populates ``self.G = 2π (Lᵀ)⁻¹`` and ``self.area = |det(L)|``.
        Called automatically by the dataclass machinery.
        """
        if self.L.ndim != 2 or tuple(self.L.shape) != (2, 2):
            raise ValueError(
                f"L must have shape (2, 2), got {tuple(self.L.shape)}."
            )
        # G @ L^T == 2π I  =>  G = 2π (L^T)^{-1}.
        self.G = 2.0 * torch.pi * torch.linalg.inv(self.L.T)
        self.area = float(abs(torch.linalg.det(self.L).item()))

    def to_reduced(self, r: torch.Tensor) -> torch.Tensor:
        """Convert real-space coordinates to reduced coordinates in ``[0, 1)²``.

        Parameters
        ----------
        r : torch.Tensor, shape (..., 2)
            Cartesian coordinates.

        Returns
        -------
        torch.Tensor, shape (..., 2)
            Fractional coordinates modulo 1 along each lattice direction.
        """
        Linv = torch.linalg.inv(self.L.to(r.device))
        return (r @ Linv) % 1.0

    def primitive_lattice_vectors(self) -> torch.Tensor:
        """Return the two primitive moiré lattice vectors used to tile this cell."""
        if self.n_cell_sides is None:
            raise ValueError(
                "Primitive lattice vectors are only available for cells "
                "constructed via `Supercell.triangular(...)`."
            )
        n1, n2 = self.n_cell_sides
        return torch.stack([self.L[0] / n1, self.L[1] / n2])

    def primitive_reciprocal_vectors(self) -> torch.Tensor:
        """Return the two primitive moiré reciprocal vectors."""
        if self.n_cell_sides is None:
            raise ValueError(
                "Primitive reciprocal vectors are only available for cells "
                "constructed via `Supercell.triangular(...)`."
            )
        n1, n2 = self.n_cell_sides
        return torch.stack([self.G[0] * n1, self.G[1] * n2])

    @staticmethod
    def _parse_n_cell_sides(n_cells: int | Sequence[int]) -> tuple[int, int]:
        if isinstance(n_cells, bool):
            raise ValueError("n_cells must be a positive integer or a length-2 sequence.")
        if isinstance(n_cells, int):
            if n_cells <= 0:
                raise ValueError(f"n_cells must be positive, got {n_cells}.")
            root = math.isqrt(n_cells)
            for n1 in range(root, 0, -1):
                if n_cells % n1 == 0:
                    return n1, n_cells // n1
            raise AssertionError("positive integers always have at least the factorization 1 x n")

        sides = list(n_cells)
        if len(sides) != 2:
            raise ValueError(f"n_cells sequence must have length 2, got {len(sides)}.")
        n1, n2 = (int(sides[0]), int(sides[1]))
        if n1 <= 0 or n2 <= 0:
            raise ValueError(f"n_cells sides must be positive, got {(n1, n2)}.")
        return n1, n2

    @classmethod
    def triangular(cls, a_M: float, n_electrons: int, n_cells: int | Sequence[int]) -> "Supercell":
        """Construct a triangular moiré supercell.

        Parameters
        ----------
        a_M : float
            Moiré lattice constant in effective Bohr radii.
        n_electrons : int
            Number of electrons at the target filling.
        n_cells : int or sequence of two ints
            Moiré primitive cells tiled into the supercell.  An integer is
            factorized into the least elongated rectangular tiling
            ``n_1 × n_2`` (for example ``12 -> 3 × 4``).  A two-entry sequence
            gives the tiling explicitly, for example ``[3, 4]``.

        Returns
        -------
        Supercell
            Fully initialised instance (``__post_init__`` is invoked).
        """
        n1, n2 = cls._parse_n_cell_sides(n_cells)
        dtype = torch.float64
        a1 = a_M * torch.tensor([1.0, 0.0], dtype=dtype)
        a2 = a_M * torch.tensor([0.5, math.sqrt(3.0) / 2.0], dtype=dtype)
        L = torch.stack([a1 * n1, a2 * n2])
        n_side = n1 if n1 == n2 else None
        return cls(
            L=L,
            n_electrons=n_electrons,
            a_M=float(a_M),
            n_cell_sides=(n1, n2),
            n_side=n_side,
        )
