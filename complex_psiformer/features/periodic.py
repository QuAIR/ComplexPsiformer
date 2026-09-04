"""Periodic single-particle feature maps.

Maps each electron coordinate ``r ∈ ℝ²`` into a translation-periodic feature
vector. Complex features are plane-wave phases; real features are their
sine and cosine components. Magnetic boundary phases are applied separately
at the orbital level.
"""

import torch
import torch.nn as nn

from ..systems.supercell import Supercell


_HARMONIC_MODES = {
    "axis",
    "half_space",
    "primitive_moire",
    "axis_plus_primitive_moire",
    "half_space_plus_primitive_moire",
}


def _dedupe_indices(indices: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        unique.append(index)
    return tuple(unique)


def _primitive_moire_indices(cell: Supercell) -> tuple[tuple[int, int], ...]:
    if cell.n_cell_sides is None:
        raise ValueError(
            "primitive moire feature modes require a cell constructed via "
            "`Supercell.triangular(...)` so n_cell_sides are known."
        )
    n1, n2 = cell.n_cell_sides
    return ((n1, 0), (0, n2), (-n1, -n2))


def _base_harmonic_indices(max_harmonic: int, mode: str) -> tuple[tuple[int, int], ...]:
    if max_harmonic < 1:
        raise ValueError(f"max_harmonic must be >= 1, got {max_harmonic}")
    if mode == "axis":
        indices: list[tuple[int, int]] = []
        for n in range(1, max_harmonic + 1):
            indices.append((n, 0))
            indices.append((0, n))
        return tuple(indices)
    if mode == "half_space":
        indices = []
        for n1 in range(-max_harmonic, max_harmonic + 1):
            for n2 in range(-max_harmonic, max_harmonic + 1):
                if n1 == 0 and n2 == 0:
                    continue
                if n1 > 0 or (n1 == 0 and n2 > 0):
                    indices.append((n1, n2))
        return tuple(indices)
    raise ValueError(f"unsupported base harmonic mode {mode!r}")


def _harmonic_indices(
    cell: Supercell,
    max_harmonic: int,
    mode: str,
) -> tuple[tuple[int, int], ...]:
    if mode not in _HARMONIC_MODES:
        modes = "', '".join(sorted(_HARMONIC_MODES))
        raise ValueError(f"unsupported harmonic mode {mode!r}; expected one of '{modes}'")
    if mode in {"axis", "half_space"}:
        return _base_harmonic_indices(max_harmonic, mode)
    if mode == "primitive_moire":
        return _primitive_moire_indices(cell)
    if mode == "axis_plus_primitive_moire":
        return _dedupe_indices(
            list(_base_harmonic_indices(max_harmonic, "axis"))
            + list(_primitive_moire_indices(cell))
        )
    if mode == "half_space_plus_primitive_moire":
        return _dedupe_indices(
            list(_base_harmonic_indices(max_harmonic, "half_space"))
            + list(_primitive_moire_indices(cell))
        )
    raise AssertionError(f"unhandled harmonic mode {mode!r}")


def _harmonic_reciprocal_vectors(
    cell: Supercell,
    max_harmonic: int,
    mode: str,
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    indices = _harmonic_indices(cell, max_harmonic, mode)
    g1, g2 = cell.G[0], cell.G[1]
    vectors = torch.stack([n1 * g1 + n2 * g2 for n1, n2 in indices], dim=0)
    return vectors, indices


class ComplexPeriodicFeatures(nn.Module):
    """Complex plane-wave features ``f(r) = [exp(i G_k·r)]``.

    Output is periodic under ``r → r + L_i`` and has unit modulus elementwise.
    This is informationally equivalent to the real (sin, cos) encoding but
    preserves phase structure.

    Attributes
    ----------
    G : torch.Tensor, shape (2, 2)
        Selected reciprocal supercell harmonics (registered as buffer).
    out_dim : int
        Output feature dimension, complex-valued.
    """

    def __init__(
        self,
        cell: Supercell,
        max_harmonic: int = 1,
        harmonic_mode: str = "axis",
    ) -> None:
        """Register ``cell.G`` as a non-trainable buffer.

        Parameters
        ----------
        cell : Supercell
            The simulation supercell providing reciprocal vectors.
        max_harmonic : int, default 1
            Highest integer reciprocal-lattice harmonic included.
        harmonic_mode : {"axis", "half_space", "primitive_moire",
            "axis_plus_primitive_moire", "half_space_plus_primitive_moire"},
            default "axis"
            ``"axis"`` keeps the legacy ``G1, G2`` features at
            ``max_harmonic=1`` and appends ``2G1, 2G2, ...`` for richer
            periodic input. ``"half_space"`` uses one representative from
            each ``±G`` pair with ``|n_i| <= max_harmonic``. The
            ``"primitive_moire"`` modes explicitly include the moiré
            primitive reciprocal star ``b1, b2, -(b1+b2)`` where
            ``b1 = n1 G1`` and ``b2 = n2 G2`` for an ``n1 x n2`` supercell.
        """
        super().__init__()
        vectors, indices = _harmonic_reciprocal_vectors(
            cell,
            max_harmonic=max_harmonic,
            mode=harmonic_mode,
        )
        self.max_harmonic = int(max_harmonic)
        self.harmonic_mode = harmonic_mode
        self.harmonic_indices = indices
        self.register_buffer("G", vectors.detach().clone())
        self.out_dim = int(vectors.shape[0])

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate the complex plane-wave feature map.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real
            Electron coordinates.

        Returns
        -------
        torch.Tensor, shape (batch, N, out_dim), complex
            Unit-modulus plane-wave features.
        """
        # phases[..., i] = G_i · r; with G shape (2, 2) and rows = G_1, G_2,
        # the matrix product r @ G^T yields a row vector of dot products.
        # Cast G to r's dtype so callers that keep the default float32 path
        # (skipping .double()) don't collide with the float64 supercell buffer
        # — downstream linear layers inherit the input dtype.
        phases = r @ self.G.to(r.dtype).T
        return torch.exp(1j * phases)


class RealPeriodicFeatures(nn.Module):
    """Real (sin, cos) feature map for selected reciprocal harmonics.

    Used by Real PsiFormer and Real SlaterNet.

    Attributes
    ----------
    G : torch.Tensor, shape (2, 2)
        Selected reciprocal supercell harmonics (registered as buffer).
    out_dim : int
        Output feature dimension, real-valued.
    """

    def __init__(
        self,
        cell: Supercell,
        max_harmonic: int = 1,
        harmonic_mode: str = "axis",
    ) -> None:
        """Register ``cell.G`` as a non-trainable buffer.

        Parameters
        ----------
        cell : Supercell
            The simulation supercell providing reciprocal vectors.
        max_harmonic : int, default 1
            Highest integer reciprocal-lattice harmonic included.
        harmonic_mode : {"axis", "half_space", "primitive_moire",
            "axis_plus_primitive_moire", "half_space_plus_primitive_moire"},
            default "axis"
            ``"axis"`` keeps the legacy ``G1, G2`` features at
            ``max_harmonic=1`` and appends ``2G1, 2G2, ...`` for richer
            periodic input. ``"half_space"`` uses one representative from
            each ``±G`` pair with ``|n_i| <= max_harmonic``. The
            ``"primitive_moire"`` modes explicitly include the moiré
            primitive reciprocal star ``b1, b2, -(b1+b2)`` where
            ``b1 = n1 G1`` and ``b2 = n2 G2`` for an ``n1 x n2`` supercell.
        """
        super().__init__()
        vectors, indices = _harmonic_reciprocal_vectors(
            cell,
            max_harmonic=max_harmonic,
            mode=harmonic_mode,
        )
        self.max_harmonic = int(max_harmonic)
        self.harmonic_mode = harmonic_mode
        self.harmonic_indices = indices
        self.register_buffer("G", vectors.detach().clone())
        self.out_dim = 2 * int(vectors.shape[0])

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate the real trigonometric feature map.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real
            Electron coordinates.

        Returns
        -------
        torch.Tensor, shape (batch, N, out_dim), real
            Concatenation of sines and cosines of ``G · r``.
        """
        phases = r @ self.G.to(r.dtype).T
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class ExtendedComplexPeriodicFeatures(nn.Module):
    """Complex plane-wave features at all ``k = n_1 G_1 + n_2 G_2`` inside shell radius ``N``.

    For each ``(n_1, n_2)`` in the chosen index set, emits one complex column
    ``exp(i k · r)``. The default (``include_conjugates=True``) emits both
    ``+k`` and ``-k``, providing the full spectral basis at the given order;
    this optional feature extension is not selected by the supplied model
    configurations, which use first-harmonic axis features without conjugates.

    Index set
    ---------
    Half-space (canonical, identical to :class:`ExtendedPeriodicFeatures`):

        I_N^{1/2} = {(n_1, n_2) ∈ ℤ² : 1 ≤ n_1² + n_2² ≤ N²,
                                       n_1 > 0  OR  (n_1 == 0 AND n_2 > 0)}

    With ``include_conjugates=True``: extend the half-space with
    ``(-n_1, -n_2)`` for every ``(n_1, n_2) ∈ I_N^{1/2}``, giving the
    full ``I_N = I_N^{1/2} ∪ (-I_N^{1/2})``.

    With ``include_zero=True``: prepend ``(0, 0)`` (the DC term, ``f = 1``).

    Final pairs are sorted by ``(n_1² + n_2², n_1, n_2)`` so a smaller-order
    output is a prefix of a larger-order output.

    Output dimension
    ----------------
    ``out_dim = |index_set|`` (complex columns):

    ===== =================================== ==================================
    order include_conjugates=False (= |I_N^{1/2}|)  include_conjugates=True (= 2·|I_N^{1/2}|)
    ===== =================================== ==================================
    1     2                                   4
    2     6                                   12
    3     14                                  28
    4     24                                  48
    5     40                                  80
    ===== =================================== ==================================

    (``+1`` for either column if ``include_zero=True``.)

    Parameters
    ----------
    cell : Supercell
        Provides reciprocal vectors ``G_1``, ``G_2`` (rows of ``cell.G``).
    order : int, default 1
        Maximum harmonic order ``N`` (must be ``≥ 1``).
    include_conjugates : bool, default True
        Emit both ``+k`` and ``-k`` columns (full reciprocal lattice within
        radius ``N``). When ``False``, emit only the half-space.
    include_zero : bool, default False
        Prepend the DC term ``(0, 0)``.

    Attributes
    ----------
    order : int
    include_conjugates : bool
    include_zero : bool
    index_pairs : list of (int, int)
        The chosen index set in shell-sorted order.
    kvecs : torch.Tensor, shape (out_dim, 2), buffer
        Stacked wave vectors ``k = n_1 G_1 + n_2 G_2``.
    out_dim : int
        Number of complex output columns.
    """

    def __init__(
        self,
        cell: Supercell,
        order: int = 1,
        include_conjugates: bool = True,
        include_zero: bool = False,
    ) -> None:
        """Build the (possibly conjugate-extended) index set, stack wavevectors, register buffer.

        Parameters
        ----------
        cell : Supercell
            Supplies reciprocal vectors.
        order : int, default 1
            Maximum harmonic order ``N``; must be ``≥ 1``.
        include_conjugates : bool, default True
            If ``True``, extend the half-space with ``(-n_1, -n_2)`` partners.
        include_zero : bool, default False
            If ``True``, prepend the DC term ``(0, 0)``.
        """
        super().__init__()
        assert order >= 1, f"order must be >= 1, got {order}"

        self.order = order
        self.include_conjugates = include_conjugates
        self.include_zero = include_zero

        half = self._build_half_space(order)
        pairs: list[tuple[int, int]] = list(half)
        if include_conjugates:
            pairs.extend((-n1, -n2) for (n1, n2) in half)
        if include_zero:
            pairs.append((0, 0))
        pairs.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))
        self.index_pairs = pairs

        # Detach+clone so the buffer is independent of cell.G's autograd /
        # device state, matching ComplexPeriodicFeatures' convention.
        G = cell.G.detach().clone()
        K = torch.stack([n1 * G[0] + n2 * G[1] for n1, n2 in pairs])
        self.register_buffer("kvecs", K)

        self.out_dim = len(pairs)

    @staticmethod
    def _build_half_space(N: int) -> list[tuple[int, int]]:
        """Enumerate the canonical half-space index set ``I_N^{1/2}``.

        Parameters
        ----------
        N : int
            Maximum harmonic order (``N ≥ 1``).

        Returns
        -------
        list of (int, int)
            All ``(n_1, n_2)`` with ``1 ≤ n_1² + n_2² ≤ N²`` satisfying
            ``n_1 > 0`` or ``(n_1 == 0 and n_2 > 0)``, sorted by
            ``(n_1² + n_2², n_1, n_2)``.
        """
        pairs: list[tuple[int, int]] = []
        for n1 in range(0, N + 1):
            for n2 in range(-N, N + 1):
                norm_sq = n1 * n1 + n2 * n2
                if norm_sq < 1 or norm_sq > N * N:
                    continue
                if n1 > 0 or (n1 == 0 and n2 > 0):
                    pairs.append((n1, n2))
        pairs.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))
        return pairs

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate ``exp(i k · r)`` for all ``k`` in the index set.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N_el, 2), real

        Returns
        -------
        torch.Tensor, shape (batch, N_el, out_dim), complex
            Unit-modulus plane-wave features.
        """
        phases = r @ self.kvecs.to(r.dtype).T
        return torch.exp(1j * phases)

    def extra_repr(self) -> str:
        """Human-readable summary used by ``print(model)``."""
        return (
            f"order={self.order}, "
            f"include_conjugates={self.include_conjugates}, "
            f"include_zero={self.include_zero}, "
            f"n_kvecs={len(self.index_pairs)}, "
            f"out_dim={self.out_dim}"
        )
