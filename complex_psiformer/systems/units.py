"""Unit conversions between physical and effective-atomic-unit systems.

All VMC code (wavefunction, sampler, energy, Ewald, moiré potential)
operates in **effective atomic units**:

* length in ``a_B* = a_B · ε_r / m*``,
* energy in ``Ha* = Ha · m* / ε_r²``,

where ``m*`` is the band effective mass (in units of ``m_e``) and
``ε_r`` is the relative dielectric constant of the host. The kinetic
operator at zero vector potential is ``T = −½ ∇²`` and the Coulomb
interaction is the plain ``1/|r|``, so no factors of ``m*`` or ``ε_r``
appear anywhere downstream.

Configuration files carry values in **physical units**
(``a_M`` in nm, ``V0`` in meV). Conversion occurs when building the
runtime; downstream wavefunction and energy routines use effective units.
"""

from dataclasses import dataclass
from typing import Any

# CODATA 2018 values, fixed once per project so tests are deterministic.
BOHR_RADIUS_NM: float = 0.0529177210903  # a_0 in nm
HARTREE_MEV: float = 27_211.386245988  # 1 Hartree in meV

DEFAULT_M_STAR: float = 0.35  # WSe2/WS2 conduction band (paper §III)
DEFAULT_EPSILON_R: float = 5.0  # 2D-averaged dielectric (hBN-encapsulated TMD)


@dataclass(frozen=True)
class EffectiveAtomicUnits:
    """Scale factors relating physical units to effective atomic units.

    Parameters
    ----------
    m_star : float
        Band effective mass in units of the free-electron mass ``m_e``.
    epsilon_r : float
        Relative dielectric constant (static, 2D-averaged).

    Notes
    -----
    Derived quantities (available as properties):

    * ``a_B_star_nm``: one effective Bohr radius in nanometres.
    * ``Ha_star_meV``: one effective Hartree in meV.

    For WSe2/WS2 heterobilayers at ``m*=0.35``, ``ε_r=5.0`` this gives
    ``a_B* ≈ 0.756 nm`` and ``Ha* ≈ 381 meV``.
    """

    m_star: float
    epsilon_r: float

    def __post_init__(self) -> None:
        if self.m_star <= 0.0:
            raise ValueError(f"m_star must be positive, got {self.m_star}.")
        if self.epsilon_r <= 0.0:
            raise ValueError(
                f"epsilon_r must be positive, got {self.epsilon_r}."
            )

    @property
    def a_B_star_nm(self) -> float:
        """Effective Bohr radius ``a_B · ε_r / m*`` in nanometres."""
        return BOHR_RADIUS_NM * self.epsilon_r / self.m_star

    @property
    def Ha_star_meV(self) -> float:
        """Effective Hartree ``Ha · m* / ε_r²`` in meV."""
        return HARTREE_MEV * self.m_star / (self.epsilon_r ** 2)

    def length_nm_to_au(self, x_nm: float) -> float:
        """Convert a length from nm to ``a_B*`` (dimensionless in code)."""
        return x_nm / self.a_B_star_nm

    def length_au_to_nm(self, x_au: float) -> float:
        """Convert a length from ``a_B*`` back to nm."""
        return x_au * self.a_B_star_nm

    def energy_meV_to_au(self, x_meV: float) -> float:
        """Convert an energy from meV to ``Ha*`` (dimensionless in code)."""
        return x_meV / self.Ha_star_meV

    def energy_au_to_meV(self, x_au: float) -> float:
        """Convert an energy from ``Ha*`` back to meV."""
        return x_au * self.Ha_star_meV


def units_from_config(cfg: dict[str, Any]) -> EffectiveAtomicUnits:
    """Build an :class:`EffectiveAtomicUnits` from a config dict.

    The config may contain a top-level ``units`` block with ``m_star``
    and ``epsilon_r``. Missing fields fall back to
    :data:`DEFAULT_M_STAR` / :data:`DEFAULT_EPSILON_R` (WSe2/WS2).
    """
    units_cfg = cfg.get("units", {}) or {}
    return EffectiveAtomicUnits(
        m_star=float(units_cfg.get("m_star", DEFAULT_M_STAR)),
        epsilon_r=float(units_cfg.get("epsilon_r", DEFAULT_EPSILON_R)),
    )


def convert_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], EffectiveAtomicUnits]:
    """Rescale physical-unit fields in ``cfg`` to effective atomic units.

    Applied conversions
    -------------------
    * ``a_M``  : nm  -> ``a_B*``
    * ``V0``   : meV -> ``Ha*``

    All other fields (``phi``, ``n_electrons``, ``batch_size``,
    optimiser hyperparameters, …) are dimensionless or unit-agnostic
    and are passed through untouched.

    Parameters
    ----------
    cfg : dict
        Raw config as loaded from YAML.

    Returns
    -------
    tuple[dict, EffectiveAtomicUnits]
        A shallow copy of ``cfg`` with ``a_M`` and ``V0`` replaced by
        their au-space values, plus the :class:`EffectiveAtomicUnits`
        instance used for the conversion (handy for pretty-printing
        final energies back in meV).
    """
    units = units_from_config(cfg)
    out = dict(cfg)
    if "a_M" in cfg:
        out["a_M"] = units.length_nm_to_au(float(cfg["a_M"]))
    if "V0" in cfg:
        out["V0"] = units.energy_meV_to_au(float(cfg["V0"]))
    return out, units
