"""Build one model and its VMC runtime from an explicit configuration."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .models import ComplexPsiFormer, ComplexSlaterNet, RealPsiFormer, RealSlaterNet
from .models.complex_psiformer.attention import gate_warmup_multiplier
from .models.magnetic_boundary_adapter import MagneticBoundaryAdapter
from .models.slater_boundary_adapter import MagneticBoundaryAdapter as SlaterBoundaryAdapter
from .systems.hamiltonian import EwaldCoulomb, MoirePotential
from .systems.scaled_ewald import ScaledEwald
from .systems.supercell import Supercell
from .systems.units import EffectiveAtomicUnits
from .vmc.energy import local_energy_complex
from .vmc.minsr_optimizer import FullComplexMinSROptimizer
from .vmc.sampler import MetropolisSampler

MODEL_CLASSES = {
    "real_slaternet": RealSlaterNet,
    "complex_slaternet": ComplexSlaterNet,
    "real_psiformer": RealPsiFormer,
    "complex_psiformer": ComplexPsiFormer,
}


def positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def finite_number(value: Any, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))


def validate_config(config: dict[str, Any]) -> None:
    """Validate the resolved configuration, without changing its values."""
    if config.get("schema_version") != 1:
        raise ValueError("unsupported configuration schema")
    if type(config.get("seed")) is not int or config["seed"] != 1:
        raise ValueError("the published configurations use seed 1")
    if config.get("model") not in MODEL_CLASSES:
        raise ValueError("unknown model")
    system, training, optimizer = (config[key] for key in ("system", "training", "optimizer"))
    if system["n_electrons"] != 12:
        raise ValueError("these model parameter fingerprints are for N=12")
    if system["n_cell_sides"] not in ([3, 6], [5, 5]):
        raise ValueError("expected the 18-cell [3,6] or 25-cell [5,5] supercell")
    positive_integer(system["n_phi"], "n_phi", allow_zero=True)
    positive_integer(system["n_images"], "n_images", allow_zero=True)
    for key in ("a_m_nm", "m_star", "epsilon_r"):
        finite_number(system[key], key, positive=True)
    for key in ("v0_mev", "phi", "lambda_c"):
        finite_number(system[key], key)
    if system["lambda_c"] < 0 or system["magnetic_gauge"] != "symmetric":
        raise ValueError("expected nonnegative lambda_c and symmetric gauge")
    if len(system["peierls_a"]) != 2:
        raise ValueError("peierls_a must contain two components")
    for value in system["peierls_a"]:
        finite_number(value, "peierls_a")
    for key in ("max_steps", "batch_size", "sampler_adapt_every", "kinetic_chunk_size",
                "checkpoint_every", "validation_every"):
        positive_integer(training[key], key)
    positive_integer(training["burn_in"], "burn_in", allow_zero=True)
    for key in ("sampler_step_size", "clip_rho", "sampler_target_acceptance"):
        finite_number(training[key], key, positive=True)
    if training["sampler_target_acceptance"] >= 1:
        raise ValueError("sampler_target_acceptance must be between 0 and 1")
    if (training["float_dtype"], training["complex_dtype"]) != ("float64", "complex128"):
        raise ValueError("the numerical contract requires float64/complex128")
    if optimizer["kind"] != "minsr":
        raise ValueError("the published optimizer is MinSR")
    for key in ("eta_0", "damping", "norm_constraint"):
        finite_number(optimizer[key], key, positive=True)
    for key in ("t_0", "score_chunk_size"):
        positive_integer(optimizer[key], key)
    model = config["model_config"]
    positive_integer(model["real_parameter_count"], "real_parameter_count")
    positive_integer(model["gate_warmup_steps"], "gate_warmup_steps", allow_zero=True)
    if "slaternet" in config["model"] and model["kwargs"]["n_det"] != 1:
        raise ValueError("SlaterNet is restricted to a single determinant")


def load_config(path: str | Path, model: str, *, n_phi: int | None = None) -> dict[str, Any]:
    """Resolve a system file and the model catalog into a portable dictionary."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    catalog = json.loads((path.parent / config.pop("model_catalog")).read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1 or model not in catalog["models"]:
        raise ValueError("unknown model or unsupported model catalog")
    allowed = config.pop("allowed_models", list(MODEL_CLASSES))
    if model not in allowed:
        raise ValueError("model is not allowed by this configuration")
    flux_values = config.pop("flux_values", None)
    if flux_values is not None:
        if type(n_phi) is not int or n_phi not in flux_values:
            raise ValueError(f"select one flux using --n-phi from {flux_values}")
        config["system"]["n_phi"] = n_phi
    elif n_phi is not None and n_phi != config["system"]["n_phi"]:
        raise ValueError("use the flux-scan configuration to select a different flux")
    config["model"] = model
    config["model_config"] = catalog["models"][model]
    validate_config(config)
    return config


def count_real_parameters(model: nn.Module) -> int:
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)


@dataclass
class Runtime:
    config: dict[str, Any]
    base_model: nn.Module
    model: nn.Module
    cell: Supercell
    units: EffectiveAtomicUnits
    coulomb: ScaledEwald
    potential: MoirePotential
    magnetic_B: float
    magnetic_origin: torch.Tensor
    optimizer: FullComplexMinSROptimizer
    sampler: MetropolisSampler
    device: torch.device

    def set_gate(self, zero_based_step: int) -> None:
        if self.config["model"] == "complex_psiformer":
            steps = self.config["model_config"]["gate_warmup_steps"]
            self.model.set_gate_multiplier(gate_warmup_multiplier(zero_based_step, steps))


def build_runtime(config: dict[str, Any], device: str | torch.device = "cpu") -> Runtime:
    """Construct using the source initialization order, then convert to double."""
    config = copy.deepcopy(config)
    validate_config(config)
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("supported devices are CPU and CUDA")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    # The source initializes parameters in the default float32 and then converts.
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("keep the default initialization dtype at torch.float32")
    torch.manual_seed(config["seed"])
    if target.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    system = config["system"]
    units = EffectiveAtomicUnits(system["m_star"], system["epsilon_r"])
    cell = Supercell.triangular(
        units.length_nm_to_au(system["a_m_nm"]), system["n_electrons"],
        system["n_cell_sides"],
    )
    ewald = EwaldCoulomb(cell, n_images=system["n_images"]).to(target).double()
    coulomb = ScaledEwald(ewald, system["lambda_c"]).to(target).double()
    potential = MoirePotential(
        cell, V0=units.energy_meV_to_au(system["v0_mev"]), phi=system["phi"],
    ).to(target).double()
    magnetic_B = 2.0 * math.pi * system["n_phi"] / cell.area
    magnetic_origin = (0.5 * (cell.L[0] + cell.L[1])).to(target)
    base_model = MODEL_CLASSES[config["model"]](
        cell, **config["model_config"]["kwargs"],
    ).to(target).double()
    model = base_model
    if system["n_phi"]:
        adapter = SlaterBoundaryAdapter if "slaternet" in config["model"] else MagneticBoundaryAdapter
        model = adapter(base_model, cell, magnetic_B, system["n_phi"], magnetic_origin).to(target).double()
    if count_real_parameters(model) != config["model_config"]["real_parameter_count"]:
        raise ValueError("model parameter count does not match the configuration")
    optimizer_kwargs = {k: v for k, v in config["optimizer"].items() if k != "kind"}
    optimizer = FullComplexMinSROptimizer(model, **optimizer_kwargs)
    sampler = MetropolisSampler(
        model, cell, step_size=config["training"]["sampler_step_size"], device=target,
    )
    return Runtime(config, base_model, model, cell, units, coulomb, potential,
                   magnetic_B, magnetic_origin, optimizer, sampler, target)


def evaluate_local_energy(runtime: Runtime, positions: torch.Tensor) -> torch.Tensor:
    """Unclipped complex local energies, in total effective Hartrees per walker."""
    chunk_size = runtime.config["training"]["kinetic_chunk_size"]
    origin = tuple(float(x.item()) for x in runtime.magnetic_origin)
    values = []
    for start in range(0, len(positions), chunk_size):
        values.append(local_energy_complex(
            runtime.model, positions[start:start + chunk_size],
            runtime.coulomb, runtime.potential,
            peierls_A=tuple(runtime.config["system"]["peierls_a"]),
            magnetic_B=runtime.magnetic_B,
            magnetic_gauge=runtime.config["system"]["magnetic_gauge"],
            magnetic_origin=origin,
        ))
    return torch.cat(values)
