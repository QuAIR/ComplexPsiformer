"""Detached exact coordinate derivatives for the published wavefunctions.

This module never changes a runtime, parameter, buffer or RNG. No autograd or
stochastic trace fallback is used. See docs/acceleration.md for smooth-domain limitations.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from complex_psiformer.models.complex_psiformer.wavefunction import ComplexPsiFormer
from complex_psiformer.models.real_psiformer import RealPsiFormer
from complex_psiformer.models.magnetic_boundary_adapter import MagneticBoundaryAdapter
from complex_psiformer.features.magnetic import TorusMagneticSections
from .complex_adapter import (
    ComplexLogPsiVGL, _validate_complex_support, _validate_coordinates,
    _complex_orbitals_from_seed, _magnetic_sections_vgl, _unsqueeze,
    _logpsi_from_orbitals,
)
from .real_adapter import _validate_real_support, _real_orbitals_from_seed
from .primitives import coordinate_seed, product
from .policy import DEFAULT_POLICY, ExecutionPolicy


def _unwrap(model):
    sections = None
    if type(model) is MagneticBoundaryAdapter:
        sections, base = model.sections, model.base_model
        if type(sections) is not TorusMagneticSections:
            raise TypeError("unsupported external magnetic sections")
        if getattr(base, "magnetic_sections", None) is not None or getattr(base, "enforce_magnetic_boundary", False):
            raise ValueError("external and internal magnetic sections cannot both be applied")
        if model.cell.n_electrons != base.n_electrons:
            raise ValueError("adapter and native electron counts differ")
    else:
        base = model
    if type(base) is ComplexPsiFormer:
        _validate_complex_support(base)
        compute = _complex_orbitals_from_seed
    elif type(base) is RealPsiFormer:
        _validate_real_support(base)
        compute = _real_orbitals_from_seed
    else:
        raise TypeError("exact compute supports published ComplexPsiFormer/RealPsiFormer and MagneticBoundaryAdapter only")
    if base.c is not None and (base.c.ndim != 1 or base.c.shape[0] != base.n_det):
        raise ValueError("determinant coefficients must have shape (n_det,)")
    return base, sections, compute


@torch.no_grad()
def log_psi_vgl(model, positions, *, policy: ExecutionPolicy = DEFAULT_POLICY) -> ComplexLogPsiVGL:
    """Exact log amplitude/phase VGL with detached outputs, for [batch,N,2]."""
    if not isinstance(policy, ExecutionPolicy):
        raise TypeError("policy must be an ExecutionPolicy")
    base, sections, compute = _unwrap(model)
    _validate_coordinates(base, positions)
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("coordinates must use float32 or float64")
    if not bool(torch.isfinite(positions).all().item()):
        raise ValueError("coordinates must be finite")
    if any(p.device != positions.device or p.dtype != positions.dtype for p in base.parameters()):
        raise ValueError("model parameters and coordinates must share device and real precision")
    coordinates = coordinate_seed(positions.detach())
    orbitals = compute(base, coordinates, matmul_backend=policy.matmul_backend)
    if sections is not None:
        orbitals = product(orbitals, _unsqueeze(_magnetic_sections_vgl(sections, coordinates), 1))
    return _logpsi_from_orbitals(orbitals, base.c, slogdet_backend=policy.slogdet_backend)


def _vector_potential(positions, *, peierls_A=(0.0, 0.0), magnetic_B=0.0,
                      magnetic_gauge="symmetric", magnetic_origin=(0.0, 0.0)):
    if magnetic_gauge != "symmetric":
        raise ValueError("only symmetric gauge is supported")
    if peierls_A is None:
        peierls_A = (0.0, 0.0)
    if len(peierls_A) != 2 or len(magnetic_origin) != 2:
        raise ValueError("vector potential and origin require two components")
    if any(not math.isfinite(float(v)) for v in (*peierls_A, *magnetic_origin, magnetic_B)):
        raise ValueError("magnetic inputs must be finite")
    relative = positions - positions.new_tensor(magnetic_origin)
    return 0.5 * float(magnetic_B) * torch.stack((-relative[..., 1], relative[..., 0]), -1) + positions.new_tensor(peierls_A)


@torch.no_grad()
def kinetic_pair(model, positions, **magnetic):
    """Return complex Laplacian and real quadratic kinetic integrands.

    Both use the same propagated gradients. Equality of their expectations
    requires domain/boundary assumptions that this implementation cannot prove.
    """
    potential = _vector_potential(positions, **magnetic)
    vgl = log_psi_vgl(model, positions)
    amplitude, covariant_phase = vgl.grad_log_abs, vgl.grad_phase_angle - potential
    amplitude_sq = amplitude.square().sum((-2, -1))
    phase_sq = covariant_phase.square().sum((-2, -1))
    real = -0.5 * (vgl.lap_log_abs + amplitude_sq - phase_sq)
    imag = -0.5 * vgl.lap_phase_angle - (amplitude * covariant_phase).sum((-2, -1))
    return torch.complex(real, imag), 0.5 * (amplitude_sq + phase_sq)


@torch.no_grad()
def evaluate_local_energy(runtime, positions, *, chunk_size=None):
    """Detached, raw complex total local energy in effective Hartrees."""
    if chunk_size is None:
        chunk_size = runtime.config["training"]["kinetic_chunk_size"]
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if positions.ndim != 3 or positions.shape[0] == 0:
        raise ValueError("positions must be a nonempty [batch,N,2] tensor")
    origin = tuple(float(x.item()) for x in runtime.magnetic_origin)
    magnetic = dict(peierls_A=tuple(runtime.config["system"]["peierls_a"]),
                    magnetic_B=runtime.magnetic_B,
                    magnetic_gauge=runtime.config["system"]["magnetic_gauge"],
                    magnetic_origin=origin)
    values = []
    for chunk in positions.detach().split(chunk_size):
        kinetic, _ = kinetic_pair(runtime.model, chunk, **magnetic)
        potential = runtime.coulomb.electron_electron(chunk)
        if runtime.potential is not None:
            potential = potential + runtime.potential(chunk).sum(-1)
        values.append(kinetic + potential.to(kinetic.real.dtype))
    return torch.cat(values).detach()


def source_identity():
    """Pinned original identity and SHA256 of UTF-8 LF-normalized port files."""
    directory = Path(__file__).resolve().parent
    identity = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    identity["port_sha256_lf"] = {
        path.name: hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()
        for path in sorted(directory.glob("*.py"))
    }
    return identity
