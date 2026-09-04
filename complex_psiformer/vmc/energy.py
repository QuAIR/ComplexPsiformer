"""Complex local-energy evaluation and optimizer-only clipping."""

import os
from contextlib import contextmanager
import torch
from torch.func import grad, jacrev, vmap
from ..systems.hamiltonian import EwaldCoulomb, MoirePotential


@contextmanager
def _temporarily_disable_parameter_grads(model: torch.nn.Module):
    states: list[tuple[torch.nn.Parameter, bool]] = []
    for param in model.parameters():
        if param.requires_grad:
            states.append((param, True))
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, requires_grad in states:
            param.requires_grad_(requires_grad)

def _env_optional_positive_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    value_norm = value.strip().lower()
    if value_norm in {"0", "none", "false", "off"}:
        return 0
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, 0, or 'none'.")
    return parsed

def _kinetic_jacrev_chunk_size(n_degrees_of_freedom: int) -> int | None:
    env_value = _env_optional_positive_int("PSIFORMER_KINETIC_JACREV_CHUNK_SIZE")
    if env_value == 0:
        return None
    if env_value is not None:
        return env_value
    if n_degrees_of_freedom >= 64:
        return 16
    return None

def _empty_cache_after_local_energy() -> bool:
    value = os.environ.get("PSIFORMER_EMPTY_CACHE_AFTER_LOCAL_ENERGY", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}

def _canonical_peierls_A(peierls_A: tuple[float, float] | list[float] | None) -> tuple[float, float]:
    if peierls_A is None:
        return (0.0, 0.0)
    if len(peierls_A) != 2:
        raise ValueError(f"peierls_A must have length 2, got {peierls_A!r}.")
    return (float(peierls_A[0]), float(peierls_A[1]))

def kinetic_energy_complex(
    model: torch.nn.Module,
    r: torch.Tensor,
    peierls_A: tuple[float, float] | list[float] | None = (0.0, 0.0),
    magnetic_B: float = 0.0,
    magnetic_gauge: str = "symmetric",
    magnetic_origin: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Per-walker complex kinetic local energy ``-1/2 * (∇²Ψ / Ψ)``.

    Both amplitude and phase derivatives are retained. Magnetic and flat
    vector potentials are included when supplied.
    """
    if r.ndim != 3:
        raise ValueError(
            f"kinetic_energy_complex expects r of shape (batch, N, 2); got {tuple(r.shape)}."
        )
    Ax, Ay = _canonical_peierls_A(peierls_A)
    B = float(magnetic_B)
    if magnetic_gauge != "symmetric":
        raise ValueError(
            f"magnetic_gauge must be 'symmetric', got {magnetic_gauge!r}"
        )
    has_peierls = (Ax != 0.0) or (Ay != 0.0)
    has_magnetic = B != 0.0

    def log_abs(rs: torch.Tensor) -> torch.Tensor:
        return model.log_psi(rs.unsqueeze(0))[0][0]

    def arg_psi(rs: torch.Tensor) -> torch.Tensor:
        _, phase = model.log_psi(rs.unsqueeze(0))
        return torch.atan2(phase[0].imag, phase[0].real)

    n_degrees_of_freedom = int(r.shape[1] * r.shape[2])
    jacrev_chunk_size = _kinetic_jacrev_chunk_size(n_degrees_of_freedom)
    use_batch_vmap = jacrev_chunk_size is None

    grad_log_abs = grad(log_abs)
    grad_arg = grad(arg_psi)

    def grad_with_aux(fn_grad):
        def wrapped(rs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            g = fn_grad(rs)
            return g, g

        return wrapped

    hess_and_grad_log_abs = jacrev(
        grad_with_aux(grad_log_abs),
        has_aux=True,
        chunk_size=jacrev_chunk_size,
    )
    hess_and_grad_arg = jacrev(
        grad_with_aux(grad_arg),
        has_aux=True,
        chunk_size=jacrev_chunk_size,
    )

    if has_peierls:
        n_electrons = r.shape[1]
        A_vec = torch.tensor([Ax, Ay], dtype=r.dtype, device=r.device)
        peierls_const = 0.5 * (Ax * Ax + Ay * Ay) * n_electrons
    if has_magnetic:
        origin_vec = torch.tensor(
            [float(magnetic_origin[0]), float(magnetic_origin[1])],
            dtype=r.dtype,
            device=r.device,
        )

    def magnetic_vector_potential(r_single: torch.Tensor) -> torch.Tensor:
        rel = r_single - origin_vec
        x = rel[..., 0]
        y = rel[..., 1]
        return torch.stack((-0.5 * B * y, 0.5 * B * x), dim=-1)

    def per_walker(r_single: torch.Tensor) -> torch.Tensor:
        H_re, g_re = hess_and_grad_log_abs(r_single)
        H_im, g_im = hess_and_grad_arg(r_single)
        lap_re = torch.einsum("iaia->", H_re)
        lap_im = torch.einsum("iaia->", H_im)
        grad_sq_re = (g_re * g_re).sum() - (g_im * g_im).sum()
        grad_sq_im = 2.0 * (g_re * g_im).sum()
        real = -0.5 * (lap_re + grad_sq_re)
        imag = -0.5 * (lap_im + grad_sq_im)
        if has_magnetic:
            A_total = magnetic_vector_potential(r_single)
            if has_peierls:
                A_total = A_total + A_vec
            real = real - (g_im * A_total).sum() + 0.5 * (A_total * A_total).sum()
            imag = imag + (g_re * A_total).sum()
            return torch.complex(real, imag)
        if has_peierls:
            real = real - (g_im * A_vec).sum() + peierls_const
            imag = imag + (g_re * A_vec).sum()
        return torch.complex(real, imag)

    if use_batch_vmap:
        try:
            return vmap(per_walker)(r)
        except (RuntimeError, NotImplementedError):
            if r.device.type == "cuda":
                torch.cuda.empty_cache()
            return torch.stack([per_walker(r[b]) for b in range(r.shape[0])])

    try:
        return torch.stack([per_walker(r[b]) for b in range(r.shape[0])])
    except RuntimeError:
        if r.device.type == "cuda":
            torch.cuda.empty_cache()
        raise

def local_energy_complex(
    model: torch.nn.Module,
    r: torch.Tensor,
    coulomb: EwaldCoulomb,
    potential: MoirePotential | None = None,
    dense_geometry=None,
    peierls_A: tuple[float, float] | list[float] | None = (0.0, 0.0),
    magnetic_B: float = 0.0,
    magnetic_gauge: str = "symmetric",
    magnetic_origin: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Compute the unclipped complex local energy for every configuration."""
    with _temporarily_disable_parameter_grads(model):
        T_loc = kinetic_energy_complex(
            model,
            r,
            peierls_A=peierls_A,
            magnetic_B=magnetic_B,
            magnetic_gauge=magnetic_gauge,
            magnetic_origin=magnetic_origin,
        ).detach()
    if r.device.type == "cuda" and _empty_cache_after_local_energy():
        torch.cuda.empty_cache()

    r_det = r.detach()
    if dense_geometry is not None:
        V_ee = coulomb.electron_electron_from_pair_geometry(dense_geometry.pair_cart.detach())
    else:
        V_ee = coulomb.electron_electron(r_det)

    V_ext = torch.zeros(r_det.shape[0], device=r.device, dtype=V_ee.dtype)
    if potential is not None:
        V_ext = potential(r_det).sum(dim=-1)

    return T_loc + torch.complex((V_ee + V_ext).to(dtype=T_loc.real.dtype), torch.zeros_like(T_loc.real))

def clip_local_energy(E_loc: torch.Tensor, rho: float = 5.0) -> torch.Tensor:
    """Clip outlier local energies to ``median ± ρ · MAD``.

    This changes only the optimization signal. Energy reporting and
    independent sampling must use the unclipped local energies.

    Parameters
    ----------
    E_loc : torch.Tensor, shape (batch,)
        Per-walker local energies.
    rho : float, default 5.0
        Width of the kept window in median-absolute-deviation units.

    Returns
    -------
    torch.Tensor, shape (batch,)
        Clipped local energies.
    """
    median = E_loc.median()
    mad = (E_loc - median).abs().median()
    return torch.clamp(E_loc, median - rho * mad, median + rho * mad)
