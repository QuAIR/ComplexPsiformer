"""Independent coordinate autograd oracles against actual published models.

No torch.func dependency is needed by these tests; the production energy
regression is skipped only when the old local PyTorch lacks torch.func.
"""
import copy
import math
from types import SimpleNamespace

import pytest
import torch

from complex_psiformer.systems.supercell import Supercell
from complex_psiformer.models.complex_psiformer.wavefunction import ComplexPsiFormer
from complex_psiformer.models.real_psiformer import RealPsiFormer
from complex_psiformer.models.magnetic_boundary_adapter import MagneticBoundaryAdapter
from complex_psiformer.exact_compute import log_psi_vgl, kinetic_pair, evaluate_local_energy, source_identity, ExecutionPolicy
from complex_psiformer.exact_compute.primitives import coordinate_seed
from complex_psiformer.exact_compute.real_adapter import _real_attention_vgl
from .test_primitives import _component_derivatives


def make_case(kind, n=2, magnetic=True, *, coefficients=False, normalization="paper"):
    # Initialization deliberately follows the published float32 -> double order.
    assert torch.get_default_dtype() == torch.float32
    with torch.random.fork_rng():
        torch.manual_seed(719 + n)
        cell = Supercell.triangular(4.0, n, (2, 2) if n == 2 else (3, 3))
        kwargs = dict(d_model=16, n_layers=1, n_heads=2, d_attn=3, d_val=5, n_det=2,
                      trainable_determinant_coefficients=coefficients)
        if kind == "complex":
            base = ComplexPsiFormer(cell, **kwargs).double()
            with torch.no_grad():
                base.blocks[0].mlp.activation.r0.copy_(torch.linspace(-0.12, 0.16, 16, dtype=torch.float64))
                base.set_gate_multiplier(0.63)
        else:
            base = RealPsiFormer(cell, **kwargs, attention_normalization=normalization).double()
        field = 2 * math.pi * 2 / cell.area
        origin = 0.5 * cell.L.sum(0)
        model = MagneticBoundaryAdapter(base, cell, field, 2, origin).double() if magnetic else base
        # Fixed interior coordinates, including a translated chart in the last walker.
        j = torch.arange(n, dtype=torch.float64)
        reduced = torch.stack(((0.113 + 0.61803398875*j) % 0.79 + 0.08,
                               (0.167 + 0.41421356237*j) % 0.73 + 0.09), -1)
        r = torch.stack((reduced @ cell.L, (reduced + torch.tensor([1.0, -1.0])) @ cell.L))
        magnetic_args = dict(magnetic_B=field, magnetic_origin=tuple(origin.tolist()), peierls_A=(0.13, -0.09))
    return model, r, magnetic_args


def autograd_log_vgl(model, r):
    results = []
    for walker in r:
        x = walker.detach().clone().requires_grad_(True)
        amplitude, phase = model.log_psi(x.unsqueeze(0))
        angle = torch.atan2(phase[0].imag, phase[0].real)
        ga, la = _component_derivatives(amplitude[0], x)
        gp, lp = _component_derivatives(angle, x)
        results.append((amplitude[0], phase[0], ga, gp, la, lp))
    return tuple(torch.stack(values).detach() for values in zip(*results))


@pytest.mark.parametrize("kind", ["complex", "real"])
@pytest.mark.parametrize("n", [2, 12])
def test_published_magnetic_value_gradient_laplacian(kind, n):
    model, r, magnetic = make_case(kind, n)
    actual = log_psi_vgl(model, r)
    expected = autograd_log_vgl(model, r)
    for name, reference in zip(actual.__dataclass_fields__, expected):
        torch.testing.assert_close(getattr(actual, name), reference, atol=2e-8, rtol=2e-8)
    ga, gp, la, lp = expected[2:]
    from complex_psiformer.exact_compute.api import _vector_potential
    a = _vector_potential(r, **magnetic)
    phase = gp - a
    expected_t = torch.complex(-0.5*(la + ga.square().sum((-2,-1)) - phase.square().sum((-2,-1))),
                              -0.5*lp - (ga*phase).sum((-2,-1)))
    t, q = kinetic_pair(model, r, **magnetic)
    torch.testing.assert_close(t, expected_t, atol=2e-8, rtol=2e-8)
    torch.testing.assert_close(q, 0.5*(ga.square()+phase.square()).sum((-2,-1)), atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("kind", ["complex", "real"])
@pytest.mark.parametrize("coefficients", [False, True])
def test_native_nonmagnetic_and_coefficients(kind, coefficients):
    model, r, _ = make_case(kind, magnetic=False, coefficients=coefficients)
    if coefficients:
        with torch.no_grad():
            model.c.copy_(torch.tensor([0.73, -0.21], dtype=torch.float64))
    actual = log_psi_vgl(model, r)
    expected = autograd_log_vgl(model, r)
    for name, reference in zip(actual.__dataclass_fields__, expected):
        torch.testing.assert_close(getattr(actual, name), reference, atol=2e-8, rtol=2e-8)
    loop = log_psi_vgl(model, r, policy=ExecutionPolicy(matmul_backend="coordinate_loop"))
    for name in actual.__dataclass_fields__:
        torch.testing.assert_close(getattr(actual, name), getattr(loop, name), atol=1e-10, rtol=1e-10)


def test_paper_attention_regression_and_explicit_standard_scaled():
    model, r, _ = make_case("real", magnetic=False)
    attention = model.blocks[0].attn
    # Treat a 2D feature stream as coordinates; Q/K dim differs from V dim.
    with torch.random.fork_rng():
        torch.manual_seed(341)
        h = torch.randn(2, 16, dtype=torch.float64)
    from complex_psiformer.exact_compute.complex_adapter import _transpose, _unsqueeze
    carrier = _unsqueeze(_transpose(coordinate_seed(h.T), 0, 1), 0)
    values = []
    for mode in ("paper", "standard_scaled"):
        attention.attention_normalization = mode
        actual = _real_attention_vgl(attention, carrier)
        from .test_primitives import _autograd_vgl, _assert_vgl_close
        expected = _autograd_vgl(lambda x: attention(x.T.unsqueeze(0)), h.T)
        _assert_vgl_close(actual, expected)
        values.append(actual.value)
    assert not torch.allclose(values[0], values[1], atol=1e-4, rtol=1e-4)


def test_no_state_rng_or_parameter_gradient_mutation_and_chunking():
    model, r, magnetic = make_case("complex")
    r = torch.cat((r, r[:1] + 0.013), 0).requires_grad_(True)
    parameters = [(p, p.requires_grad, p.grad) for p in model.parameters()]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    # Deterministic stand-ins isolate the same additive potential contract.
    runtime = SimpleNamespace(model=model, magnetic_B=magnetic["magnetic_B"],
        magnetic_origin=r.new_tensor(magnetic["magnetic_origin"]),
        coulomb=SimpleNamespace(electron_electron=lambda x: x.square().sum((-2,-1))),
        potential=lambda x: x[..., 0],
        config={"training": {"kinetic_chunk_size": 2}, "system": {
            "peierls_a": magnetic["peierls_A"], "magnetic_gauge": "symmetric"}})
    config = copy.deepcopy(runtime.config)
    actual = evaluate_local_energy(runtime, r)
    whole = evaluate_local_energy(runtime, r, chunk_size=3)
    t, q = kinetic_pair(model, r, **magnetic)
    torch.testing.assert_close(actual, whole, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(actual, t + r.square().sum((-2,-1)) + r[...,0].sum(-1))
    assert not any(t.requires_grad for t in (actual, whole, q))
    assert runtime.config == config
    assert torch.equal(rng, torch.random.get_rng_state())
    assert r.grad is None
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    for p, requires_grad, gradient in parameters:
        assert p.requires_grad == requires_grad and p.grad is gradient
    assert torch.is_grad_enabled()


def test_production_energy_parity():
    pytest.importorskip("torch.func")
    from complex_psiformer.vmc.energy import kinetic_energy_complex, local_energy_complex
    from complex_psiformer.systems.hamiltonian import EwaldCoulomb, MoirePotential
    from complex_psiformer.systems.scaled_ewald import ScaledEwald
    for kind in ("real", "complex"):
        model, r, magnetic = make_case(kind)
        t, q = kinetic_pair(model, r, **magnetic)
        torch.testing.assert_close(t, kinetic_energy_complex(model, r, **magnetic), atol=2e-8, rtol=2e-8)
        _, _, ga, gp, _, _ = autograd_log_vgl(model, r)
        from complex_psiformer.exact_compute.api import _vector_potential
        q_reference = 0.5 * (ga.square() + (gp - _vector_potential(r, **magnetic)).square()).sum((-2, -1))
        torch.testing.assert_close(q, q_reference, atol=2e-8, rtol=2e-8)
        coulomb = ScaledEwald(EwaldCoulomb(model.cell, n_images=1).double(), 0.73).double()
        potential = MoirePotential(model.cell, V0=0.017, phi=0.21).double()
        runtime = SimpleNamespace(model=model, coulomb=coulomb, potential=potential,
            magnetic_B=magnetic["magnetic_B"], magnetic_origin=r.new_tensor(magnetic["magnetic_origin"]),
            config={"training": {"kinetic_chunk_size": 1}, "system": {
                "peierls_a": magnetic["peierls_A"], "magnetic_gauge": "symmetric"}})
        torch.testing.assert_close(evaluate_local_energy(runtime, r),
            local_energy_complex(model, r, coulomb, potential, **magnetic), atol=2e-8, rtol=2e-8)


def test_explicit_unsupported_inputs_and_source_identity():
    model, r, _ = make_case("complex")
    with pytest.raises(TypeError, match="supports published"):
        log_psi_vgl(torch.nn.Linear(2, 2), r)
    with pytest.raises(ValueError, match="nonempty"):
        log_psi_vgl(model, r[:0])
    with pytest.raises(ValueError, match="finite"):
        log_psi_vgl(model, r * float("nan"))
    with pytest.raises(ValueError, match="symmetric"):
        kinetic_pair(model, r, magnetic_gauge="landau")
    model.base_model.blocks[0].attn.heads[0].score_mode = "abs"
    with pytest.raises(ValueError, match="abs_dot"):
        log_psi_vgl(model, r)
    identity = source_identity()
    assert identity["commit"] == "9828f69871ed47baecf07fbc3bfba88969db0cd5"
    assert "api.py" in identity["port_sha256_lf"]
    assert all(len(value) == 64 for value in identity["port_sha256_lf"].values())
