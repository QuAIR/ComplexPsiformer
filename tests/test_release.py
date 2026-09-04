"""Small CPU checks; temporary outputs are not experimental results."""
from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from complex_psiformer.checkpoint import read_checkpoint
from complex_psiformer.features.magnetic import symmetric_gauge_transition_phase
from complex_psiformer.models import ComplexSlaterNet, RealSlaterNet
from complex_psiformer.models.complex_psiformer.attention import ComplexAttentionHead
from complex_psiformer.models.real_psiformer import RealMultiHeadAttention
from complex_psiformer.runtime import (MODEL_CLASSES, build_runtime, count_real_parameters,
                                      load_config)
from complex_psiformer.sample import sample
from complex_psiformer.train import train
from complex_psiformer.vmc.energy import kinetic_energy_complex
from complex_psiformer.vmc.minsr_optimizer import (FullComplexMinSROptimizer,
                                                  build_centered_sample_system)

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def configuration(model: str, system: str = "18cell") -> dict:
    return load_config(ROOT / "configs" / f"{system}.json", model)


def tiny_configuration(model: str) -> dict:
    config = configuration(model)
    config["training"].update(batch_size=2, kinetic_chunk_size=1)
    return config


class ReleaseTests(unittest.TestCase):
    def test_configurations_and_parameter_counts(self) -> None:
        for system in ("18cell", "25cell"):
            for model in MODEL_CLASSES:
                with self.subTest(system=system, model=model):
                    config = configuration(model, system)
                    runtime = build_runtime(config)
                    self.assertEqual(config["seed"], 1)
                    self.assertEqual(count_real_parameters(runtime.model),
                                     config["model_config"]["real_parameter_count"])
                    self.assertTrue(all(p.dtype == torch.float64 for p in runtime.model.parameters()))
        path = ROOT / "configs" / "flux_scan.json"
        with self.assertRaises(ValueError):
            load_config(path, "complex_psiformer")
        for flux in range(5):
            config = load_config(path, "complex_psiformer", n_phi=flux)
            runtime = build_runtime(config)
            log_abs, phase = runtime.model.log_psi(runtime.sampler.initialize(2))
            self.assertTrue(torch.isfinite(log_abs).all() and torch.isfinite(phase).all())
            self.assertAlmostEqual(runtime.magnetic_B * runtime.cell.area, 2 * math.pi * flux)

    def test_single_determinant_and_config_guards(self) -> None:
        runtime = build_runtime(configuration("real_slaternet"))
        for cls in (RealSlaterNet, ComplexSlaterNet):
            for n_det in (0, 2, 4, True):
                with self.subTest(cls=cls.__name__, n_det=n_det), self.assertRaises(ValueError):
                    cls(runtime.cell, n_det=n_det)
            with self.assertRaises(ValueError):
                cls(runtime.cell, trainable_determinant_coefficients=True)
        for key, value in (("batch_size", 0), ("clip_rho", float("nan")),
                           ("float_dtype", "float32")):
            config = configuration("complex_psiformer")
            config["training"][key] = value
            with self.assertRaises(ValueError):
                build_runtime(config)

    def test_antisymmetry_and_magnetic_boundary(self) -> None:
        for model in MODEL_CLASSES:
            with self.subTest(model=model), torch.no_grad():
                runtime = build_runtime(configuration(model))
                r = runtime.sampler.initialize(2)
                log_abs, phase = runtime.model.log_psi(r)
                permuted = r.clone()
                permuted[:, [0, 1]] = permuted[:, [1, 0]]
                permuted_log, permuted_phase = runtime.model.log_psi(permuted)
                torch.testing.assert_close(permuted_log, log_abs, atol=2e-7, rtol=0)
                torch.testing.assert_close(permuted_phase, -phase, atol=2e-7, rtol=0)
                for vector in runtime.cell.L:
                    shifted = r.clone()
                    shifted[:, 0] += vector
                    shifted_log, shifted_phase = runtime.model.log_psi(shifted)
                    expected = symmetric_gauge_transition_phase(
                        r[:, 0], vector, runtime.magnetic_B, runtime.magnetic_origin,
                    )
                    torch.testing.assert_close(shifted_log, log_abs, atol=2e-7, rtol=0)
                    torch.testing.assert_close(shifted_phase, phase * expected, atol=2e-7, rtol=0)

    def test_attention_equations(self) -> None:
        torch.manual_seed(1)
        complex_head = ComplexAttentionHead(8, 3, 4).double()
        h = torch.complex(torch.randn(2, 5, 8, dtype=torch.float64),
                          torch.randn(2, 5, 8, dtype=torch.float64))
        q, k, v = complex_head.Wq(h), complex_head.Wk(h), complex_head.Wv(h)
        score = torch.abs(q.conj() @ k.transpose(-1, -2)) / math.sqrt(3)
        expected = torch.softmax(score, dim=-1).to(torch.complex128) @ v
        torch.testing.assert_close(complex_head(h), expected)
        real_head = RealMultiHeadAttention(8, 2, 3, 4).double()
        x = h.real
        q, k, v = real_head.Wqkv(x).split([6, 6, 8], dim=-1)
        q = q.reshape(2, 5, 2, 3).transpose(1, 2)
        k = k.reshape(2, 5, 2, 3).transpose(1, 2)
        v = v.reshape(2, 5, 2, 4).transpose(1, 2)
        value = torch.softmax(q @ k.transpose(-1, -2), dim=-1) @ v / 2
        expected = real_head.Wo(value.transpose(1, 2).reshape(2, 5, 8))
        torch.testing.assert_close(real_head(x), expected)

    def test_phase_sensitive_kinetic_energy(self) -> None:
        class PlaneWave(nn.Module):
            def log_psi(self, r):
                k = r.new_tensor([0.3, -0.4])
                angle = (r * k).sum((-1, -2))
                return angle * 0, torch.exp(1j * angle)

        r = torch.randn(2, 3, 2, dtype=torch.float64)
        energy = kinetic_energy_complex(PlaneWave(), r, peierls_A=(0.1, 0.2))
        expected = 3 * ((0.3 - 0.1) ** 2 + (-0.4 - 0.2) ** 2) / 2
        torch.testing.assert_close(energy.real, torch.full((2,), expected, dtype=torch.float64))
        torch.testing.assert_close(energy.imag, torch.zeros(2, dtype=torch.float64))

    def test_minsr_full_amplitude_and_phase_update(self) -> None:
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.1, 0.2], dtype=torch.float64))

            def log_psi(self, x):
                return x * self.weight[0], torch.exp(1j * x * self.weight[1])

        model = ToyModel()
        optimizer = FullComplexMinSROptimizer(
            model, eta_0=0.03, t_0=2000, damping=0.03, norm_constraint=5e-4,
        )
        x = torch.tensor([-1, 0.25, 2.0], dtype=torch.float64)
        energy = torch.tensor([1 + 2j, -0.5 + 0.2j, 0.3 - 1j], dtype=torch.complex128)
        amp_scores = torch.stack((x, torch.zeros_like(x)), dim=1)
        phase_scores = amp_scores.flip(1)
        system = build_centered_sample_system(amp_scores, phase_scores, energy)
        metric = system.X.T @ system.X
        delta = -0.06 * torch.linalg.solve(
            metric + 0.03 * torch.eye(2, dtype=torch.float64), system.X.T @ system.y,
        )
        qgt_norm = torch.linalg.vector_norm(system.X @ delta).square().item()
        if qgt_norm > 5e-4:
            delta *= math.sqrt(5e-4 / qgt_norm)
        before = model.weight.detach().clone()
        optimizer.step(energy, *model.log_psi(x))
        torch.testing.assert_close(model.weight, before + delta)
        self.assertEqual(optimizer.t, 1)

    def test_training_checkpoint_and_frozen_sampling(self) -> None:
        with tempfile.TemporaryDirectory(prefix="complex_psiformer_test_") as temporary:
            root = Path(temporary)
            for model in MODEL_CLASSES:
                with self.subTest(model=model):
                    config = tiny_configuration(model)
                    checkpoint = train(config, root / model, stop_at=1)
                    payload = read_checkpoint(checkpoint)
                    self.assertEqual(payload["completed_step"], 1)
                    self.assertEqual(payload["optimizer_state"], {"t": 1})
                    self.assertEqual(payload["config"], config)
                    with self.assertRaises(FileExistsError):
                        train(config, root / model, stop_at=1)
            checkpoint = root / "complex_psiformer" / "checkpoint_step000001.pt"
            result = sample(checkpoint, root / "sampling", burn_in=1, sweeps=2, thin=1)
            data = torch.load(result, weights_only=True)
            self.assertEqual(data["energy_raw_complex"].shape, (2, 2))
            self.assertTrue(torch.isfinite(data["energy_raw_complex"]).all())
            self.assertEqual(read_checkpoint(checkpoint)["completed_step"], 1)
            damaged = read_checkpoint(checkpoint)
            damaged["source_sha256"] = "0" * 64
            broken_path = root / "mismatch.pt"
            torch.save(damaged, broken_path)
            with self.assertRaises(ValueError):
                read_checkpoint(broken_path)

    def test_exact_cpu_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="complex_psiformer_resume_") as temporary:
            root = Path(temporary)
            config = tiny_configuration("complex_psiformer")
            # Exercise adaptation and warmup-state restoration inside the same runner.
            config["training"]["sampler_adapt_every"] = 1
            complete = read_checkpoint(train(config, root / "complete", stop_at=2))
            first = train(config, root / "first", stop_at=1)
            resumed = read_checkpoint(train(config, root / "resumed", stop_at=2, resume=first))
            for name, value in complete["model_state"].items():
                self.assertTrue(torch.equal(value, resumed["model_state"][name]), name)
            for key in ("walkers", "cached_log", "torch_rng_state"):
                self.assertTrue(torch.equal(complete[key], resumed[key]), key)
            self.assertEqual(complete["optimizer_state"], resumed["optimizer_state"])
            self.assertEqual(complete["sampler_state"], resumed["sampler_state"])
            for expected, actual in zip(complete["history"], resumed["history"]):
                self.assertEqual({k: v for k, v in expected.items() if k != "elapsed_seconds"},
                                 {k: v for k, v in actual.items() if k != "elapsed_seconds"})
            changed = copy.deepcopy(config)
            changed["optimizer"]["eta_0"] *= 2
            with self.assertRaises(ValueError):
                train(changed, root / "invalid", stop_at=2, resume=first)


if __name__ == "__main__":
    unittest.main()
