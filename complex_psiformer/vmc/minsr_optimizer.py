"""Full-parameter, full-complex sample-space MinSR optimizer."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

PHASE_UNIT_TOLERANCE = 1.0e-6


@dataclass(slots=True)
class CenteredSampleSystem:
    """Centered real sample-space design matrix and complex-energy target."""

    X: torch.Tensor
    y: torch.Tensor
    energy_centered: torch.Tensor
    diagnostics: dict[str, float]


def build_centered_sample_system(
    amplitude_scores: torch.Tensor,
    phase_scores: torch.Tensor,
    E_loc_complex: torch.Tensor,
    *,
    destination: torch.Tensor | None = None,
) -> CenteredSampleSystem:
    """Center scores and complex energy under the full-complex MinSR convention."""
    if amplitude_scores.ndim != 2 or phase_scores.shape != amplitude_scores.shape:
        raise ValueError("amplitude_scores and phase_scores must have matching shape (batch, parameters)")
    batch, parameter_count = amplitude_scores.shape
    if batch == 0:
        raise ValueError("sample system must contain at least one sample")
    if not torch.is_complex(E_loc_complex) or E_loc_complex.shape != (batch,):
        raise ValueError("E_loc_complex must be complex with shape (batch,)")
    if phase_scores.device != amplitude_scores.device:
        raise ValueError("amplitude and phase scores must be on the same device")

    expected_shape = (2 * batch, parameter_count)
    if destination is None:
        X = torch.empty(expected_shape, dtype=torch.float64, device=amplitude_scores.device)
    else:
        if destination.shape != expected_shape or destination.dtype != torch.float64:
            raise ValueError("destination must be float64 with shape (2 * batch, parameters)")
        X = destination
    X[:batch].copy_(amplitude_scores.to(device=X.device, dtype=X.dtype))
    X[batch:].copy_(phase_scores.to(device=X.device, dtype=X.dtype))
    X[:batch].sub_(X[:batch].mean(dim=0, keepdim=True))
    X[batch:].sub_(X[batch:].mean(dim=0, keepdim=True))
    X.div_(math.sqrt(batch))

    energy = E_loc_complex.detach().to(device=X.device, dtype=torch.complex128)
    energy_relative = energy - energy[0]
    energy_centered = energy_relative - energy_relative.mean()
    y = torch.cat((energy_centered.real, energy_centered.imag), dim=0) / math.sqrt(batch)
    diagnostics = {
        "energy_center_residual_abs": float(torch.abs(energy_centered.mean()).item()),
    }
    return CenteredSampleSystem(X=X, y=y, energy_centered=energy_centered, diagnostics=diagnostics)


class FullComplexMinSROptimizer:
    """Apply all-parameter MinSR with unit-phase tolerance ``1e-6``."""

    def __init__(
        self,
        model: nn.Module,
        eta_0: float = 10.0,
        t_0: int = 100_000,
        damping: float = 1.0e-3,
        norm_constraint: float = 1.0e-3,
        score_chunk_size: int = 16,
    ) -> None:
        if t_0 <= 0:
            raise ValueError("t_0 must be positive")
        if damping <= 0.0:
            raise ValueError("damping must be positive")
        if norm_constraint < 0.0:
            raise ValueError("norm_constraint must be non-negative")
        if score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")

        named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not named_params:
            raise ValueError("model has no trainable parameters")
        if any(torch.is_complex(param) for _, param in named_params):
            raise TypeError("FullComplexMinSROptimizer requires real trainable parameters")
        devices = {param.device for _, param in named_params}
        if len(devices) != 1:
            raise ValueError("all trainable parameters must be on one device")

        self.model = model
        self.eta_0 = float(eta_0)
        self.t_0 = int(t_0)
        self.damping = float(damping)
        self.norm_constraint = float(norm_constraint)
        self.score_chunk_size = int(score_chunk_size)
        self.t = 0
        self.last_step_profile: dict[str, Any] = {}
        self._param_names = [name for name, _ in named_params]
        self._params = [param for _, param in named_params]
        self._parameter_count = sum(param.numel() for param in self._params)

    def learning_rate(self) -> float:
        """Return ``eta_0 / (1 + t / t_0)`` for the next update."""
        return self.eta_0 / (1.0 + self.t / self.t_0)

    def _write_chunked_jacobian(
        self,
        outputs: torch.Tensor,
        destination: torch.Tensor,
        *,
        retain_after: bool,
    ) -> dict[str, int]:
        batch = int(outputs.shape[0])
        stats = {
            "calls": 0,
            "max_chunk_size": 0,
            "max_seed_bytes": 0,
            "max_flattened_bytes": 0,
            "max_explicit_live_bytes": 0,
        }
        if not outputs.requires_grad:
            destination.zero_()
            return stats

        for start in range(0, batch, self.score_chunk_size):
            stop = min(start + self.score_chunk_size, batch)
            selected = outputs[start:stop]
            chunk_size = stop - start
            grad_outputs = torch.eye(chunk_size, dtype=selected.dtype, device=selected.device)
            keep_graph = retain_after or stop < batch
            grads = torch.autograd.grad(
                selected,
                self._params,
                grad_outputs=grad_outputs,
                retain_graph=keep_graph,
                allow_unused=True,
                is_grads_batched=True,
            )
            pieces: list[torch.Tensor] = []
            for grad, param in zip(grads, self._params):
                if grad is None:
                    pieces.append(
                        torch.zeros(chunk_size, param.numel(), dtype=torch.float64, device=outputs.device)
                    )
                else:
                    pieces.append(grad.reshape(chunk_size, -1).to(dtype=torch.float64))
            flat_grads = torch.cat(pieces, dim=1)
            destination[start:stop].copy_(flat_grads)
            live_storages: dict[tuple[torch.device, int], int] = {}
            for tensor in (*[grad for grad in grads if grad is not None], *pieces, flat_grads):
                storage = tensor.untyped_storage()
                live_storages[(tensor.device, storage.data_ptr())] = storage.nbytes()
            stats["calls"] += 1
            stats["max_chunk_size"] = max(stats["max_chunk_size"], chunk_size)
            stats["max_seed_bytes"] = max(stats["max_seed_bytes"], grad_outputs.numel() * grad_outputs.element_size())
            stats["max_flattened_bytes"] = max(
                stats["max_flattened_bytes"],
                flat_grads.numel() * flat_grads.element_size(),
            )
            stats["max_explicit_live_bytes"] = max(
                stats["max_explicit_live_bytes"],
                sum(live_storages.values()),
            )
        return stats

    @staticmethod
    def _validate_step_inputs(
        E_loc_complex: torch.Tensor,
        log_abs: torch.Tensor,
        phase: torch.Tensor,
    ) -> int:
        if not isinstance(E_loc_complex, torch.Tensor) or not torch.is_complex(E_loc_complex):
            raise TypeError("E_loc_complex must be a complex tensor")
        if not isinstance(log_abs, torch.Tensor) or torch.is_complex(log_abs):
            raise TypeError("log_abs must be a real tensor")
        if not isinstance(phase, torch.Tensor) or not torch.is_complex(phase):
            raise TypeError("phase must be a complex unit-phase tensor")
        if E_loc_complex.ndim != 1:
            raise ValueError("E_loc_complex must have shape (batch,)")
        if log_abs.ndim != 1:
            raise ValueError("log_abs must have shape (batch,)")
        if phase.ndim != 1:
            raise ValueError("phase must have shape (batch,)")
        if log_abs.shape != E_loc_complex.shape or phase.shape != E_loc_complex.shape:
            raise ValueError("E_loc_complex, log_abs, and phase must have the same shape")
        if E_loc_complex.numel() == 0:
            raise ValueError("step inputs must contain at least one sample")
        if not torch.isfinite(phase).all().item():
            raise ValueError("phase must contain only finite values")
        phase_modulus = torch.abs(phase)
        if torch.any(phase_modulus == 0.0).item():
            raise ValueError("phase must be nonzero for every sample")
        max_deviation = float(torch.max(torch.abs(phase_modulus - 1.0)).item())
        if max_deviation > PHASE_UNIT_TOLERANCE:
            raise ValueError(
                "phase must have unit modulus within tolerance "
                f"{PHASE_UNIT_TOLERANCE:g}; maximum deviation was {max_deviation:.6g}"
            )
        return int(E_loc_complex.shape[0])

    def step(
        self,
        E_loc_complex: torch.Tensor,
        log_abs: torch.Tensor,
        phase: torch.Tensor,
    ) -> float:
        """Apply an all-parameter MinSR update and return mean real energy."""
        total_start = time.perf_counter()
        batch = self._validate_step_inputs(E_loc_complex, log_abs, phase)
        device = self._params[0].device
        if log_abs.device != device or phase.device != device:
            raise ValueError("model outputs and trainable parameters must be on the same device")

        phase_angle = torch.atan2(phase.imag, phase.real)
        X = torch.empty(2 * batch, self._parameter_count, dtype=torch.float64, device=device)
        score_start = time.perf_counter()
        amplitude_vjp = self._write_chunked_jacobian(
            log_abs,
            X[:batch],
            retain_after=phase_angle.requires_grad,
        )
        phase_vjp = self._write_chunked_jacobian(phase_angle, X[batch:], retain_after=False)
        score_seconds = time.perf_counter() - score_start

        energy = E_loc_complex.detach().to(device=device, dtype=torch.complex128)
        system = build_centered_sample_system(X[:batch], X[batch:], energy, destination=X)
        X = system.X
        y = system.y
        amplitude_force = 2.0 * X[:batch].T @ y[:batch]
        phase_force = 2.0 * X[batch:].T @ y[batch:]

        learning_rate = self.learning_rate()
        solve_start = time.perf_counter()
        kernel = X @ X.T
        regularized_kernel = kernel + self.damping * torch.eye(
            2 * batch,
            dtype=kernel.dtype,
            device=kernel.device,
        )
        alpha = torch.linalg.solve(regularized_kernel, y)
        raw_delta = -2.0 * learning_rate * X.T @ alpha
        raw_qgt_norm = torch.linalg.vector_norm(X @ raw_delta).square()
        trust_scale = 1.0
        if raw_qgt_norm.item() > self.norm_constraint:
            trust_scale = math.sqrt(self.norm_constraint / raw_qgt_norm.item())
        delta = raw_delta * trust_scale
        applied_qgt_norm = torch.linalg.vector_norm(X @ delta).square()
        solve_seconds = time.perf_counter() - solve_start

        apply_start = time.perf_counter()
        with torch.no_grad():
            offset = 0
            for param in self._params:
                next_offset = offset + param.numel()
                update = delta[offset:next_offset].reshape_as(param).to(dtype=param.dtype)
                param.add_(update)
                offset = next_offset
        apply_seconds = time.perf_counter() - apply_start

        self.t += 1
        total_seconds = time.perf_counter() - total_start
        timing = {
            "score_seconds": float(score_seconds),
            "solve_seconds": float(solve_seconds),
            "apply_seconds": float(apply_seconds),
            "total_seconds": float(total_seconds),
        }
        self.last_step_profile = {
            "metric_mode": "full_complex_sample_space_minsr",
            "batch_size": batch,
            "sample_kernel_dimension": 2 * batch,
            "covered_tensor_count": len(self._params),
            "covered_parameter_count": self._parameter_count,
            "covered_params": self._parameter_count,
            "parameter_names": list(self._param_names),
            "fallback_params": 0,
            "fallback_parameter_count": 0,
            "configured_score_chunk_size": self.score_chunk_size,
            "amplitude_vjp_calls": amplitude_vjp["calls"],
            "phase_vjp_calls": phase_vjp["calls"],
            "max_vjp_chunk_size": max(amplitude_vjp["max_chunk_size"], phase_vjp["max_chunk_size"]),
            "max_vjp_seed_bytes": max(amplitude_vjp["max_seed_bytes"], phase_vjp["max_seed_bytes"]),
            "max_flattened_vjp_bytes": max(
                amplitude_vjp["max_flattened_bytes"], phase_vjp["max_flattened_bytes"]
            ),
            "max_explicit_live_vjp_bytes": max(
                amplitude_vjp["max_explicit_live_bytes"], phase_vjp["max_explicit_live_bytes"]
            ),
            "score_storage_bytes": X.numel() * X.element_size(),
            "projected_score_storage_bytes_batch_384": 2 * 384 * self._parameter_count * X.element_size(),
            "phase_unit_tolerance": PHASE_UNIT_TOLERANCE,
            **system.diagnostics,
            "damping": self.damping,
            "learning_rate": learning_rate,
            "raw_qgt_norm": float(raw_qgt_norm.item()),
            "trust_scale": trust_scale,
            "applied_qgt_norm": float(applied_qgt_norm.item()),
            "update_norm": float(torch.linalg.vector_norm(delta).item()),
            "amplitude_force_norm": float(torch.linalg.vector_norm(amplitude_force).item()),
            "phase_force_norm": float(torch.linalg.vector_norm(phase_force).item()),
            "timing": timing,
        }
        return float(energy.real.mean().item())

    def state_dict(self) -> dict[str, int]:
        """Return checkpointable optimizer state."""
        return {"t": self.t}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore checkpointable optimizer state."""
        if "t" not in state_dict:
            raise KeyError("optimizer state is missing 't'")
        t = int(state_dict["t"])
        if t < 0:
            raise ValueError("optimizer step counter must be non-negative")
        self.t = t
