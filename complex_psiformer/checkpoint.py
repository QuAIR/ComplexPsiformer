"""Portable checkpoints for this release's runner (not legacy bundle formats)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import torch

from .runtime import Runtime, validate_config
from .compute import compute_policy, validate_compute_policy


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_hash() -> str:
    """Fingerprint installed Python source; normalize line endings for portability."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def environment_contract(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if cuda else platform.machine(),
        "cuda": torch.version.cuda if cuda else None,
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "kinetic_jacrev_chunk_size": os.environ.get("PSIFORMER_KINETIC_JACREV_CHUNK_SIZE", ""),
    }


def assert_finite(value: Any, path: str = "state") -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"nonfinite tensor in {path}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"nonfinite scalar in {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    return value


def atomic_save(path: Path, value: Any, *, tensor_file: bool = False) -> None:
    """Write atomically within a run-owned directory."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        if tensor_file:
            torch.save(value, temporary)
        else:
            Path(temporary).write_text(
                json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(runtime: Runtime, path: Path, walkers: torch.Tensor,
                    cached_log: torch.Tensor, history: list[dict[str, Any]],
                    *, compute: dict[str, Any] | None = None) -> None:
    compute = compute_policy(runtime.config) if compute is None else compute
    validate_compute_policy(runtime.config, compute)
    payload = {
        "schema_version": 1,
        "compute_policy": compute,
        "kind": "complex_psiformer_training",
        "source_sha256": source_hash(),
        "config": runtime.config,
        "config_sha256": canonical_hash(runtime.config),
        "environment": environment_contract(runtime.device),
        "completed_step": runtime.optimizer.t,
        "model_state": _cpu_tree(runtime.base_model.state_dict()),
        "optimizer_state": runtime.optimizer.state_dict(),
        "sampler_state": {
            "step_size": runtime.sampler.step_size,
            "acceptance_count": runtime.sampler._acceptance_count,
            "proposal_count": runtime.sampler._proposal_count,
        },
        "walkers": _cpu_tree(walkers),
        "cached_log": _cpu_tree(cached_log),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(runtime.device) if runtime.device.type == "cuda" else None,
        "history": history,
    }
    assert_finite(payload)
    atomic_save(path, payload, tensor_file=True)


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("kind") != "complex_psiformer_training":
        raise ValueError("unsupported checkpoint format")
    if payload["source_sha256"] != source_hash():
        raise ValueError("checkpoint source hash differs from the installed implementation")
    validate_config(payload["config"])
    validate_compute_policy(payload["config"], payload.get("compute_policy"))
    if payload["config_sha256"] != canonical_hash(payload["config"]):
        raise ValueError("checkpoint configuration hash mismatch")
    step = payload["completed_step"]
    history = payload["history"]
    if type(step) is not int or not 0 < step <= payload["config"]["training"]["max_steps"]:
        raise ValueError("invalid completed_step")
    if payload["optimizer_state"] != {"t": step}:
        raise ValueError("optimizer step mismatch")
    if len(history) != step or any(row["step"] != i + 1 for i, row in enumerate(history)):
        raise ValueError("incomplete or inconsistent history")
    if any(row["optimizer_step"] != row["step"] for row in history):
        raise ValueError("history optimizer-step mismatch")
    batch = payload["config"]["training"]["batch_size"]
    n_el = payload["config"]["system"]["n_electrons"]
    if payload["walkers"].shape != (batch, n_el, 2) or payload["walkers"].dtype != torch.float64:
        raise ValueError("walker shape or dtype mismatch")
    if payload["cached_log"].shape != (batch,) or payload["cached_log"].dtype != torch.float64:
        raise ValueError("cached log-amplitude shape or dtype mismatch")
    sampler = payload["sampler_state"]
    if sampler["step_size"] <= 0 or not 0 <= sampler["acceptance_count"] <= sampler["proposal_count"]:
        raise ValueError("invalid sampler state")
    assert_finite(payload)
    return payload


def restore_training(runtime: Runtime, payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if payload["config"] != runtime.config:
        raise ValueError("resume configuration differs from the checkpoint")
    if payload["environment"] != environment_contract(runtime.device):
        raise ValueError("exact resume requires the same recorded execution environment")
    runtime.base_model.load_state_dict(payload["model_state"], strict=True)
    runtime.optimizer.load_state_dict(payload["optimizer_state"])
    state = payload["sampler_state"]
    runtime.sampler.step_size = state["step_size"]
    runtime.sampler._acceptance_count = state["acceptance_count"]
    runtime.sampler._proposal_count = state["proposal_count"]
    torch.set_rng_state(payload["torch_rng_state"])
    if runtime.device.type == "cuda":
        torch.cuda.set_rng_state(payload["cuda_rng_state"], runtime.device)
    return payload["walkers"].to(runtime.device), payload["cached_log"].to(runtime.device)
