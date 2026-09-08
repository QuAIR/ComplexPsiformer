"""Compute scheduling recorded separately from the scientific configuration."""
from __future__ import annotations


def compute_policy(config, backend="baseline", chunk_size=None):
    if backend not in ("baseline", "forward_vgl"):
        raise ValueError("unsupported compute backend")
    if backend == "baseline":
        if chunk_size is not None:
            raise ValueError("baseline uses the configured kinetic_chunk_size")
        return {"backend": backend, "chunk_size": config["training"]["kinetic_chunk_size"]}
    if config["model"] not in ("complex_psiformer", "real_psiformer"):
        raise ValueError("forward_vgl supports Complex/Real PsiFormer only")
    batch = config["training"]["batch_size"]
    chunk_size = batch if chunk_size is None else chunk_size
    if type(chunk_size) is not int or not 1 <= chunk_size <= batch:
        raise ValueError("compute chunk size must be an integer from 1 through batch_size")
    from .exact_compute import source_identity
    return {"backend": backend, "chunk_size": chunk_size, "source": source_identity()}


def validate_compute_policy(config, policy):
    if not isinstance(policy, dict):
        raise ValueError("missing or invalid compute policy")
    backend = policy.get("backend")
    chunk = policy.get("chunk_size") if backend == "forward_vgl" else None
    if policy != compute_policy(config, backend, chunk):
        raise ValueError("compute policy/source identity mismatch")


def evaluate_energy(runtime, positions, policy):
    if policy["backend"] == "baseline":
        from .runtime import evaluate_local_energy
        return evaluate_local_energy(runtime, positions)
    from .exact_compute import evaluate_local_energy
    return evaluate_local_energy(runtime, positions, chunk_size=policy["chunk_size"])
