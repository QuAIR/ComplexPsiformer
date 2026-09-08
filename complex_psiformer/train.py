"""Train one configured wavefunction with full amplitude-and-phase MinSR."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (assert_finite, atomic_save, canonical_hash, read_checkpoint,
                         restore_training, save_checkpoint, source_hash)
from .runtime import (MODEL_CLASSES, build_runtime, load_config,
                      positive_integer, validate_config)
from .vmc.energy import clip_local_energy
from .compute import compute_policy, evaluate_energy


def train(config: dict[str, Any], output: str | Path, *, device: str = "cpu",
          stop_at: int | None = None, resume: str | Path | None = None,
          compute_backend: str = "baseline", compute_chunk_size: int | None = None) -> Path:
    """Run to an absolute step; every invocation writes into a new directory."""
    validate_config(config)
    policy = compute_policy(config, compute_backend, compute_chunk_size)
    settings = config["training"]
    stop_at = settings["max_steps"] if stop_at is None else stop_at
    positive_integer(stop_at, "stop_at")
    if stop_at > settings["max_steps"]:
        raise ValueError("stop_at exceeds max_steps")
    # Validate before building or creating any output directory.
    payload = read_checkpoint(resume) if resume is not None else None
    if payload is not None:
        if payload["compute_policy"] != policy:
            raise ValueError("resume compute policy differs from the checkpoint")
        if payload["config"] != config:
            raise ValueError("resume configuration differs from the checkpoint")
        if payload["completed_step"] >= stop_at:
            raise ValueError("checkpoint is already at or beyond stop_at")
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new output directory; existing runs are never overwritten")
    runtime = build_runtime(config, device)
    if payload is not None:
        walkers, cached_log = restore_training(runtime, payload)
        history = list(payload["history"])
        start_step = payload["completed_step"]
    else:
        runtime.set_gate(0)
        walkers = runtime.sampler.initialize(settings["batch_size"])
        walkers, cached_log = runtime.sampler.burn_in(walkers, settings["burn_in"])
        history, start_step = [], 0
    output.mkdir(parents=True, exist_ok=False)
    atomic_save(output / "experiment_manifest.json", {
        "config": config, "config_sha256": canonical_hash(config),
        "source_sha256": source_hash(), "compute_policy": policy,
        "resume_from_step": start_step,
    })
    started = time.perf_counter()
    elapsed_offset = history[-1]["elapsed_seconds"] if history else 0.0
    checkpoint_path = None
    for zero_based_step in range(start_step, stop_at):
        runtime.set_gate(zero_based_step)
        with torch.no_grad():
            cached_log = runtime.model(walkers)
        walkers, cached_log, acceptance = runtime.sampler.step(walkers, cached_log.detach())
        positions = walkers.detach().requires_grad_(True)
        log_abs, phase = runtime.model.log_psi(positions)
        energy_raw = evaluate_energy(runtime, positions, policy).detach()
        assert_finite((energy_raw, log_abs, phase), "training inputs")
        signal = torch.complex(
            clip_local_energy(energy_raw.real.to(torch.float64), rho=settings["clip_rho"]),
            clip_local_energy(energy_raw.imag.to(torch.float64), rho=settings["clip_rho"]),
        )
        optimized_mean = runtime.optimizer.step(signal, log_abs, phase)
        assert_finite(runtime.base_model.state_dict(), "model")
        completed = zero_based_step + 1
        if completed % settings["sampler_adapt_every"] == 0:
            runtime.sampler.adapt_step_size(settings["sampler_target_acceptance"])
        record = {
            "step": completed,
            "energy_raw_real": float(energy_raw.real.mean().item()),
            "energy_variance": float(energy_raw.real.var(unbiased=False).item()),
            "energy_raw_imag_std": float(energy_raw.imag.std(unbiased=False).item()),
            "energy_optimized": float(optimized_mean),
            "acceptance": float(acceptance),
            "step_size": float(runtime.sampler.step_size),
            "optimizer_step": runtime.optimizer.t,
            "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
        }
        assert_finite(record, "diagnostics")
        history.append(record)
        if completed % settings["validation_every"] == 0 or completed == stop_at:
            # These are training diagnostics, not independent energy evaluations.
            atomic_save(output / f"validation_step{completed:06d}.json", record)
        if completed % settings["checkpoint_every"] == 0 or completed == stop_at:
            suffix = "final" if completed == settings["max_steps"] else f"step{completed:06d}"
            checkpoint_path = output / f"checkpoint_{suffix}.pt"
            save_checkpoint(runtime, checkpoint_path, walkers, cached_log, history,
                            compute=policy)
            atomic_save(output / "history.json", history)
            print(json.dumps(record, allow_nan=False), flush=True)
    if checkpoint_path is None:
        raise RuntimeError("no checkpoint produced")
    return checkpoint_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CLASSES, required=True)
    parser.add_argument("--n-phi", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--compute-backend", choices=("baseline", "forward_vgl"), default="baseline")
    parser.add_argument("--compute-chunk-size", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config, args.model, n_phi=args.n_phi)
    checkpoint = train(config, args.output, device=args.device,
                       stop_at=args.stop_at, resume=args.resume,
                       compute_backend=args.compute_backend, compute_chunk_size=args.compute_chunk_size)
    print(f"Saved {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
