"""Sample a saved, frozen wavefunction without optimization."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import assert_finite, atomic_save, read_checkpoint
from .runtime import build_runtime, evaluate_local_energy, positive_integer


def sample(checkpoint: str | Path, output: str | Path, *, device: str = "cpu",
           burn_in: int, sweeps: int, thin: int) -> Path:
    """Return raw complex energies, retaining both sweep and walker axes."""
    positive_integer(burn_in, "burn_in", allow_zero=True)
    positive_integer(sweeps, "sweeps")
    positive_integer(thin, "thin")
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new sampling output directory")
    payload = read_checkpoint(checkpoint)
    runtime = build_runtime(payload["config"], device)
    runtime.base_model.load_state_dict(payload["model_state"], strict=True)
    runtime.model.eval()
    # Freeze parameters, not coordinates: the kinetic energy needs spatial derivatives.
    for param in runtime.model.parameters():
        param.requires_grad_(False)
    before = {name: value.detach().clone()
              for name, value in runtime.base_model.state_dict().items()}
    runtime.sampler.step_size = payload["sampler_state"]["step_size"]
    # Fresh uniform walkers; retain the saved gate and do not perform any optimizer update.
    torch.manual_seed(payload["config"]["seed"])
    walkers = runtime.sampler.initialize(payload["config"]["training"]["batch_size"])
    walkers, cached_log = runtime.sampler.burn_in(walkers, burn_in)
    output.mkdir(parents=True, exist_ok=False)
    raw, acceptance = [], []
    for _ in range(sweeps):
        accepted = 0.0
        for _ in range(thin):
            walkers, cached_log, fraction = runtime.sampler.step(walkers, cached_log)
            accepted += fraction
        energy = evaluate_local_energy(runtime, walkers.detach().requires_grad_(True)).detach()
        assert_finite(energy, "sampled local energy")
        raw.append(energy.cpu())
        acceptance.append(accepted / thin)
    for name, value in runtime.base_model.state_dict().items():
        if not torch.equal(value, before[name]):
            raise RuntimeError("frozen model state changed during sampling")
    result = {
        "config": payload["config"],
        "completed_step": payload["completed_step"],
        "source_sha256": payload["source_sha256"],
        "protocol": {"seed": payload["config"]["seed"], "burn_in": burn_in,
                     "sweeps": sweeps, "thin": thin},
        "energy_raw_complex": torch.stack(raw),
        "acceptance": acceptance,
        "final_walkers": walkers.detach().cpu(),
        "energy_unit": "Ha* (total energy)",
        "axes": ["retained_sweep", "walker"],
    }
    target = output / "raw_samples.pt"
    atomic_save(target, result, tensor_file=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--burn-in", type=int, required=True)
    parser.add_argument("--sweeps", type=int, required=True)
    parser.add_argument("--thin", type=int, required=True)
    args = parser.parse_args(argv)
    target = sample(args.checkpoint, args.output, device=args.device,
                    burn_in=args.burn_in, sweeps=args.sweeps, thin=args.thin)
    print(f"Saved unclipped samples to {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
