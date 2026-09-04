"""Build one configured model and check its forward interface (no training)."""
import argparse
import json

import torch

from complex_psiformer.runtime import (MODEL_CLASSES, build_runtime, count_real_parameters,
                                      load_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/18cell.json")
    parser.add_argument("--model", choices=MODEL_CLASSES, default="complex_psiformer")
    parser.add_argument("--n-phi", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config, args.model, n_phi=args.n_phi)
    runtime = build_runtime(config, args.device)
    positions = runtime.sampler.initialize(2)
    with torch.no_grad():
        log_abs, phase = runtime.model.log_psi(positions)
    assert torch.isfinite(log_abs).all() and torch.isfinite(phase).all()
    print(json.dumps({
        "model": config["model_config"]["name"],
        "seed": config["seed"], "n_phi": config["system"]["n_phi"],
        "real_parameters": count_real_parameters(runtime.model),
        "log_amplitude_shape": list(log_abs.shape),
        "phase_dtype": str(phase.dtype),
    }, indent=2))


if __name__ == "__main__":
    main()
