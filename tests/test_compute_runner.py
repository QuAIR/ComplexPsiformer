"""CPU regression for the public accelerated training/sampling/checkpoint path."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from complex_psiformer.checkpoint import read_checkpoint
from complex_psiformer.compute import compute_policy
from complex_psiformer.runtime import load_config
from complex_psiformer.sample import sample
from complex_psiformer.train import train

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def config_for(model):
    config = load_config(ROOT / "configs/18cell.json", model)
    config["training"].update(batch_size=2, kinetic_chunk_size=1, burn_in=2,
                              sampler_adapt_every=1)
    return config


@pytest.mark.parametrize("model", ["real_psiformer", "complex_psiformer"])
def test_update_sampling_and_exact_resume(tmp_path, model):
    config = config_for(model)
    base_path = train(config, tmp_path / "base", stop_at=2)
    base = read_checkpoint(base_path)
    fast = read_checkpoint(train(config, tmp_path / "fast", stop_at=2,
                                 compute_backend="forward_vgl"))
    for name, value in base["model_state"].items():
        torch.testing.assert_close(value, fast["model_state"][name], atol=2e-9, rtol=2e-9)
    assert torch.equal(base["walkers"], fast["walkers"])
    assert torch.equal(base["torch_rng_state"], fast["torch_rng_state"])
    assert base["sampler_state"] == fast["sampler_state"]
    for a, b in zip(base["history"], fast["history"]):
        assert a["energy_raw_real"] == pytest.approx(b["energy_raw_real"], abs=2e-8, rel=2e-8)
    first = train(config, tmp_path / "first", stop_at=1, compute_backend="forward_vgl")
    resumed = read_checkpoint(train(config, tmp_path / "resume", stop_at=2,
                                    resume=first, compute_backend="forward_vgl"))
    for name, value in fast["model_state"].items():
        assert torch.equal(value, resumed["model_state"][name])
    for key in ("walkers", "cached_log", "torch_rng_state"):
        assert torch.equal(fast[key], resumed[key])
    assert fast["sampler_state"] == resumed["sampler_state"]
    assert fast["optimizer_state"] == resumed["optimizer_state"]
    for a, b in zip(fast["history"], resumed["history"]):
        assert {k: v for k, v in a.items() if k != "elapsed_seconds"} == {
            k: v for k, v in b.items() if k != "elapsed_seconds"}
    for options in ({}, {"compute_backend": "forward_vgl", "compute_chunk_size": 1}):
        with pytest.raises(ValueError, match="resume compute policy"):
            train(config, tmp_path / "invalid", stop_at=2, resume=first, **options)
        assert not (tmp_path / "invalid").exists()
    outputs = []
    for backend in ("baseline", "forward_vgl"):
        path = sample(base_path, tmp_path / backend, burn_in=1, sweeps=2, thin=1,
                      compute_backend=backend)
        outputs.append(torch.load(path, weights_only=True))
    assert torch.equal(outputs[0]["final_walkers"], outputs[1]["final_walkers"])
    torch.testing.assert_close(outputs[0]["energy_raw_complex"], outputs[1]["energy_raw_complex"],
                               atol=2e-8, rtol=2e-8)
    assert outputs[1]["training_compute_policy"]["backend"] == "baseline"
    assert outputs[1]["compute_policy"]["backend"] == "forward_vgl"
    assert fast["compute_policy"]["chunk_size"] == config["training"]["batch_size"]
    damaged = torch.load(first, weights_only=True)
    damaged["compute_policy"]["source"]["commit"] = "0" * 40
    torch.save(damaged, tmp_path / "damaged.pt")
    with pytest.raises(ValueError, match="compute policy/source"):
        read_checkpoint(tmp_path / "damaged.pt")


def test_invalid_backend_rejected_before_run(tmp_path):
    with pytest.raises(ValueError, match="PsiFormer only"):
        train(config_for("real_slaternet"), tmp_path / "bad", stop_at=1,
              compute_backend="forward_vgl")
    assert not (tmp_path / "bad").exists()
    config = config_for("complex_psiformer")
    for chunk in (0, -1, True, 1.5, 3):
        with pytest.raises(ValueError, match="chunk size"):
            compute_policy(config, "forward_vgl", chunk)
    with pytest.raises(ValueError, match="baseline uses"):
        compute_policy(config, "baseline", 1)


def test_public_cli_forwards_compute_options(tmp_path):
    from complex_psiformer.train import main as train_main
    from complex_psiformer.sample import main as sample_main
    with patch("complex_psiformer.train.train", return_value=tmp_path / "saved.pt") as run:
        assert train_main(["--config", str(ROOT / "configs/25cell.json"),
                           "--model", "complex_psiformer", "--output", str(tmp_path / "train"),
                           "--compute-backend", "forward_vgl", "--compute-chunk-size", "384"]) == 0
        assert run.call_args.kwargs["compute_backend"] == "forward_vgl"
        assert run.call_args.kwargs["compute_chunk_size"] == 384
    with patch("complex_psiformer.sample.sample", return_value=tmp_path / "raw.pt") as run:
        assert sample_main(["--checkpoint", str(tmp_path / "saved.pt"),
                            "--output", str(tmp_path / "sample"), "--burn-in", "1",
                            "--sweeps", "2", "--thin", "1", "--compute-backend", "forward_vgl",
                            "--compute-chunk-size", "256"]) == 0
        assert run.call_args.kwargs["compute_backend"] == "forward_vgl"
        assert run.call_args.kwargs["compute_chunk_size"] == 256
