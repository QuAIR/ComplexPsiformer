# ComplexPsiformer

Neural wavefunctions for two-dimensional magnetic moiré systems.

This repository provides model construction, physical and numerical parameters,
and minimal VMC training and frozen-network sampling entry points. It does not
include trained weights, experimental results, or plotting assets.

## Models

All supplied configurations use `seed = 1` and `N = 12` electrons.

| Model | Hidden width | Layers | Determinants | Real trainable parameters |
|---|---:|---:|---:|---:|
| Real SlaterNet | 64 real | 3 | 1 | 14,272 |
| Complex SlaterNet | 32 complex | 3 | 1 | 13,600 |
| Real PsiFormer | 64 real | 3 | 4 | 43,456 |
| Complex PsiFormer | 32 complex | 3 | 4 | 40,480 |

Both SlaterNets are restricted to a single determinant. “Real” and “Complex”
refer to the hidden representations; the orbital readout and magnetic boundary
conditions can make the full wavefunction complex in either case.

See [model construction](docs/models.md) and
[experimental parameters](docs/parameters.md) for equations and settings.

## Installation

Use Python 3.10 or later and PyTorch 2.10.0. From this checkout:

~~~sh
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

For GPU execution, use a CUDA-enabled PyTorch 2.10.0 build compatible with your
driver. Device selection is explicit; the programs do not select servers,
allocate GPUs, or launch jobs in the background.

## Build a model without training

~~~sh
python -m examples.build_model --config configs/18cell.json --model complex_psiformer
~~~

This checks a two-walker forward pass and prints the parameter count, without
any optimizer update. Available model keys are `real_slaternet`,
`complex_slaternet`, `real_psiformer`, and `complex_psiformer`.

## Training

Version 0.2.0 adds the optional accelerated `forward_vgl` backend for Complex/Real
PsiFormer. Add `--compute-backend forward_vgl --compute-chunk-size 384` to the
training or sampling command. The default remains `baseline`; see
[acceleration, validation, and checkpoint compatibility](docs/acceleration.md).

The following command starts one training run, using 10,000 MinSR updates and
384 walkers:

~~~sh
python -m complex_psiformer.train --config configs/18cell.json --model complex_psiformer --device cuda:0 --output runs/18cell_complex
~~~

Use `configs/25cell.json` for the 25-cell system, or select another model key.
Both system files use `n_phi = 2`.

The flux configuration selects one 18-cell Complex PsiFormer run at a time:

~~~sh
python -m complex_psiformer.train --config configs/flux_scan.json --model complex_psiformer --n-phi 3 --device cuda:0 --output runs/flux3_complex
~~~

The supported flux values are 0, 1, 2, 3, and 4; `--n-phi` is required for this
configuration. Other sectors are not launched automatically.

### Checkpointing and continuation

`--stop-at` is an absolute update count and does not change `max_steps`.
Checkpoints are written every 250 updates and at the requested stopping point.
A checkpoint at `max_steps` is named `checkpoint_final.pt`.

To continue a checkpoint, pass the same configuration and model and choose a
**new** output directory:

~~~sh
python -m complex_psiformer.train --config configs/18cell.json --model complex_psiformer --device cuda:0 --resume runs/18cell_complex/checkpoint_step001000.pt --stop-at 10000 --output runs/18cell_complex_continued
~~~

The runner restores model buffers, optimizer counter, walkers, sampler state,
history, and random-number-generator state. It rejects configuration, source,
or recorded execution-environment mismatches. Existing runs are never
overwritten. Continuation is tested bitwise on CPU in the same environment;
bitwise agreement across different hardware is not promised.

Checkpoints use this repository's runner format. Older research-bundle formats
are not accepted implicitly.

## Frozen-network sampling

Sampling loads a saved network, freezes its parameters, and starts fresh uniform
walkers. It does not optimize or change the saved attention gate. Burn-in,
retained sweeps, and thinning must be specified explicitly.

This short command illustrates the interface; its small counts are **not**
a production uncertainty-estimation protocol:

~~~sh
python -m complex_psiformer.sample --checkpoint runs/18cell_complex/checkpoint_final.pt --device cuda:0 --burn-in 10 --sweeps 2 --thin 1 --output samples/18cell_complex
~~~

The output retains unclipped complex local energies with separate
`[retained_sweep, walker]` axes. Energies are total energies in effective
Hartrees (`Ha*`), not energy per electron. Equilibration, autocorrelation,
and uncertainty convergence must be assessed before reporting estimates.

## Tests

~~~sh
python -m unittest discover -s tests -v
~~~

Small CPU checks cover configurations, parameter counts, single-determinant
restrictions, antisymmetry, magnetic boundary phases, attention equations,
phase-sensitive kinetic energy, MinSR, frozen sampling, and exact continuation.
Temporary small-batch outputs are software checks, not experimental results.

## Layout

- `complex_psiformer/models/`: four model implementations.
- `complex_psiformer/features/`: periodic inputs and magnetic sections.
- `complex_psiformer/systems/`: supercells, units, Ewald interaction, and potential.
- `complex_psiformer/vmc/`: sampler, complex local energy, and MinSR.
- `configs/`: model catalog and seed-1 system configurations.
- `examples/` and `tests/`: minimal usage and CPU checks.

[Source provenance](docs/provenance.md) records the frozen implementations.
