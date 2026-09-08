# Exact forward coordinate derivatives

Version 0.2.0 provides an optional `forward_vgl` energy backend for the published
Complex and Real PsiFormer models. It propagates values, all first coordinate
derivatives, and the Laplacian through the model without constructing a full
coordinate Hessian. The model equations, magnetic carrier, float64/complex128
precision, sampling budget, and FullComplexMinSR update are preserved.

## Training and sampling

```sh
python -m complex_psiformer.train --config configs/25cell.json --model complex_psiformer --device cuda:0 --compute-backend forward_vgl --compute-chunk-size 384 --output runs/25cell_complex_vgl
python -m complex_psiformer.sample --checkpoint runs/25cell_complex_vgl/checkpoint_final.pt --device cuda:0 --compute-backend forward_vgl --compute-chunk-size 384 --burn-in 20000 --sweeps 1024 --thin 20 --output samples/25cell_complex_vgl
```

These commands start computation when invoked. The sampling counts illustrate
explicit budgeting; they do not by themselves certify equilibration or an error
bar. Sampling returns the original raw complex Laplacian local energies.

The default backend is `baseline`, which remains available for all four models.
`forward_vgl` supports Complex/Real PsiFormer, including the published magnetic
boundary adapter; SlaterNets must use `baseline`.

For `forward_vgl`, an omitted `--compute-chunk-size` uses the configured batch
size (384 for the supplied configurations). Choose a smaller positive integer
if memory is limited; this changes calculation scheduling, not the number of
walkers. `baseline` uses the configuration's original `kinetic_chunk_size`
and rejects the new chunk override.

Manifests and checkpoints record the compute policy and derivative-source
identity separately from the scientific configuration. To resume, repeat the
same backend and chunk settings as the saved run. Sampling may explicitly choose
a different backend and records both the training and evaluation policies.
As with the previous release, checkpoints are tied to their source fingerprint:
checkpoints created before this release must be used with their original checkout.

## API

```python
from complex_psiformer.exact_compute import log_psi_vgl, kinetic_pair
```

`log_psi_vgl(model, positions)` takes a nonempty `[batch, N, 2]` coordinate tensor
and returns detached log amplitude, unit complex phase, their first derivatives,
and their Laplacians. `kinetic_pair` returns the complex Laplacian kinetic energy
and the real quadratic kinetic integrand, sharing the propagated gradients.
Its magnetic arguments are `peierls_A`, `magnetic_B`, `magnetic_gauge="symmetric"`,
and `magnetic_origin`. Parameter-score differentiation for MinSR continues
through the original model outside this detached calculation.

The derivative rules preserve the published unscaled Real attention logits and
scaled values, Complex `abs_dot`, Kerr denominator, RMS cap, determinant
coefficients and orbital readout signs. Derivative propagation uses ordinary
unconjugated complex products. In particular,
`lap(AB) = lap(A)B + 2 sum_a (d_a A)(d_a B) + A lap(B)`.

## Validation and performance evidence

The included tests compare analytic primitives and N=2/N=12 model derivatives
against independent analytic/autograd references. Public-runner tests check
Complex/Real energy and parameter-update parity, unchanged walkers and RNG,
frozen sampling, bitwise CPU continuation within one backend, and rejection of
changed compute policies. Run them with the pinned PyTorch environment:

```sh
python -m pip install pytest
python -m pytest -q tests
```

Release validation on 2026-09-08: **35 tests passed** on Windows CPU with Python
3.12.14 and PyTorch 2.10.0+cpu. Wheel construction, installed-package imports,
provenance loading and CLI availability also passed. GPU timing was not rerun
for this packaging release.

The earlier integration benchmark on an RTX 3090, using 384 configurations,
measured median complete optimization steps of 12.1965 → 5.7040 seconds for
Complex PsiFormer (2.14×) and 4.3907 → 2.8052 seconds for Real PsiFormer (1.57×).
Those measurements used a 256-sized forward chunk, two warmup pairs and five
alternating baseline/forward pairs on trained N=12 seed-1 state copies. Maximum
raw-energy discrepancies were 2.99e-12 and 2.51e-14 Ha* total, and maximum
parameter-update discrepancies were 5.56e-17 and 1.12e-16. They are prior
integration measurements, not a benchmark of every configuration or a promised
speedup for this release's 384-sized default. No new GPU timing is claimed here.

## Provenance and limits

The implementation is adapted from
[Quantum-Matter's training-time-optimization commit](https://github.com/Erdong-Huang/Quantum-Matter/tree/9828f69871ed47baecf07fbc3bfba88969db0cd5)
and ported to the published PsiFormer definitions. Original file hashes and commit
are included in `complex_psiformer/exact_compute/provenance.json`. The tested
experiment integration was promoted into the installable package without
changing its derivative formulas.

Classical derivatives require smooth local points. Zero overlaps, zero Kerr
inputs, cap kinks, determinant nodes and magnetic chart seams require separate
analysis. Individually singular determinants are unsupported even when their
mixture might be finite. Unsupported inputs fail explicitly; there is no automatic
smoothing, jitter, approximate trace or removal of samples. Agreement at tested
points establishes numerical implementation parity there; it does not establish
global regularity, finite population variance, or equality of quadratic and
Laplacian expectation values.
