# Source provenance

This release assembles the models from their frozen research implementations.
It retains their numerical model, boundary, Hamiltonian, sampler, energy, and
MinSR kernels.

| Source family | Source commit | Source archive SHA-256 |
|---|---|---|
| Real and Complex SlaterNet | `5dd20e62e88a1dad2f2c545d3eae871ca32c1c7b` | `b4e0ab38b50b002ba152c44b4db4c4268c7b83931cd1c3b1b4168ce79be9356a` |
| Real and Complex PsiFormer | `71c7ef65d3fe8d2155e3f1eba5a5493e37c7a698` | `1166dbee370711c2aa3fd1e8153869ebc380c483f3de15ca2e5b3468aa4eff55` |

The SlaterNet family keeps its source-matched magnetic-section evaluator and
boundary adapter. Shared kernels are identical between the two source archives
where reused. Separate section implementations are retained, not substituted.

Publication changes are limited to class/module naming, supplied constructor
defaults, single-determinant SlaterNet restrictions, documentation, configuration
loading, and self-contained execution/checkpoint entry points. Complex-energy
functions and the interaction-scaling wrapper are extracted without numerical
changes.

Initialized parameter tensors, forward outputs, and complex local energies
were checked against the frozen sources in both supercells. Core definitions
were compared after accounting for naming, documentation, the stated defaults,
and single-determinant guards. These are implementation checks, not physical
results.

[source_manifest.json](../provenance/source_manifest.json) records original
source members and their hashes, without private execution paths or data.

The runner defines its own checkpoint schema and source/configuration
fingerprints. Automatic conversion of older experiment bundles is not claimed.
