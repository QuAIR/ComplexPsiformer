# Uniform-gas optimization histories for Table S3 and Figure S1

Two separately optimized twelve-electron states at Landau filling 3/10 and
Landau-level mixing strength kappa = 9 have different first-shell angular
contrast. Display labels identify the final pair correlations.

`training_energy.csv` contains all 20,000 updates for each state. The atomic-unit
columns contain the unclipped real local energy averaged over the training
walkers, before the parameter update. The other columns convert the total
effective-Hartree energies to meV per electron using
`27211.386245988 * 0.35 / 5**2 / 12`.

`plot_training.py` reproduces `uniform_training_states.pdf` and its PNG preview
using NumPy and Matplotlib. Run it in this directory. Both panels use means over
nonoverlapping blocks of 100 updates. Panel (a) shows the entire training record
and panel (b) enlarges updates 15,001 through 20,000. The curves have no error
bands. Table S3 reports separate fixed-parameter quadratic-form estimates and
their conditional Monte Carlo errors, not the final points of these curves.

`provenance.json` records the original history hashes, continuation segments,
terminal state hashes, and frozen-evaluation checkpoint hashes. The continuation
for the negligible-contrast state was joined to its verified 15,000-update parent
record. Each assembled trajectory contains exactly one entry for updates 1
through 20,000, and both terminal states match the Table S3 evaluation manifests.

These curves illustrate training outcomes. They do not establish symmetry
eigenvalues or attribute the energy difference to a particular initialization.
The existing Table S3 values and their sampling diagnostics are unchanged.
The raw Monte Carlo configurations and model checkpoints remain separately held
by the authors. No files in the earlier data snapshots are replaced.
