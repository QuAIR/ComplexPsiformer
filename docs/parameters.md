# Experimental parameters

The JSON configurations are the machine-readable settings. All supplied runs
use `seed = 1`; no result table is included.

## Systems

| Setting | 18-cell | 25-cell |
|---|---|---|
| Primitive-cell tiling | `[3, 6]` | `[5, 5]` |
| Electrons | 12 | 12 |
| Comparison flux $n_\phi$ | 2 | 2 |
| Moiré lattice constant | 8.031 nm | 8.031 nm |
| Effective mass $m^*/m_e$ | 0.35 | 0.35 |
| Relative dielectric constant | 5 | 5 |
| Potential amplitude $V_0$ | 15 meV | 15 meV |
| Potential phase $\phi$ | 0.7854 rad | 0.7854 rad |
| Interaction multiplier $\lambda_C$ | 1 | 1 |
| Ewald cutoff `n_images` | 5 | 5 |
| Gauge | symmetric | symmetric |
| Flat vector potential | `[0, 0]` | `[0, 0]` |

Primitive vectors are $\mathbf a_1=a_M(1,0)$ and
$\mathbf a_2=a_M(1/2,\sqrt{3}/2)$. Supercell vectors are
$\mathbf L_1=n_1\mathbf a_1$, $\mathbf L_2=n_2\mathbf a_2$.

[flux_scan.json](../configs/flux_scan.json) specifies the 18-cell Complex
PsiFormer for $n_\phi\in\{0,1,2,3,4\}$, keeping other settings unchanged.
Flux is through the whole supercell, not each primitive cell.

## Units and Hamiltonian

Effective units are

$$
a_B^*=a_B\frac{\epsilon_r}{m^*/m_e},\qquad
\mathrm{Ha}^*=\mathrm{Ha}\frac{m^*/m_e}{\epsilon_r^2}.
$$

The runtime converts physical inputs once. The Hamiltonian uses
$\frac12(-i\nabla-\mathbf A)^2$, the periodic Ewald interaction including
background/self terms, and

$$
V(\mathbf r)=-2V_0\sum_{j=1}^3\cos(\mathbf b_j\cdot\mathbf r+\phi),\qquad
(\mathbf b_1,\mathbf b_2,\mathbf b_3)
=(n_1\mathbf G_1,n_2\mathbf G_2,-n_1\mathbf G_1-n_2\mathbf G_2).
$$

The field is $B=2\pi n_\phi/|\det L|$, with symmetric-gauge origin at the
supercell center. Local energies include amplitude and phase derivatives and
are **total** energies in $\mathrm{Ha}^*$.

## Training

| Setting | Value |
|---|---:|
| Optimizer | full amplitude-and-phase MinSR |
| Updates | 10,000 |
| Parallel walkers | 384 |
| Initial training burn-in | 0 |
| Metropolis proposals per update | 1 full-configuration proposal per walker |
| Initial Gaussian proposal scale | 0.5 effective Bohr |
| Proposal-scale adaptation interval | 50 updates |
| Target acceptance | 0.5 |
| Local-energy walker chunk | 24 |
| Score-Jacobian chunk | 16 |
| Initial learning rate $\eta_0$ | 0.03 |
| Learning-rate scale $t_0$ | 2,000 |
| Damping | 0.03 |
| QGT squared-norm constraint | $5\times10^{-4}$ |
| Optimizer clipping multiplier $\rho$ | 3 |
| Checkpoint / training-diagnostic interval | 250 updates |
| Real / complex computation dtype | float64 / complex128 |

Learning rate is $\eta_t=\eta_0/(1+t/t_0)$. Complex PsiFormer has a 500-update
attention-gate warmup; the other models have none.
Cached log amplitudes are recomputed before each proposal under the current
parameters and gate.

Initialization follows the source order: construct parameters with PyTorch's
default float32, then convert the model to double. The runtime rejects a
changed global default dtype.

Proposal scale is adapted as
$s\leftarrow s[1+0.1(a-a_{\mathrm{target}})]$ using the accumulated acceptance
fraction since the previous adaptation.

## MinSR and energy records

Here $E$ denotes the clipped optimizer signal, not the reported raw energy.
For batch size $B_s$, the centered amplitude and phase scores are stacked:

$$
X=\frac{1}{\sqrt{B_s}}
\begin{bmatrix}
\partial_\theta\log|\Psi|-\langle\partial_\theta\log|\Psi|\rangle\\
\partial_\theta\arg\Psi-\langle\partial_\theta\arg\Psi\rangle
\end{bmatrix},
\qquad
y=\frac{1}{\sqrt{B_s}}
\begin{bmatrix}\operatorname{Re}(E-\langle E\rangle)\\
\operatorname{Im}(E-\langle E\rangle)\end{bmatrix}.
$$

The update
$\delta\theta=-2\eta X^\mathsf T(XX^\mathsf T+\lambda I)^{-1}y$
is rescaled if $\|X\delta\theta\|^2$ exceeds the constraint.

For the optimizer signal only, real and imaginary energies are clipped
separately to the within-batch median plus/minus $\rho$ times the **median
absolute deviation** from that median. Energy records retain the raw values:

- `energy_raw_real`: mean unclipped real local energy.
- `energy_variance`: raw real-energy variance, with denominator $B_s$.
- `energy_raw_imag_std`: raw imaginary-energy standard deviation, with denominator $B_s$.
- `energy_optimized`: mean real energy after optimizer clipping.

A record describes the sampled state **before** its update; the checkpoint
contains parameters **after** that update. `validation_step*.json` files are
training diagnostics, not independent frozen-network energy measurements.

Frozen-network sampling performs no optimization, no clipping, and no
proposal-scale adaptation. Burn-in, retained sweep count, and thinning are
explicit inputs. The saved sweep/walker axes support later autocorrelation-aware
analysis; the sampler does not attach a standard error to correlated samples.
