# Model construction

[models.json](../configs/models.json) defines the supplied settings.

## Wavefunction and magnetic boundary

Coordinates have shape `[batch, N, 2]`. Orbital matrices have electron index $i$
as row and orbital index $j$ as column:

$$
\Psi(\mathbf R)=\sum_{m=1}^{M}\det\Phi^{(m)}(\mathbf R).
$$

All determinant coefficients are fixed to one. SlaterNets use $M=1$ and
PsiFormers use $M=4$. `model.log_psi(R)` returns a stabilized log amplitude
and complex unit phase. `model(R)` returns only the log amplitude.

At nonzero integer flux, the native orbital is multiplied by a fixed section,

$$
\Phi^{(m)}_{ij}(\mathbf R)=
\widetilde\Phi^{(m)}_{ij}(\mathbf R)s_j(\mathbf r_i),
$$

implementing the symmetric-gauge torus transition law

$$
\Psi(\ldots,\mathbf r_i+\mathbf L_a,\ldots)=
\exp\!\left[\frac{iB}{2}
\bigl(L_{a,x}(y_i-o_y)-L_{a,y}(x_i-o_x)\bigr)\right]\Psi(\mathbf R).
$$

Here $B|\det L|=2\pi n_\phi$ and
$\mathbf o=(\mathbf L_1+\mathbf L_2)/2$.
Section labels cycle through flux degeneracy and then Landau-level index.
The image sum has radius 8. Each family retains its source-matched section
implementation. At zero flux the bare periodic network is used.

## Inputs and orbital readout

The reciprocal supercell vectors satisfy
$\mathbf G_a\cdot\mathbf L_b=2\pi\delta_{ab}$. Both axes use the first harmonic:

$$
f_{\mathbb R}(\mathbf r)=
(\sin(\mathbf G_1\cdot\mathbf r),\sin(\mathbf G_2\cdot\mathbf r),
 \cos(\mathbf G_1\cdot\mathbf r),\cos(\mathbf G_2\cdot\mathbf r)),
\qquad
f_{\mathbb C}(\mathbf r)=
(e^{i\mathbf G_1\cdot\mathbf r},e^{i\mathbf G_2\cdot\mathbf r}).
$$

The embedding has no bias or embedding activation. Hidden width is 64 over
the reals or 32 over the complexes.

Real streams use independent real orbital projections:
$\widetilde\Phi_{ij}=w_{j,\mathrm{re}}\cdot h_i+
i w_{j,\mathrm{im}}\cdot h_i$.
Complex streams use
$\widetilde\Phi_{ij}=\sum_d h_{i,d}
(w_{j,d,\mathrm{re}}-i w_{j,d,\mathrm{im}})$.
Thus “Real” describes the hidden representation, not a restriction to
real-valued wavefunctions.

## Real SlaterNet

Three electron-wise residual layers implement

$$
h_i^{\ell+1}=h_i^\ell+\tanh(W_\ell h_i^\ell+b_\ell).
$$

There is no attention or dependence on other electrons before determinant
assembly. Both orbital projections use the full hidden width. The constructor
allows only a single determinant with fixed unit coefficient.

## Complex SlaterNet

The embedding and residual layers use free real-linear maps:

$$
\begin{pmatrix}\operatorname{Re}F(z)\\ \operatorname{Im}F(z)\end{pmatrix}
=
\begin{pmatrix}W_{AA}&W_{AB}\\W_{BA}&W_{BB}\end{pmatrix}
\begin{pmatrix}\operatorname{Re}z\\ \operatorname{Im}z\end{pmatrix}
+\begin{pmatrix}b_{\mathrm{re}}\\b_{\mathrm{im}}\end{pmatrix}.
$$

Three layers implement $h^{\ell+1}=h^\ell+\sigma(F_\ell(h^\ell))$, with

$$
\sigma(z)=\tanh(|z|-r_0)\frac{z}{|z|+10^{-8}}.
$$

The per-neuron threshold $r_0$ is learned. The signed radial factor can reverse
phase by $\pi$; this is a U(1)-equivariant activation, not an everywhere-positive
modulus map. No conjugate-augmented inputs are used. The output has one
determinant.

## Real PsiFormer

Each of three blocks uses two all-electron attention heads, followed by one
residual tanh MLP. Query/key and value widths are both 16 per head.
The published normalization is

$$
\alpha_{ij}=\operatorname{softmax}_j(q_i\cdot k_j),\qquad
a_i=\frac{1}{\sqrt{d_{\mathrm{val}}}}\sum_j\alpha_{ij}v_j.
$$

Head outputs are concatenated, projected to width 64, and added to the input
stream. A residual MLP completes the block. Four orbital matrices feed the
fixed-unit determinant sum.

## Complex PsiFormer

The two heads in each block have
$d_{\mathrm{attn}}=d_{\mathrm{val}}=8$:

$$
\alpha_{ij}=\operatorname{softmax}_j
\left(\frac{|q_i^\dagger k_j|}{\sqrt{d_{\mathrm{attn}}}}\right),
\qquad a_i=\sum_j\alpha_{ij}v_j.
$$

Query/key/value and output projections are free real-linear maps with no
bias. Their cross-component matrices are initialized to zero and remain
trainable.

For projected attention output $u$ and input stream $h$, the residual branch
uses the RMS cap

$$
u\longmapsto u\min\left(1,
\frac{\sqrt{\langle|h|^2\rangle+\epsilon}}
{\sqrt{\langle|u|^2\rangle+\epsilon}+\epsilon}\right),
\qquad\epsilon=10^{-6}.
$$

The averages run over the electron and hidden-feature axes within each walker.
A positive softplus gate, fixed at its unit initialization, multiplies the
branch. A separate non-trainable multiplier
$\min(\max(t/500,0),1)$ implements warmup at zero-based optimizer step $t$.
A residual complex MLP then completes the block.

Three blocks feed four determinants. All trainable tensors are stored as
real tensors; counts include every real component. Optimization differentiates
both amplitude and phase.
