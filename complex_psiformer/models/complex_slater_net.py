"""Complex SlaterNet with one Slater determinant."""

import torch
import torch.nn as nn

from ..features.periodic import (
    ComplexPeriodicFeatures,
    ExtendedComplexPeriodicFeatures,
)
from ..systems.supercell import Supercell
from .kerr_mlp import (
    ComplexLinear,
    ComplexMLP,
    FreeComplexLinear,
    KerrActivation,
    SplitTanhActivation,
)


class ComplexSlaterNet(nn.Module):
    """Single-determinant wavefunction with a residual electron-wise MLP."""

    def __init__(
        self,
        cell: Supercell,
        d_model: int = 32,
        n_layers: int = 3,
        n_det: int = 1,
        order: int = 1,
        include_conjugates: bool = False,
        init_real: bool = False,
        init_imag_scale: float = 1.0,
        embed_activation: str = "none",
        embed_bias: bool = False,
        activation: str = "kerr",
        embedding_kind: str = "free",
        per_block_kind: str = "free",
        trainable_determinant_coefficients: bool = False,
    ) -> None:
        """Construct the single-determinant model from the stated parameters."""
        super().__init__()
        if isinstance(n_det, bool) or n_det != 1:
            raise ValueError("SlaterNet supports n_det=1 only in this package")
        if trainable_determinant_coefficients:
            raise ValueError("The single determinant has a fixed unit coefficient")
        self.n_electrons = cell.n_electrons
        self.n_det = n_det
        self.d_model = d_model
        self.n_layers = n_layers
        self.order = order
        self.include_conjugates = include_conjugates
        self.trainable_determinant_coefficients = bool(
            trainable_determinant_coefficients
        )

        # Legacy compatibility: at order=1 with no conjugates, use the original
        # 2-feature ComplexPeriodicFeatures so existing checkpoints / numerics
        # remain bit-for-bit identical to the pre-extended path.
        if order == 1 and not include_conjugates:
            self.features = ComplexPeriodicFeatures(cell)
        else:
            self.features = ExtendedComplexPeriodicFeatures(
                cell, order=order, include_conjugates=include_conjugates,
            )

        self.embed_activation_kind = embed_activation
        self.embed_bias = embed_bias
        self.embedding_kind = embedding_kind
        self.per_block_kind = per_block_kind
        if embedding_kind == "complex":
            self.embed: nn.Module = ComplexLinear(
                self.features.out_dim, d_model, bias=embed_bias,
            )
        elif embedding_kind == "free":
            self.embed = FreeComplexLinear(
                self.features.out_dim, d_model, bias=embed_bias,
            )
        else:
            raise ValueError(
                f"embedding_kind must be 'complex' or 'free', "
                f"got {embedding_kind!r}."
            )
        if embed_activation == "none":
            self.embed_act: nn.Module = nn.Identity()
        elif embed_activation == "kerr":
            self.embed_act = KerrActivation(d_model)
        elif embed_activation == "split_tanh":
            self.embed_act = SplitTanhActivation()
        else:
            raise ValueError(
                f"embed_activation must be 'none', 'kerr', or 'split_tanh', "
                f"got {embed_activation!r}."
            )
        self.activation_kind = activation
        self.blocks = nn.ModuleList([
            ComplexMLP(d_model, activation=activation, linear_kind=per_block_kind)
            for _ in range(n_layers)
        ])

        scale = d_model ** -0.5
        self.w_re = nn.Parameter(
            torch.randn(n_det, cell.n_electrons, d_model) * scale
        )
        self.w_im = nn.Parameter(
            torch.randn(n_det, cell.n_electrons, d_model) * scale
        )
        determinant_coefficients = torch.ones(n_det) / n_det
        if self.trainable_determinant_coefficients:
            self.c = nn.Parameter(determinant_coefficients)
        else:
            self.register_buffer("c", determinant_coefficients)

        if init_real:
            self._zero_all_imaginary()
        elif init_imag_scale != 1.0:
            with torch.no_grad():
                for p in self.imag_params():
                    p.mul_(init_imag_scale)

    def _compute_orbitals(self, r: torch.Tensor) -> torch.Tensor:
        """Compute the per-determinant orbital matrix ``φ_jᵐ(r_i)``.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        torch.Tensor, shape (batch, n_det, N, N), complex
            Element ``[b, m, i, j]`` is orbital ``j`` of determinant ``m``
            evaluated at electron ``i``.
        """
        f = self.features(r)
        h = self.embed(f)
        # Optional embed-side nonlinearity (default Identity is a no-op so
        # this preserves bit-for-bit behaviour when embed_activation="none").
        h = self.embed_act(h)
        for block in self.blocks:
            h = block(h)

        # Materialize ``w`` with its conjugate already baked in (rather
        # than calling ``.conj()`` at the end) so no conj-view sneaks
        # into the autograd graph — ``torch.func.jacrev`` in
        # :func:`kinetic_energy` lacks a batching rule for that view.
        w_conj = torch.complex(self.w_re, -self.w_im)
        phi = torch.einsum("bid,mjd->bmij", h, w_conj)
        return phi

    def log_psi(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``log|Ψ|`` and the complex phase of ``Ψ`` separately.

        Uses ``torch.linalg.slogdet`` and factors the largest ``logdet``
        out of the determinant sum to avoid over/underflow.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        log_psi_abs : torch.Tensor, shape (batch,), real
        phase_psi : torch.Tensor, shape (batch,), complex, unit modulus
        """
        phi = self._compute_orbitals(r)
        signs, logdets = torch.linalg.slogdet(phi)

        max_logdet = logdets.max(dim=-1, keepdim=True).values
        det_vals = signs * torch.exp(logdets - max_logdet)
        c = self.c.to(det_vals.dtype)
        psi = (c * det_vals).sum(dim=-1)

        abs_psi = torch.abs(psi)
        log_psi_abs = torch.log(abs_psi) + max_logdet.squeeze(-1)
        phase_psi = psi / (abs_psi + 1e-30)
        return log_psi_abs, phase_psi

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Return ``log|Ψ|`` only (sufficient for Metropolis sampling).

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        torch.Tensor, shape (batch,), real
        """
        log_psi_abs, _ = self.log_psi(r)
        return log_psi_abs

    def count_parameters(self) -> dict:
        """Return a parameter-count breakdown by component (complex = 2 reals).

        Follows the accounting in ``docs/complex_slater_net_plan.md`` §4.4
        and the Experiment-G H5 extension::

            # CR-constrained ComplexLinear:
            embedding_complex   = 2 · M · d_model              # W_re + W_im
            mlp_block_complex   = 2 · d_model · (d_model + 1) + (Kerr r0 ? d : 0)

            # Unconstrained FreeComplexLinear (4 independent real W matrices):
            embedding_free      = 4 · M · d_model
            mlp_block_free      = 4 · d_model · d_model + 2 · d_model
                                  + (Kerr r0 ? d : 0)

        Bias contributions (``+ 2 · d_model``) are added once per layer when
        the corresponding ``bias`` flag is set, regardless of layer kind.

        Returns
        -------
        dict
            Keys: ``embedding``, ``mlp``, ``projections``, ``c_coeffs``,
            ``total``.
        """
        M = self.features.out_dim
        d = self.d_model
        n_layers = self.n_layers
        n_el = self.n_electrons
        n_det = self.n_det

        # Embedding: W matrices (CR-tied or 4 independent) + optional bias
        # + optional Kerr r0 when embed_activation="kerr".
        if self.embedding_kind == "free":
            embedding = 4 * M * d
        else:
            embedding = 2 * M * d
        if self.embed_bias:
            embedding += 2 * d
        if isinstance(self.embed_act, KerrActivation):
            embedding += d

        # Per-block linear weight count: 2*d*d (CR-tied) or 4*d*d (free),
        # always with a bias of 2*d (ComplexMLP allocates ComplexLinear
        # with default bias=True, and FreeComplexLinear is constructed
        # explicitly with bias=True). Plus the optional Kerr r0.
        if self.per_block_kind == "free":
            mlp_w = 4 * d * d
        else:
            mlp_w = 2 * d * d
        block_has_r0 = bool(self.blocks) and isinstance(
            self.blocks[0].activation, KerrActivation,
        )
        mlp_block = mlp_w + 2 * d + (d if block_has_r0 else 0)
        mlp = n_layers * mlp_block
        projections = 2 * d * n_el * n_det
        c_coeffs = n_det
        total = embedding + mlp + projections + c_coeffs

        return {
            "embedding": embedding,
            "mlp": mlp,
            "projections": projections,
            "c_coeffs": c_coeffs,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Curriculum API (Plan §3 Experiment B). Mirrors FreeComplexPsiformer.
    # ------------------------------------------------------------------

    def imag_params(self) -> list[nn.Parameter]:
        """Every imaginary-valued trainable tensor in the model.

        For CR-constrained :class:`ComplexLinear` layers, the imaginary
        parameters are the ``W_im`` weight matrix (and ``b_im`` if
        present): zeroing them puts the layer on the real submanifold.

        For unconstrained :class:`FreeComplexLinear` layers, the analogue
        is the pair of cross-coupling matrices ``W_AB`` (Im → Re) and
        ``W_BA`` (Re → Im): zeroing both puts the layer on the
        decoupled-real-and-imag submanifold (the same submanifold where
        a CR-constrained ``W_im = 0`` lands). ``b_im`` is included when
        present for the same reason.

        The per-layer Kerr threshold ``activation.r0`` is also included
        — it is a modulus cutoff that lives on the imaginary-channel
        side of the parameter split. The orbital-head ``w_im`` is
        intentionally excluded — same convention as
        :meth:`FreeComplexPsiformer.imag_params`.
        """
        tensors: list[nn.Parameter] = []
        if isinstance(self.embed, FreeComplexLinear):
            tensors.append(self.embed.W_AB)
            tensors.append(self.embed.W_BA)
        else:
            tensors.append(self.embed.W_im)
        if self.embed.b_im is not None:
            tensors.append(self.embed.b_im)
        # Embed-side Kerr (when embed_activation="kerr") owns its own r0
        # threshold; SplitTanh and Identity have no learnable parameters.
        if isinstance(self.embed_act, KerrActivation):
            tensors.append(self.embed_act.r0)
        for block in self.blocks:
            if isinstance(block.linear, FreeComplexLinear):
                tensors.append(block.linear.W_AB)
                tensors.append(block.linear.W_BA)
            else:
                tensors.append(block.linear.W_im)
            if block.linear.b_im is not None:
                tensors.append(block.linear.b_im)
            # Only Kerr blocks own a learnable r0 threshold; SplitTanh
            # blocks have no parameters and therefore no imag entry here.
            if isinstance(block.activation, KerrActivation):
                tensors.append(block.activation.r0)
        return tensors

    def _zero_all_imaginary(self) -> None:
        """Zero every tensor returned by :meth:`imag_params`."""
        with torch.no_grad():
            for p in self.imag_params():
                p.zero_()

    def freeze_all_imaginary(self) -> None:
        """Set ``requires_grad=False`` on every imaginary-valued tensor.

        Combined with a prior :meth:`_zero_all_imaginary` (or
        ``init_real=True``), and optionally
        :meth:`init_embedding_to_match_real`, this locks the model on
        the SlaterNet-equivalent submanifold during the Phase-A
        curriculum step. Release with :meth:`unfreeze_all_imaginary`.
        """
        for p in self.imag_params():
            p.requires_grad_(False)

    def unfreeze_all_imaginary(self) -> None:
        """Re-enable gradient flow through every imaginary-valued tensor."""
        for p in self.imag_params():
            p.requires_grad_(True)

    def init_embedding_to_match_real(self, W_real_target: torch.Tensor) -> None:
        """CR-symmetric init of the order=1 embedding from a real target.

        Sets ``self.embed.W_re`` and ``self.embed.W_im`` so that, applied
        to the four order=1 complex plane-wave columns produced by
        :class:`ExtendedComplexPeriodicFeatures(order=1, include_conjugates=True)`,
        the embedding output has

            Re(h_complex) ≡ W_real_target @ [sin(G₁·r), sin(G₂·r),
                                             cos(G₁·r), cos(G₂·r)]
            Im(h_complex) ≡ 0

        for every input ``r``. This is the "CR-symmetric trick" of plan
        §3.1: the imaginary half of the complex weight is non-zero by
        design — it pairs with the imaginary half of ``exp(±i k·r)`` to
        contribute additional real terms — so the model is on the real
        submanifold in *output space* even though ``embed.W_im`` is not
        zero in *parameter space*.

        Parameters
        ----------
        W_real_target : torch.Tensor, shape (d_model, 4)
            The 4 columns must correspond to ``RealPeriodicFeatures``'s
            order:
            ``[sin(G₁·r), sin(G₂·r), cos(G₁·r), cos(G₂·r)]``.
            (This matches ``SlaterNet.net[0].weight`` at order=1 with no
            transpose.)

        Notes
        -----
        Requires ``order == 1`` and ``include_conjugates == True`` so the
        embedding sees the conjugate-extended four-feature basis. The
        complex feature column ordering at this configuration is
        ``[exp(-iG₁·r), exp(-iG₂·r), exp(+iG₂·r), exp(+iG₁·r)]`` (shell-
        sorted by ``ExtendedComplexPeriodicFeatures``); the algebra
        below is derived in this ordering.
        """
        if self.order != 1 or not self.include_conjugates:
            raise ValueError(
                f"init_embedding_to_match_real requires order=1 and "
                f"include_conjugates=True, got order={self.order}, "
                f"include_conjugates={self.include_conjugates}."
            )
        if isinstance(self.embed, FreeComplexLinear):
            raise ValueError(
                "init_embedding_to_match_real assumes a CR-constrained "
                "ComplexLinear embedding (W_re/W_im); the FreeComplexLinear "
                "embed (W_AA/W_AB/W_BA/W_BB) does not enjoy the same "
                "conjugate-pair algebra. Use embedding_kind='complex' "
                "for the curriculum experiments."
            )
        if W_real_target.shape != (self.d_model, 4):
            raise ValueError(
                f"W_real_target must have shape (d_model={self.d_model}, 4), "
                f"got {tuple(W_real_target.shape)}."
            )

        target = W_real_target.detach().to(
            dtype=self.embed.W_re.dtype, device=self.embed.W_re.device,
        )
        s_G1 = target[:, 0]   # sin(G₁·r) coefficients per output neuron
        s_G2 = target[:, 1]
        c_G1 = target[:, 2]
        c_G2 = target[:, 3]

        # Complex feature columns at order=1, include_conjugates=True
        # in shell-sorted order: k0=-G1, k1=-G2, k2=+G2, k3=+G1.
        with torch.no_grad():
            self.embed.W_re.zero_()
            self.embed.W_im.zero_()
            # Re(h) = (W_re[:,0]+W_re[:,3]) cos(G₁) + (W_re[:,1]+W_re[:,2]) cos(G₂)
            #       + (W_im[:,0]-W_im[:,3]) sin(G₁) + (W_im[:,1]-W_im[:,2]) sin(G₂)
            # Im(h) = (W_re[:,3]-W_re[:,0]) sin(G₁) + (W_re[:,2]-W_re[:,1]) sin(G₂)
            #       + (W_im[:,0]+W_im[:,3]) cos(G₁) + (W_im[:,1]+W_im[:,2]) cos(G₂)
            # Im(h) ≡ 0 ⇒ W_re[:,3]=W_re[:,0], W_re[:,2]=W_re[:,1],
            #             W_im[:,3]=-W_im[:,0], W_im[:,2]=-W_im[:,1]
            # Match Re(h) to target ⇒ 2 W_re[:,0]=c_G1, 2 W_re[:,1]=c_G2,
            #                          2 W_im[:,0]=s_G1, 2 W_im[:,1]=s_G2.
            self.embed.W_re[:, 0] = c_G1 / 2.0
            self.embed.W_re[:, 3] = c_G1 / 2.0
            self.embed.W_re[:, 1] = c_G2 / 2.0
            self.embed.W_re[:, 2] = c_G2 / 2.0
            self.embed.W_im[:, 0] = s_G1 / 2.0
            self.embed.W_im[:, 3] = -s_G1 / 2.0
            self.embed.W_im[:, 1] = s_G2 / 2.0
            self.embed.W_im[:, 2] = -s_G2 / 2.0
