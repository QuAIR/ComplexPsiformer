"""Real PsiFormer with periodic electron features and real self-attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.periodic import RealPeriodicFeatures
from ..systems.supercell import Supercell


class RealMultiHeadAttention(nn.Module):
    """Real-valued multi-head dot-product self-attention.

    For each head ``h`` the operation is

        score[b, h, i, j] = q[b, h, i] . k[b, h, j]
        alpha[b, h, i, j] = softmax_j(score[b, h, i, j])
        out[b, h, i]      = Σ_j alpha[b, h, i, j] * v[b, h, j] / sqrt(d_val)

    Per-head outputs are concatenated along the feature axis and
    projected back to ``d_model`` with ``W_o``. All tensors are real.

    These equations describe ``attention_normalization='paper'``, the
    published setting. The optional ``standard_scaled`` branch instead
    scales logits by ``1/sqrt(d_attn)`` and leaves values unscaled.

    Attributes
    ----------
    n_heads : int
        Number of attention heads.
    d_attn : int
        Query/key dimension per head.
    d_val : int
        Value dimension per head.
    Wqkv : nn.Linear
        Fused Q/K/V projection (``d_model → n_heads * (2·d_attn + d_val)``),
        no bias. Packing the three projections into one GEMM halves the
        launch overhead on GPU at the cost of a single split afterwards;
        the total parameter count is identical to three separate linears.
    Wo : nn.Linear
        Output projection (``n_heads * d_val → d_model``), no bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_attn: int,
        d_val: int,
        attention_normalization: str = "paper",
    ) -> None:
        """Allocate the fused QKV projection and the output projection.

        Parameters
        ----------
        d_model : int
            Input/output embedding dimension.
        n_heads : int
            Number of independent attention heads.
        d_attn : int
            Query/key dimension per head.
        d_val : int
            Value dimension per head.
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_attn = d_attn
        self.d_val = d_val
        if attention_normalization not in ("standard_scaled", "paper"):
            raise ValueError(
                "attention_normalization must be 'standard_scaled' or 'paper', "
                f"got {attention_normalization!r}"
            )
        self.attention_normalization = attention_normalization
        self._qkv_out = n_heads * (2 * d_attn + d_val)
        self.Wqkv = nn.Linear(d_model, self._qkv_out, bias=False)
        self.Wo = nn.Linear(n_heads * d_val, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head real dot-product self-attention.

        Parameters
        ----------
        h : torch.Tensor, shape (batch, N, d_model), real
        key_padding_mask : torch.Tensor | None, shape (batch, N), bool
            Optional padding mask. ``True`` entries are excluded from the key
            axis of the softmax. No padding mask is used in the published
            full-electron attention configuration.

        Returns
        -------
        torch.Tensor, shape (batch, N, d_model), real
        """
        batch, n_el, _ = h.shape

        # One GEMM produces Q, K, V packed along the last axis; split after.
        qkv = self.Wqkv(h)
        q, k, v = qkv.split(
            [self.n_heads * self.d_attn, self.n_heads * self.d_attn, self.n_heads * self.d_val],
            dim=-1,
        )
        # Reshape into (batch, heads, N, d_*) so the matmuls below are
        # batched GEMMs dispatched directly to cuBLAS.
        q = q.view(batch, n_el, self.n_heads, self.d_attn).transpose(1, 2)
        k = k.view(batch, n_el, self.n_heads, self.d_attn).transpose(1, 2)
        v = v.view(batch, n_el, self.n_heads, self.d_val).transpose(1, 2)

        scale = self.d_attn**-0.5
        # score = (1/sqrt(d_attn)) · q · kᵀ ; matmul avoids einsum overhead.
        score = torch.matmul(q, k.transpose(-2, -1))
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, n_el):
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"({batch}, {n_el}), got {tuple(key_padding_mask.shape)}"
                )
            score = score.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(score.dtype).min,
            )
        if self.attention_normalization == "paper":
            alpha = F.softmax(score, dim=-1)
            out = torch.matmul(alpha, v) * (self.d_val ** -0.5)
        else:
            alpha = F.softmax(score * scale, dim=-1)
            out = torch.matmul(alpha, v)

        # (batch, n_heads, N, d_val) → (batch, N, n_heads * d_val) → d_model.
        out = out.transpose(1, 2).reshape(batch, n_el, self.n_heads * self.d_val)
        return self.Wo(out)


class RealAttentionBlock(nn.Module):
    """One real transformer block: multi-head attention + residual MLP stack.

    Implements the two update equations from Sec. III C of the paper

        f_i^{l+1} = h_i^l + W_o · concat_h[ SelfAttn_i^h({h_j^l}) ]
        h_i^{l+1} = f_i^{l+1} + tanh( W^{l+1} f_i^{l+1} + b^{l+1} )

    both with residual connections. The published configuration uses one
    per-block MLP layer, with its own residual.

    Attributes
    ----------
    attn : RealMultiHeadAttention
        Multi-head dot-product attention sub-layer.
    mlp_layers : nn.ModuleList[nn.Linear]
        ``n_mlp_per_block`` dense layers (``d_model → d_model``), each with
        bias; applied with a ``tanh`` residual update in :meth:`forward`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_attn: int,
        d_val: int,
        n_mlp_per_block: int = 1,
        attention_normalization: str = "paper",
    ) -> None:
        """Allocate the attention sub-layer and ``n_mlp_per_block`` dense layers.

        Parameters
        ----------
        d_model : int
            Input/output embedding dimension.
        n_heads : int
            Number of attention heads.
        d_attn : int
            Query/key dimension per head.
        d_val : int
            Value dimension per head.
        n_mlp_per_block : int, default 1
            Number of ``tanh``-residual dense layers between attention
            sub-layers. Paper uses 1; supported values are any ``≥ 0``.
        """
        super().__init__()
        self.attn = RealMultiHeadAttention(
            d_model,
            n_heads,
            d_attn,
            d_val,
            attention_normalization=attention_normalization,
        )
        self.mlp_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_mlp_per_block)])

    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one attention sub-layer followed by the residual MLP stack.

        Parameters
        ----------
        h : torch.Tensor, shape (batch, N, d_model), real
        key_padding_mask : torch.Tensor | None, shape (batch, N), bool
            Optional padding mask forwarded into the attention sub-layer.

        Returns
        -------
        torch.Tensor, shape (batch, N, d_model), real
        """
        f = h + self.attn(h, key_padding_mask=key_padding_mask)
        for layer in self.mlp_layers:
            f = f + torch.tanh(layer(f))
        return f


class RealPsiFormer(nn.Module):
    """Real attention streams, complex orbital readout, and a fixed determinant sum."""

    def __init__(
        self,
        cell: Supercell,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 2,
        d_attn: int = 16,
        d_val: int = 16,
        n_det: int = 4,
        n_mlp_per_block: int = 1,
        feature_max_harmonic: int = 1,
        feature_harmonic_mode: str = "axis",
        attention_normalization: str = "paper",
        trainable_determinant_coefficients: bool = False,
        determinant_combination: str = "sum",
    ) -> None:
        """Wire up features, embedding, attention stack, and orbital heads.

        Parameters
        ----------
        cell : Supercell
            Simulation supercell providing ``G`` and ``n_electrons``.
        d_model : int, default 64
            Hidden embedding dimension.
        n_layers : int, default 3
            Number of :class:`RealAttentionBlock` layers.
        n_heads : int, default 2
            Attention heads per block.
        d_attn : int, default 16
            Query/key dimension per head.
        d_val : int, default 16
            Value dimension per head.
        n_det : int, default 4
            Number of Slater determinants summed to form ``Ψ``.
        n_mlp_per_block : int, default 1
            Per-block MLP layers (see :class:`RealAttentionBlock`).
        feature_max_harmonic : int, default 1
            Highest reciprocal-lattice harmonic included in the input
            sine/cosine feature map.
        feature_harmonic_mode : {"axis", "half_space"}, default "axis"
            Which reciprocal-lattice harmonics to include.
        determinant_combination : {"mean", "sum"}, default "sum"
            Initial determinant weights in
            ``Psi = sum_m c_m det(Phi_m)``. ``"mean"`` preserves the legacy
            ``c_m = 1 / n_det`` initialization; ``"sum"`` uses ``c_m = 1``.
            The weights remain fixed when
            ``trainable_determinant_coefficients=False``.
        """
        super().__init__()
        self.n_electrons = cell.n_electrons
        self.n_det = n_det
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_attn = d_attn
        self.d_val = d_val
        self.n_mlp_per_block = n_mlp_per_block
        self.feature_max_harmonic = feature_max_harmonic
        self.feature_harmonic_mode = feature_harmonic_mode
        self.attention_normalization = attention_normalization
        self.trainable_determinant_coefficients = bool(
            trainable_determinant_coefficients
        )
        if determinant_combination not in {"mean", "sum"}:
            raise ValueError("determinant_combination must be 'mean' or 'sum'")
        self.determinant_combination = determinant_combination

        self.features = RealPeriodicFeatures(
            cell,
            max_harmonic=feature_max_harmonic,
            harmonic_mode=feature_harmonic_mode,
        )
        self.embed = nn.Linear(self.features.out_dim, d_model, bias=False)
        self.blocks = nn.ModuleList(
            [
                RealAttentionBlock(
                    d_model,
                    n_heads,
                    d_attn,
                    d_val,
                    n_mlp_per_block,
                    attention_normalization=attention_normalization,
                )
                for _ in range(n_layers)
            ]
        )

        scale = d_model**-0.5
        self.w_re = nn.Parameter(torch.randn(n_det, cell.n_electrons, d_model) * scale)
        self.w_im = nn.Parameter(torch.randn(n_det, cell.n_electrons, d_model) * scale)
        determinant_coefficients = torch.ones(n_det)
        if determinant_combination == "mean":
            determinant_coefficients = determinant_coefficients / n_det
        if self.trainable_determinant_coefficients:
            self.c = nn.Parameter(determinant_coefficients)
        else:
            self.register_buffer("c", determinant_coefficients)

    def _electron_streams(self, r: torch.Tensor) -> torch.Tensor:
        """Feature map + embedding + attention stack.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        torch.Tensor, shape (batch, N, d_model), real
        """
        f = self.features(r)
        h = self.embed(f)
        for block in self.blocks:
            h = block(h)
        return h

    def _orbitals(self, r: torch.Tensor) -> torch.Tensor:
        """Compute the per-determinant complex orbital matrix ``φ_jᵐ(r_i)``.

        Each orbital is assembled from two real projections of the real
        stream ``h_i^L``:
        ``φ_jᵐ(r_i) = w_re[m, j] · h_i^L + i · w_im[m, j] · h_i^L``.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        torch.Tensor, shape (batch, n_det, N, N), complex
            Row index is electron ``i``, column index is orbital ``j``;
            the convention matches :class:`RealSlaterNet` and
            :class:`ComplexPsiFormer` so the Slater determinant flips
            sign on a row permutation (antisymmetry).
        """
        h = self._electron_streams(r)
        # Fuse the real/imag orbital projections into one einsum: stack
        # the two weight tensors along a new leading axis so a single
        # kernel produces both parts and `torch.complex` just rewraps
        # them. Saves a kernel launch vs. the separate-einsum version.
        w_stack = torch.stack((self.w_re, self.w_im), dim=0)  # (2, n_det, N, d_model)
        phi_stack = torch.einsum("bid,rmjd->rbmij", h, w_stack)  # (2, b, n_det, N, N)
        return torch.complex(phi_stack[0], phi_stack[1])

    def log_psi(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``log|Ψ|`` and the complex phase of ``Ψ`` separately.

        The complex orbital matrices are evaluated with ``slogdet`` and
        combined using a stabilized sum of determinants.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2), real

        Returns
        -------
        log_psi_abs : torch.Tensor, shape (batch,), real
        phase_psi : torch.Tensor, shape (batch,), complex, unit modulus
        """
        phi = self._orbitals(r)
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
        """Break down the trainable-parameter count by component.

        Formula (paper Appendix C, assumes ``n_mlp_per_block == 1``):

            embedding   = feature_out_dim * d_model
            per_layer   = 2·n_heads·d_attn·d_model
                        + 2·n_heads·d_val ·d_model
                        + d_model·(d_model + 1)
            projections = 2 · d_model · n_electrons · n_det
            c_coeffs    = n_det
            total       = embedding + n_layers·per_layer
                        + projections + c_coeffs

        Returns
        -------
        dict
            Keys: ``'embedding'``, ``'attention'``, ``'projections'``,
            ``'c_coeffs'``, ``'total'``.
        """
        d = self.d_model
        Nh = self.n_heads
        da = self.d_attn
        dv = self.d_val
        N = self.n_electrons
        M = self.n_det
        L = self.n_layers

        embedding = self.features.out_dim * d
        per_layer = 2 * Nh * da * d + 2 * Nh * dv * d + d * (d + 1)
        projections = 2 * d * N * M
        c_coeffs = M if self.trainable_determinant_coefficients else 0
        total = embedding + L * per_layer + projections + c_coeffs

        return {
            "embedding": embedding,
            "attention": L * per_layer,
            "projections": projections,
            "c_coeffs": c_coeffs,
            "total": total,
        }
