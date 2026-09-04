"""Metropolis-Hastings sampler for ``|Ψ(R)|²`` on a periodic supercell.
"""

import torch

from ..systems.supercell import Supercell


class MetropolisSampler:
    """Batched Metropolis-Hastings sampler with isotropic Gaussian proposals.

    Maintains an internal acceptance counter that feeds
    :meth:`adapt_step_size`, which tunes the proposal scale toward a
    target acceptance rate (default 50 %).

    Attributes
    ----------
    model : torch.nn.Module
        Provides ``log|Ψ|(r)`` via ``model(r)``.
    cell : Supercell
        Used for uniform initialisation and periodic-image wrapping.
    step_size : float
        Standard deviation of the Gaussian proposal (in effective Bohr).
    device : str or torch.device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cell: Supercell,
        step_size: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        """Bind the sampler to a model and initialise counters.

        Parameters
        ----------
        model : torch.nn.Module
            Wavefunction returning ``log|Ψ|`` on a batched coordinate tensor.
        cell : Supercell
            Simulation cell for wrapping and initialisation.
        step_size : float, default 0.1
            Initial proposal standard deviation.
        device : str or torch.device, default 'cpu'
            Device on which walkers live.
        """
        self.model = model
        self.cell = cell
        self.step_size = float(step_size)
        self.device = device
        self._acceptance_count = 0
        self._proposal_count = 0

    @torch.no_grad()
    def initialize(self, batch_size: int) -> torch.Tensor:
        """Sample ``batch_size`` walkers uniformly inside the supercell.

        Parameters
        ----------
        batch_size : int
            Number of parallel walkers.

        Returns
        -------
        torch.Tensor, shape (batch_size, n_electrons, 2), real
            Cartesian initial positions.
        """
        L = self.cell.L.to(self.device)
        u = torch.rand(
            batch_size, self.cell.n_electrons, 2, dtype=L.dtype, device=self.device
        )
        return u @ L

    @torch.no_grad()
    def step(
        self, r: torch.Tensor, log_psi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Propose one Gaussian move per walker and accept/reject.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2)
            Current walker positions.
        log_psi : torch.Tensor, shape (batch,)
            ``log|Ψ|`` cached from the previous step.

        Returns
        -------
        r_new : torch.Tensor, shape (batch, N, 2)
        log_psi_new : torch.Tensor, shape (batch,)
        acceptance_fraction : float
            Fraction of walkers that accepted this step.
        """
        batch = r.shape[0]
        proposal = self._wrap(r + self.step_size * torch.randn_like(r))

        log_psi_prop = self.model(proposal)
        log_accept = 2.0 * (log_psi_prop - log_psi)
        u = torch.rand(batch, device=r.device, dtype=log_psi.dtype)
        accept = torch.log(u) < log_accept

        r_new = torch.where(accept[:, None, None], proposal, r)
        log_psi_new = torch.where(accept, log_psi_prop, log_psi)

        self._acceptance_count += int(accept.sum().item())
        self._proposal_count += batch

        return r_new, log_psi_new, accept.float().mean().item()

    def _wrap(self, r: torch.Tensor) -> torch.Tensor:
        """Apply the minimum-image convention (reduced-coord ``mod 1``).

        Parameters
        ----------
        r : torch.Tensor, shape (..., 2)

        Returns
        -------
        torch.Tensor, shape (..., 2)
            Coordinates wrapped into the supercell.
        """
        u = self.cell.to_reduced(r)
        return u @ self.cell.L.to(r.device)

    @torch.no_grad()
    def burn_in(
        self, r: torch.Tensor, n_steps: int = 1000
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``n_steps`` Metropolis updates without recording statistics.

        Parameters
        ----------
        r : torch.Tensor, shape (batch, N, 2)
            Initial walker positions.
        n_steps : int, default 1000

        Returns
        -------
        r_final : torch.Tensor, shape (batch, N, 2)
        log_psi_final : torch.Tensor, shape (batch,)
        """
        log_psi = self.model(r)
        for _ in range(n_steps):
            r, log_psi, _ = self.step(r, log_psi)
        return r, log_psi

    def adapt_step_size(self, target_rate: float = 0.5) -> None:
        """Adjust ``step_size`` toward a target acceptance rate.

        Uses the cumulative acceptance fraction since the last call and
        resets the internal counters. No-op if no proposals have been
        made since the last adaptation.

        Parameters
        ----------
        target_rate : float, default 0.5
            Desired acceptance fraction.
        """
        if self._proposal_count == 0:
            return
        rate = self._acceptance_count / self._proposal_count
        self.step_size *= 1.0 + 0.1 * (rate - target_rate)
        self._acceptance_count = 0
        self._proposal_count = 0
