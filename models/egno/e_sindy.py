"""
Equivariant SINDy (E-SINDy) for E-GNO.

Sparse identification of the governing PDE from the type-consistent
equivariant library. Uses iterative hard thresholding (IHT) for
exact sparsity enforcement.

Section 4.5 of the paper.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional


class ESINDy(nn.Module):
    """
    Equivariant Sparse Identification of Nonlinear Dynamics.

    Solves:
        ξ* = argmin_ξ (1/NS) Σ_{i,s} ||∂_t u(x_i, t_s) - Θ^[l*]_i(t_s) ξ||²
             + λ ||ξ||_1

    with type-consistency constraint (only type-l* library terms are used).
    """

    def __init__(self, n_library_terms: int, hidden_channels: int = 64,
                 lambda_sparse: float = 0.1, threshold: float = 0.05,
                 n_iht_steps: int = 10, lr_iht: float = 0.01):
        """
        Args:
            n_library_terms: maximum number of library terms P
            hidden_channels: channel dimension C
            lambda_sparse: L1 sparsity weight
            threshold: hard threshold δ for IHT
            n_iht_steps: number of IHT iterations
            lr_iht: step size for IHT gradient steps
        """
        super().__init__()
        self.n_library_terms = n_library_terms
        self.hidden_channels = hidden_channels
        self.lambda_sparse = lambda_sparse
        self.threshold = threshold
        self.n_iht_steps = n_iht_steps
        self.lr_iht = lr_iht

        # Learnable coefficients ξ (initialized to small values)
        self.xi = nn.Parameter(torch.randn(n_library_terms) * 0.1)

        # Mask for active terms (updated during IHT)
        self.register_buffer('active_mask', torch.ones(n_library_terms, dtype=torch.bool))

    def forward(self, theta: torch.Tensor, du_dt: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Apply the sparse operator: Θ^[l*] @ ξ*

        Args:
            theta: (N, P, D) type-consistent library evaluated at nodes
            du_dt: (N, D) optional time derivative for computing residual

        Returns:
            prediction: (N, D) = Θ @ ξ (the predicted RHS of the PDE)
        """
        P = theta.shape[1]
        xi = self.xi[:P] * self.active_mask[:P].float()

        # Θ @ ξ: weighted sum of library terms
        # theta: (N, P, D), xi: (P,)
        prediction = torch.einsum('npd,p->nd', theta, xi)
        return prediction

    def compute_loss(self, theta: torch.Tensor, du_dt: torch.Tensor
                     ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the E-SINDy loss (Eq. 8).

        Args:
            theta: (N, P, D) library matrix
            du_dt: (N, D) time derivative target

        Returns:
            loss: scalar loss value
            info: dict with diagnostics
        """
        P = theta.shape[1]
        xi = self.xi[:P]

        # Prediction
        pred = torch.einsum('npd,p->nd', theta, xi * self.active_mask[:P].float())

        # Reconstruction loss
        recon_loss = ((du_dt - pred) ** 2).mean()

        # L1 sparsity
        l1_loss = self.lambda_sparse * torch.abs(xi).sum()

        total_loss = recon_loss + l1_loss

        # Diagnostics
        n_active = (torch.abs(xi) > self.threshold).sum().item()
        info = {
            'recon_loss': recon_loss.item(),
            'l1_loss': l1_loss.item(),
            'n_active_terms': n_active,
            'xi': xi.detach().cpu(),
        }

        return total_loss, info

    @torch.no_grad()
    def iht_step(self, theta: torch.Tensor, du_dt: torch.Tensor):
        """
        Perform one step of Iterative Hard Thresholding (Algorithm 1).

        Updates self.xi in-place and updates the active_mask.

        Args:
            theta: (N, P, D) library matrix
            du_dt: (N, D) time derivative target
        """
        P = theta.shape[1]
        xi = self.xi[:P].clone()

        # Gradient step
        pred = torch.einsum('npd,p->nd', theta, xi)
        residual = du_dt - pred  # (N, D)
        # Gradient of ||du_dt - Θξ||² w.r.t. ξ
        grad = -2.0 * torch.einsum('npd,nd->p', theta, residual) / theta.shape[0]
        xi = xi - self.lr_iht * grad

        # Hard threshold
        mask = torch.abs(xi) >= self.threshold
        xi = xi * mask.float()

        self.xi.data[:P] = xi
        self.active_mask[:P] = mask

    @torch.no_grad()
    def run_iht(self, theta: torch.Tensor, du_dt: torch.Tensor):
        """
        Run full IHT procedure (Algorithm 1 in paper).

        Args:
            theta: (N, P, D) library matrix
            du_dt: (N, D) time derivative target
        """
        P = theta.shape[1]

        # Step 1: Initialize with least-squares solution
        # ξ = (Θ^T Θ)^{-1} Θ^T du_dt
        # Reshape theta to (N*D, P) and du_dt to (N*D,)
        N, _, D = theta.shape
        Theta_flat = theta.reshape(N * D, P)
        y_flat = du_dt.reshape(N * D)

        try:
            xi_ls = torch.linalg.lstsq(Theta_flat, y_flat).solution
        except Exception:
            xi_ls = torch.zeros(P, device=theta.device)

        self.xi.data[:P] = xi_ls

        # Step 2: IHT iterations
        for _ in range(self.n_iht_steps):
            self.iht_step(theta, du_dt)

    def get_discovered_pde(self, term_names: List[str]) -> dict:
        """
        Return the discovered PDE in symbolic form.

        Args:
            term_names: list of library term names (from E-DOL)

        Returns:
            dict mapping term_name -> coefficient value for active terms
        """
        P = len(term_names)
        xi = self.xi[:P].detach().cpu()
        mask = self.active_mask[:P].cpu()

        pde = {}
        for i, name in enumerate(term_names):
            if mask[i] and abs(xi[i].item()) > self.threshold:
                pde[name] = xi[i].item()

        return pde

    def sparsity_loss(self) -> torch.Tensor:
        """Return L1 sparsity penalty on ξ."""
        return self.lambda_sparse * torch.abs(self.xi).sum()

    def get_active_terms_count(self) -> int:
        """Return number of active (non-zero) terms."""
        return (torch.abs(self.xi) > self.threshold).sum().item()
