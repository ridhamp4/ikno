"""
Physics-Informed Neural Network (PINN) baseline.

Standard MLP-based PINN that takes (x, t) as input and predicts
the field u(x, t). Physics is enforced through PDE residual loss.

Reference: Raissi et al., "PINNs: A deep learning framework for solving
forward and inverse problems", JCP 2019.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for vector-valued PDEs.

    Maps (x, t) -> u(x, t) and enforces PDE constraints via
    automatic differentiation.
    """

    def __init__(self, field_dim: int = 3, hidden_dim: int = 128,
                 n_layers: int = 5, activation: str = 'tanh'):
        super().__init__()
        self.field_dim = field_dim

        act_fn = {
            'tanh': nn.Tanh(),
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
        }[activation]

        # Input: (x, y, z, t) -> 4 dims
        layers = [nn.Linear(4, hidden_dim), act_fn]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
        layers.append(nn.Linear(hidden_dim, field_dim))

        self.net = nn.Sequential(*layers)

        # Initialize with Xavier
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict field value at spatial-temporal location.

        Args:
            x: (N, 3) spatial coordinates
            t: (N, 1) or scalar time

        Returns:
            u: (N, field_dim) predicted field
        """
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        xt = torch.cat([x, t], dim=-1)  # (N, 4)
        return self.net(xt)

    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor
                           ) -> Dict[str, torch.Tensor]:
        """
        Compute spatial and temporal derivatives via autograd.

        Args:
            x: (N, 3) coordinates (requires_grad=True)
            t: (N, 1) time (requires_grad=True)

        Returns:
            dict with 'u', 'u_t', 'u_x', 'u_xx', etc.
        """
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        u = self.forward(x, t)  # (N, d)

        derivs = {'u': u}

        # Time derivative: ∂u/∂t
        u_t = torch.zeros_like(u)
        for d in range(self.field_dim):
            grad_outputs = torch.zeros_like(u)
            grad_outputs[:, d] = 1.0
            grads = torch.autograd.grad(u, t, grad_outputs=grad_outputs,
                                         create_graph=True, retain_graph=True)[0]
            u_t[:, d] = grads.squeeze(-1)
        derivs['u_t'] = u_t

        # Spatial gradients: ∂u_d/∂x_j
        u_x = torch.zeros(u.shape[0], self.field_dim, 3, device=u.device)
        for d in range(self.field_dim):
            grad_outputs = torch.zeros_like(u)
            grad_outputs[:, d] = 1.0
            grads = torch.autograd.grad(u, x, grad_outputs=grad_outputs,
                                         create_graph=True, retain_graph=True)[0]
            u_x[:, d, :] = grads
        derivs['u_x'] = u_x

        # Laplacian: ∇²u (sum of second spatial derivatives)
        u_xx = torch.zeros_like(u)
        for d in range(self.field_dim):
            for j in range(3):
                grad_outputs = torch.zeros(u.shape[0], self.field_dim, 3,
                                           device=u.device)
                grad_outputs[:, d, j] = 1.0
                grads2 = torch.autograd.grad(u_x, x, grad_outputs=grad_outputs,
                                              create_graph=True, retain_graph=True)[0]
                u_xx[:, d] += grads2[:, j]
        derivs['u_xx'] = u_xx

        return derivs


class PINNBaseline(nn.Module):
    """
    PINN baseline wrapper with PDE-specific loss functions.
    """

    def __init__(self, pde_type: str = 'maxwell', field_dim: int = 3,
                 hidden_dim: int = 128, n_layers: int = 5,
                 lambda_pde: float = 1.0, lambda_data: float = 1.0):
        super().__init__()
        self.pde_type = pde_type
        self.field_dim = field_dim
        self.lambda_pde = lambda_pde
        self.lambda_data = lambda_data

        # For Maxwell: predict both E and B (6 components)
        out_dim = 6 if pde_type == 'maxwell' else field_dim
        if pde_type == 'elasticity':
            out_dim = 6  # u and v

        self.pinn = PINN(
            field_dim=out_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    def data_loss(self, coords: torch.Tensor, times: torch.Tensor,
                  u_true: torch.Tensor) -> torch.Tensor:
        """MSE loss on observed data points."""
        T, N, d = u_true.shape
        total_loss = torch.tensor(0.0, device=coords.device)

        for t_idx in range(T):
            t = times[t_idx].expand(N, 1)
            x = coords.requires_grad_(True)
            u_pred = self.pinn(x, t)[:, :d]
            total_loss = total_loss + ((u_pred - u_true[t_idx]) ** 2).mean()

        return total_loss / T

    def pde_residual_loss(self, coords: torch.Tensor, times: torch.Tensor,
                          n_collocation: int = 1000) -> torch.Tensor:
        """PDE residual loss at collocation points."""
        device = coords.device
        N = coords.shape[0]

        # Sample collocation points
        idx = torch.randint(0, N, (n_collocation,), device=device)
        t_idx = torch.randint(0, len(times), (n_collocation,), device=device)

        x_coll = coords[idx].requires_grad_(True)
        t_coll = times[t_idx].unsqueeze(-1).requires_grad_(True)

        derivs = self.pinn.compute_derivatives(x_coll, t_coll)
        u_t = derivs['u_t']
        u_xx = derivs['u_xx']
        u = derivs['u']

        if self.pde_type == 'maxwell':
            # ∂E/∂t = ∇×B, ∂B/∂t = -∇×E (simplified)
            # For PINN, we approximate with Laplacian-based residual
            residual = (u_t - u_xx) ** 2
        elif self.pde_type == 'navier_stokes':
            # ∂u/∂t = ν∇²u - (u·∇)u (simplified, pressure-free)
            nu = 0.005
            u_x = derivs['u_x']  # (N, 3, 3)
            advection = torch.einsum('nd,nde->ne', u[:, :3], u_x[:, :3, :])
            residual = (u_t[:, :3] - nu * u_xx[:, :3] + advection) ** 2
        elif self.pde_type == 'elasticity':
            # ∂²u/∂t² = cs²∇²u (simplified)
            cs2 = 1.0
            residual = (u_t[:, :3] - cs2 * u_xx[:, :3]) ** 2
        else:
            residual = (u_t - u_xx) ** 2

        return residual.mean()

    def compute_loss(self, coords: torch.Tensor, times: torch.Tensor,
                     u_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Total PINN loss = data + PDE residual."""
        data_loss = self.data_loss(coords, times, u_true)
        pde_loss = self.pde_residual_loss(coords, times)
        total = self.lambda_data * data_loss + self.lambda_pde * pde_loss

        info = {
            'data_loss': data_loss.item(),
            'pde_loss': pde_loss.item(),
            'total_loss': total.item(),
        }
        return total, info

    def predict(self, coords: torch.Tensor, t: float) -> torch.Tensor:
        """Predict field at given time."""
        t_tensor = torch.tensor(t, device=coords.device).expand(coords.shape[0], 1)
        with torch.no_grad():
            return self.pinn(coords, t_tensor)[:, :self.field_dim]
