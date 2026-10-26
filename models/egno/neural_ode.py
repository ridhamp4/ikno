"""
Neural ODE rollout for E-GNO.

Embeds the discovered sparse operator into a Neural ODE for stable
multi-step trajectory prediction. Uses the adjoint method for
memory-efficient backpropagation.

Section 4.6 of the paper.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class EquivariantResidualNet(nn.Module):
    """
    Small equivariant network for the residual ε_i(t) in Eq. 12.
    Maps type-1 features to type-1 corrections.
    """

    def __init__(self, hidden_channels: int = 64, field_dim: int = 3):
        super().__init__()
        self.field_dim = field_dim
        if field_dim % 3 != 0:
            raise ValueError('Residual state must contain packed 3D vectors')
        self.n_vector_fields = field_dim // 3
        n_invariants = self.n_vector_fields * (self.n_vector_fields + 1) // 2
        # Invariant features -> a scalar mixing matrix between vector fields.
        self.scalar_net = nn.Sequential(
            nn.Linear(n_invariants, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, self.n_vector_fields ** 2),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (N, 3) vector field

        Returns:
            residual: (N, 3) equivariant correction
        """
        n = u.shape[0]
        vectors = u.reshape(n, self.n_vector_fields, 3)
        gram = torch.einsum('nkd,nld->nkl', vectors, vectors)
        invariants = torch.cat([
            gram[:, i, j:j + 1]
            for i in range(self.n_vector_fields)
            for j in range(i, self.n_vector_fields)
        ], dim=-1)
        mixing = self.scalar_net(invariants).reshape(
            n, self.n_vector_fields, self.n_vector_fields)
        residual = torch.einsum('nab,nbd->nad', mixing, vectors)
        return residual.reshape(n, self.field_dim)


class NeuralODEFunction(nn.Module):
    """
    The ODE function f(u, t) = Θ^[l*](u) ξ* + ε(u, t).

    This wraps the E-GNO pipeline (encoder -> MS-SHMP -> E-DOL -> E-SINDy)
    into a callable ODE right-hand side.
    """

    def __init__(self, egno_forward_fn, residual_net: Optional[EquivariantResidualNet] = None):
        """
        Args:
            egno_forward_fn: callable(u) -> du_dt_pred that runs the full
                E-GNO pipeline (encoder, MS-SHMP, E-DOL, E-SINDy)
            residual_net: optional residual correction network
        """
        super().__init__()
        self.egno_forward_fn = egno_forward_fn
        self.residual_net = residual_net

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        ODE right-hand side: du/dt = F(u).

        Args:
            t: scalar time (unused if autonomous ODE)
            u: (N, 3) current field state

        Returns:
            du_dt: (N, 3) time derivative
        """
        du_dt = self.egno_forward_fn(u)

        if self.residual_net is not None:
            du_dt = du_dt + self.residual_net(u)

        return du_dt


class NeuralODERollout(nn.Module):
    """
    Neural ODE rollout using simple RK4 integration.

    Integrates the E-GNO dynamics forward in time for multi-step prediction.
    Supports backpropagation through the solver.
    """

    def __init__(self, field_dim: int = 3, use_residual: bool = True,
                 hidden_channels: int = 64):
        super().__init__()
        self.field_dim = field_dim
        self.use_residual = use_residual

        if use_residual:
            self.residual_net = EquivariantResidualNet(
                hidden_channels=hidden_channels,
                field_dim=field_dim,
            )
        else:
            self.residual_net = None

    def rk4_step(self, f, u: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Single RK4 step.

        Args:
            f: callable(t, u) -> du_dt
            u: (N, d) current state
            dt: time step size

        Returns:
            u_next: (N, d) state at t + dt
        """
        t_dummy = torch.tensor(0.0, device=u.device)
        k1 = f(t_dummy, u)
        k2 = f(t_dummy, u + 0.5 * dt * k1)
        k3 = f(t_dummy, u + 0.5 * dt * k2)
        k4 = f(t_dummy, u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def euler_step(self, f, u: torch.Tensor, dt: float) -> torch.Tensor:
        """Simple Euler step for faster training."""
        t_dummy = torch.tensor(0.0, device=u.device)
        return u + dt * f(t_dummy, u)

    def rollout(self, ode_fn: NeuralODEFunction, u0: torch.Tensor,
                dt: float, n_steps: int, method: str = 'rk4'
                ) -> torch.Tensor:
        """
        Integrate the ODE forward for n_steps.

        Args:
            ode_fn: the ODE right-hand side function
            u0: (N, d) initial condition
            dt: time step size
            n_steps: number of steps to integrate
            method: 'rk4' or 'euler'

        Returns:
            trajectory: (n_steps+1, N, d) full trajectory including u0
        """
        step_fn = self.rk4_step if method == 'rk4' else self.euler_step

        trajectory = [u0]
        u = u0
        for _ in range(n_steps):
            u = step_fn(ode_fn, u, dt)
            trajectory.append(u)

        return torch.stack(trajectory, dim=0)

    def forward(self, egno_forward_fn, u0: torch.Tensor,
                dt: float, n_steps: int, method: str = 'rk4'
                ) -> torch.Tensor:
        """
        Full rollout with the E-GNO operator.

        Args:
            egno_forward_fn: callable(u) -> du_dt
            u0: (N, d) initial condition
            dt: time step size
            n_steps: number of integration steps
            method: integration method ('rk4' or 'euler')

        Returns:
            trajectory: (n_steps+1, N, d)
        """
        ode_fn = NeuralODEFunction(egno_forward_fn, self.residual_net)
        return self.rollout(ode_fn, u0, dt, n_steps, method)
