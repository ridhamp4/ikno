"""
EGNN baseline adapter.

Adapts the existing EGNN code (baselines/egnn-main/) for PDE prediction
on our datasets. EGNN is E(n)-equivariant but not SO(3)-typed.

Reference: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021.
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Tuple

# Import the EGNN model using importlib to avoid conflicts with our models package
import importlib.util
_egnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'egnn-main', 'models', 'egnn_clean', 'egnn_clean.py')
if os.path.exists(_egnn_path):
    _spec = importlib.util.spec_from_file_location("egnn_clean_module", _egnn_path)
    _egnn_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_egnn_module)
    EGNN_Base = _egnn_module.EGNN
else:
    # Fallback: define a minimal EGNN inline
    EGNN_Base = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Avoid circular import by importing graph_utils directly
_graph_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'models', 'graph_utils.py')
_spec2 = importlib.util.spec_from_file_location("graph_utils_module", _graph_utils_path)
_graph_utils = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_graph_utils)
radius_graph = _graph_utils.radius_graph


class EGNNBaseline(nn.Module):
    """
    EGNN baseline for PDE prediction on irregular meshes.

    Uses the E(n)-equivariant GNN to predict du/dt from current state
    and coordinates.
    """

    def __init__(self, field_dim: int = 3, hidden_dim: int = 128,
                 n_layers: int = 4, r_max: float = 0.1):
        super().__init__()
        self.field_dim = field_dim
        self.hidden_dim = hidden_dim
        self.r_max = r_max

        self.egnn = EGNN_Base(
            in_node_nf=field_dim,
            hidden_nf=hidden_dim,
            out_node_nf=field_dim,
            in_edge_nf=1,  # edge distance
            n_layers=n_layers,
            residual=True,
            attention=True,
            normalize=True,
        )

        # Learnable output scale (helps match target magnitude quickly)
        self.output_scale = nn.Parameter(10.0 * torch.ones(field_dim))

    def forward(self, u: torch.Tensor, coords: torch.Tensor
                ) -> torch.Tensor:
        """
        Predict du/dt using EGNN.

        Args:
            u: (N, field_dim) current field values (used as node features)
            coords: (N, 3) node coordinates

        Returns:
            du_dt: (N, field_dim) predicted time derivative
        """
        # Build radius graph
        edge_index, edge_vec = radius_graph(coords, 0.0, self.r_max)

        if edge_index.shape[1] == 0:
            return torch.zeros_like(u)

        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        edges = [edge_index[0], edge_index[1]]

        # EGNN forward
        h_out, x_out = self.egnn(u, coords, edges, edge_dist)

        # h_out is the predicted du/dt
        return h_out * self.output_scale

    def compute_loss(self, u_traj: torch.Tensor, coords: torch.Tensor,
                     dt: float) -> torch.Tensor:
        """One-step prediction loss (MSE on du/dt)."""
        u0 = u_traj[0]
        u1_true = u_traj[1]
        du_dt_true = (u1_true - u0) / dt
        du_dt_pred = self.forward(u0, coords)
        return ((du_dt_pred - du_dt_true) ** 2).mean()

    def rollout(self, u0: torch.Tensor, coords: torch.Tensor,
                dt: float, n_steps: int) -> torch.Tensor:
        """Multi-step rollout."""
        trajectory = [u0]
        u = u0
        for _ in range(n_steps):
            du_dt = self.forward(u, coords)
            u = u + dt * du_dt
            trajectory.append(u)
        return torch.stack(trajectory, dim=0)
