"""
Graph Network Simulator (GNS) baseline.

Standard message-passing GNN for simulating physical dynamics on
irregular meshes. No equivariance enforcement.

Reference: Sanchez-Gonzalez et al., "Learning to simulate complex physics
with graph networks", ICML 2020.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
import os

import importlib.util
_graph_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'models', 'graph_utils.py')
_spec = importlib.util.spec_from_file_location("graph_utils_module", _graph_utils_path)
_graph_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_graph_utils)
radius_graph = _graph_utils.radius_graph


class MLP(nn.Module):
    """Simple MLP block."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, activation: nn.Module = nn.SiLU()):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GNSLayer(nn.Module):
    """Single GNS message-passing layer."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.edge_mlp = MLP(2 * node_dim + edge_dim, hidden_dim, hidden_dim)
        self.node_mlp = MLP(node_dim + hidden_dim, node_dim, hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        # Edge model
        edge_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        edge_msg = self.edge_mlp(edge_input)

        # Aggregate
        agg = torch.zeros(h.shape[0], edge_msg.shape[1],
                          device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_msg), edge_msg)

        # Node update
        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(node_input)
        return h + h_new  # residual connection


class GNS(nn.Module):
    """
    Graph Network Simulator.

    Encode -> Process (message passing) -> Decode architecture.
    """

    def __init__(self, field_dim: int = 3, hidden_dim: int = 128,
                 n_layers: int = 6, r_max: float = 0.1):
        super().__init__()
        self.field_dim = field_dim
        self.hidden_dim = hidden_dim
        self.r_max = r_max

        # Encoder
        self.node_encoder = MLP(field_dim, hidden_dim, hidden_dim)
        self.edge_encoder = MLP(4, hidden_dim, hidden_dim)  # 3 (direction) + 1 (distance)

        # Processor
        self.layers = nn.ModuleList([
            GNSLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        # Decoder
        self.decoder = MLP(hidden_dim, field_dim, hidden_dim)

        # Learnable output scale (helps match target magnitude quickly)
        self.output_scale = nn.Parameter(10.0 * torch.ones(field_dim))

    def forward(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Predict du/dt from current state.

        Args:
            u: (N, field_dim) current field
            coords: (N, 3) node coordinates

        Returns:
            du_dt: (N, field_dim) predicted time derivative
        """
        # Build graph
        edge_index, edge_vec = radius_graph(coords, 0.0, self.r_max)

        if edge_index.shape[1] == 0:
            return torch.zeros_like(u)

        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        edge_dir = edge_vec / edge_dist.clamp(min=1e-8)
        edge_attr = torch.cat([edge_dir, edge_dist], dim=-1)  # (E, 4)

        # Encode
        h = self.node_encoder(u)
        edge_feat = self.edge_encoder(edge_attr)

        # Process
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)

        # Decode
        du_dt = self.decoder(h) * self.output_scale
        return du_dt

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
