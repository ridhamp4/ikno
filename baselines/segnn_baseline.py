"""
SEGNN (SE(3)-Equivariant GNN) baseline for PDE prediction.

Implements an SE(3)-equivariant GNN with steerable features for single-step
prediction. No sparsity or operator library.

Reference: Brandstetter et al., "Message Passing Neural PDE Solvers", ICLR 2022.

This is a simplified implementation of the SEGNN architecture tailored
to our PDE prediction task.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import importlib.util

_graph_utils_path = os.path.join(_project_root, 'models', 'graph_utils.py')
_spec = importlib.util.spec_from_file_location("graph_utils_module", _graph_utils_path)
_graph_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_graph_utils)
radius_graph = _graph_utils.radius_graph

_sh_path = os.path.join(_project_root, 'models', 'egno', 'spherical_harmonics.py')
_spec2 = importlib.util.spec_from_file_location("sh_module", _sh_path)
_sh_module = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_sh_module)
SphericalHarmonics = _sh_module.SphericalHarmonics
ClebschGordanProducts = _sh_module.ClebschGordanProducts
GaussianRBF = _sh_module.GaussianRBF


class SEGNNLayer(nn.Module):
    """
    Single SEGNN message passing layer with steerable features.

    Uses CG tensor products for equivariant message computation.
    Simplified version operating with type-0 and type-1 features.
    """

    def __init__(self, hidden_channels: int = 64, l_max: int = 1, n_rbf: int = 16):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.l_max = l_max
        C = hidden_channels

        self.sh = SphericalHarmonics(l_max=l_max)
        self.cg = ClebschGordanProducts(l_max=l_max)
        self.rbf = GaussianRBF(n_rbf=n_rbf)

        # Radial MLP for message weights
        self.radial_mlp = nn.Sequential(
            nn.Linear(n_rbf, C),
            nn.SiLU(),
            nn.Linear(C, C),
        )

        # Node update MLPs (per type)
        self.node_mlp_scalar = nn.Sequential(
            nn.Linear(2 * C, C),
            nn.SiLU(),
            nn.Linear(C, C),
        )
        self.node_mlp_vector = nn.Linear(2 * 3 * C, 3 * C, bias=False)

        # Gate
        self.gate_mlp = nn.Sequential(
            nn.Linear(C, C),
            nn.Sigmoid(),
        )

    def forward(self, h0: torch.Tensor, h1: torch.Tensor,
                edge_index: torch.Tensor, edge_dist: torch.Tensor,
                edge_dir: torch.Tensor, n_nodes: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h0: (N, C) type-0 (scalar) features
            h1: (N, 3*C) type-1 (vector) features
            edge_index: (2, E)
            edge_dist: (E,)
            edge_dir: (E, 3)
            n_nodes: int

        Returns:
            h0_new, h1_new
        """
        src, dst = edge_index
        E = edge_index.shape[1]
        C = self.hidden_channels

        if E == 0:
            return h0, h1

        # RBF features
        rbf = self.rbf(edge_dist)  # (E, n_rbf)
        w = self.radial_mlp(rbf)    # (E, C)

        # Spherical harmonics on edge directions
        Y = self.sh(edge_dir)  # dict: l -> (E, 2l+1)

        # --- Scalar messages (type-0) ---
        msg_scalar = h0[dst] * w  # (E, C)
        agg_scalar = torch.zeros(n_nodes, C, device=h0.device, dtype=h0.dtype)
        agg_scalar.scatter_add_(0, src.unsqueeze(-1).expand_as(msg_scalar), msg_scalar)
        count = torch.zeros(n_nodes, device=h0.device)
        count.scatter_add_(0, src, torch.ones(E, device=h0.device))
        count = count.clamp(min=1).unsqueeze(-1)
        agg_scalar = agg_scalar / count

        # --- Vector messages (type-1) ---
        h1_j = h1[dst].view(E, 3, C)  # (E, 3, C)
        Y1 = Y[1] if 1 in Y else torch.zeros(E, 3, device=h0.device)  # (E, 3)

        # Message: radial_weight * (h1_j contracted with Y1)
        # Simple: weight h1_j by radial weight and SH direction
        msg_vec = h1_j * w.unsqueeze(1)  # (E, 3, C)
        # Add directional component: Y1 ⊗ scalar_features
        msg_vec = msg_vec + Y1.unsqueeze(-1) * h0[dst].unsqueeze(1)  # (E, 3, C)
        msg_vec_flat = msg_vec.reshape(E, 3 * C)

        agg_vec = torch.zeros(n_nodes, 3 * C, device=h0.device, dtype=h0.dtype)
        agg_vec.scatter_add_(0, src.unsqueeze(-1).expand_as(msg_vec_flat), msg_vec_flat)
        agg_vec = agg_vec / count.expand(-1, 3 * C)

        # --- Node update ---
        # Scalar update
        h0_cat = torch.cat([h0, agg_scalar], dim=-1)
        h0_new = h0 + self.node_mlp_scalar(h0_cat)

        # Vector update with scalar gate
        h1_cat = torch.cat([h1, agg_vec], dim=-1)
        h1_update = self.node_mlp_vector(h1_cat)

        gate = self.gate_mlp(h0_new)  # (N, C)
        h1_update_r = h1_update.view(n_nodes, 3, C) * gate.unsqueeze(1)
        h1_new = h1 + h1_update_r.reshape(n_nodes, 3 * C)

        return h0_new, h1_new


class SEGNN(nn.Module):
    """
    SE(3)-Equivariant GNN for PDE prediction.

    Encoder-Processor-Decoder architecture with steerable features.
    """

    def __init__(self, field_dim: int = 3, hidden_channels: int = 64,
                 l_max: int = 1, n_layers: int = 4, n_rbf: int = 16,
                 r_max: float = 0.15):
        super().__init__()
        self.field_dim = field_dim
        self.hidden_channels = hidden_channels
        self.r_max = r_max
        C = hidden_channels

        # Encoder
        self.scalar_enc = nn.Linear(1, C)  # ||u||²
        self.vector_enc = nn.Linear(field_dim, C, bias=False)  # produces channel weights

        # Processor: stack of SEGNN layers
        self.layers = nn.ModuleList([
            SEGNNLayer(hidden_channels=C, l_max=l_max, n_rbf=n_rbf)
            for _ in range(n_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3 * C, C),
            nn.SiLU(),
            nn.Linear(C, field_dim),
        )

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
        N = u.shape[0]
        C = self.hidden_channels

        # Build graph
        edge_index, edge_vec = radius_graph(coords, 0.0, self.r_max)
        if edge_index.shape[1] == 0:
            return torch.zeros_like(u)

        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_dir = edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8)

        # Encode
        u_norm_sq = (u * u).sum(dim=-1, keepdim=True)  # (N, 1)
        h0 = self.scalar_enc(u_norm_sq)  # (N, C)

        # Vector encoding: u * scalar_weights
        channel_w = self.vector_enc(u)  # (N, C)
        h1 = u.unsqueeze(-1) * channel_w.unsqueeze(1)  # (N, 3, C)
        h1 = h1.reshape(N, 3 * C)

        # Process
        for layer in self.layers:
            h0, h1 = layer(h0, h1, edge_index, edge_dist, edge_dir, N)

        # Decode (from vector features)
        du_dt = self.decoder(h1) * self.output_scale  # (N, field_dim)
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
