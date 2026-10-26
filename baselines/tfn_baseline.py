"""A compact Tensor Field Network baseline implemented with e3nn.

The model carries even scalar features and odd vector features. Messages are
fully connected tensor products between node irreps and spherical harmonics,
with distance-conditioned weights. Every learned map respects O(3), and hence
SO(3), equivariance. This baseline is intentionally more general than the
PDE-specialized isotropic kernel.
"""

import torch
import torch.nn as nn

from models.graph_utils import radius_graph

try:
    from e3nn import o3
except ImportError:  # Keep the rest of the repository importable without e3nn.
    o3 = None


class _TFNLayer(nn.Module):
    def __init__(self, hidden_channels, n_rbf, radial_hidden, r_max):
        super().__init__()
        if o3 is None:
            raise ImportError(
                'TFNBaseline requires e3nn. Install it or run through the '
                'workspace-local .venv-e3nn environment.')
        self.hidden_channels = hidden_channels
        self.r_max = r_max
        self.irreps = o3.Irreps(
            f'{hidden_channels}x0e + {hidden_channels}x1o')
        self.irreps_sh = o3.Irreps.spherical_harmonics(1)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.irreps, self.irreps_sh, self.irreps,
            shared_weights=False)
        self.radial = nn.Sequential(
            nn.Linear(n_rbf, radial_hidden),
            nn.SiLU(),
            nn.Linear(radial_hidden, self.tensor_product.weight_numel),
        )
        self.self_linear = o3.Linear(self.irreps, self.irreps)
        self.message_linear = o3.Linear(self.irreps, self.irreps)
        self.gates = nn.Linear(hidden_channels, hidden_channels)
        centers = torch.linspace(0.0, r_max, n_rbf)
        self.register_buffer('rbf_centers', centers)
        spacing = r_max / max(n_rbf - 1, 1)
        self.rbf_gamma = 1.0 / max(spacing * spacing, 1e-8)

    def forward(self, features, edge_index, edge_vec):
        if edge_index.shape[1] == 0:
            return features
        src, dst = edge_index
        distance = torch.norm(edge_vec, dim=-1).clamp(min=1e-8)
        direction = edge_vec / distance.unsqueeze(-1)
        harmonics = o3.spherical_harmonics(
            self.irreps_sh, direction, normalize=True,
            normalization='component')
        rbf = torch.exp(
            -self.rbf_gamma *
            (distance.unsqueeze(-1) - self.rbf_centers) ** 2)
        weights = self.radial(rbf)
        messages = self.tensor_product(features[dst], harmonics, weights)
        aggregate = torch.zeros_like(features)
        aggregate.scatter_add_(
            0, src.unsqueeze(-1).expand_as(messages), messages)
        counts = torch.zeros(
            features.shape[0], device=features.device, dtype=features.dtype)
        counts.scatter_add_(0, src, torch.ones_like(distance))
        aggregate = aggregate / counts.clamp(min=1).unsqueeze(-1)

        updated = self.self_linear(features) + self.message_linear(aggregate)
        channels = self.hidden_channels
        scalars = torch.nn.functional.silu(updated[:, :channels])
        vectors = updated[:, channels:].reshape(-1, channels, 3)
        vectors = vectors * torch.sigmoid(self.gates(scalars)).unsqueeze(-1)
        return torch.cat([scalars, vectors.reshape(features.shape[0], -1)], dim=-1)


class TFNBaseline(nn.Module):
    """Typed scalar/vector Tensor Field Network for one-frame PDE dynamics."""

    def __init__(self, field_dim=3, hidden_channels=8, n_layers=3,
                 n_rbf=8, radial_hidden=32, r_max=0.4,
                 max_neighbors=64):
        super().__init__()
        if o3 is None:
            raise ImportError(
                'TFNBaseline requires e3nn. Install it or run through the '
                'workspace-local .venv-e3nn environment.')
        if field_dim != 3:
            raise ValueError('TFNBaseline currently supports one 3D vector field')
        self.field_dim = field_dim
        self.hidden_channels = hidden_channels
        self.r_max = r_max
        self.max_neighbors = max_neighbors
        self.irreps_hidden = o3.Irreps(
            f'{hidden_channels}x0e + {hidden_channels}x1o')
        self.scalar_encoder = nn.Sequential(
            nn.Linear(1, hidden_channels), nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels))
        self.vector_encoder = o3.Linear(
            o3.Irreps('1x1o'), o3.Irreps(f'{hidden_channels}x1o'))
        self.layers = nn.ModuleList([
            _TFNLayer(hidden_channels, n_rbf, radial_hidden, r_max)
            for _ in range(n_layers)
        ])
        self.decoder = o3.Linear(self.irreps_hidden, o3.Irreps('1x1o'))
        self.output_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, u, coords):
        edge_index, edge_vec = radius_graph(
            coords, 0.0, self.r_max, self.max_neighbors)
        scalars = self.scalar_encoder((u * u).sum(dim=-1, keepdim=True))
        vectors = self.vector_encoder(u)
        features = torch.cat([scalars, vectors], dim=-1)
        for layer in self.layers:
            features = layer(features, edge_index, edge_vec)
        return self.decoder(features) * self.output_scale

    def compute_loss(self, trajectory, coords, dt):
        derivative = (trajectory[1] - trajectory[0]) / dt
        return ((self(trajectory[0], coords) - derivative) ** 2).mean()

    def rollout(self, state, coords, dt, n_steps):
        trajectory = [state]
        for _ in range(n_steps):
            state = state + dt * self(state, coords)
            trajectory.append(state)
        return torch.stack(trajectory)
