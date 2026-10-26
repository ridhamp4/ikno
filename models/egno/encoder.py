"""
Equivariant Node Encoder for E-GNO.

Lifts observed vector field u(x_i, t) into typed SO(3) features
h_i^{(l)} for l = 0, 1, ..., L_max.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class EquivariantEncoder(nn.Module):
    """
    Equivariant node encoding: lifts vector field observations into
    SO(3)-typed features.

    For a 3D vector field u ∈ R^3:
        - Type-0 (scalar): ||u||, ||u||^2, or learned scalar features
        - Type-1 (vector): u itself
        - Type-2 (tensor): outer product features (optional)

    Each type has C_l channels, controlled by hidden_channels.
    """

    def __init__(self, in_field_dim: int = 3, hidden_channels: int = 64,
                 l_max: int = 2, n_scalar_features: int = 0):
        """
        Args:
            in_field_dim: dimension of input field (3 for vector field)
            hidden_channels: number of channels per type
            l_max: maximum representation order
            n_scalar_features: additional scalar features per node (e.g., density)
        """
        super().__init__()
        self.in_field_dim = in_field_dim
        self.hidden_channels = hidden_channels
        self.l_max = l_max
        if in_field_dim % 3 != 0:
            raise ValueError('EquivariantEncoder expects one or more 3D vector fields')
        self.n_vector_fields = in_field_dim // 3

        # The Gram matrix of the input vectors contains all pairwise SO(3)
        # invariants and permits coupled systems such as (E, B) and (u, v).
        scalar_in = (self.n_vector_fields * (self.n_vector_fields + 1) // 2
                     + n_scalar_features)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_in, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Invariant-conditioned mixing weights. Spatial components are never
        # mixed by an unrestricted matrix, so this map is exactly equivariant.
        self.vector_weight_net = nn.Sequential(
            nn.Linear(scalar_in, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, self.n_vector_fields * hidden_channels),
        )

        # Type-2 encoder (if l_max >= 2): from symmetric traceless part
        if l_max >= 2:
            self.tensor_weight_net = nn.Linear(scalar_in, hidden_channels, bias=False)

    def forward(self, u: torch.Tensor,
                scalar_features: Optional[torch.Tensor] = None) -> Dict[int, torch.Tensor]:
        """
        Encode field observations into typed features.

        Args:
            u: (N, 3) vector field values at nodes
            scalar_features: (N, n_scalar) optional additional scalar features

        Returns:
            dict mapping l -> (N, (2l+1) * C) typed feature tensors
        """
        N = u.shape[0]
        features = {}

        if u.shape[-1] != self.in_field_dim:
            raise ValueError(f'Expected state dimension {self.in_field_dim}, got {u.shape[-1]}')
        vectors = u.reshape(N, self.n_vector_fields, 3)
        gram = torch.einsum('nkd,nld->nkl', vectors, vectors)
        invariant_parts = []
        for i in range(self.n_vector_fields):
            for j in range(i, self.n_vector_fields):
                invariant_parts.append(gram[:, i, j:i + 1 if j == i else j + 1])
        invariant_features = torch.cat(invariant_parts, dim=-1)
        if scalar_features is not None:
            scalar_in = torch.cat([invariant_features, scalar_features], dim=-1)
        else:
            scalar_in = invariant_features
        features[0] = self.scalar_encoder(scalar_in)  # (N, C)

        channel_weights = self.vector_weight_net(scalar_in).reshape(
            N, self.n_vector_fields, self.hidden_channels)
        h1 = torch.einsum('nkd,nkc->ndc', vectors, channel_weights)
        features[1] = h1.reshape(N, 3 * self.hidden_channels)  # (N, 3*C)

        # Type-2: symmetric traceless tensor features
        if self.l_max >= 2:
            # Compute symmetric traceless part of u ⊗ u
            # S = u ⊗ u - (1/3)||u||^2 I
            base_vector = h1.mean(dim=-1)
            uu = base_vector.unsqueeze(-1) * base_vector.unsqueeze(-2)
            trace = (base_vector * base_vector).sum(dim=-1, keepdim=True)
            S = uu - (trace / 3.0).unsqueeze(-1) * torch.eye(3, device=u.device)

            # Extract 5 independent components (Voigt notation)
            # [S_xx, S_yy, S_xy, S_xz, S_yz] (S_zz = -S_xx - S_yy)
            s5 = torch.stack([
                S[:, 0, 0], S[:, 1, 1],
                S[:, 0, 1], S[:, 0, 2], S[:, 1, 2]
            ], dim=-1)  # (N, 5)

            t2_weights = self.tensor_weight_net(scalar_in)
            h2_full = s5.unsqueeze(-1) * t2_weights.unsqueeze(1)  # (N, 5, C)
            features[2] = h2_full.reshape(N, 5 * self.hidden_channels)

        return features

    def output_dims(self) -> Dict[int, int]:
        """Return the output dimension for each type l."""
        dims = {
            0: self.hidden_channels,
            1: 3 * self.hidden_channels,
        }
        if self.l_max >= 2:
            dims[2] = 5 * self.hidden_channels
        return dims
