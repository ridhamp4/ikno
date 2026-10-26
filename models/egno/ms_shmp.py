"""
Multi-Scale Spherical-Harmonic Message Passing (MS-SHMP) for E-GNO.

Implements the core equivariant message passing described in Section 4.3:
- Edge spherical harmonics
- Radial basis expansion
- Clebsch-Gordan tensor product messages
- Mean aggregation
- Gated scale fusion
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .spherical_harmonics import SphericalHarmonics, ClebschGordanProducts, GaussianRBF, SmoothCutoff


class RadialNetwork(nn.Module):
    """Learnable radial weights from RBF expansion."""

    def __init__(self, n_rbf: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, rbf: torch.Tensor) -> torch.Tensor:
        return self.net(rbf)


class IrrepLinear(nn.Module):
    """Linear channel mixing shared across representation components."""

    def __init__(self, rep_dim: int, channels: int):
        super().__init__()
        self.rep_dim = rep_dim
        self.channels = channels
        self.channel_map = nn.Linear(channels, channels, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        n = value.shape[0]
        return self.channel_map(value.reshape(n, self.rep_dim, self.channels)).reshape(
            n, self.rep_dim * self.channels)


class SHMPLayer(nn.Module):
    """
    Single-scale Spherical-Harmonic Message Passing layer.

    For a given scale k, computes equivariant messages using CG tensor
    products between node features and edge spherical harmonics.
    """

    def __init__(self, hidden_channels: int, l_max: int = 2,
                 n_rbf: int = 16, scale_idx: int = 0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.l_max = l_max
        self.scale_idx = scale_idx

        self.sh = SphericalHarmonics(l_max=l_max)
        self.cg = ClebschGordanProducts(l_max=l_max)
        self.rbf = GaussianRBF(n_rbf=n_rbf)

        # Radial networks for each (l1, l2) -> l3 coupling
        self.radial_nets = nn.ModuleDict()
        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                    key = f'rad_{l1}_{l2}_{l3}'
                    self.radial_nets[key] = RadialNetwork(
                        n_rbf, hidden_channels
                    )

    def forward(self, h: Dict[int, torch.Tensor],
                edge_index: torch.Tensor,
                edge_dist: torch.Tensor,
                edge_dir: torch.Tensor,
                n_nodes: int) -> Dict[int, torch.Tensor]:
        """
        Compute messages and aggregate at one scale.

        Args:
            h: dict l -> (N, (2l+1)*C) node features
            edge_index: (2, E) edges [src, dst]
            edge_dist: (E,) edge lengths
            edge_dir: (E, 3) unit edge directions
            n_nodes: total number of nodes

        Returns:
            aggregated: dict l -> (N, (2l+1)*C) aggregated messages
        """
        src, dst = edge_index
        E = edge_index.shape[1]

        if E == 0:
            return {l: torch.zeros_like(v) for l, v in h.items()}

        # The production experiments use l<=1.  Work directly in the
        # Cartesian scalar/vector basis here: dot products are invariant and
        # scalar-vector products and cross products are SO(3)-equivariant.
        # This avoids mixing Cartesian node features with an incompatible
        # real-spherical-harmonic basis.
        if self.l_max == 1:
            rbf = self.rbf(edge_dist)
            # Mean aggregation and the radial network already encode scale.
            # Unbounded 1/r^(k+1) factors caused catastrophic activations on
            # normalized irregular meshes, especially at the second scale.
            a_ij = torch.ones_like(edge_dist)
            h0 = h[0][dst]
            h1 = h[1][dst].reshape(E, 3, self.hidden_channels)
            direction = edge_dir.unsqueeze(-1)

            w000 = self.radial_nets['rad_0_0_0'](rbf)
            w011 = self.radial_nets['rad_0_1_1'](rbf)
            w101 = self.radial_nets['rad_1_0_1'](rbf)
            w110 = self.radial_nets['rad_1_1_0'](rbf)
            w111 = self.radial_nets['rad_1_1_1'](rbf)

            dot = (h1 * direction).sum(dim=1)
            scalar_msg = w000 * h0 + w110 * dot
            vector_msg = (
                direction * (w011 * h0).unsqueeze(1)
                + h1 * w101.unsqueeze(1)
                + torch.cross(h1, direction.expand_as(h1), dim=1)
                  * w111.unsqueeze(1)
            )
            scalar_msg = scalar_msg * a_ij.unsqueeze(-1)
            vector_msg = vector_msg * a_ij[:, None, None]

            count = torch.zeros(n_nodes, device=edge_index.device, dtype=h0.dtype)
            count.scatter_add_(0, src, torch.ones(E, device=edge_index.device,
                                                  dtype=h0.dtype))
            count = count.clamp(min=1.0)
            agg0 = torch.zeros(n_nodes, self.hidden_channels, device=h0.device,
                               dtype=h0.dtype)
            agg0.scatter_add_(0, src.unsqueeze(-1).expand_as(scalar_msg), scalar_msg)
            flat_vector = vector_msg.reshape(E, 3 * self.hidden_channels)
            agg1 = torch.zeros(n_nodes, 3 * self.hidden_channels, device=h0.device,
                               dtype=h0.dtype)
            agg1.scatter_add_(0, src.unsqueeze(-1).expand_as(flat_vector), flat_vector)
            return {
                0: agg0 / count.unsqueeze(-1),
                1: agg1 / count.unsqueeze(-1),
            }

        # Step 1: Edge spherical harmonics
        Y = self.sh(edge_dir)  # dict l -> (E, 2l+1)

        # Step 2: RBF expansion
        rbf = self.rbf(edge_dist)  # (E, n_rbf)

        # Distance-dependent normalization: a_{ij}^{(k)} = 1/||x_j - x_i||^{k+1}
        k = self.scale_idx
        a_ij = torch.ones_like(edge_dist)

        # Step 3: Compute equivariant messages for each output type l3
        messages = {l: [] for l in range(self.l_max + 1)}

        for l1 in range(self.l_max + 1):
            if l1 not in h:
                continue
            dim_l1 = 2 * l1 + 1
            h_j = h[l1][dst]  # (E, dim_l1 * C)
            C = h_j.shape[1] // dim_l1

            for l2 in range(self.l_max + 1):
                if l2 not in Y:
                    continue
                for l3 in range(abs(l1 - l2), min(l1 + l2, self.l_max) + 1):
                    key = f'rad_{l1}_{l2}_{l3}'
                    dim_l3 = 2 * l3 + 1

                    # Radial weights
                    w = self.radial_nets[key](rbf)  # (E, C)

                    # CG tensor product: (h_j^{l1} ⊗_CG^{l3} Y^{l2})
                    tp = self.cg.tensor_product(h_j, Y[l2], l1, l2, l3)  # (E, dim_l3 * C)

                    # Apply radial weights (channel-wise scaling)
                    tp_r = tp.view(E, dim_l3, C) * w.unsqueeze(1)  # (E, dim_l3, C)

                    # Apply distance normalization
                    msg = a_ij.unsqueeze(-1).unsqueeze(-1) * tp_r  # (E, dim_l3, C)
                    messages[l3].append(msg.reshape(E, dim_l3 * C))

        # Step 4: Sum messages per output type and aggregate
        aggregated = {}
        for l3 in range(self.l_max + 1):
            if not messages[l3]:
                dim_l3 = 2 * l3 + 1
                aggregated[l3] = torch.zeros(n_nodes, dim_l3 * self.hidden_channels,
                                              device=edge_index.device)
                continue

            # Sum all CG couplings for this output type
            total_msg = torch.stack(messages[l3], dim=0).sum(dim=0)  # (E, dim_l3*C)

            # Mean aggregation over neighbors
            dim_l3 = 2 * l3 + 1
            agg = torch.zeros(n_nodes, dim_l3 * self.hidden_channels,
                              device=edge_index.device, dtype=total_msg.dtype)
            # Scatter add
            agg.scatter_add_(0, src.unsqueeze(-1).expand_as(total_msg), total_msg)
            # Count neighbors for mean
            count = torch.zeros(n_nodes, device=edge_index.device)
            count.scatter_add_(0, src, torch.ones(E, device=edge_index.device))
            count = count.clamp(min=1.0).unsqueeze(-1)
            agg = agg / count

            aggregated[l3] = agg

        return aggregated


class MSSHMP(nn.Module):
    """
    Multi-Scale Spherical-Harmonic Message Passing.

    Processes K resolution-stratified graphs, then fuses multi-scale
    features with learnable weights and a scalar gate.
    """

    def __init__(self, hidden_channels: int = 64, l_max: int = 2,
                 n_scales: int = 3, n_rbf: int = 16, n_layers: int = 2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.l_max = l_max
        self.n_scales = n_scales
        self.n_layers = n_layers

        # Per-scale SHMP layers
        self.scale_layers = nn.ModuleList()
        for p in range(n_layers):
            layers_p = nn.ModuleList()
            for k in range(n_scales):
                layers_p.append(SHMPLayer(
                    hidden_channels=hidden_channels,
                    l_max=l_max,
                    n_rbf=n_rbf,
                    scale_idx=k,
                ))
            self.scale_layers.append(layers_p)

        # Scale fusion weights: W_tilde^{(k,l)}
        self.fusion_weights = nn.ParameterDict()
        for l in range(l_max + 1):
            self.fusion_weights[str(l)] = nn.Parameter(
                torch.ones(n_scales) / n_scales
            )

        # Self-connection weights: W^{(l)}_self
        self.self_weights = nn.ModuleDict()
        for l in range(l_max + 1):
            self.self_weights[str(l)] = IrrepLinear(2 * l + 1, hidden_channels)

        # Gating MLP: produces scalar gate from invariant norms
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_channels),
            nn.Sigmoid(),
        )

    def forward(self, h: Dict[int, torch.Tensor],
                graphs: List[Dict[str, torch.Tensor]],
                n_nodes: int) -> Tuple[Dict[int, torch.Tensor],
                                       List[Dict[int, torch.Tensor]]]:
        """
        Multi-scale message passing with scale fusion.

        Args:
            h: initial typed features {l: (N, (2l+1)*C)}
            graphs: list of K graph dicts from build_multiscale_graphs
            n_nodes: number of nodes

        Returns:
            h_new: updated typed features
            all_aggregates: list of per-scale aggregates (for E-DOL)
        """
        all_aggregates_per_layer = []

        for p in range(self.n_layers):
            scale_aggregates = []

            # Process each scale
            for k in range(self.n_scales):
                g = graphs[k]
                if g['edge_index'].shape[1] == 0:
                    agg_k = {l: torch.zeros_like(h[l]) for l in h}
                else:
                    agg_k = self.scale_layers[p][k](
                        h, g['edge_index'], g['edge_dist'], g['edge_dir'], n_nodes
                    )
                scale_aggregates.append(agg_k)

            all_aggregates_per_layer.append(scale_aggregates)

            # Fuse multi-scale features (Eq. 10-11)
            h_new = {}
            for l in range(self.l_max + 1):
                if l not in h:
                    continue
                dim_l = (2 * l + 1) * self.hidden_channels
                str_l = str(l)

                # Weighted sum of scale aggregates (Eq. 10)
                w = torch.softmax(self.fusion_weights[str_l], dim=0)
                h_tilde = torch.zeros(n_nodes, dim_l, device=h[l].device,
                                       dtype=h[l].dtype)
                for k in range(self.n_scales):
                    if l in scale_aggregates[k]:
                        h_tilde = h_tilde + w[k] * scale_aggregates[k][l]

                # Self-connection + aggregated features (Eq. 11)
                h_combined = self.self_weights[str_l](h[l]) + h_tilde

                # Scalar gate from invariant norms
                h0_norm_sq = (h[0] ** 2).mean(dim=-1, keepdim=True)
                h1_norm_sq = torch.zeros_like(h0_norm_sq)
                if 1 in h and h_tilde.shape[1] > 0:
                    h1_vals = scale_aggregates[0].get(1, torch.zeros(n_nodes, 1, device=h[l].device))
                    h1_norm_sq = (h1_vals ** 2).mean(dim=-1, keepdim=True)

                gate_input = torch.cat([h0_norm_sq, h1_norm_sq], dim=-1)
                gate = self.gate_mlp(gate_input)  # (N, C)

                # Apply gate channel-wise
                dim_rep = 2 * l + 1
                h_combined_r = h_combined.view(n_nodes, dim_rep, self.hidden_channels)
                h_combined_r = h_combined_r * gate.unsqueeze(1)
                h_new[l] = h_combined_r.reshape(n_nodes, dim_l)

            h = h_new

        # Return the last layer's scale aggregates for E-DOL
        return h, all_aggregates_per_layer[-1]
