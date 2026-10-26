"""
Equivariant Differential Operator Library (E-DOL) for E-GNO.

Assembles a typed library of differential-operator approximations from
the multi-scale MS-SHMP aggregates. Each library term carries an SO(3)
representation type (l=0, 1, or 2).

Library terms (Section 4.4):
    θ^{(0,1)}: u_hat (identity, type-1)
    θ^{(1,1)}: ∇u approximation (scale-1 type-1 aggregate)
    θ^{(2,1)}: ∇²u approximation (scale-2 type-1 aggregate)
    θ^{(3,1)}: ∇(∇·u) (divergence × gradient, type-1)
    θ^{(4,1)}: (u·∇)u advection term (type-1)
    θ^{(5,1)}: ∇×(∇×u) curl-of-curl (type-1)
    θ^{(6,0)}: ||u||² (type-0 scalar)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from .spherical_harmonics import ClebschGordanProducts


class EDOL(nn.Module):
    """
    Equivariant Differential Operator Library.

    Constructs a typed library from multi-scale equivariant aggregates
    for use in E-SINDy sparse identification.
    """

    def __init__(self, hidden_channels: int = 64, l_max: int = 2,
                 n_scales: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.l_max = l_max
        self.n_scales = n_scales
        self.cg = ClebschGordanProducts(l_max=l_max)

        # Learnable projections for each library term
        C = hidden_channels
        # Project scale aggregates to consistent channel dim for library
        self.proj_identity = nn.Linear(3 * C, C, bias=False)  # type-1 identity
        self.proj_grad = nn.Linear(3 * C, C, bias=False)       # type-1 gradient
        self.proj_laplacian = nn.Linear(3 * C, C, bias=False)  # type-1 Laplacian

        # For nonlinear library terms, we use CG products of aggregates
        # These produce the same channel dim after projection
        self.proj_div_grad = nn.Linear(C, C, bias=False)
        self.proj_advection = nn.Linear(C, C, bias=False)
        self.proj_curl_curl = nn.Linear(C, C, bias=False)

        # Library term metadata: (name, type_l, description)
        self.library_info = [
            ('identity', 1, 'u'),
            ('gradient', 1, '∇u'),
            ('laplacian', 1, '∇²u'),
            ('div_grad', 1, '∇(∇·u)'),
            ('advection', 1, '(u·∇)u'),
            ('curl_curl', 1, '∇×(∇×u)'),
            ('norm_sq', 0, '||u||²'),
        ]

    def forward(self, h: Dict[int, torch.Tensor],
                scale_aggregates: List[Dict[int, torch.Tensor]],
                u: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, int]]:
        """
        Build the differential operator library.

        Args:
            h: current node features {l: (N, (2l+1)*C)}
            scale_aggregates: list of K dicts, each {l: (N, (2l+1)*C)}
            u: (N, 3) raw vector field values

        Returns:
            library: dict mapping term_name -> (values, type_l)
                values: (N, (2*type_l+1) * C_lib) library column
                type_l: SO(3) representation type of this term
        """
        N = u.shape[0]
        C = self.hidden_channels
        library = {}
        if u.shape[-1] % 3 != 0:
            raise ValueError('E-DOL expects packed 3D vector fields')
        # Nonlinear terms currently act on the primary vector field. Coupled
        # cross-field differential terms are handled separately by future
        # explicit operator libraries; this avoids treating a packed state as
        # a single higher-dimensional vector.
        u_primary = u.reshape(N, -1, 3)[:, 0]

        # --- Type-1 (vector) terms ---

        # θ^{(0,1)}: Identity (the field itself)
        # h[1] is (N, 3*C) type-1 features
        if 1 in h:
            library['identity'] = (h[1], 1)

        # θ^{(1,1)}: Gradient approximation (scale-1 type-1 aggregate)
        if len(scale_aggregates) > 0 and 1 in scale_aggregates[0]:
            library['gradient'] = (scale_aggregates[0][1], 1)

        # θ^{(2,1)}: Laplacian approximation (scale-2 type-1 aggregate)
        if len(scale_aggregates) > 1 and 1 in scale_aggregates[1]:
            library['laplacian'] = (scale_aggregates[1][1], 1)

        # θ^{(3,1)}: ∇(∇·u) - divergence times gradient
        # Computed as CG product of type-0 (divergence ≈ scale-1 type-0)
        # and type-1 (gradient ≈ scale-1 type-1)
        if (len(scale_aggregates) > 0 and
                0 in scale_aggregates[0] and 1 in scale_aggregates[0]):
            div_approx = scale_aggregates[0][0]  # (N, C) type-0
            grad_approx = scale_aggregates[0][1]  # (N, 3*C) type-1
            # CG product of type-0 ⊗ type-1 -> type-1
            # This is just scalar * vector (type-0 acts as channel-wise scaling)
            grad_r = grad_approx.view(N, 3, C)
            div_grad = grad_r * div_approx.unsqueeze(1)  # (N, 3, C)
            library['div_grad'] = (div_grad.reshape(N, 3 * C), 1)

        # θ^{(4,1)}: Advection (u·∇)u
        # CG product of scale-1 type-1 with itself -> type-1
        if len(scale_aggregates) > 0 and 1 in scale_aggregates[0]:
            m1 = scale_aggregates[0][1]  # (N, 3*C)
            # Advection: contract one vector index to get scalar, multiply by other
            m1_r = m1.view(N, 3, C)
            # Use u as the advecting velocity
            # (u · ∇)u ≈ u_j * ∂u/∂x_j  →  contract u with gradient features
            u_expanded = u_primary.unsqueeze(-1)  # (N, 3, 1)
            advection = (u_expanded * m1_r).sum(dim=1, keepdim=True)  # (N, 1, C)
            # This gives a scalar per channel; multiply by u direction for type-1
            adv_vec = u_primary.unsqueeze(-1) * advection  # (N, 3, C)
            library['advection'] = (adv_vec.reshape(N, 3 * C), 1)

        # θ^{(5,1)}: Curl of curl ∇×(∇×u)
        # Approximated by CG self-product of gradient features
        if len(scale_aggregates) > 0 and 1 in scale_aggregates[0]:
            m1 = scale_aggregates[0][1]  # (N, 3*C)
            m1_r = m1.view(N, 3, C)
            # Cross product structure: approximate curl by antisymmetric part
            # curl_curl ≈ ∇(∇·u) - ∇²u
            if 'div_grad' in library and 'laplacian' in library:
                cc = library['div_grad'][0] - library['laplacian'][0]
                library['curl_curl'] = (cc, 1)

        # --- Type-0 (scalar) terms ---

        # θ^{(6,0)}: ||u||²
        u_norm_sq = (u_primary * u_primary).sum(dim=-1, keepdim=True)  # (N, 1)
        u_norm_sq_expanded = u_norm_sq.expand(N, C)
        library['norm_sq'] = (u_norm_sq_expanded, 0)

        # Divergence (type-0, from scale-1 type-0 aggregate)
        if len(scale_aggregates) > 0 and 0 in scale_aggregates[0]:
            library['divergence'] = (scale_aggregates[0][0], 0)

        return library

    def get_type_consistent_library(self, library: Dict[str, Tuple[torch.Tensor, int]],
                                     target_type: int = 1
                                     ) -> Tuple[torch.Tensor, List[str]]:
        """
        Extract the type-consistent sub-library Θ^[l*] for a given target type.

        Args:
            library: full library from forward()
            target_type: target SO(3) type (1 for vector PDEs)

        Returns:
            theta: (N, P, D) library matrix where P = number of type-consistent terms,
                   D = (2*target_type+1) * C
            term_names: list of P term names
        """
        terms = []
        names = []
        for name, (values, type_l) in library.items():
            if type_l == target_type:
                terms.append(values)
                names.append(name)

        if not terms:
            return None, []

        # Stack into library matrix: (N, P, D)
        theta = torch.stack(terms, dim=1)
        return theta, names
