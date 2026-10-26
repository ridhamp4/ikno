"""
E-GNO: Equivariant Graph Neural Operator - Full Model.

Combines all components:
1. Equivariant Node Encoder (§4.1)
2. Multi-Scale Graph Construction (§4.2)
3. MS-SHMP (§4.3)
4. E-DOL (§4.4)
5. E-SINDy (§4.5)
6. Neural ODE Rollout (§4.6)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from .encoder import EquivariantEncoder
from .ms_shmp import MSSHMP
from .e_dol import EDOL
from .e_sindy import ESINDy
from .neural_ode import NeuralODERollout
from ..graph_utils import build_multiscale_graphs


class EquivariantVectorDecoder(nn.Module):
    """Channel-only linear map from type-1 features to 3D vector fields."""

    def __init__(self, hidden_channels: int, n_vector_fields: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_vector_fields = n_vector_fields
        self.channel_map = nn.Linear(hidden_channels, n_vector_fields, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        typed = h.reshape(n, 3, self.hidden_channels)
        vectors = self.channel_map(typed).permute(0, 2, 1)
        return vectors.reshape(n, 3 * self.n_vector_fields)


class EGNO(nn.Module):
    """
    E-GNO: Equivariant Graph Neural Operator.

    Full pipeline: input field -> encoder -> MS-SHMP -> E-DOL -> E-SINDy -> Neural ODE -> prediction.
    """

    def __init__(self,
                 field_dim: int = 3,
                 hidden_channels: int = 64,
                 l_max: int = 2,
                 n_scales: int = 3,
                 n_mp_layers: int = 2,
                 n_rbf: int = 16,
                 n_library_terms: int = 10,
                 lambda_sparse: float = 0.1,
                 threshold: float = 0.05,
                 n_iht_steps: int = 10,
                 use_residual: bool = True,
                 use_operator_library: bool = True,
                 r_scales: Optional[List[Tuple[float, float]]] = None,
                 max_neighbors: int = 64):
        """
        Args:
            field_dim: dimension of input/output field (3 for vector field)
            hidden_channels: channel dimension C
            l_max: maximum SO(3) representation order
            n_scales: number of multi-scale graphs K
            n_mp_layers: number of message passing layers P
            n_rbf: number of radial basis functions
            n_library_terms: max library terms for E-SINDy
            lambda_sparse: L1 sparsity coefficient
            threshold: IHT threshold δ
            n_iht_steps: number of IHT iterations
            use_residual: whether to use residual network in Neural ODE
            r_scales: manual radius scale ranges (auto if None)
            max_neighbors: max neighbors per node per scale
        """
        super().__init__()

        self.field_dim = field_dim
        if field_dim % 3 != 0:
            raise ValueError('EGNO field_dim must describe one or more 3D vectors')
        self.n_vector_fields = field_dim // 3
        self.hidden_channels = hidden_channels
        self.l_max = l_max
        self.n_scales = n_scales
        self.r_scales = r_scales
        self.max_neighbors = max_neighbors
        self.use_operator_library = use_operator_library

        # 1. Equivariant Encoder
        self.encoder = EquivariantEncoder(
            in_field_dim=field_dim,
            hidden_channels=hidden_channels,
            l_max=l_max,
        )

        # 2. MS-SHMP
        self.ms_shmp = MSSHMP(
            hidden_channels=hidden_channels,
            l_max=l_max,
            n_scales=n_scales,
            n_rbf=n_rbf,
            n_layers=n_mp_layers,
        )

        # 3. E-DOL
        self.e_dol = EDOL(
            hidden_channels=hidden_channels,
            l_max=l_max,
            n_scales=n_scales,
        )

        # 4. E-SINDy
        self.e_sindy = ESINDy(
            n_library_terms=n_library_terms,
            hidden_channels=hidden_channels,
            lambda_sparse=lambda_sparse,
            threshold=threshold,
            n_iht_steps=n_iht_steps,
        )

        # 5. Neural ODE Rollout
        self.ode_rollout = NeuralODERollout(
            field_dim=field_dim,
            use_residual=use_residual,
            hidden_channels=hidden_channels,
        )

        # Output decoder: from type-1 hidden features to field dim
        self.decoder = EquivariantVectorDecoder(hidden_channels, self.n_vector_fields)
        self.direct_decoder = EquivariantVectorDecoder(
            hidden_channels, self.n_vector_fields)

        # Cache for graphs (recomputed when coords change)
        self._cached_graphs = None
        self._cached_coords_hash = None

    def build_graphs(self, coords: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Build or retrieve cached multi-scale graphs."""
        coords_hash = hash(coords.data_ptr())
        if self._cached_graphs is not None and self._cached_coords_hash == coords_hash:
            return self._cached_graphs

        graphs = build_multiscale_graphs(
            coords, K=self.n_scales,
            r_scales=self.r_scales,
            max_neighbors=self.max_neighbors,
        )
        self._cached_graphs = graphs
        self._cached_coords_hash = coords_hash
        return graphs

    def compute_rhs(self, u: torch.Tensor, coords: torch.Tensor
                    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the right-hand side F(u) of the discovered PDE.

        This runs the full E-GNO pipeline (encoder -> MS-SHMP -> E-DOL -> E-SINDy).

        Args:
            u: (N, field_dim) current field state
            coords: (N, 3) node coordinates

        Returns:
            du_dt: (N, field_dim) predicted time derivative
            info: dict with library, discovered PDE, etc.
        """
        N = u.shape[0]
        graphs = self.build_graphs(coords)

        # 1. Encode
        h = self.encoder(u)

        # 2. MS-SHMP
        h_updated, scale_aggregates = self.ms_shmp(h, graphs, N)

        if not self.use_operator_library:
            du_dt = self.direct_decoder(h_updated[1])
            return du_dt, {
                'library': {}, 'theta': None, 'term_names': [],
                'h_updated': h_updated, 'scale_aggregates': scale_aggregates,
            }

        # 3. E-DOL: build library
        library = self.e_dol(h_updated, scale_aggregates, u)

        # 4. Extract type-consistent sub-library (type-1 for vector PDEs)
        theta, term_names = self.e_dol.get_type_consistent_library(library, target_type=1)

        # 5. E-SINDy: sparse identification
        if theta is not None and theta.shape[1] > 0:
            # Use hidden features from library
            du_dt_hidden = self.e_sindy(theta)  # (N, 3*C)
            # Decode to field dimension
            du_dt = self.decoder(du_dt_hidden)  # (N, field_dim)
        else:
            du_dt = torch.zeros(N, self.field_dim, device=u.device)

        info = {
            'library': library,
            'theta': theta,
            'term_names': term_names,
            'h_updated': h_updated,
            'scale_aggregates': scale_aggregates,
        }

        return du_dt, info

    def forward(self, u0: torch.Tensor, coords: torch.Tensor,
                dt: float = 0.01, n_steps: int = 1,
                method: str = 'rk4') -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: predict trajectory from initial condition.

        Args:
            u0: (N, field_dim) initial field state
            coords: (N, 3) node coordinates
            dt: time step for ODE integration
            n_steps: number of integration steps
            method: 'rk4' or 'euler'

        Returns:
            trajectory: (n_steps+1, N, field_dim) predicted trajectory
            info: dict with diagnostics
        """
        # Build graphs once
        self.build_graphs(coords)

        # Create ODE RHS function
        def rhs_fn(u):
            du_dt, _ = self.compute_rhs(u, coords)
            return du_dt

        # Rollout with Neural ODE
        trajectory = self.ode_rollout(rhs_fn, u0, dt, n_steps, method)

        # Get info from last step
        _, info = self.compute_rhs(u0, coords)

        return trajectory, info

    def compute_loss(self, u_true: torch.Tensor, coords: torch.Tensor,
                     dt: float, lambda_eq: float = 0.0,
                     n_rollout_steps: int = 1,
                     method: str = 'euler',
                     max_time_windows: int = 4,
                     normalize_by_increment: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the full training loss (Eq. 13).

        L = trajectory_loss + λ₁ * sparsity + λ₂ * equivariance_reg

        Args:
            u_true: (T, N, field_dim) ground truth trajectory
            coords: (N, 3) node coordinates
            dt: time step
            lambda_eq: equivariance regularizer weight
            n_rollout_steps: number of rollout steps for training

        Returns:
            total_loss: scalar loss
            info: dict with loss components
        """
        T, N, D = u_true.shape
        n_steps = min(n_rollout_steps, T - 1)

        n_windows = T - n_steps
        if n_windows <= 0:
            raise ValueError('Trajectory is too short for the requested rollout')
        if max_time_windows and n_windows > max_time_windows:
            starts = torch.randperm(n_windows, device=u_true.device)[:max_time_windows].tolist()
        else:
            starts = list(range(n_windows))

        traj_loss = torch.tensor(0.0, device=u_true.device)
        sindy_loss = torch.tensor(0.0, device=u_true.device)
        for start in starts:
            trajectory, model_info = self.forward(
                u_true[start], coords, dt, n_steps, method=method)
            for step in range(1, n_steps + 1):
                prediction_error = ((
                    u_true[start + step] - trajectory[step]) ** 2).mean()
                if normalize_by_increment:
                    increment_energy = ((
                        u_true[start + step] - u_true[start]) ** 2).mean()
                    prediction_error = prediction_error / increment_energy.clamp(min=1e-12)
                traj_loss = traj_loss + prediction_error

            if model_info['theta'] is not None:
                du_dt_est = (u_true[start + 1] - u_true[start]) / dt
                du_dt_hidden = self.e_sindy(model_info['theta'])
                du_dt_pred_field = self.decoder(du_dt_hidden)
                sindy_loss = sindy_loss + ((du_dt_pred_field - du_dt_est) ** 2).mean()

        traj_loss = traj_loss / (len(starts) * n_steps)
        sindy_loss = sindy_loss / len(starts)

        # Sparsity loss
        sparse_loss = self.e_sindy.sparsity_loss()

        # Total loss (all components now O(1))
        total_loss = traj_loss + sparse_loss + 0.1 * sindy_loss

        info = {
            'traj_loss': traj_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'sindy_loss': sindy_loss.item(),
            'total_loss': total_loss.item(),
            'n_active_terms': self.e_sindy.get_active_terms_count(),
        }

        return total_loss, info

    def run_iht(self, u_true: torch.Tensor, coords: torch.Tensor, dt: float):
        """
        Run IHT to discover sparse PDE after training.

        Args:
            u_true: (T, N, field_dim) trajectory data
            coords: (N, 3) node coordinates
            dt: time step
        """
        with torch.no_grad():
            # Estimate time derivative
            du_dt_est = (u_true[1] - u_true[0]) / dt  # (N, field_dim)

            # Compute library
            _, info = self.compute_rhs(u_true[0], coords)

            if info['theta'] is not None:
                # Project theta from hidden space (N, P, 3*C) to field space (N, P, field_dim)
                theta_hidden = info['theta']  # (N, P, 3*C)
                N, P, D_hidden = theta_hidden.shape
                # Decode each library column to field space
                theta_field = self.decoder(
                    theta_hidden.reshape(N * P, D_hidden)
                ).reshape(N, P, self.field_dim)
                self.e_sindy.run_iht(theta_field, du_dt_est)

    def get_discovered_pde(self, coords: torch.Tensor, u: torch.Tensor) -> Dict:
        """
        Return the discovered PDE in symbolic form.

        Args:
            coords: (N, 3) coordinates
            u: (N, field_dim) field values

        Returns:
            dict mapping operator_name -> coefficient
        """
        with torch.no_grad():
            _, info = self.compute_rhs(u, coords)
            if info['term_names']:
                return self.e_sindy.get_discovered_pde(info['term_names'])
        return {}

    def equivariance_error(self, u: torch.Tensor, coords: torch.Tensor,
                           n_rotations: int = 10) -> float:
        """
        Compute equivariance error: E_R ||F(R·u) - R·F(u)||.

        Args:
            u: (N, 3) field values
            coords: (N, 3) coordinates
            n_rotations: number of random rotations to sample

        Returns:
            mean equivariance error
        """
        errors = []
        with torch.no_grad():
            if u.shape[-1] % 3 != 0:
                raise ValueError('Equivariance evaluation requires packed 3D vector fields')
            n_vectors = u.shape[-1] // 3

            def rotate_fields(value, rotation):
                shape = value.shape
                packed = value.reshape(shape[0], n_vectors, 3)
                return (packed @ rotation.T).reshape(shape)

            # F(u)
            fu, _ = self.compute_rhs(u, coords)
            base_graphs = self.build_graphs(coords)

            for _ in range(n_rotations):
                # Random rotation matrix
                R = random_rotation_matrix(device=u.device)

                # Rotated inputs
                coords_rot = coords @ R.T
                u_rot = rotate_fields(u, R)

                # F(R·u, R·x)
                # Radius/kNN topology is rotation invariant. Reuse the exact
                # discrete topology and rotate its geometric attributes to
                # avoid floating-point neighbor-boundary changes.
                self._cached_graphs = [{
                    **graph,
                    'edge_vec': graph['edge_vec'] @ R.T,
                    'edge_dir': graph['edge_dir'] @ R.T,
                } for graph in base_graphs]
                self._cached_coords_hash = hash(coords_rot.data_ptr())
                fu_rot, _ = self.compute_rhs(u_rot, coords_rot)

                # R·F(u)
                rfu = rotate_fields(fu, R)

                # Error
                numerator = ((fu_rot - rfu) ** 2).mean().sqrt()
                denominator = ((fu_rot ** 2).mean().sqrt()
                               + (rfu ** 2).mean().sqrt()).clamp(min=1e-8)
                err = numerator / denominator
                errors.append(err.item())

            self._cached_graphs = None
        return sum(errors) / len(errors)


def random_rotation_matrix(device='cpu') -> torch.Tensor:
    """Generate a random 3D rotation matrix (uniform on SO(3)).

    QR decomposition runs on CPU (not supported on MPS), result moved to target device.
    """
    q, r = torch.linalg.qr(torch.randn(3, 3))  # always on CPU
    d = torch.diag(torch.sign(torch.diag(r)))
    q = q @ d
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(device)
