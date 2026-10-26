"""Exact SO(3)-equivariant isotropic kernels for coupled vector PDEs."""

import torch
import torch.nn as nn

from .graph_utils import radius_graph


class IsotropicKernelOperator(nn.Module):
    """Learn a linear isotropic operator on one or more 3D vector fields.

    The most general pairwise linear SO(3)-equivariant kernel is a radial
    combination of ``v`` and ``r_hat (r_hat dot v)``.  A learned matrix mixes
    source and target vector fields without mixing Cartesian components.
    """

    _BASIS_NAMES = ('difference', 'longitudinal', 'source', 'cross')

    def __init__(self, field_dim=6, hidden_dim=64, r_max=0.3,
                 max_neighbors=32, basis_terms=None):
        super().__init__()
        if field_dim % 3:
            raise ValueError('field_dim must contain packed 3D vectors')
        self.field_dim = field_dim
        self.n_vectors = field_dim // 3
        self.r_max = r_max
        self.max_neighbors = max_neighbors
        active_terms = set(self._BASIS_NAMES if basis_terms is None else basis_terms)
        unknown_terms = active_terms.difference(self._BASIS_NAMES)
        if unknown_terms:
            raise ValueError(f'Unknown isotropic basis terms: {sorted(unknown_terms)}')
        if not active_terms:
            raise ValueError('At least one isotropic basis term must be active')
        self.basis_terms = tuple(
            name for name in self._BASIS_NAMES if name in active_terms)
        self.register_buffer('basis_mask', torch.tensor([
            float(name in active_terms) for name in self._BASIS_NAMES
        ]), persistent=False)
        # Coefficients are functions only of SO(3) invariants.  Four vector
        # bases per source/target pair make the kernel nonlinear enough for
        # advection while preserving exact rotation equivariance.
        n_invariants = 1 + 5 * self.n_vectors
        n_coefficients = 4 * self.n_vectors * self.n_vectors
        self.radial = nn.Sequential(
            nn.Linear(n_invariants, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_coefficients),
        )
        # Start from the known kinematic relation and learn only the spatial
        # correction.  This also avoids a large random acceleration at epoch 0.
        nn.init.zeros_(self.radial[-1].weight)
        nn.init.zeros_(self.radial[-1].bias)
        self.self_mixing = nn.Parameter(torch.zeros(
            self.n_vectors, self.n_vectors))
        # For first-order wave systems the kinematic relation du/dt=v is exact.
        if self.n_vectors == 2:
            with torch.no_grad():
                self.self_mixing[0, 1] = 1.0

    def forward(self, state, coords):
        n = state.shape[0]
        vectors = state.reshape(n, self.n_vectors, 3)
        edge_index, edge_vec = radius_graph(
            coords, 0.0, self.r_max, self.max_neighbors)
        if edge_index.shape[1] == 0:
            return torch.einsum('ab,nbd->nad', self.self_mixing, vectors).reshape(
                n, self.field_dim)
        src, dst = edge_index
        distance = torch.norm(edge_vec, dim=-1).clamp(min=1e-8)
        direction = edge_vec / distance.unsqueeze(-1)
        # Differential operators annihilate constant fields.  Building the
        # kernel from neighbor differences imposes that property exactly and
        # is the mesh-free analogue of a finite-difference stencil.
        neighbor = vectors[dst] - vectors[src]
        source = vectors[src]
        longitudinal = direction[:, None, :] * (
            neighbor * direction[:, None, :]).sum(dim=-1, keepdim=True)
        invariants = torch.cat([
            distance[:, None],
            torch.norm(source, dim=-1),
            torch.norm(neighbor, dim=-1),
            (source * direction[:, None, :]).sum(dim=-1),
            (neighbor * direction[:, None, :]).sum(dim=-1),
            (source * neighbor).sum(dim=-1),
        ], dim=-1)
        coefficients = self.radial(invariants).reshape(
            -1, self.n_vectors, self.n_vectors, 4)
        coefficients = coefficients * self.basis_mask
        source_cross_direction = torch.cross(
            source, direction[:, None, :].expand_as(source), dim=-1)
        messages = (
            torch.einsum('eab,ebd->ead', coefficients[..., 0], neighbor)
            + torch.einsum('eab,ebd->ead', coefficients[..., 1], longitudinal)
            + torch.einsum('eab,ebd->ead', coefficients[..., 2], source)
            + torch.einsum('eab,ebd->ead', coefficients[..., 3],
                           source_cross_direction)
        )
        aggregate = torch.zeros(n, self.n_vectors, 3, device=state.device,
                                dtype=state.dtype)
        aggregate.scatter_add_(
            0, src[:, None, None].expand_as(messages), messages)
        counts = torch.zeros(n, device=state.device, dtype=state.dtype)
        counts.scatter_add_(0, src, torch.ones_like(distance))
        aggregate = aggregate / counts.clamp(min=1).reshape(n, 1, 1)
        self_term = torch.einsum('ab,nbd->nad', self.self_mixing, vectors)
        return (self_term + aggregate).reshape(n, self.field_dim)

    def compute_loss(self, trajectory, coords, dt):
        derivative = (trajectory[1] - trajectory[0]) / dt
        return ((self(trajectory[0], coords) - derivative) ** 2).mean()

    def rollout(self, state, coords, dt, n_steps):
        trajectory = [state]
        for _ in range(n_steps):
            state = state + dt * self(state, coords)
            trajectory.append(state)
        return torch.stack(trajectory)

    def equivariance_error(self, state, coords, n_rotations=5):
        """Normalized SO(3) operator error (zero up to floating point)."""
        from .egno.egno_model import random_rotation_matrix
        reference = self(state, coords).reshape(
            state.shape[0], self.n_vectors, 3)
        errors = []
        with torch.no_grad():
            for _ in range(n_rotations):
                rotation = random_rotation_matrix(device=state.device)
                state_rotated = (state.reshape(
                    state.shape[0], self.n_vectors, 3) @ rotation.T).reshape_as(state)
                output_rotated = self(
                    state_rotated, coords @ rotation.T).reshape_as(reference)
                expected = reference @ rotation.T
                numerator = (output_rotated - expected).square().mean().sqrt()
                denominator = (output_rotated.square().mean().sqrt()
                               + expected.square().mean().sqrt()).clamp(min=1e-8)
                errors.append(float(numerator / denominator))
        return sum(errors) / len(errors)
