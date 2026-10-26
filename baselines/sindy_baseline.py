"""
SINDy baseline for PDE discovery on irregular meshes.
Uses PySINDy (https://github.com/dynamicslab/pysindy) as backend.

Constructs finite-difference approximations of differential operators
on the irregular point cloud using nearest-neighbor stencils.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import KDTree


class FiniteDifferenceLibrary:
    """
    Approximate differential operators on irregular meshes using
    weighted least-squares finite differences.
    """

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def fit(self, coords: np.ndarray):
        """Build KD-tree for neighbor queries."""
        self.tree = KDTree(coords)
        self.coords = coords

    def gradient(self, field: np.ndarray) -> np.ndarray:
        """
        Approximate gradient ∇f at each node.

        Args:
            field: (N,) or (N, d) field values

        Returns:
            grad: (N, 3) or (N, d, 3) gradient approximation
        """
        N = self.coords.shape[0]
        is_vector = field.ndim == 2
        if not is_vector:
            field = field[:, None]

        d = field.shape[1]
        grad = np.zeros((N, d, 3))

        for i in range(N):
            _, idx = self.tree.query(self.coords[i], k=self.n_neighbors + 1)
            idx = idx[1:]  # exclude self
            dx = self.coords[idx] - self.coords[i]  # (K, 3)
            df = field[idx] - field[i]  # (K, d)

            # Weighted least squares: min ||W(dx @ β - df)||^2
            W = np.diag(1.0 / (np.linalg.norm(dx, axis=1) + 1e-8))
            A = W @ dx
            b = W @ df
            try:
                beta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                grad[i] = beta.T  # (d, 3)
            except np.linalg.LinAlgError:
                pass

        if not is_vector:
            return grad[:, 0, :]
        return grad

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """
        Approximate Laplacian ∇²f at each node.

        Args:
            field: (N,) or (N, d) field values

        Returns:
            lap: (N,) or (N, d) Laplacian approximation
        """
        N = self.coords.shape[0]
        is_vector = field.ndim == 2
        if not is_vector:
            field = field[:, None]
        d = field.shape[1]
        lap = np.zeros((N, d))

        for i in range(N):
            _, idx = self.tree.query(self.coords[i], k=self.n_neighbors + 1)
            idx = idx[1:]
            dx = self.coords[idx] - self.coords[i]
            df = field[idx] - field[i]
            r = np.linalg.norm(dx, axis=1)

            # Laplacian ≈ (2/d_dim) * mean((f(x_j) - f(x_i)) / r_ij^2) * correction
            weights = 1.0 / (r ** 2 + 1e-8)
            for c in range(d):
                lap[i, c] = np.sum(weights * df[:, c]) / np.sum(weights) * 6.0

        if not is_vector:
            return lap[:, 0]
        return lap

    def divergence(self, field: np.ndarray) -> np.ndarray:
        """Approximate divergence ∇·f for vector field (N, 3)."""
        grad = self.gradient(field)  # (N, 3, 3)
        return np.trace(grad, axis1=1, axis2=2)  # (N,)

    def curl(self, field: np.ndarray) -> np.ndarray:
        """Approximate curl ∇×f for vector field (N, 3)."""
        grad = self.gradient(field)  # (N, 3, 3)
        # curl_x = ∂f_z/∂y - ∂f_y/∂z
        # curl_y = ∂f_x/∂z - ∂f_z/∂x
        # curl_z = ∂f_y/∂x - ∂f_x/∂y
        curl = np.zeros_like(field)
        curl[:, 0] = grad[:, 2, 1] - grad[:, 1, 2]
        curl[:, 1] = grad[:, 0, 2] - grad[:, 2, 0]
        curl[:, 2] = grad[:, 1, 0] - grad[:, 0, 1]
        return curl


class SINDyBaseline:
    """
    SINDy baseline for PDE discovery on irregular meshes.

    Builds a library of candidate terms using finite differences,
    then performs sparse regression (STLSQ) to identify active terms.
    """

    def __init__(self, n_neighbors: int = 20, threshold: float = 0.1,
                 max_iter: int = 20, alpha: float = 0.01):
        self.fd_lib = FiniteDifferenceLibrary(n_neighbors=n_neighbors)
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha
        self.xi = None
        self.term_names = None

    def build_library(self, coords: np.ndarray, u: np.ndarray
                      ) -> Tuple[np.ndarray, List[str]]:
        """
        Build the SINDy library from field data.

        Args:
            coords: (N, 3) node coordinates
            u: (N, 3) vector field values

        Returns:
            Theta: (N*3, P) library matrix
            term_names: list of P term names
        """
        self.fd_lib.fit(coords)

        terms = []
        names = []

        # Identity: u
        terms.append(u.reshape(-1))
        names.append('u')

        # Laplacian: ∇²u
        lap_u = self.fd_lib.laplacian(u)
        terms.append(lap_u.reshape(-1))
        names.append('∇²u')

        # Curl: ∇×u
        curl_u = self.fd_lib.curl(u)
        terms.append(curl_u.reshape(-1))
        names.append('∇×u')

        # Divergence gradient: ∇(∇·u)
        div_u = self.fd_lib.divergence(u)
        grad_div = self.fd_lib.gradient(div_u)
        terms.append(grad_div.reshape(-1))
        names.append('∇(∇·u)')

        # Advection: (u·∇)u
        grad_u = self.fd_lib.gradient(u)  # (N, 3, 3)
        advection = np.einsum('nd,nde->ne', u, grad_u)
        terms.append(advection.reshape(-1))
        names.append('(u·∇)u')

        # ||u||² * u (nonlinear)
        u_norm_sq = np.sum(u ** 2, axis=1, keepdims=True)
        terms.append((u_norm_sq * u).reshape(-1))
        names.append('||u||²u')

        Theta = np.stack(terms, axis=1)  # (N*3, P)
        return Theta, names

    def fit(self, coords: np.ndarray, u: np.ndarray, du_dt: np.ndarray):
        """
        Fit SINDy model using Sequential Thresholded Least Squares (STLSQ).

        Args:
            coords: (N, 3) coordinates
            u: (N, 3) field values
            du_dt: (N, 3) time derivative
        """
        Theta, self.term_names = self.build_library(coords, u)
        y = du_dt.reshape(-1)

        # STLSQ
        P = Theta.shape[1]
        xi = np.linalg.lstsq(Theta, y, rcond=None)[0]

        for _ in range(self.max_iter):
            small = np.abs(xi) < self.threshold
            xi[small] = 0.0
            big_idx = np.where(~small)[0]
            if len(big_idx) == 0:
                break
            xi[big_idx] = np.linalg.lstsq(
                Theta[:, big_idx], y, rcond=None
            )[0]

        self.xi = xi

    def predict(self, coords: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Predict du/dt using discovered model."""
        Theta, _ = self.build_library(coords, u)
        return (Theta @ self.xi).reshape(-1, 3)

    def get_discovered_pde(self) -> Dict[str, float]:
        """Return discovered PDE terms and coefficients."""
        if self.xi is None or self.term_names is None:
            return {}
        pde = {}
        for i, name in enumerate(self.term_names):
            if abs(self.xi[i]) > self.threshold:
                pde[name] = float(self.xi[i])
        return pde

    def evaluate(self, coords: np.ndarray, u_traj: np.ndarray,
                 dt: float) -> Dict[str, float]:
        """
        Evaluate SINDy baseline on trajectory data.

        Args:
            coords: (N, 3)
            u_traj: (T, N, 3) trajectory
            dt: timestep

        Returns:
            dict with rel_l2, pde_recovery info
        """
        T, N, d = u_traj.shape

        # Use first timestep for fitting
        u0 = u_traj[0]
        du_dt = (u_traj[1] - u_traj[0]) / dt
        self.fit(coords, u0, du_dt)

        # One-step prediction error
        pred = self.predict(coords, u0)
        u1_pred = u0 + dt * pred

        rel_l2 = np.linalg.norm(u1_pred - u_traj[1]) / (np.linalg.norm(u_traj[1]) + 1e-8)

        return {
            'rel_l2': rel_l2,
            'discovered_pde': self.get_discovered_pde(),
            'n_active_terms': sum(1 for v in self.xi if abs(v) > self.threshold),
        }
