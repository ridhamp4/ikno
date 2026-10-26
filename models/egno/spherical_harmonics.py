"""
Spherical harmonics, Clebsch-Gordan coefficients, and Wigner-D matrices
for SO(3)-equivariant operations.

Uses e3nn as backend for CG coefficients and spherical harmonics computation.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, List, Optional
from functools import lru_cache


# ---------------------------------------------------------------------------
#  Real Spherical Harmonics
# ---------------------------------------------------------------------------

def _associated_legendre(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """Compute associated Legendre polynomial P_l^m(x) (unnormalized)."""
    if l == m:
        if l == 0:
            return torch.ones_like(x)
        return -(2 * l - 1) * torch.sqrt(1 - x * x) * _associated_legendre(l - 1, l - 1, x)
    if l == m + 1:
        return x * (2 * m + 1) * _associated_legendre(m, m, x)
    return ((2 * l - 1) * x * _associated_legendre(l - 1, m, x)
            - (l + m - 1) * _associated_legendre(l - 2, m, x)) / (l - m)


def _real_sh_normalization(l: int, m: int) -> float:
    """Normalization constant for real spherical harmonics."""
    return math.sqrt(
        (2 * l + 1) / (4 * math.pi) *
        math.factorial(l - abs(m)) / math.factorial(l + abs(m))
    )


class SphericalHarmonics(nn.Module):
    """
    Compute real spherical harmonics Y^(l)_m(r_hat) for unit vectors.

    For a direction vector r_hat = (x, y, z) with |r_hat| = 1:
        Y^(l)_m returns a (2l+1)-dim vector for each input direction.

    Output transforms as Y^(l)(R @ r_hat) = D^(l)(R) @ Y^(l)(r_hat).
    """

    def __init__(self, l_max: int = 2):
        super().__init__()
        self.l_max = l_max

    def forward(self, r_hat: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Compute spherical harmonics for unit direction vectors.

        Args:
            r_hat: (N, 3) unit direction vectors

        Returns:
            dict mapping l -> (N, 2l+1) spherical harmonic coefficients
        """
        x, y, z = r_hat[:, 0], r_hat[:, 1], r_hat[:, 2]
        r_xy = torch.sqrt(x * x + y * y).clamp(min=1e-10)
        cos_theta = z
        phi = torch.atan2(y, x)

        result = {}
        for l in range(self.l_max + 1):
            Y_l = torch.zeros(r_hat.shape[0], 2 * l + 1,
                              device=r_hat.device, dtype=r_hat.dtype)
            for m_idx, m in enumerate(range(-l, l + 1)):
                abs_m = abs(m)
                P_lm = _associated_legendre(l, abs_m, cos_theta)
                norm = _real_sh_normalization(l, m)

                if m > 0:
                    Y_l[:, m_idx] = norm * math.sqrt(2) * P_lm * torch.cos(m * phi)
                elif m < 0:
                    Y_l[:, m_idx] = norm * math.sqrt(2) * P_lm * torch.sin(abs_m * phi)
                else:
                    Y_l[:, m_idx] = norm * P_lm

            result[l] = Y_l

        return result


# ---------------------------------------------------------------------------
#  Clebsch-Gordan Coefficients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _cg_coefficient(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> float:
    """
    Compute a single real Clebsch-Gordan coefficient C^{l3 m3}_{l1 m1, l2 m2}.
    Uses the Racah formula.
    """
    if m1 + m2 != m3:
        return 0.0
    if abs(l1 - l2) > l3 or l3 > l1 + l2:
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0

    # Use the standard formula for CG coefficients
    prefactor = math.sqrt(
        (2 * l3 + 1) *
        math.factorial(l1 + l2 - l3) *
        math.factorial(l1 - l2 + l3) *
        math.factorial(-l1 + l2 + l3) /
        math.factorial(l1 + l2 + l3 + 1)
    ) * math.sqrt(
        math.factorial(l3 + m3) * math.factorial(l3 - m3) *
        math.factorial(l1 + m1) * math.factorial(l1 - m1) *
        math.factorial(l2 + m2) * math.factorial(l2 - m2)
    )

    s_min = max(0, l2 - l3 - m1, l1 - l3 + m2)
    s_max = min(l1 + l2 - l3, l1 - m1, l2 + m2)

    total = 0.0
    for s in range(s_min, s_max + 1):
        total += ((-1) ** s /
                  (math.factorial(s) *
                   math.factorial(l1 + l2 - l3 - s) *
                   math.factorial(l1 - m1 - s) *
                   math.factorial(l2 + m2 - s) *
                   math.factorial(l3 - l2 + m1 + s) *
                   math.factorial(l3 - l1 - m2 + s)))

    return prefactor * total


def precompute_cg_matrix(l1: int, l2: int, l3: int) -> torch.Tensor:
    """
    Precompute the CG coefficient matrix for coupling l1 x l2 -> l3.

    Returns:
        (2*l1+1, 2*l2+1, 2*l3+1) tensor of CG coefficients
    """
    cg = torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    for m1_idx, m1 in enumerate(range(-l1, l1 + 1)):
        for m2_idx, m2 in enumerate(range(-l2, l2 + 1)):
            for m3_idx, m3 in enumerate(range(-l3, l3 + 1)):
                cg[m1_idx, m2_idx, m3_idx] = _cg_coefficient(l1, m1, l2, m2, l3, m3)
    return cg


class ClebschGordanProducts(nn.Module):
    """
    Clebsch-Gordan tensor product for SO(3) features.

    Computes (f^(l1) ⊗_CG^{l3} g^(l2)) for equivariant feature coupling.
    """

    def __init__(self, l_max: int = 2):
        super().__init__()
        self.l_max = l_max
        # Precompute and register all CG matrices
        for l1 in range(l_max + 1):
            for l2 in range(l_max + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                    cg = precompute_cg_matrix(l1, l2, l3)
                    self.register_buffer(f'cg_{l1}_{l2}_{l3}', cg)

    def product(self, f: torch.Tensor, g: torch.Tensor,
                l1: int, l2: int, l3: int) -> torch.Tensor:
        """
        Compute CG product: (f^(l1) ⊗_CG^{l3} g^(l2)).

        Args:
            f: (..., 2*l1+1, C1) type-l1 features
            g: (..., 2*l2+1, C2) type-l2 features
            l1, l2, l3: representation orders

        Returns:
            (..., 2*l3+1, C1) type-l3 features (contracted over l2 channels)
        """
        cg = getattr(self, f'cg_{l1}_{l2}_{l3}')  # (2l1+1, 2l2+1, 2l3+1)
        cg = cg.to(f.device)

        # f: (..., 2l1+1, C), g: (..., 2l2+1)
        # Contract: sum_{m1,m2} C^{l3,m3}_{l1,m1,l2,m2} * f_{m1} * g_{m2}
        if g.dim() == f.dim() - 1:
            # g has no channel dim (e.g., spherical harmonics)
            # f: (..., 2l1+1, C), g: (..., 2l2+1)
            # result: (..., 2l3+1, C)
            return torch.einsum('abc,...ac,...b->...cc',
                                cg.unsqueeze(-1).expand(-1, -1, -1, f.shape[-1]),
                                f, g).diagonal(dim1=-2, dim2=-1).unsqueeze(-1).squeeze(-1)

        # General case: both have channels
        # f: (..., 2l1+1, C), g: (..., 2l2+1, C)
        return torch.einsum('abc,...ac,...bc->...cc',
                            cg.unsqueeze(-1).expand(-1, -1, -1, f.shape[-1]),
                            f, g).diagonal(dim1=-2, dim2=-1).unsqueeze(-1).squeeze(-1)

    def tensor_product(self, f: torch.Tensor, g: torch.Tensor,
                       l1: int, l2: int, l3: int) -> torch.Tensor:
        """
        Simplified CG tensor product for message passing.

        Args:
            f: (N, 2*l1+1 * C) type-l1 features (channels interleaved)
            g: (N, 2*l2+1) type-l2 features (spherical harmonics, no channels)

        Returns:
            (N, 2*l3+1 * C) type-l3 features
        """
        cg = getattr(self, f'cg_{l1}_{l2}_{l3}').to(f.device)  # (2l1+1, 2l2+1, 2l3+1)

        dim_l1 = 2 * l1 + 1
        dim_l2 = 2 * l2 + 1
        dim_l3 = 2 * l3 + 1
        N = f.shape[0]
        C = f.shape[1] // dim_l1

        # Reshape f to (N, 2l1+1, C)
        f_r = f.view(N, dim_l1, C)
        # g is (N, 2l2+1)

        # result_{n, m3, c} = sum_{m1, m2} cg[m1, m2, m3] * f[n, m1, c] * g[n, m2]
        result = torch.einsum('ijk,nid,nj->nkd', cg, f_r, g)
        return result.reshape(N, dim_l3 * C)


# ---------------------------------------------------------------------------
#  Wigner-D Matrices (for testing equivariance)
# ---------------------------------------------------------------------------

def wigner_d_matrix(l: int, alpha: float, beta: float, gamma: float) -> torch.Tensor:
    """
    Compute real Wigner-D matrix D^(l)(R) for Euler angles (alpha, beta, gamma).
    Uses ZYZ convention.

    Returns:
        (2l+1, 2l+1) rotation matrix in the irrep-l basis
    """
    # Small d-matrix (rotation about y-axis by beta)
    d = torch.zeros(2 * l + 1, 2 * l + 1)
    for m1_idx, m1 in enumerate(range(-l, l + 1)):
        for m2_idx, m2 in enumerate(range(-l, l + 1)):
            d[m1_idx, m2_idx] = _small_d(l, m1, m2, beta)

    # Full D-matrix: D^l_{m1,m2} = e^{-i*m1*alpha} * d^l_{m1,m2} * e^{-i*m2*gamma}
    # For real spherical harmonics, we need the real form
    # This is a simplification; for testing we use rotation matrices directly
    return d


def _small_d(l: int, m1: int, m2: int, beta: float) -> float:
    """Wigner small-d matrix element."""
    cos_b2 = math.cos(beta / 2)
    sin_b2 = math.sin(beta / 2)

    s_min = max(0, m2 - m1)
    s_max = min(l + m2, l - m1)

    total = 0.0
    for s in range(s_min, s_max + 1):
        total += ((-1) ** (m1 - m2 + s) *
                  cos_b2 ** (2 * l + m2 - m1 - 2 * s) *
                  sin_b2 ** (m1 - m2 + 2 * s) /
                  (math.factorial(l + m2 - s) *
                   math.factorial(s) *
                   math.factorial(m1 - m2 + s) *
                   math.factorial(l - m1 - s)))

    prefactor = math.sqrt(
        math.factorial(l + m1) * math.factorial(l - m1) *
        math.factorial(l + m2) * math.factorial(l - m2)
    )
    return prefactor * total


# ---------------------------------------------------------------------------
#  Radial Basis Functions
# ---------------------------------------------------------------------------

class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion."""

    def __init__(self, n_rbf: int = 16, r_min: float = 0.0, r_max: float = 1.0,
                 trainable: bool = False):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(r_min, r_max, n_rbf)
        width = (r_max - r_min) / n_rbf
        widths = torch.full((n_rbf,), width)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: (...,) distances

        Returns:
            (..., n_rbf) RBF features
        """
        r = r.unsqueeze(-1)
        return torch.exp(-0.5 * ((r - self.centers) / self.widths) ** 2)


class BesselRBF(nn.Module):
    """Bessel radial basis functions (as used in DimeNet)."""

    def __init__(self, n_rbf: int = 16, r_max: float = 1.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.r_max = r_max
        freq = torch.arange(1, n_rbf + 1).float() * math.pi / r_max
        self.register_buffer('freq', freq)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r.unsqueeze(-1).clamp(min=1e-8)
        return math.sqrt(2.0 / self.r_max) * torch.sin(self.freq * r) / r


class SmoothCutoff(nn.Module):
    """Smooth polynomial cutoff function."""

    def __init__(self, r_max: float, p: int = 6):
        super().__init__()
        self.r_max = r_max
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.r_max
        return torch.where(
            x < 1.0,
            1.0 - ((self.p + 1) * (self.p + 2) / 2) * x ** self.p
            + self.p * (self.p + 2) * x ** (self.p + 1)
            - (self.p * (self.p + 1) / 2) * x ** (self.p + 2),
            torch.zeros_like(x)
        )
