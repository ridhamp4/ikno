"""
utils.py  –  Shared utilities for E-GNO dataset generation.
Mesh generation, spectral helpers, interpolation, and storage.
Dependencies: numpy, scipy
"""

import numpy as np
import os, json, time


# ─────────────────────────────────────────────────────────────
#  Irregular mesh generation
# ─────────────────────────────────────────────────────────────

def halton_sequence(n, base):
    """Generate a 1-D Halton low-discrepancy sequence of length n."""
    seq = np.zeros(n)
    num, den = 0, 1
    for i in range(n):
        x = den - num
        if x == 1:
            num, den = 1, den * base
        else:
            y = den // base
            while x <= y:
                y //= base
            num = (base + 1) * y - x
        seq[i] = num / den
    return seq


def generate_irregular_mesh(n_nodes, seed=42):
    """
    Generate a quasi-random 3-D point cloud in [0,1]^3
    using a Halton sequence (low-discrepancy, better coverage
    than pure uniform random).

    Returns
    -------
    coords : ndarray, shape (n_nodes, 3), dtype float32
    """
    rng = np.random.default_rng(seed)
    # Mix Halton with small jitter for true irregularity
    x = halton_sequence(n_nodes, 2)
    y = halton_sequence(n_nodes, 3)
    z = halton_sequence(n_nodes, 5)
    coords = np.stack([x, y, z], axis=1)
    # Add small jitter (~1/sqrt(N) scale)
    jitter_scale = 0.4 / np.sqrt(n_nodes)
    coords += rng.normal(0, jitter_scale, coords.shape)
    coords = np.clip(coords, 0.01, 0.99).astype(np.float32)
    return coords


# ─────────────────────────────────────────────────────────────
#  Grid construction helpers
# ─────────────────────────────────────────────────────────────

def make_grid(N, L=2*np.pi):
    """
    Build a 3-D uniform grid on [0, L)^3 with N points per side.

    Returns
    -------
    x, y, z  : 1-D arrays of grid coordinates
    kx, ky, kz : 3-D wavenumber arrays (for spectral ops, complex dtype
                  to avoid numpy 2.x mixed-type operator bugs with Python 3.14)
    """
    dx = L / N
    x = np.linspace(0, L - dx, N)
    kvals = np.fft.fftfreq(N, d=1.0/N)          # wavenumbers in 1/L units
    kx, ky, kz = np.meshgrid(kvals, kvals, kvals, indexing='ij')
    # Cast to complex to avoid mixed float*complex operator bugs
    kx = kx.astype(np.complex128)
    ky = ky.astype(np.complex128)
    kz = kz.astype(np.complex128)
    return x, kvals, kx, ky, kz


def rfft3(f):
    """3-D FFT of real-valued 3-D field f (N,N,N) → complex (N,N,N//2+1)."""
    return np.fft.rfftn(f)

def irfft3(fh, N):
    """Inverse of rfft3."""
    return np.fft.irfftn(fh, s=(N, N, N))


# ─────────────────────────────────────────────────────────────
#  Spectral differential operators  (full complex FFT version)
# ─────────────────────────────────────────────────────────────

def fftn3(f):
    return np.fft.fftn(f)

def ifftn3(fh):
    return np.fft.ifftn(fh).real


def curl_hat(Fx_h, Fy_h, Fz_h, kx, ky, kz):
    """
    Spectral curl:  (∇×F)^  in Fourier space.
    Returns three arrays (curlx_h, curly_h, curlz_h).
    """
    # (∇×F)_x = ∂Fz/∂y - ∂Fy/∂z
    # (∇×F)_y = ∂Fx/∂z - ∂Fz/∂x
    # (∇×F)_z = ∂Fy/∂x - ∂Fx/∂y
    cx = 1j*ky*Fz_h - 1j*kz*Fy_h
    cy = 1j*kz*Fx_h - 1j*kx*Fz_h
    cz = 1j*kx*Fy_h - 1j*ky*Fx_h
    return cx, cy, cz


def div_hat(Fx_h, Fy_h, Fz_h, kx, ky, kz):
    """Spectral divergence → scalar field in Fourier space."""
    return 1j*kx*Fx_h + 1j*ky*Fy_h + 1j*kz*Fz_h


def grad_hat(phi_h, kx, ky, kz):
    """Spectral gradient of scalar → vector field in Fourier space."""
    return 1j*kx*phi_h, 1j*ky*phi_h, 1j*kz*phi_h


def laplacian_hat(Fx_h, Fy_h, Fz_h, kx, ky, kz):
    """Spectral vector Laplacian → three Fourier arrays."""
    k2 = kx**2 + ky**2 + kz**2
    return -k2*Fx_h, -k2*Fy_h, -k2*Fz_h


def leray_project(ux_h, uy_h, uz_h, kx, ky, kz):
    """
    Helmholtz-Hodge projection onto divergence-free subspace.
    P̂ᵢⱼ = δᵢⱼ - kᵢkⱼ/|k|²
    Sets the k=0 mode (mean) to zero.
    """
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # avoid division by zero; k=0 mode set to 0 below
    # k·û (dot product in Fourier space, no 'i' factor)
    kdotu = kx * ux_h + ky * uy_h + kz * uz_h
    factor = kdotu / k2
    ux_h = ux_h - kx * factor
    uy_h = uy_h - ky * factor
    uz_h = uz_h - kz * factor
    # Zero mean flow
    ux_h[0, 0, 0] = 0.0
    uy_h[0, 0, 0] = 0.0
    uz_h[0, 0, 0] = 0.0
    return ux_h, uy_h, uz_h


def dealias_filter(N):
    """
    2/3-rule dealiasing filter mask for an N^3 grid.
    Returns a complex mask (N,N,N): 1+0j for kept modes, 0+0j for filtered.
    Complex dtype avoids numpy 2.x mixed-type operator bugs with Python 3.14.
    """
    k_max = N // 3
    kv = np.fft.fftfreq(N, d=1.0/N)
    kx, ky, kz = np.meshgrid(kv, kv, kv, indexing='ij')
    mask = (np.abs(kx) < k_max) & (np.abs(ky) < k_max) & (np.abs(kz) < k_max)
    return mask.astype(np.complex128)


# ─────────────────────────────────────────────────────────────
#  RK4 integrator
# ─────────────────────────────────────────────────────────────

def rk4_step(state, rhs_fn, dt):
    """
    Classic 4th-order Runge-Kutta step.
    state : tuple of arrays (arbitrarily many fields)
    rhs_fn : callable(state) → tuple of arrays (same shape as state)
    """
    k1 = rhs_fn(state)
    k2 = rhs_fn(tuple(s + 0.5*dt*k for s, k in zip(state, k1)))
    k3 = rhs_fn(tuple(s + 0.5*dt*k for s, k in zip(state, k2)))
    k4 = rhs_fn(tuple(s + dt*k for s, k in zip(state, k3)))
    return tuple(
        s + (dt/6.0)*(k1i + 2*k2i + 2*k3i + k4i)
        for s, k1i, k2i, k3i, k4i in zip(state, k1, k2, k3, k4)
    )


# ─────────────────────────────────────────────────────────────
#  Smooth initial condition generation
# ─────────────────────────────────────────────────────────────

def smooth_random_field_hat(N, k_max=8, k_min=2, seed=None, dtype=complex):
    """
    Generate a smooth random scalar field in Fourier space.
    Energy concentrated in low wavenumbers k_min ≤ |k| ≤ k_max.
    """
    rng = np.random.default_rng(seed)
    kv = np.fft.fftfreq(N, d=1.0/N)
    kx, ky, kz = np.meshgrid(kv, kv, kv, indexing='ij')
    k_norm = np.sqrt(kx**2 + ky**2 + kz**2)
    # Band-pass energy spectrum
    envelope = np.exp(-0.5 * ((k_norm - (k_min+k_max)/2) / 3)**2)
    envelope[k_norm < 1] = 0.0
    # Random phases
    amplitude = envelope * (rng.standard_normal((N, N, N)) +
                            1j * rng.standard_normal((N, N, N)))
    # Enforce Hermitian symmetry for real physical field
    field_real = np.fft.ifftn(amplitude).real
    fh = np.fft.fftn(field_real)  # guaranteed Hermitian
    return fh


def smooth_divergence_free_field(N, k_max=6, seed=None):
    """
    Generate a smooth divergence-free vector field via the curl of
    a random smooth vector potential  A:  u = ∇×A.
    This ensures ∇·u = 0 exactly (up to machine precision).

    Returns three physical-space fields ux, uy, uz  each (N,N,N).
    """
    _, _, kx, ky, kz = make_grid(N)
    Ax_h = smooth_random_field_hat(N, k_max=k_max, seed=seed)
    Ay_h = smooth_random_field_hat(N, k_max=k_max, seed=None if seed is None else seed+1)
    Az_h = smooth_random_field_hat(N, k_max=k_max, seed=None if seed is None else seed+2)
    ux_h, uy_h, uz_h = curl_hat(Ax_h, Ay_h, Az_h, kx, ky, kz)
    ux = np.fft.ifftn(ux_h).real
    uy = np.fft.ifftn(uy_h).real
    uz = np.fft.ifftn(uz_h).real
    # Normalise to unit RMS velocity
    rms = np.sqrt(np.mean(ux**2 + uy**2 + uz**2))
    if rms > 1e-10:
        ux, uy, uz = ux/rms, uy/rms, uz/rms
    return ux, uy, uz


# ─────────────────────────────────────────────────────────────
#  Trilinear interpolation from grid to irregular mesh (pure numpy)
# ─────────────────────────────────────────────────────────────

def _trilinear_interp(field_3d, pts_scaled):
    """
    Pure-numpy trilinear interpolation.

    Parameters
    ----------
    field_3d   : (N,N,N) float array on a uniform grid in [0,N)
    pts_scaled : (M,3) float array, coordinates in [0,N-1] (continuous index)

    Returns
    -------
    (M,) interpolated values
    """
    N = field_3d.shape[0]
    pts = np.clip(pts_scaled, 0, N - 1 - 1e-6)

    x0 = np.floor(pts[:, 0]).astype(int)
    y0 = np.floor(pts[:, 1]).astype(int)
    z0 = np.floor(pts[:, 2]).astype(int)
    x1 = np.minimum(x0 + 1, N - 1)
    y1 = np.minimum(y0 + 1, N - 1)
    z1 = np.minimum(z0 + 1, N - 1)

    xd = pts[:, 0] - x0
    yd = pts[:, 1] - y0
    zd = pts[:, 2] - z0

    # 8-corner values
    c000 = field_3d[x0, y0, z0];  c001 = field_3d[x0, y0, z1]
    c010 = field_3d[x0, y1, z0];  c011 = field_3d[x0, y1, z1]
    c100 = field_3d[x1, y0, z0];  c101 = field_3d[x1, y0, z1]
    c110 = field_3d[x1, y1, z0];  c111 = field_3d[x1, y1, z1]

    # Trilinear blend
    c00 = c000*(1-xd) + c100*xd;  c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd;  c11 = c011*(1-xd) + c111*xd
    c0  = c00*(1-yd)  + c10*yd;   c1  = c01*(1-yd)  + c11*yd
    return (c0*(1-zd) + c1*zd).astype(np.float32)


def grid_to_mesh(field_3d, grid_1d, mesh_coords_phys):
    """
    Interpolate a 3-D field from a uniform grid to an irregular mesh.

    Parameters
    ----------
    field_3d         : (N,N,N) physical-space field
    grid_1d          : (N,) array of grid coordinates (e.g. [0, dx, 2dx, ...])
    mesh_coords_phys : (M,3) mesh coordinates in the same units as grid_1d

    Returns
    -------
    values : (M,) float32 interpolated at mesh points
    """
    N   = len(grid_1d)
    dx  = grid_1d[1] - grid_1d[0]
    # Convert physical coordinates to continuous grid indices
    pts_idx = (mesh_coords_phys - grid_1d[0]) / dx
    return _trilinear_interp(field_3d, pts_idx)


def vector_field_to_mesh(fields_3d, grid_1d, mesh_coords_phys):
    """
    Interpolate a list of 3 scalar fields (vector components) to an
    irregular mesh.

    Parameters
    ----------
    fields_3d        : list of 3 arrays, each (N,N,N)
    grid_1d          : (N,) grid coordinate array
    mesh_coords_phys : (M,3) physical coordinates

    Returns
    -------
    (M, 3) float32 array
    """
    components = [grid_to_mesh(f, grid_1d, mesh_coords_phys) for f in fields_3d]
    return np.stack(components, axis=1)   # (M, 3)


# ─────────────────────────────────────────────────────────────
#  Noise injection
# ─────────────────────────────────────────────────────────────

def add_noise(field, noise_level=0.01, seed=None):
    """
    Add Gaussian noise at `noise_level` fraction of the field's
    standard deviation.  field : (...) float array.
    """
    rng = np.random.default_rng(seed)
    sigma = noise_level * np.std(field)
    return (field + rng.standard_normal(field.shape).astype(field.dtype) * sigma).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  Storage
# ─────────────────────────────────────────────────────────────

def save_trajectory(save_path, coords, fields_dict, times, metadata=None):
    """
    Save a single trajectory to a .npz file.

    Parameters
    ----------
    save_path   : str, path ending in .npz
    coords      : (N,3) float32 — spatial coordinates
    fields_dict : dict of str → (T,N,d) float32 arrays
    times       : (T,) float32 — time stamps
    metadata    : optional dict of scalars/strings
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {'coords': coords, 'times': times}
    save_dict.update(fields_dict)
    if metadata:
        # encode metadata as JSON string in a scalar array
        save_dict['metadata_json'] = np.array(json.dumps(metadata))
    np.savez_compressed(save_path, **save_dict)


def print_progress(i, total, start_time, label=''):
    elapsed = time.time() - start_time
    rate = (i+1) / elapsed if elapsed > 0 else 0
    eta = (total - i - 1) / rate if rate > 0 else 0
    print(f'\r  {label} [{i+1:3d}/{total}]  '
          f'{elapsed:.1f}s elapsed  ETA {eta:.0f}s', end='', flush=True)
