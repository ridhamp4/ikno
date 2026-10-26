"""
generate_elasticity.py  –  Dataset generator for 3-D linear elasticity
                            (elastic wave equation).

Solves the isotropic elastic wave equation on a periodic domain [0, 2π]³
using a pseudo-spectral method written as a first-order system.

Physics
-------
    ρ ∂²u/∂t² = μ ∇²u + (λ+μ) ∇(∇·u)

    Equivalent to Lamé's equations with P-wave speed cp and S-wave speed cs:

    ∂²u/∂t² = cs² ∇²u + (cp²−cs²) ∇(∇·u)

    where  cp² = (λ + 2μ)/ρ,   cs² = μ/ρ

    Written as a first-order system for RK4 integration:

    ∂u/∂t = v
    ∂v/∂t = cs² ∇²u + (cp²−cs²) ∇(∇·u)

    In Fourier space:
    ∂û/∂t = v̂
    ∂v̂/∂t = −cs² |k|² û − (cp²−cs²) k (k·û)

Output layout
-------------
    data/elasticity/
        train/traj_000.npz … traj_099.npz
        test/traj_000.npz  … traj_019.npz
    Each .npz contains:
        coords : (N_nodes, 3)    float32 — spatial coordinates in [0,1]³
        u      : (T, N_nodes, 3) float32 — displacement field
        v      : (T, N_nodes, 3) float32 — velocity field (∂u/∂t)
        times  : (T,)            float32 — time values

Dependencies: numpy, scipy
"""

import numpy as np
import os, time, argparse
from utils import (
    make_grid, fftn3, smooth_random_field_hat, smooth_divergence_free_field,
    generate_irregular_mesh, vector_field_to_mesh,
    add_noise, save_trajectory, rk4_step, print_progress,
    div_hat, grad_hat,
)


# ─────────────────────────────────────────────────────────────
#  Elastic wave RHS  (spectral)
# ─────────────────────────────────────────────────────────────

def elasticity_rhs(state, kx, ky, kz, cp2, cs2):
    """
    Spectral RHS for the elastic wave equation (first-order form).

    state = (ux_h, uy_h, uz_h, vx_h, vy_h, vz_h)
    where u = displacement, v = velocity = ∂u/∂t.

    ∂u/∂t = v
    ∂v/∂t = cs²∇²u + (cp²−cs²)∇(∇·u)
    """
    ux_h, uy_h, uz_h, vx_h, vy_h, vz_h = state
    k2 = kx**2 + ky**2 + kz**2

    # ∂u/∂t = v
    dux_h = vx_h
    duy_h = vy_h
    duz_h = vz_h

    # ∇·u  (divergence of displacement, in Fourier space)
    divu_h = div_hat(ux_h, uy_h, uz_h, kx, ky, kz)

    # ∇(∇·u)  (gradient of divergence)
    grad_divu_x, grad_divu_y, grad_divu_z = grad_hat(divu_h, kx, ky, kz)

    # ∂v/∂t = cs²∇²u + (cp²−cs²)∇(∇·u)
    dvx_h = -cs2 * k2 * ux_h + (cp2 - cs2) * grad_divu_x
    dvy_h = -cs2 * k2 * uy_h + (cp2 - cs2) * grad_divu_y
    dvz_h = -cs2 * k2 * uz_h + (cp2 - cs2) * grad_divu_z

    return (dux_h, duy_h, duz_h, dvx_h, dvy_h, dvz_h)


# ─────────────────────────────────────────────────────────────
#  Initial condition generator
# ─────────────────────────────────────────────────────────────

def make_elastic_ic(N, ic_type='mixed', k_max=5, seed=None):
    """
    Generate initial conditions for the elastic wave equation.

    ic_type:
        'p_wave'   — purely compressional initial displacement (∇·u ≠ 0, curl u = 0)
        's_wave'   — purely shear initial displacement (∇·u = 0, curl u ≠ 0)
        'mixed'    — mix of both (most general)

    Returns ux, uy, uz, vx, vy, vz  each (N,N,N).
    The initial velocity is set to zero for simplicity (wave packet problem).
    """
    _, _, kx, ky, kz = make_grid(N)

    if ic_type == 's_wave':
        # Divergence-free displacement (pure shear wave)
        ux, uy, uz = smooth_divergence_free_field(N, k_max=k_max, seed=seed)

    elif ic_type == 'p_wave':
        # Compressional displacement: u = ∇φ  for smooth scalar φ
        phi_h = smooth_random_field_hat(N, k_max=k_max, seed=seed)
        gx_h, gy_h, gz_h = grad_hat(phi_h, kx, ky, kz)
        ux = np.fft.ifftn(gx_h).real
        uy = np.fft.ifftn(gy_h).real
        uz = np.fft.ifftn(gz_h).real
        rms = np.sqrt(np.mean(ux**2 + uy**2 + uz**2)) + 1e-12
        ux, uy, uz = ux/rms, uy/rms, uz/rms

    else:  # 'mixed'
        ux_s, uy_s, uz_s = smooth_divergence_free_field(N, k_max=k_max, seed=seed)
        phi_h = smooth_random_field_hat(N, k_max=k_max,
                                        seed=None if seed is None else seed+1)
        gx_h, gy_h, gz_h = grad_hat(phi_h, kx, ky, kz)
        ux_p = np.fft.ifftn(gx_h).real
        uy_p = np.fft.ifftn(gy_h).real
        uz_p = np.fft.ifftn(gz_h).real
        rms_p = np.sqrt(np.mean(ux_p**2+uy_p**2+uz_p**2)) + 1e-12
        ux_p, uy_p, uz_p = ux_p/rms_p, uy_p/rms_p, uz_p/rms_p
        # Combine 70% shear, 30% compressional
        ux = 0.7*ux_s + 0.3*ux_p
        uy = 0.7*uy_s + 0.3*uy_p
        uz = 0.7*uz_s + 0.3*uz_p

    # Zero initial velocity → wave packet evolution
    vx = np.zeros_like(ux)
    vy = np.zeros_like(uy)
    vz = np.zeros_like(uz)

    return ux, uy, uz, vx, vy, vz


# ─────────────────────────────────────────────────────────────
#  Single trajectory solver
# ─────────────────────────────────────────────────────────────

def solve_elasticity(N, cp, cs, dt, n_steps, save_every,
                     ic_type='mixed', seed=None):
    """
    Solve the elastic wave equation for one trajectory.

    Returns list of (ux, uy, uz, vx, vy, vz) physical-space snapshots.
    """
    _, _, kx, ky, kz = make_grid(N)
    cp2, cs2 = cp**2, cs**2

    ux, uy, uz, vx, vy, vz = make_elastic_ic(N, ic_type=ic_type, seed=seed)
    state = (
        fftn3(ux), fftn3(uy), fftn3(uz),
        fftn3(vx), fftn3(vy), fftn3(vz),
    )

    rhs = lambda s: elasticity_rhs(s, kx, ky, kz, cp2, cs2)
    snapshots = []

    # Add numerical dissipation (spectral viscosity) to prevent aliasing
    # at high wavenumbers — apply a mild spectral filter every N_filter steps
    k2 = kx**2 + ky**2 + kz**2
    k_nyq = N // 2
    spectral_filter = np.exp(-36 * (np.sqrt(k2) / k_nyq)**36)  # super-Gaussian

    for step in range(n_steps):
        if step % save_every == 0:
            snap = tuple(np.fft.ifftn(fh).real for fh in state)
            snapshots.append(snap)
        state = rk4_step(state, rhs, dt)
        # Apply spectral filter every 10 steps to control aliasing
        if step % 10 == 9:
            state = tuple(fh * spectral_filter for fh in state)

    return snapshots


# ─────────────────────────────────────────────────────────────
#  Dataset generation loop
# ─────────────────────────────────────────────────────────────

# Available material configurations (P-wave and S-wave speeds)
MATERIAL_CONFIGS = [
    # (name, cp, cs)  — representative elastic materials
    ('granite',    6.0, 3.5),
    ('sandstone',  4.5, 2.5),
    ('steel',      5.9, 3.2),
    ('rubber',     1.8, 0.9),
    ('concrete',   4.2, 2.4),
]


def generate_elasticity_dataset(
    out_dir='data/elasticity',
    N=64,
    n_nodes=6000,
    n_train=100,
    n_test=20,
    T_steps=50,
    dt=0.003,
    save_every=2,
    noise_level=0.01,
    base_seed=0,
    ic_type='mixed',       # 'mixed', 'p_wave', or 's_wave'
    fixed_material=None,
):
    """
    Generate the full linear elasticity dataset.

    Material properties are cycled across trajectories to produce
    diverse P/S wave speed combinations, increasing the difficulty of
    PDE discovery (the model must recover the Lamé coefficients).
    """
    grid_1d, _, _, _, _ = make_grid(N)
    times = np.arange(T_steps, dtype=np.float32) * dt * save_every

    t0 = time.time()
    for split, n_traj in [('train', n_train), ('test', n_test)]:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f'\n[Elasticity] Generating {split} set ({n_traj} trajectories)...')
        split_start = time.time()

        for i in range(n_traj):
            seed = base_seed + (0 if split == 'train' else n_train) + i
            if fixed_material is None:
                mat_name, cp, cs = MATERIAL_CONFIGS[i % len(MATERIAL_CONFIGS)]
            else:
                matches = [item for item in MATERIAL_CONFIGS if item[0] == fixed_material]
                if not matches:
                    raise ValueError(
                        f'Unknown fixed_material={fixed_material!r}; choose from '
                        f'{[item[0] for item in MATERIAL_CONFIGS]}')
                mat_name, cp, cs = matches[0]
            # Normalise wave speeds for numerical stability (keep cp,cs in [1,7])
            cp_norm = cp / 6.0 * 2.0    # rescale to ~[0.3, 2.0]
            cs_norm = cs / 6.0 * 2.0

            # --- 1. Solve on regular grid ---
            snapshots = solve_elasticity(
                N=N, cp=cp_norm, cs=cs_norm,
                dt=dt,
                n_steps=T_steps * save_every,
                save_every=save_every,
                ic_type=ic_type,
                seed=seed,
            )

            # --- 2. Irregular mesh ---
            coords = generate_irregular_mesh(n_nodes, seed=seed + 50000)

            # --- 3. Interpolate displacement and velocity to mesh ---
            u_traj = np.zeros((T_steps, n_nodes, 3), dtype=np.float32)
            v_traj = np.zeros((T_steps, n_nodes, 3), dtype=np.float32)
            for t_idx, snap in enumerate(snapshots):
                ux, uy, uz, vx, vy, vz = snap
                u_traj[t_idx] = vector_field_to_mesh(
                    [ux, uy, uz], grid_1d, coords * (2 * np.pi)
                )
                v_traj[t_idx] = vector_field_to_mesh(
                    [vx, vy, vz], grid_1d, coords * (2 * np.pi)
                )

            # --- 4. Noise ---
            u_traj = add_noise(u_traj, noise_level=noise_level, seed=seed)
            v_traj = add_noise(v_traj, noise_level=noise_level, seed=seed+1)

            # --- 5. Save ---
            path = os.path.join(split_dir, f'traj_{i:03d}.npz')
            save_trajectory(
                path,
                coords=coords,
                fields_dict={'u': u_traj, 'v': v_traj},
                times=times,
                metadata={
                    'pde': 'elasticity',
                    'grid_N': N, 'dt': dt,
                    'material': mat_name,
                    'cp': cp_norm, 'cs': cs_norm,
                    'cp2_minus_cs2': cp_norm**2 - cs_norm**2,  # key PDE param
                    'ic_type': ic_type,
                    'noise_level': noise_level,
                    'seed': seed,
                }
            )
            print_progress(i, n_traj, split_start, label=f'Elast/{split}')

        print()

    total = time.time() - t0
    print(f'\n[Elasticity] Done. Total time: {total:.1f}s')
    print(f'             Output directory: {os.path.abspath(out_dir)}')


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 3D linear elasticity dataset for E-GNO'
    )
    parser.add_argument('--out_dir',  default='data/elasticity')
    parser.add_argument('--N',        type=int,   default=64)
    parser.add_argument('--n_nodes',  type=int,   default=6000)
    parser.add_argument('--n_train',  type=int,   default=100)
    parser.add_argument('--n_test',   type=int,   default=20)
    parser.add_argument('--T_steps',  type=int,   default=50)
    parser.add_argument('--dt',       type=float, default=0.003)
    parser.add_argument('--noise',    type=float, default=0.01)
    parser.add_argument('--ic_type',  default='mixed',
                        choices=['mixed', 'p_wave', 's_wave'])
    parser.add_argument('--seed',     type=int,   default=0)
    args = parser.parse_args()

    generate_elasticity_dataset(
        out_dir=args.out_dir,
        N=args.N, n_nodes=args.n_nodes,
        n_train=args.n_train, n_test=args.n_test,
        T_steps=args.T_steps, dt=args.dt,
        noise_level=args.noise,
        ic_type=args.ic_type,
        base_seed=args.seed,
    )
