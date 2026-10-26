"""
generate_maxwell.py  –  Dataset generator for Maxwell's equations.

Solves the vacuum Maxwell equations on a periodic domain [0, 2π]³
using a pseudo-spectral method, then sub-samples the solution onto
an irregular 3-D point cloud.

Physics
-------
    ∂E/∂t =  ∇ × B          (Faraday → Ampere in Gaussian units, c=1)
    ∂B/∂t = -∇ × E          (Faraday's law)

Divergence constraints (∇·E = 0, ∇·B = 0) are enforced by initialising
both fields as curls of smooth random vector potentials.

Output layout
-------------
    data/maxwell/
        train/traj_000.npz … traj_099.npz
        test/traj_000.npz  … traj_019.npz
    Each .npz contains:
        coords : (N_nodes, 3)  float32  — spatial coordinates in [0,1]³
        E      : (T, N_nodes, 3) float32 — electric field at each timestep
        B      : (T, N_nodes, 3) float32 — magnetic field at each timestep
        times  : (T,)           float32  — time values

Dependencies: numpy, scipy
"""

import numpy as np
import os, time, argparse
from utils import (
    make_grid, fftn3, ifftn3, curl_hat, smooth_divergence_free_field,
    smooth_random_field_hat, generate_irregular_mesh, vector_field_to_mesh,
    add_noise, save_trajectory, rk4_step, print_progress
)


# ─────────────────────────────────────────────────────────────
#  Maxwell RHS  (spectral)
# ─────────────────────────────────────────────────────────────

def maxwell_rhs(state, kx, ky, kz):
    """
    Right-hand side of Maxwell's equations in Fourier space.

    state = (Ex_h, Ey_h, Ez_h, Bx_h, By_h, Bz_h)
    Returns d/dt of the same tuple.
    """
    Ex_h, Ey_h, Ez_h, Bx_h, By_h, Bz_h = state

    # ∂E/∂t = ∇×B
    dEx_h, dEy_h, dEz_h = curl_hat(Bx_h, By_h, Bz_h, kx, ky, kz)

    # ∂B/∂t = -∇×E
    cEx_h, cEy_h, cEz_h = curl_hat(Ex_h, Ey_h, Ez_h, kx, ky, kz)
    dBx_h = -cEx_h
    dBy_h = -cEy_h
    dBz_h = -cEz_h

    return (dEx_h, dEy_h, dEz_h, dBx_h, dBy_h, dBz_h)


def divergence_clean(field_h, kx, ky, kz):
    """
    Project a Fourier-space vector field to its divergence-free part.
    (Helmholtz decomposition: keep only the solenoidal component.)
    """
    from utils import div_hat, grad_hat
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    Fx_h, Fy_h, Fz_h = field_h
    div_h = div_hat(Fx_h, Fy_h, Fz_h, kx, ky, kz)
    # Remove the compressible part: F → F - ∇(∇·F / |k|²)
    phi_h = div_h / k2
    gx_h, gy_h, gz_h = grad_hat(phi_h, kx, ky, kz)
    return Fx_h - gx_h, Fy_h - gy_h, Fz_h - gz_h


# ─────────────────────────────────────────────────────────────
#  Initial condition generator
# ─────────────────────────────────────────────────────────────

def make_initial_condition(N, k_max=6, seed=None):
    """
    Generate divergence-free initial E and B fields.
    Both fields are initialized as  ∇ × A  for smooth random A.
    Returns six (N,N,N) physical-space arrays.
    """
    _, _, kx, ky, kz = make_grid(N)
    Ex, Ey, Ez = smooth_divergence_free_field(N, k_max=k_max, seed=seed)
    Bx, By, Bz = smooth_divergence_free_field(N, k_max=k_max,
                                               seed=None if seed is None else seed + 100)
    # Scale B to have similar energy as E
    rms_E = np.sqrt(np.mean(Ex**2 + Ey**2 + Ez**2))
    rms_B = np.sqrt(np.mean(Bx**2 + By**2 + Bz**2))
    if rms_B > 1e-10:
        Bx, By, Bz = Bx*(rms_E/rms_B), By*(rms_E/rms_B), Bz*(rms_E/rms_B)
    return Ex, Ey, Ez, Bx, By, Bz


# ─────────────────────────────────────────────────────────────
#  Single trajectory solver
# ─────────────────────────────────────────────────────────────

def solve_maxwell(N, dt, n_steps, save_every, seed):
    """
    Solve Maxwell's equations for one trajectory.

    Returns list of (Ex,Ey,Ez,Bx,By,Bz) snapshots,
    each a tuple of three (N,N,N) arrays.
    """
    _, _, kx, ky, kz = make_grid(N)
    Ex, Ey, Ez, Bx, By, Bz = make_initial_condition(N, seed=seed)

    # Transform to Fourier space
    state = (
        fftn3(Ex), fftn3(Ey), fftn3(Ez),
        fftn3(Bx), fftn3(By), fftn3(Bz),
    )

    rhs = lambda s: maxwell_rhs(s, kx, ky, kz)
    snapshots = []

    for step in range(n_steps):
        if step % save_every == 0:
            # Transform back to physical space for this snapshot
            snaps = tuple(np.fft.ifftn(fh).real for fh in state)
            snapshots.append(snaps)
        state = rk4_step(state, rhs, dt)

    return snapshots


# ─────────────────────────────────────────────────────────────
#  Dataset generation loop
# ─────────────────────────────────────────────────────────────

def generate_maxwell_dataset(
    out_dir='data/maxwell',
    N=64,               # Grid size per side (64³ default)
    n_nodes=5000,       # Irregular mesh nodes
    n_train=100,
    n_test=20,
    T_steps=50,         # Number of saved timesteps per trajectory
    dt=0.005,           # Time step
    save_every=1,       # Save every n_steps PDE steps between snapshots
    noise_level=0.01,
    base_seed=0,
):
    """
    Generate the full Maxwell dataset.

    Effective simulation time per trajectory:
        t_total = dt * save_every * T_steps
    """
    grid_1d, _, _, _, _ = make_grid(N)
    times = np.arange(T_steps, dtype=np.float32) * dt * save_every

    t0 = time.time()
    for split, n_traj in [('train', n_train), ('test', n_test)]:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f'\n[Maxwell] Generating {split} set ({n_traj} trajectories)...')
        split_start = time.time()

        for i in range(n_traj):
            seed = base_seed + (0 if split == 'train' else n_train) + i

            # --- 1. Solve on regular grid ---
            snapshots = solve_maxwell(
                N=N, dt=dt,
                n_steps=T_steps * save_every,
                save_every=save_every,
                seed=seed,
            )

            # --- 2. Generate irregular mesh (different mesh per trajectory) ---
            coords = generate_irregular_mesh(n_nodes, seed=seed + 50000)

            # --- 3. Interpolate E and B to the irregular mesh ---
            E_traj = np.zeros((T_steps, n_nodes, 3), dtype=np.float32)
            B_traj = np.zeros((T_steps, n_nodes, 3), dtype=np.float32)
            for t_idx, snap in enumerate(snapshots):
                Ex, Ey, Ez, Bx, By, Bz = snap
                # coords are in [0,1]^3; scale to physical domain [0, 2π]^3
                coords_phys = coords * (2 * np.pi)
                E_traj[t_idx] = vector_field_to_mesh(
                    [Ex, Ey, Ez], grid_1d, coords_phys
                )
                B_traj[t_idx] = vector_field_to_mesh(
                    [Bx, By, Bz], grid_1d, coords_phys
                )

            # --- 4. Add observation noise ---
            E_traj = add_noise(E_traj, noise_level=noise_level, seed=seed)
            B_traj = add_noise(B_traj, noise_level=noise_level, seed=seed+1)

            # --- 5. Save ---
            path = os.path.join(split_dir, f'traj_{i:03d}.npz')
            save_trajectory(
                path,
                coords=coords,
                fields_dict={'E': E_traj, 'B': B_traj},
                times=times,
                metadata={
                    'pde': 'maxwell',
                    'grid_N': N,
                    'dt': dt,
                    'noise_level': noise_level,
                    'seed': seed,
                }
            )
            print_progress(i, n_traj, split_start, label=f'Maxwell/{split}')

        print()  # newline after progress bar

    total = time.time() - t0
    print(f'\n[Maxwell] Done. Total time: {total:.1f}s')
    print(f'          Output directory: {os.path.abspath(out_dir)}')
    _print_dataset_stats(out_dir, n_train, n_test, n_nodes, T_steps)


def _print_dataset_stats(out_dir, n_train, n_test, n_nodes, T_steps):
    train_size = sum(
        os.path.getsize(os.path.join(out_dir, 'train', f))
        for f in os.listdir(os.path.join(out_dir, 'train'))
        if f.endswith('.npz')
    )
    print(f'\n  Train files : {n_train}  '
          f'({train_size/1e6:.1f} MB compressed)')
    print(f'  Test  files : {n_test}')
    print(f'  Shape per traj : E and B each (T={T_steps}, N={n_nodes}, 3)')


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Maxwell equations dataset for E-GNO'
    )
    parser.add_argument('--out_dir', default='data/maxwell')
    parser.add_argument('--N',         type=int,   default=64,
                        help='Grid points per side (64 recommended)')
    parser.add_argument('--n_nodes',   type=int,   default=5000,
                        help='Irregular mesh nodes per trajectory')
    parser.add_argument('--n_train',   type=int,   default=100)
    parser.add_argument('--n_test',    type=int,   default=20)
    parser.add_argument('--T_steps',   type=int,   default=50,
                        help='Saved timesteps per trajectory')
    parser.add_argument('--dt',        type=float, default=0.005)
    parser.add_argument('--noise',     type=float, default=0.01,
                        help='Relative observation noise level')
    parser.add_argument('--seed',      type=int,   default=0)
    args = parser.parse_args()

    generate_maxwell_dataset(
        out_dir=args.out_dir,
        N=args.N,
        n_nodes=args.n_nodes,
        n_train=args.n_train,
        n_test=args.n_test,
        T_steps=args.T_steps,
        dt=args.dt,
        noise_level=args.noise,
        base_seed=args.seed,
    )
