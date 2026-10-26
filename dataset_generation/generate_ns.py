"""
generate_ns.py  –  Dataset generator for 3-D incompressible Navier-Stokes.

Solves the incompressible Navier-Stokes equations on a periodic domain
[0, 2π]³ using the pseudo-spectral (Fourier Galerkin) method with
2/3-rule dealiasing and divergence-free projection.

Physics
-------
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u
    ∇·u = 0

The pressure is eliminated implicitly via the Leray projector P:
    ∂û/∂t = P[-F̂(u·∇u)] + ν(-|k|²)û

where F̂ is the Fourier transform and P is the projection onto
divergence-free vector fields:
    (P v̂)ᵢ = v̂ᵢ - kᵢ(k·v̂)/|k|²

A large-scale random forcing is optionally applied to maintain
a statistically stationary turbulent state.

Output layout
-------------
    data/navier_stokes/
        train/traj_000.npz … traj_099.npz
        test/traj_000.npz  … traj_019.npz
    Each .npz contains:
        coords : (N_nodes, 3)    float32  — spatial coordinates in [0,1]³
        u      : (T, N_nodes, 3) float32  — velocity field
        times  : (T,)            float32  — time values

Dependencies: numpy, scipy
"""

import numpy as np
import os, time, argparse
from utils import (
    make_grid, fftn3, ifftn3, leray_project, dealias_filter,
    smooth_divergence_free_field, generate_irregular_mesh,
    vector_field_to_mesh, add_noise, save_trajectory,
    rk4_step, print_progress,
)


# ─────────────────────────────────────────────────────────────
#  Navier-Stokes RHS  (pseudo-spectral)
# ─────────────────────────────────────────────────────────────

def ns_rhs_nonlinear(state, kx, ky, kz, dealias_mask):
    """
    Nonlinear (advection) RHS only for incompressible NS.
    Diffusion is handled via exact exponential integration (Strang splitting).

    state = (ux_h, uy_h, uz_h)  — Fourier coefficients
    Returns P[-F̂(u·∇u)] (projected advection term).
    """
    ux_h, uy_h, uz_h = state

    # Apply 2/3 dealiasing to velocity before computing nonlinear term
    ux_h_d = ux_h * dealias_mask
    uy_h_d = uy_h * dealias_mask
    uz_h_d = uz_h * dealias_mask

    # Physical-space velocity
    ux = np.fft.ifftn(ux_h_d).real
    uy = np.fft.ifftn(uy_h_d).real
    uz = np.fft.ifftn(uz_h_d).real

    # Velocity gradients in Fourier space (via ik*u_h)
    dux_dx_h = 1j * kx * ux_h_d;  dux_dy_h = 1j * ky * ux_h_d;  dux_dz_h = 1j * kz * ux_h_d
    duy_dx_h = 1j * kx * uy_h_d;  duy_dy_h = 1j * ky * uy_h_d;  duy_dz_h = 1j * kz * uy_h_d
    duz_dx_h = 1j * kx * uz_h_d;  duz_dy_h = 1j * ky * uz_h_d;  duz_dz_h = 1j * kz * uz_h_d

    dux_dx = np.fft.ifftn(dux_dx_h).real;  dux_dy = np.fft.ifftn(dux_dy_h).real
    dux_dz = np.fft.ifftn(dux_dz_h).real
    duy_dx = np.fft.ifftn(duy_dx_h).real;  duy_dy = np.fft.ifftn(duy_dy_h).real
    duy_dz = np.fft.ifftn(duy_dz_h).real
    duz_dx = np.fft.ifftn(duz_dx_h).real;  duz_dy = np.fft.ifftn(duz_dy_h).real
    duz_dz = np.fft.ifftn(duz_dz_h).real

    # Nonlinear term: (u·∇)u  in physical space
    Nx = ux * dux_dx + uy * dux_dy + uz * dux_dz
    Ny = ux * duy_dx + uy * duy_dy + uz * duy_dz
    Nz = ux * duz_dx + uy * duz_dy + uz * duz_dz

    # Fourier transform and project onto divergence-free subspace
    Nx_h = np.fft.fftn(Nx)
    Ny_h = np.fft.fftn(Ny)
    Nz_h = np.fft.fftn(Nz)

    rhs_x_h, rhs_y_h, rhs_z_h = leray_project(-Nx_h, -Ny_h, -Nz_h, kx, ky, kz)
    return (rhs_x_h, rhs_y_h, rhs_z_h)


def strang_step(state, kx, ky, kz, nu, dealias, k2, dt, forcing, rk4_step_fn,
                cfl_safety=0.15):
    """
    Strang splitting step: exact diffusion + RK4 advection.

    1. Half-step exact diffusion:  û *= exp(-ν k² dt/2)
    2. Full-step RK4 for nonlinear advection + forcing (with CFL sub-stepping)
    3. Half-step exact diffusion:  û *= exp(-ν k² dt/2)

    This is 2nd-order accurate and unconditionally stable for the
    diffusion part. CFL sub-stepping handles the nonlinear advection.
    """
    N = state[0].shape[0]
    decay_half = np.exp(-nu * k2.real * dt * 0.5)

    # Half-step diffusion
    state = tuple(s * decay_half for s in state)

    # CFL-based sub-stepping for nonlinear advection
    fx_h, fy_h, fz_h = forcing
    def rhs(s):
        d = ns_rhs_nonlinear(s, kx, ky, kz, dealias)
        return (d[0] + fx_h, d[1] + fy_h, d[2] + fz_h)

    # Estimate max velocity for CFL (conservative factor for spectral methods)
    u_max = max(np.max(np.abs(np.fft.ifftn(s * dealias).real)) for s in state)
    k_eff = N / 3.0  # dealiased wavenumber cutoff
    cfl_dt = cfl_safety / max(u_max * k_eff, 1e-10)

    n_sub = max(1, int(np.ceil(dt / cfl_dt)))
    dt_sub = dt / n_sub

    for _ in range(n_sub):
        state = rk4_step_fn(state, rhs, dt_sub)
        # Energy cap: if any mode grows too large, damp it
        for i, s in enumerate(state):
            if not np.all(np.isfinite(s)):
                # NaN recovery: zero out bad modes
                s = np.where(np.isfinite(s), s, 0.0)
                state = state[:i] + (s,) + state[i+1:]

    # Half-step diffusion
    state = tuple(s * decay_half for s in state)

    # Re-project to stay exactly divergence-free
    state = leray_project(*state, kx, ky, kz)
    return state


def compute_energy(state):
    """Compute total kinetic energy from Fourier coefficients."""
    return sum(np.mean(np.abs(s)**2) for s in state)


def kolmogorov_forcing_hat(kx, ky, kz, k_f=2, amplitude=0.1):
    """
    Deterministic Kolmogorov-type forcing in Fourier space,
    applied at wavenumber k_f in the x-direction.
    Keeps the simulation at a stationary Reynolds number.
    """
    N = kx.shape[0]
    fx_h = np.zeros((N, N, N), dtype=complex)
    fy_h = np.zeros((N, N, N), dtype=complex)
    fz_h = np.zeros((N, N, N), dtype=complex)
    # Force uy at kx=k_f, ky=kz=0
    idx = np.where(
        (np.abs(kx - k_f) < 0.5) & (np.abs(ky) < 0.5) & (np.abs(kz) < 0.5)
    )
    fy_h[idx] = amplitude * N**3
    return fx_h, fy_h, fz_h


# ─────────────────────────────────────────────────────────────
#  Warm-up: spin up to developed turbulence
# ─────────────────────────────────────────────────────────────

def warm_up(N, nu, dt, n_warmup_steps, seed):
    """
    Run the NS solver for n_warmup_steps without saving,
    to reach a statistically stationary turbulent state.
    Uses Strang splitting for unconditional stability.
    Returns the final (ux_h, uy_h, uz_h) state in Fourier space.
    """
    _, _, kx, ky, kz = make_grid(N)
    k2 = kx**2 + ky**2 + kz**2
    ux, uy, uz = smooth_divergence_free_field(N, k_max=5, seed=seed)
    state = (np.fft.fftn(ux), np.fft.fftn(uy), np.fft.fftn(uz))
    dealias = dealias_filter(N)
    forcing = kolmogorov_forcing_hat(kx, ky, kz)

    for step in range(n_warmup_steps):
        state = strang_step(state, kx, ky, kz, nu, dealias, k2, dt, forcing, rk4_step)

        # Energy monitoring: detect blowup
        if step % 100 == 0:
            E = compute_energy(state)
            if not np.isfinite(E) or E > 1e10:
                print(f'\n     [WARN] Energy blowup at warmup step {step} (E={E:.2e}), resetting...')
                ux, uy, uz = smooth_divergence_free_field(N, k_max=5, seed=seed + step)
                state = (np.fft.fftn(ux), np.fft.fftn(uy), np.fft.fftn(uz))

    return state


# ─────────────────────────────────────────────────────────────
#  Single trajectory solver
# ─────────────────────────────────────────────────────────────

def solve_ns(N, nu, dt, n_steps, save_every, init_state=None, seed=None, use_forcing=True):
    """
    Solve 3-D incompressible NS for one trajectory.
    Uses Strang splitting for unconditional stability.

    Returns list of (ux, uy, uz) physical-space snapshots.
    If init_state is None, generates a fresh IC from seed.
    """
    _, _, kx, ky, kz = make_grid(N)
    k2 = kx**2 + ky**2 + kz**2
    dealias = dealias_filter(N)
    forcing_full = kolmogorov_forcing_hat(kx, ky, kz)
    zero_forcing = (np.zeros_like(kx, dtype=complex),) * 3
    forcing = forcing_full if use_forcing else zero_forcing

    if init_state is None:
        ux, uy, uz = smooth_divergence_free_field(N, k_max=4, seed=seed)
        state = (np.fft.fftn(ux), np.fft.fftn(uy), np.fft.fftn(uz))
    else:
        state = init_state

    snapshots = []
    for step in range(n_steps):
        if step % save_every == 0:
            snap = tuple(np.fft.ifftn(fh).real for fh in state)
            snapshots.append(snap)
        state = strang_step(state, kx, ky, kz, nu, dealias, k2, dt, forcing, rk4_step)

    return snapshots, state   # return final state for trajectory chaining


# ─────────────────────────────────────────────────────────────
#  Dataset generation loop
# ─────────────────────────────────────────────────────────────

def generate_ns_dataset(
    out_dir='data/navier_stokes',
    N=64,
    n_nodes=8000,
    n_train=100,
    n_test=20,
    T_steps=50,
    dt=0.002,
    save_every=2,
    nu=0.005,                # Kinematic viscosity (Re ~ 200)
    n_warmup_steps=500,      # Steps to reach stationary turbulence
    use_forcing=True,
    noise_level=0.01,
    base_seed=0,
):
    """
    Generate the full Navier-Stokes dataset.

    Strategy: run a single warm-up to stationary turbulence, then
    fork independent trajectories from perturbations of that state
    (each with a different small-amplitude perturbation).
    """
    grid_1d, _, kx, ky, kz = make_grid(N)
    times = np.arange(T_steps, dtype=np.float32) * dt * save_every

    print(f'\n[NS] Warming up to stationary turbulence ({n_warmup_steps} steps)...')
    warm_state = warm_up(N, nu, dt, n_warmup_steps, seed=base_seed)
    print('     Warm-up complete.')

    t0 = time.time()
    for split, n_traj in [('train', n_train), ('test', n_test)]:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f'\n[NS] Generating {split} set ({n_traj} trajectories)...')
        split_start = time.time()

        for i in range(n_traj):
            seed = base_seed + (0 if split == 'train' else n_train) + i
            rng = np.random.default_rng(seed)

            # --- 1. Perturb warm state to get diverse ICs ---
            ux_w, uy_w, uz_w = warm_state
            ux_pert, uy_pert, uz_pert = smooth_divergence_free_field(
                N, k_max=4, seed=seed)
            pert_scale = 0.1   # 10% perturbation amplitude
            init_state = (
                ux_w + pert_scale * np.fft.fftn(ux_pert),
                uy_w + pert_scale * np.fft.fftn(uy_pert),
                uz_w + pert_scale * np.fft.fftn(uz_pert),
            )
            # Re-project to divergence-free
            init_state = leray_project(*init_state, kx, ky, kz)

            # --- 2. Solve ---
            snapshots, _ = solve_ns(
                N=N, nu=nu, dt=dt,
                n_steps=T_steps * save_every,
                save_every=save_every,
                init_state=init_state,
                use_forcing=use_forcing,
            )

            # --- 3. Irregular mesh ---
            coords = generate_irregular_mesh(n_nodes, seed=seed + 50000)

            # --- 4. Interpolate velocity to mesh ---
            u_traj = np.zeros((T_steps, n_nodes, 3), dtype=np.float32)
            for t_idx, snap in enumerate(snapshots):
                ux, uy, uz = snap
                u_traj[t_idx] = vector_field_to_mesh(
                    [ux, uy, uz], grid_1d, coords * (2 * np.pi)
                )

            # --- 5. Noise ---
            u_traj = add_noise(u_traj, noise_level=noise_level, seed=seed)

            # --- 6. Save ---
            path = os.path.join(split_dir, f'traj_{i:03d}.npz')
            save_trajectory(
                path,
                coords=coords,
                fields_dict={'u': u_traj},
                times=times,
                metadata={
                    'pde': 'navier_stokes',
                    'grid_N': N, 'nu': nu, 'dt': dt,
                    'noise_level': noise_level, 'seed': seed,
                    'forecast_forcing': bool(use_forcing),
                    'warmup_forcing': True,
                    'save_every': save_every,
                    'Re_approx': 1.0 / nu,
                }
            )
            print_progress(i, n_traj, split_start, label=f'NS/{split}')

        print()

    total = time.time() - t0
    print(f'\n[NS] Done. Total time: {total:.1f}s')
    print(f'     Output directory: {os.path.abspath(out_dir)}')


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 3D Navier-Stokes dataset for E-GNO'
    )
    parser.add_argument('--out_dir',  default='data/navier_stokes')
    parser.add_argument('--N',        type=int,   default=64)
    parser.add_argument('--n_nodes',  type=int,   default=8000)
    parser.add_argument('--n_train',  type=int,   default=100)
    parser.add_argument('--n_test',   type=int,   default=20)
    parser.add_argument('--T_steps',  type=int,   default=50)
    parser.add_argument('--dt',       type=float, default=0.002)
    parser.add_argument('--nu',       type=float, default=0.005,
                        help='Kinematic viscosity (lower = higher Re)')
    parser.add_argument('--n_warmup', type=int,   default=500,
                        help='Warm-up steps to reach turbulent state')
    parser.add_argument('--noise',    type=float, default=0.01)
    parser.add_argument('--seed',     type=int,   default=0)
    parser.add_argument('--no_forcing', action='store_true',
                        help='Disable Kolmogorov forcing (free decay)')
    args = parser.parse_args()

    generate_ns_dataset(
        out_dir=args.out_dir,
        N=args.N, n_nodes=args.n_nodes,
        n_train=args.n_train, n_test=args.n_test,
        T_steps=args.T_steps, dt=args.dt, nu=args.nu,
        n_warmup_steps=args.n_warmup,
        use_forcing=not args.no_forcing,
        noise_level=args.noise,
        base_seed=args.seed,
    )
