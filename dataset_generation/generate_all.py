"""
generate_all.py  –  Master script: generate all three E-GNO datasets.

Usage examples
--------------
  # Full generation (all three PDEs, paper settings)
  python generate_all.py

  # Quick smoke-test (tiny grid, few trajectories, ~30 seconds)
  python generate_all.py --quick

  # Only one PDE
  python generate_all.py --pde maxwell

  # Custom output root and noise level
  python generate_all.py --out_root ./my_data --noise 0.05

  # Faster generation with smaller grid (some accuracy loss)
  python generate_all.py --N 32

Dataset structure produced
--------------------------
  <out_root>/
      maxwell/
          train/traj_000.npz … traj_099.npz
          test/ traj_000.npz … traj_019.npz
      navier_stokes/
          train/…   test/…
      elasticity/
          train/…   test/…

Loading a trajectory
--------------------
  import numpy as np
  d = np.load('maxwell/train/traj_000.npz')
  coords = d['coords']   # (N_nodes, 3)  spatial coordinates in [0,1]^3
  E      = d['E']        # (T, N_nodes, 3) electric field
  B      = d['B']        # (T, N_nodes, 3) magnetic field
  times  = d['times']    # (T,)

  d = np.load('navier_stokes/train/traj_000.npz')
  u = d['u']             # (T, N_nodes, 3) velocity

  d = np.load('elasticity/train/traj_000.npz')
  u = d['u']             # (T, N_nodes, 3) displacement
  v = d['v']             # (T, N_nodes, 3) velocity = ∂u/∂t

System requirements
-------------------
  - Python ≥ 3.8
  - numpy ≥ 1.21
  - scipy ≥ 1.7
  - ~4 GB RAM for N=64 full generation
  - ~500 MB RAM for N=32 quick test
  - CPU-only; no GPU required.
  - Estimated runtimes (single core, modern laptop):
      N=64, 100 train + 20 test per PDE  ≈  25–40 min total
      N=32, 10  train +  5 test per PDE  ≈  1–2   min total
"""

import argparse, os, sys, time, json

# ── ensure utils.py is importable ──────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_maxwell     import generate_maxwell_dataset
from generate_ns          import generate_ns_dataset
from generate_elasticity  import generate_elasticity_dataset


# ─────────────────────────────────────────────────────────────
#  Paper-quality default configs
# ─────────────────────────────────────────────────────────────

PAPER_CONFIGS = {
    'maxwell': dict(
        N=64, n_nodes=5000, n_train=100, n_test=20,
        T_steps=50, dt=0.005, save_every=1, noise_level=0.01,
    ),
    'navier_stokes': dict(
        N=64, n_nodes=8000, n_train=100, n_test=20,
        T_steps=50, dt=0.002, save_every=2, nu=0.02,
        n_warmup_steps=500, use_forcing=True, noise_level=0.01,
    ),
    'elasticity': dict(
        N=64, n_nodes=6000, n_train=100, n_test=20,
        T_steps=50, dt=0.003, save_every=2, noise_level=0.01,
        ic_type='mixed',
    ),
}

# Quick smoke-test configs (small + fast)
QUICK_CONFIGS = {
    'maxwell': dict(
        N=32, n_nodes=500, n_train=10, n_test=5,
        T_steps=20, dt=0.01, save_every=1, noise_level=0.01,
    ),
    'navier_stokes': dict(
        N=32, n_nodes=500, n_train=10, n_test=5,
        T_steps=20, dt=0.005, save_every=2, nu=0.05,
        n_warmup_steps=50, use_forcing=True, noise_level=0.01,
    ),
    'elasticity': dict(
        N=32, n_nodes=500, n_train=10, n_test=5,
        T_steps=20, dt=0.005, save_every=2, noise_level=0.01,
        ic_type='mixed',
    ),
}

GENERATORS = {
    'maxwell':       generate_maxwell_dataset,
    'navier_stokes': generate_ns_dataset,
    'elasticity':    generate_elasticity_dataset,
}


def estimate_disk_usage(configs):
    """Rough disk usage estimate in MB."""
    total_mb = 0
    for name, cfg in configs.items():
        n_traj = cfg['n_train'] + cfg['n_test']
        T = cfg['T_steps']
        N = cfg.get('n_nodes', 5000)
        # Fields: 2 vector fields × 3 comps × float32
        n_fields = {'maxwell': 6, 'navier_stokes': 3, 'elasticity': 6}[name]
        raw_bytes = n_traj * T * N * n_fields * 4
        compressed_mb = raw_bytes / 1e6 * 0.25   # ~4x compression
        total_mb += compressed_mb
    return total_mb


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate all E-GNO datasets (Maxwell, NS, Elasticity)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--out_root', default='data',
                        help='Root output directory')
    parser.add_argument('--pde', default='all',
                        choices=['all', 'maxwell', 'navier_stokes', 'elasticity'],
                        help='Which PDE to generate')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke-test with tiny grid and few trajectories')
    parser.add_argument('--N',      type=int,   default=None,
                        help='Override grid size per side')
    parser.add_argument('--noise',  type=float, default=None,
                        help='Override noise level')
    parser.add_argument('--n_train',type=int,   default=None,
                        help='Override number of training trajectories')
    parser.add_argument('--n_test', type=int,   default=None,
                        help='Override number of test trajectories')
    parser.add_argument('--seed',   type=int,   default=42)
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly replace existing trajectory files for selected PDEs')
    args = parser.parse_args()

    # Select base config
    base_configs = QUICK_CONFIGS if args.quick else PAPER_CONFIGS

    # Apply any CLI overrides
    configs = {}
    for name, cfg in base_configs.items():
        cfg = cfg.copy()
        if args.N       is not None: cfg['N']       = args.N
        if args.noise   is not None: cfg['noise_level'] = args.noise
        if args.n_train is not None: cfg['n_train'] = args.n_train
        if args.n_test  is not None: cfg['n_test']  = args.n_test
        cfg['base_seed'] = args.seed
        configs[name] = cfg

    # Which PDEs to generate
    pdenames = list(GENERATORS.keys()) if args.pde == 'all' else [args.pde]

    # Summary
    mode = 'QUICK SMOKE-TEST' if args.quick else 'PAPER-QUALITY'
    print('=' * 60)
    print(f'  E-GNO Dataset Generator  [{mode}]')
    print('=' * 60)
    est_mb = estimate_disk_usage({k: configs[k] for k in pdenames})
    print(f'  PDEs to generate : {pdenames}')
    print(f'  Output root      : {os.path.abspath(args.out_root)}')
    print(f'  Estimated disk   : ~{est_mb:.0f} MB')
    print('=' * 60)

    wall_start = time.time()
    for name in pdenames:
        cfg = configs[name]
        out_dir = os.path.join(args.out_root, name)
        existing = []
        for split in ('train', 'val', 'test'):
            split_dir = os.path.join(out_dir, split)
            if os.path.isdir(split_dir):
                existing.extend(os.path.join(split_dir, f)
                                for f in os.listdir(split_dir) if f.endswith('.npz'))
        if existing and not args.overwrite:
            raise FileExistsError(
                f'{out_dir} already contains {len(existing)} trajectories. '
                'Use a new --out_root or pass --overwrite explicitly; mixing runs is unsafe.')
        if args.overwrite:
            for path in existing:
                os.remove(path)
        print(f'\n{"─"*60}')
        print(f'  PDE: {name.upper()}')
        print(f'  Grid: {cfg["N"]}³  |  '
              f'Nodes: {cfg.get("n_nodes")}  |  '
              f'T_steps: {cfg["T_steps"]}')
        print(f'  Train: {cfg["n_train"]}  Test: {cfg["n_test"]}  '
              f'Noise: {cfg["noise_level"]*100:.0f}%')
        print(f'{"─"*60}')
        GENERATORS[name](out_dir=out_dir, **cfg)
        with open(os.path.join(out_dir, 'dataset_manifest.json'), 'w') as f:
            json.dump({
                'pde': name,
                'mode': 'quick' if args.quick else 'paper',
                'configuration': cfg,
                'split_policy': 'disjoint sequential seeds',
            }, f, indent=2)

    wall_total = time.time() - wall_start
    print('\n' + '=' * 60)
    print(f'  ALL DONE  —  Total wall time: {wall_total/60:.1f} min')
    print(f'  Data saved to: {os.path.abspath(args.out_root)}')
    print('=' * 60)

    # Print a quick loading example
    print('\n  Quick-load example:')
    print('  >>> import numpy as np')
    example_pde = pdenames[0]
    print(f"  >>> d = np.load('{args.out_root}/{example_pde}/train/traj_000.npz')")
    print("  >>> d['coords'].shape   # (N_nodes, 3)")
    print("  >>> list(d.keys())      # all available fields")


if __name__ == '__main__':
    main()
