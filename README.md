# IKNO

Official implementation of **Exact SO(3)-Equivariant Isotropic Kernels for
Rotation-Robust Neural Dynamics**, accepted to the NeurReps 2026 Proceedings
Track.

The Invariant-Conditioned Isotropic Kernel Neural Operator (IKNO) predicts local
vector-field dynamics on irregular point clouds. Rotation-invariant scalar edge
features weight vector generators that transform equivariantly under proper
three-dimensional rotations. The resulting operator is exactly SO(3)-equivariant
up to floating-point precision.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick validation

```bash
pytest -q test_smoke.py test_scientific_validity.py test_neurreps_extensions.py
```

## Generate data

Generate a small Navier--Stokes dataset for a quick run:

```bash
python dataset_generation/generate_all.py \
  --pde navier_stokes \
  --quick \
  --out_root dataset_generation/data
```

Generated datasets, checkpoints, and experiment outputs are intentionally not
tracked by Git.

## Train IKNO

```bash
python train.py \
  --model iso_kernel \
  --pde navier_stokes \
  --data_dir dataset_generation/data \
  --epochs 100 \
  --n_nodes 128 \
  --r_max 0.4 \
  --max_neighbors 64
```

The targeted NeurReps controls, including rotation augmentation, the
parameter-matched GNS, Tensor Field Network, and IKNO basis ablations, are
defined in `run_neurreps_extensions.py`.

## Paper

The camera-ready manuscript source is in `NeurReps_paper/`.

## Citation

```bibtex
@InProceedings{patel26,
  title  = {{E}xact {$\mathrm{SO}(3)$}-{E}quivariant {I}sotropic {K}ernels for {R}otation-{R}obust {N}eural {D}ynamics},
  author = {Patel, Ridham},
  year   = {2026}
}
```
