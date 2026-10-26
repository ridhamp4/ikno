"""Regression tests for the publication-critical protocol and symmetry fixes."""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from data.dataset import validate_dataset
from models.egno import EGNO
from models.isotropic_kernel import IsotropicKernelOperator
from train import get_batch_dt, get_primary_field


def _write_trajectory(root, split, seed, dt=0.01):
    directory = os.path.join(root, 'maxwell', split)
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.savez_compressed(
        os.path.join(directory, f'traj_{seed:03d}.npz'),
        coords=rng.random((12, 3), dtype=np.float32),
        times=np.arange(3, dtype=np.float32) * dt,
        E=rng.standard_normal((3, 12, 3)).astype(np.float32),
        B=rng.standard_normal((3, 12, 3)).astype(np.float32),
        metadata_json=np.array(json.dumps({'seed': seed, 'dt': dt})),
    )


def test_integrity_rejects_seed_leakage():
    with tempfile.TemporaryDirectory() as root:
        _write_trajectory(root, 'train', 7)
        _write_trajectory(root, 'test', 7)
        with pytest.raises(ValueError, match='seed leakage'):
            validate_dataset(root, 'maxwell')


def test_complete_markov_states_are_packed():
    batch = {
        'E': torch.randn(2, 3, 5, 3), 'B': torch.randn(2, 3, 5, 3),
        'u': torch.randn(2, 3, 5, 3), 'v': torch.randn(2, 3, 5, 3),
    }
    assert get_primary_field('maxwell', batch).shape[-1] == 6
    assert get_primary_field('elasticity', batch).shape[-1] == 6
    assert get_primary_field('navier_stokes', batch).shape[-1] == 3


def test_dt_comes_from_saved_times_and_conflicts_fail():
    batch = {'times': torch.tensor([0.0, 0.02, 0.04])}
    assert get_batch_dt(batch, {'dt': None}) == pytest.approx(0.02)
    with pytest.raises(ValueError, match='conflicts'):
        get_batch_dt(batch, {'dt': 0.01})


def test_coupled_egno_is_so3_equivariant():
    torch.manual_seed(0)
    model = EGNO(field_dim=6, hidden_channels=8, l_max=1, n_scales=2,
                 n_mp_layers=1, n_rbf=4, n_library_terms=8,
                 use_residual=True, max_neighbors=8)
    state = torch.randn(24, 6)
    coords = torch.rand(24, 3)
    assert model.equivariance_error(state, coords, n_rotations=5) < 1e-5


def test_nonlinear_isotropic_kernel_is_so3_equivariant():
    torch.manual_seed(1)
    model = IsotropicKernelOperator(
        field_dim=3, hidden_dim=16, r_max=0.5, max_neighbors=12)
    state = torch.randn(32, 3)
    coords = torch.rand(32, 3)
    assert model.equivariance_error(state, coords, n_rotations=5) < 1e-5
