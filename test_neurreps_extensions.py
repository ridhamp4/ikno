import torch

from models.egno.egno_model import random_rotation_matrix
from models.isotropic_kernel import IsotropicKernelOperator


def _rotate(values, rotation):
    return values @ rotation.T


def test_every_ikno_basis_ablation_remains_equivariant():
    coordinates = torch.randn(32, 3) * 0.1
    velocity = torch.randn(32, 3)
    rotation = random_rotation_matrix()
    variants = [
        ['difference', 'longitudinal', 'source'],
        ['difference', 'longitudinal', 'cross'],
        ['difference', 'longitudinal'],
        ['difference'],
    ]
    for terms in variants:
        model = IsotropicKernelOperator(
            field_dim=3, hidden_dim=8, r_max=0.4,
            max_neighbors=31, basis_terms=terms)
        expected = _rotate(model(velocity, coordinates), rotation)
        actual = model(
            _rotate(velocity, rotation), _rotate(coordinates, rotation))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_tfn_baseline_is_equivariant_when_e3nn_is_available():
    try:
        from baselines.tfn_baseline import TFNBaseline, o3
    except ImportError:
        return
    if o3 is None:
        return
    coordinates = torch.randn(32, 3) * 0.1
    velocity = torch.randn(32, 3)
    rotation = random_rotation_matrix()
    model = TFNBaseline(
        hidden_channels=4, n_layers=1, r_max=0.4,
        max_neighbors=31)
    expected = _rotate(model(velocity, coordinates), rotation)
    actual = model(
        _rotate(velocity, rotation), _rotate(coordinates, rotation))
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
