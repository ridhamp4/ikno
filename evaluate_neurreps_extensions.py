"""Evaluate preselected NeurReps controls on the frozen final holdout."""

import json
import os

import numpy as np
import torch

from data.dataset import create_dataloaders
from evaluate import compute_equivariance_error
from models.egno.egno_model import random_rotation_matrix
from train import get_batch_dt, get_model, get_primary_field


MODEL_SPECS = {
    'ikno': {
        'model_name': 'iso_kernel', 'hidden_dim': 64,
        'checkpoint': 'results_corrected/ns_clean_nonlinear_iso_seed_{seed}/best_model.pt',
    },
    'gns': {
        'model_name': 'gns', 'hidden_dim': 32, 'n_layers': 6,
        'checkpoint': 'results_corrected/ns_clean_gns_r04_seed_{seed}/best_model.pt',
    },
    'component_egnn': {
        'model_name': 'egnn', 'hidden_dim': 32, 'n_layers': 4,
        'checkpoint': 'results_corrected/ns_clean_egnn_r04_seed_{seed}/best_model.pt',
    },
    'gns_aug': {
        'model_name': 'gns', 'hidden_dim': 32, 'n_layers': 6,
        'rotation_augmentation': True,
        'checkpoint': 'results_neurreps_extensions/gns_aug/seed_{seed}/best_model.pt',
    },
    'component_egnn_aug': {
        'model_name': 'egnn', 'hidden_dim': 32, 'n_layers': 4,
        'rotation_augmentation': True,
        'checkpoint': 'results_neurreps_extensions/egnn_aug/seed_{seed}/best_model.pt',
    },
    'gns_small': {
        'model_name': 'gns', 'hidden_dim': 9, 'n_layers': 6,
        'checkpoint': 'results_neurreps_extensions/gns_small/seed_{seed}/best_model.pt',
    },
    'tfn': {
        'model_name': 'tfn', 'hidden_channels': 8, 'n_layers': 3,
        'n_rbf': 8, 'radial_hidden': 32,
        'checkpoint': 'results_neurreps_extensions/tfn/seed_{seed}/best_model.pt',
    },
}


def rotate_vectors(values, rotation):
    return values @ rotation.T


def relative_error(prediction, target):
    return float(torch.norm(prediction - target) /
                 torch.norm(target).clamp(min=1e-8))


def rollout(model, state, coords, dt, steps):
    if hasattr(model, 'rollout'):
        return model.rollout(state, coords, dt, steps)[-1]
    for _ in range(steps):
        state = state + dt * model(state, coords)
    return state


def load_holdout(data_dir):
    _, loader = create_dataloaders(
        data_dir, 'navier_stokes', batch_size=1,
        n_nodes_subsample=128, t_steps=20, num_workers=0)
    samples = []
    for batch in loader:
        samples.append((
            batch['coords'][0], get_primary_field('navier_stokes', batch)[0],
            get_batch_dt(batch, {})))
    return samples


def evaluate_run(model, model_name, samples, rotations, rollout_rotations):
    original_one = []
    original_ten = []
    per_rotation_one = [[] for _ in rotations]
    per_rotation_ten = [[] for _ in range(rollout_rotations)]
    equivariance = []

    model.eval()
    with torch.no_grad():
        for coords, trajectory, dt in samples:
            u0 = trajectory[0]
            original_one.append(relative_error(
                u0 + dt * model(u0, coords), trajectory[1]))
            original_ten.append(relative_error(
                rollout(model, u0, coords, dt, 10), trajectory[10]))

            # Use a fixed seed per model/run around this generic diagnostic.
            equivariance.append(compute_equivariance_error(
                model, u0, coords, model_name, dt, n_rotations=20))

            for rotation_index, rotation in enumerate(rotations):
                coords_rotated = rotate_vectors(coords, rotation)
                u0_rotated = rotate_vectors(u0, rotation)
                target_one = rotate_vectors(trajectory[1], rotation)
                prediction_one = u0_rotated + dt * model(
                    u0_rotated, coords_rotated)
                per_rotation_one[rotation_index].append(
                    relative_error(prediction_one, target_one))
                if rotation_index < rollout_rotations:
                    target_ten = rotate_vectors(trajectory[10], rotation)
                    prediction_ten = rollout(
                        model, u0_rotated, coords_rotated, dt, 10)
                    per_rotation_ten[rotation_index].append(
                        relative_error(prediction_ten, target_ten))

    original_one_mean = float(np.mean(original_one))
    rotation_one = np.asarray([np.mean(values) for values in per_rotation_one])
    rotation_ten = np.asarray([np.mean(values) for values in per_rotation_ten])
    delta = rotation_one - original_one_mean
    return {
        'one_step': original_one_mean,
        'ten_step': float(np.mean(original_ten)),
        'rotated_one_step': float(np.mean(rotation_one)),
        'rotated_ten_step': float(np.mean(rotation_ten)),
        'equivariance_error': float(np.mean(equivariance)),
        'rotation_delta_mean': float(np.mean(delta)),
        'rotation_delta_median': float(np.median(delta)),
        'rotation_delta_p95': float(np.quantile(delta, 0.95)),
        'rotation_delta_max': float(np.max(delta)),
        'rotation_one_step_values': rotation_one.tolist(),
    }


def main():
    data_dir = 'dataset_generation/data_ns_final_holdout'
    output_dir = 'results_neurreps_extensions/final_holdout'
    os.makedirs(output_dir, exist_ok=True)
    samples = load_holdout(data_dir)
    torch.manual_seed(20260824)
    rotations = [random_rotation_matrix() for _ in range(1000)]
    results = {}

    for label, specification in MODEL_SPECS.items():
        results[label] = {}
        for seed in range(3):
            config = {
                'pde_name': 'navier_stokes', 'field_dim': 3,
                'r_max': 0.4, 'max_neighbors': 64,
                **specification,
            }
            model = get_model(config['model_name'], config, 'cpu')
            checkpoint_path = specification['checkpoint'].format(seed=seed)
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Hold the operator-equivariance rotations fixed across methods.
            torch.manual_seed(20260824)
            print(f'Evaluating {label}, seed {seed}...')
            metrics = evaluate_run(
                model, config['model_name'], samples, rotations,
                rollout_rotations=20)
            metrics['n_params'] = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
            results[label][str(seed)] = metrics
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as handle:
                json.dump(results, handle, indent=2)


if __name__ == '__main__':
    main()
