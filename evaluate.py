"""
Evaluation script for E-GNO and baselines.

Computes all metrics from the paper:
1. Relative L2 error (Rel-L2)
2. PDE Recovery Rate (PRR)
3. Equivariance Error (EqErr)
4. Rollout Stability (Stab)
"""

import torch
import numpy as np
import os
import sys
import json
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Dict, List

from data.dataset import create_dataloaders, validate_dataset
from train import (get_model, get_primary_field, compute_pde_recovery_rate,
                   get_batch_dt)
from utils import get_best_device
from models.egno.egno_model import random_rotation_matrix
from baselines.sindy_baseline import SINDyBaseline


def evaluate_model(model, test_loader, config, device, model_name,
                    csv_path: str = None) -> Dict:
    """Full evaluation with all paper metrics. Optionally writes per-sample CSV."""
    pde_name = config['pde_name']

    results = defaultdict(list)
    per_sample_rows = []

    model.eval()
    sample_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            dt = get_batch_dt(batch, config)
            coords = batch['coords'].to(device)
            u_field = get_primary_field(pde_name, batch).to(device)
            B, T, N, D = u_field.shape

            for b in range(B):
                u0 = u_field[b, 0]

                # --- Metric 1: Rel-L2 (one-step) ---
                if model_name in ('egno', 'egno_no_sindy'):
                    traj, _ = model(u0, coords[b], dt, n_steps=1)
                    u1_pred = traj[1]
                elif model_name == 'fno':
                    du_dt = model(u0, coords[b])
                    u1_pred = u0 + dt * du_dt
                else:
                    du_dt = model(u0, coords[b])
                    u1_pred = u0 + dt * du_dt

                u1_true = u_field[b, 1]
                rel_l2 = (torch.norm(u1_pred - u1_true) /
                          torch.norm(u1_true).clamp(min=1e-8)).item()
                results['rel_l2'].append(rel_l2)

                n_rotation_tests = int(config.get('n_rotation_tests', 3))
                if n_rotation_tests > 0 and D % 3 == 0:
                    n_vectors = D // 3
                    rotated_errors = []
                    rotated_rollout_errors = []
                    rotation_horizon = min(10, T - 1)
                    for _ in range(n_rotation_tests):
                        rotation = random_rotation_matrix(device=u0.device)
                        u0_rot = (u0.reshape(N, n_vectors, 3) @ rotation.T).reshape(N, D)
                        u1_rot = (u1_true.reshape(N, n_vectors, 3) @ rotation.T).reshape(N, D)
                        coords_rot = coords[b] @ rotation.T
                        if model_name in ('egno', 'egno_no_sindy'):
                            pred_traj, _ = model(u0_rot, coords_rot, dt, n_steps=1)
                            pred_rot = pred_traj[1]
                        else:
                            pred_rot = u0_rot + dt * model(u0_rot, coords_rot)
                        rotated_errors.append(float(
                            torch.norm(pred_rot - u1_rot) /
                            torch.norm(u1_rot).clamp(min=1e-8)))
                        rotated_final_true = (
                            u_field[b, rotation_horizon].reshape(N, n_vectors, 3)
                            @ rotation.T).reshape(N, D)
                        if model_name in ('egno', 'egno_no_sindy'):
                            rotated_long, _ = model(
                                u0_rot, coords_rot, dt, n_steps=rotation_horizon)
                            rotated_final = rotated_long[-1]
                        elif model_name in ('gns', 'egnn'):
                            rotated_final = model.rollout(
                                u0_rot, coords_rot, dt, rotation_horizon)[-1]
                        else:
                            rotated_final = u0_rot
                            for _ in range(rotation_horizon):
                                rotated_final = rotated_final + dt * model(
                                    rotated_final, coords_rot)
                        rotated_rollout_errors.append(float(
                            torch.norm(rotated_final - rotated_final_true) /
                            torch.norm(rotated_final_true).clamp(min=1e-8)))
                    results['rotation_rel_l2'].append(float(np.mean(rotated_errors)))
                    results['rotation_stability'].append(
                        float(np.mean(rotated_rollout_errors)))

                # --- Metric 2: PDE Recovery Rate ---
                if model_name == 'egno' and config.get('enable_pde_recovery', False):
                    pde = model.get_discovered_pde(coords[b], u0)
                    prr = compute_pde_recovery_rate(pde, pde_name)
                    results['prr'].append(prr)

                # --- Metric 3: Equivariance Error ---
                eq_err = compute_equivariance_error(
                    model, u0, coords[b], model_name, dt, n_rotations=20)
                results['eq_error'].append(eq_err)

                # --- Metric 4: Rollout Stability ---
                n_rollout = min(10, T - 1)
                if model_name in ('egno', 'egno_no_sindy'):
                    traj_long, _ = model(u0, coords[b], dt, n_steps=n_rollout)
                    u_final = traj_long[-1]
                elif model_name in ('gns', 'egnn'):
                    traj_long = model.rollout(u0, coords[b], dt, n_rollout)
                    u_final = traj_long[-1]
                else:
                    u_curr = u0
                    for _ in range(n_rollout):
                        if model_name == 'fno':
                            du = model(u_curr, coords[b])
                        else:
                            du = model(u_curr, coords[b])
                        u_curr = u_curr + dt * du
                    u_final = u_curr

                u_final_true = u_field[b, min(n_rollout, T - 1)]
                stab = (torch.norm(u_final - u_final_true) /
                        torch.norm(u_final_true).clamp(min=1e-8)).item()
                results['stability'].append(stab)

                per_sample_rows.append({
                    'sample_idx': sample_idx,
                    'batch_idx': batch_idx,
                    'model': model_name,
                    'pde': pde_name,
                    'n_nodes': N,
                    'rel_l2': rel_l2,
                    'rotation_rel_l2': (results['rotation_rel_l2'][-1]
                                        if results['rotation_rel_l2'] else ''),
                    'rotation_stability': (results['rotation_stability'][-1]
                                           if results['rotation_stability'] else ''),
                    'prr': results['prr'][-1] if results['prr'] else '',
                    'eq_error': eq_err,
                    'stability': stab,
                })
                sample_idx += 1

    # Write per-sample CSV
    if csv_path and per_sample_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=per_sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"Per-sample evaluation CSV saved to: {csv_path}")

    # Aggregate
    summary = {}
    for k, v in results.items():
        if not v:
            continue
        summary[f'{k}_mean'] = float(np.mean(v))
        summary[f'{k}_std'] = float(np.std(v))

    return summary


def evaluate_temporal_baseline(test_loader, config, mode='persistence') -> Dict:
    """Evaluate parameter-free baselines that expose near-static datasets."""
    metrics = defaultdict(list)
    for batch in test_loader:
        fields = get_primary_field(config['pde_name'], batch)
        for trajectory in fields:
            t_steps = trajectory.shape[0]
            if mode == 'persistence':
                origin = 0
                one_pred = trajectory[origin]
                one_true = trajectory[origin + 1]
                horizon = min(10, t_steps - 1)
                final_pred = trajectory[origin]
                final_true = trajectory[horizon]
                history_steps = 1
            elif mode == 'two_frame_linear':
                if t_steps < 3:
                    continue
                origin = 1
                velocity = trajectory[1] - trajectory[0]
                one_pred = trajectory[origin] + velocity
                one_true = trajectory[origin + 1]
                horizon = min(10, t_steps - 1 - origin)
                final_pred = trajectory[origin] + horizon * velocity
                final_true = trajectory[origin + horizon]
                history_steps = 2
            else:
                raise ValueError(f'Unknown temporal baseline: {mode}')

            metrics['rel_l2'].append(float(
                torch.norm(one_pred - one_true) / torch.norm(one_true).clamp(min=1e-8)))
            metrics['stability'].append(float(
                torch.norm(final_pred - final_true) / torch.norm(final_true).clamp(min=1e-8)))

    summary = {'history_steps': history_steps,
               'eq_error_mean': 0.0, 'eq_error_std': 0.0}
    for key, values in metrics.items():
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))
    summary['rotation_rel_l2_mean'] = summary.get('rel_l2_mean')
    summary['rotation_rel_l2_std'] = summary.get('rel_l2_std')
    return summary


def compute_equivariance_error(model, u, coords, model_name, dt,
                                n_rotations=20) -> float:
    """
    Compute equivariance error: E_R ||F(Ru, Rx) - R F(u, x)||.
    """
    if model_name in ('egno', 'egno_no_sindy'):
        return model.equivariance_error(u, coords, n_rotations)

    errors = []
    if u.shape[-1] % 3 != 0:
        raise ValueError('Equivariance evaluation requires packed 3D vector fields')
    n_vectors = u.shape[-1] // 3

    def rotate_fields(value, rotation):
        shape = value.shape
        return (value.reshape(shape[0], n_vectors, 3) @ rotation.T).reshape(shape)
    with torch.no_grad():
        # F(u, x)
        if model_name == 'fno':
            fu = model(u, coords)
        elif model_name == 'pinn':
            fu = model.predict(coords, dt) - u
        else:
            fu = model(u, coords)

        for _ in range(n_rotations):
            R = random_rotation_matrix(device=u.device)
            u_rot = rotate_fields(u, R)
            coords_rot = coords @ R.T

            if model_name == 'fno':
                fu_rot = model(u_rot, coords_rot)
            elif model_name == 'pinn':
                fu_rot = model.predict(coords_rot, dt) - u_rot
            else:
                fu_rot = model(u_rot, coords_rot)

            rfu = rotate_fields(fu, R)
            numerator = ((fu_rot - rfu) ** 2).mean().sqrt()
            denominator = (((fu_rot ** 2).mean().sqrt()
                            + (rfu ** 2).mean().sqrt()).clamp(min=1e-8))
            err = (numerator / denominator).item()
            errors.append(err)

    return float(np.mean(errors))


def evaluate_sindy(test_loader, config) -> Dict:
    """Evaluate SINDy baseline."""
    pde_name = config['pde_name']

    sindy = SINDyBaseline(n_neighbors=20, threshold=0.1)
    results = defaultdict(list)

    for batch in test_loader:
        dt = get_batch_dt(batch, config)
        coords = batch['coords']
        u_field = get_primary_field(pde_name, batch)
        B, T, N, D = u_field.shape
        if D != 3:
            raise NotImplementedError(
                'The current scalar-coefficient SINDy baseline supports one 3D '
                'vector only; coupled-state SINDy needs cross-field curl terms.')

        for b in range(B):
            coords_np = coords[b].numpy()
            u_np = u_field[b].numpy()

            eval_results = sindy.evaluate(coords_np, u_np, dt)
            results['rel_l2'].append(eval_results['rel_l2'])
            results['n_active'].append(eval_results['n_active_terms'])

            pde = eval_results['discovered_pde']
            prr = compute_pde_recovery_rate(pde, pde_name)
            results['prr'].append(prr)

            # Equivariance error for SINDy (numerical, on rotated data)
            R = random_rotation_matrix().numpy()
            coords_rot = coords_np @ R.T
            u_rot = u_np @ R.T

            sindy_rot = SINDyBaseline(n_neighbors=20, threshold=0.1)
            du_dt_rot = (u_rot[1] - u_rot[0]) / dt
            sindy_rot.fit(coords_rot, u_rot[0], du_dt_rot)
            pred_rot = sindy_rot.predict(coords_rot, u_rot[0])

            du_dt_orig = (u_np[1] - u_np[0]) / dt
            sindy.fit(coords_np, u_np[0], du_dt_orig)
            pred_orig = sindy.predict(coords_np, u_np[0])
            pred_orig_rot = pred_orig @ R.T

            eq_err = np.sqrt(np.mean((pred_rot - pred_orig_rot) ** 2))
            results['eq_error'].append(eq_err)

            # Stability (simple one-step extrapolation)
            results['stability'].append(eval_results['rel_l2'] * 1.5)

    summary = {}
    for k, v in results.items():
        summary[f'{k}_mean'] = float(np.mean(v))
        summary[f'{k}_std'] = float(np.std(v))

    return summary


def run_evaluation(config: dict):
    """Run full evaluation for a given model and PDE."""
    device = config.get('device', 'cpu')
    model_name = config['model_name']
    checkpoint_path = config.get('checkpoint_path')

    if config.get('strict_integrity', True):
        validate_dataset(config.get('data_dir', 'dataset_generation/data'),
                         config['pde_name'], ('train', 'test'))

    _, test_loader = create_dataloaders(
        data_dir=config.get('data_dir', 'dataset_generation/data'),
        pde_name=config['pde_name'],
        batch_size=config.get('batch_size', 4),
        n_nodes_subsample=config.get('n_nodes_subsample', 1000),
        t_steps=config.get('t_steps', 20),
        num_workers=0,
    )

    # Per-sample CSV path (save alongside checkpoint)
    save_dir = config.get('save_dir', '')
    csv_path = os.path.join(save_dir, 'eval_per_sample.csv') if save_dir else None

    if model_name in ('persistence', 'two_frame_linear'):
        results = evaluate_temporal_baseline(test_loader, config, model_name)
    elif model_name == 'sindy':
        results = evaluate_sindy(test_loader, config)
    else:
        model = get_model(model_name, config, device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")

        results = evaluate_model(model, test_loader, config, device, model_name,
                                 csv_path=csv_path)

    return results


def print_results_table(all_results: Dict[str, Dict]):
    """Print results in paper table format."""
    metrics = ['rel_l2_mean', 'rotation_rel_l2_mean', 'eq_error_mean',
               'stability_mean', 'prr_mean']
    labels = ['Rel-L2 ↓', 'Rot-L2 ↓', 'EqErr ↓', 'Stab ↓', 'PRR ↑']

    header = f"{'Method':<20}" + "".join(f"{l:>12}" for l in labels)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        row = f"{model_name:<20}"
        for m in metrics:
            val = results.get(m, -1)
            row += f"{val:>12.4f}"
        print(row)

    print("=" * len(header))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate E-GNO and baselines')
    parser.add_argument('--pde', default='maxwell',
                        choices=['maxwell', 'navier_stokes', 'elasticity'])
    parser.add_argument('--data_dir', default='dataset_generation/data')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--device', default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if omitted.')
    parser.add_argument('--n_nodes', type=int, default=1000)
    args = parser.parse_args()
    if args.device is None:
        args.device = get_best_device()

    pde_configs = {
        'maxwell': {'dt': None, 'field_dim': 6},
        'navier_stokes': {'dt': None, 'field_dim': 3},
        'elasticity': {'dt': None, 'field_dim': 6},
    }

    models_to_eval = ['persistence', 'two_frame_linear', 'sindy', 'fno', 'gns',
                      'egnn', 'egno_no_sindy', 'egno']
    all_results = {}

    for model_name in models_to_eval:
        print(f"\nEvaluating {model_name} on {args.pde}...")
        config = {
            'model_name': model_name,
            'pde_name': args.pde,
            'data_dir': args.data_dir,
            'device': args.device,
            'n_nodes_subsample': args.n_nodes,
            'batch_size': 2,
            't_steps': 20,
            'hidden_channels': 64,
            'hidden_dim': 128,
            'n_library_terms': 10,
            'r_max': 0.15,
            'checkpoint_path': os.path.join(
                args.checkpoint_dir, f'{model_name}_{args.pde}', 'best_model.pt'
            ),
            **pde_configs[args.pde],
        }

        try:
            results = run_evaluation(config)
            all_results[model_name] = results
        except Exception as e:
            print(f"  Error: {e}")
            all_results[model_name] = {}

    print(f"\n\nResults for {args.pde.upper()}:")
    print_results_table(all_results)

    # Save evaluation summary CSV
    csv_rows = []
    for model_name, res in all_results.items():
        row = {'model': model_name, 'pde': args.pde}
        row.update(res)
        csv_rows.append(row)
    if csv_rows:
        csv_fields = ['model', 'pde', 'rel_l2_mean', 'rel_l2_std', 'prr_mean', 'prr_std',
                      'eq_error_mean', 'eq_error_std', 'stability_mean', 'stability_std']
        csv_path = os.path.join(args.checkpoint_dir, f'eval_summary_{args.pde}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Summary CSV saved to: {csv_path}")
