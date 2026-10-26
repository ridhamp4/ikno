"""
Training pipeline for E-GNO and all baselines.

Supports:
- Multi-step rollout training for E-GNO
- E-SINDy IHT scheduling
- Equivariance error monitoring
- PDE recovery rate evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import csv
import time
import random
import platform
from typing import Dict, Optional, Tuple
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_dataloaders, create_train_val_test_dataloaders
from utils import get_best_device
from models.egno import EGNO
from baselines.fno_baseline import FNOBaseline
from baselines.pinn_baseline import PINNBaseline
from baselines.gns_baseline import GNS
from baselines.egnn_baseline import EGNNBaseline
from baselines.segnn_baseline import SEGNN
from baselines.tfn_baseline import TFNBaseline
from models.isotropic_kernel import IsotropicKernelOperator
from models.egno.egno_model import random_rotation_matrix


def get_model(model_name: str, config: dict, device: str = 'cpu') -> nn.Module:
    """Create model by name."""
    expected_dim = {'maxwell': 6, 'navier_stokes': 3, 'elasticity': 6}.get(
        config.get('pde_name'))
    field_dim = config.get('field_dim', expected_dim or 3)
    if expected_dim is not None and field_dim != expected_dim:
        raise ValueError(
            f"{config['pde_name']} requires complete state dimension {expected_dim}; "
            f"received field_dim={field_dim}")
    if model_name == 'egno':
        model = EGNO(
            field_dim=field_dim,
            hidden_channels=config.get('hidden_channels', 64),
            l_max=config.get('l_max', 2),
            n_scales=config.get('n_scales', 3),
            n_mp_layers=config.get('n_mp_layers', 2),
            n_rbf=config.get('n_rbf', 16),
            n_library_terms=config.get('n_library_terms', 10),
            lambda_sparse=config.get('lambda_sparse', 0.1),
            threshold=config.get('threshold', 0.05),
            use_residual=config.get('use_residual', True),
            max_neighbors=config.get('max_neighbors', 64),
        )
    elif model_name == 'egno_no_sindy':
        model = EGNO(
            field_dim=field_dim,
            hidden_channels=config.get('hidden_channels', 64),
            l_max=config.get('l_max', 2),
            n_scales=config.get('n_scales', 3),
            n_mp_layers=config.get('n_mp_layers', 2),
            n_rbf=config.get('n_rbf', 16),
            n_library_terms=config.get('n_library_terms', 10),
            lambda_sparse=0.0,  # No sparsity
            threshold=config.get('threshold', 0.05),
            use_residual=config.get('use_residual', True),
            use_operator_library=False,
            max_neighbors=config.get('max_neighbors', 64),
        )
    elif model_name == 'fno':
        model = FNOBaseline(
            field_dim=field_dim,
            grid_size=config.get('grid_size', 32),
            width=config.get('fno_width', 20),
            modes=config.get('fno_modes', 8),
        )
    elif model_name == 'pinn':
        model = PINNBaseline(
            pde_type=config.get('pde_name', 'maxwell'),
            field_dim=field_dim,
            hidden_dim=config.get('hidden_dim', 128),
        )
    elif model_name == 'gns':
        model = GNS(
            field_dim=field_dim,
            hidden_dim=config.get('hidden_dim', 128),
            n_layers=config.get('n_layers', 6),
            r_max=config.get('r_max', 0.1),
        )
    elif model_name == 'egnn':
        model = EGNNBaseline(
            field_dim=field_dim,
            hidden_dim=config.get('hidden_dim', 128),
            n_layers=config.get('n_layers', 4),
            r_max=config.get('r_max', 0.1),
        )
    elif model_name == 'segnn':
        model = SEGNN(
            field_dim=field_dim,
            hidden_channels=config.get('hidden_channels', 64),
            l_max=config.get('l_max', 1),
            n_layers=config.get('n_layers', 4),
            r_max=config.get('r_max', 0.3),
        )
    elif model_name == 'tfn':
        model = TFNBaseline(
            field_dim=field_dim,
            hidden_channels=config.get('hidden_channels', 8),
            n_layers=config.get('n_layers', 3),
            n_rbf=config.get('n_rbf', 8),
            radial_hidden=config.get('radial_hidden', 32),
            r_max=config.get('r_max', 0.3),
            max_neighbors=config.get('max_neighbors', 64),
        )
    elif model_name == 'iso_kernel':
        model = IsotropicKernelOperator(
            field_dim=field_dim,
            hidden_dim=config.get('hidden_dim', 64),
            r_max=config.get('r_max', 0.3),
            max_neighbors=config.get('max_neighbors', 32),
            basis_terms=config.get('basis_terms'),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def get_primary_field(pde_name: str, batch: dict) -> torch.Tensor:
    """Extract the complete Markov state, packing 3D vectors by field."""
    if pde_name == 'maxwell':
        return torch.cat([batch['E'], batch['B']], dim=-1)
    elif pde_name == 'navier_stokes':
        return batch['u']
    elif pde_name == 'elasticity':
        return torch.cat([batch['u'], batch['v']], dim=-1)
    raise ValueError(f"Unknown PDE: {pde_name}")


def get_batch_dt(batch: dict, config: dict) -> float:
    """Read the saved interval from data and reject a conflicting override."""
    times = batch['times']
    if times.numel() < 2:
        raise ValueError('A trajectory needs at least two saved time points')
    intervals = times[1:] - times[:-1]
    if not torch.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-8):
        raise ValueError('Non-uniform saved time intervals are not supported')
    data_dt = float(intervals[0].item())
    configured_dt = config.get('dt')
    if configured_dt is not None and abs(float(configured_dt) - data_dt) > max(1e-8, 1e-4 * data_dt):
        if config.get('strict_dt', True):
            raise ValueError(f'Configured dt={configured_dt} conflicts with data dt={data_dt}')
    return data_dt


def train_step_egno(model: EGNO, batch: dict, config: dict,
                    device: str) -> Tuple[torch.Tensor, Dict]:
    """Single training step for E-GNO."""
    pde_name = config['pde_name']
    dt = get_batch_dt(batch, config)
    n_rollout = config.get('n_rollout_steps', 1)

    coords = batch['coords'].to(device)  # (B, N, 3)
    u_field = get_primary_field(pde_name, batch).to(device)  # (B, T, N, 3)

    B, T, N, D = u_field.shape

    total_loss = torch.tensor(0.0, device=device)
    info = defaultdict(float)

    ode_method = config.get('ode_method', 'euler')

    for b in range(B):
        loss_b, info_b = model.compute_loss(
            u_field[b], coords[b], dt,
            n_rollout_steps=min(n_rollout, T - 1),
            method=ode_method,
            max_time_windows=config.get('n_train_windows', 4),
            normalize_by_increment=config.get('normalize_by_increment', False),
        )
        total_loss = total_loss + loss_b
        for k, v in info_b.items():
            info[k] += v

    total_loss = total_loss / B
    info = {k: v / B for k, v in info.items()}
    return total_loss, dict(info)


def train_step_baseline(model: nn.Module, batch: dict, config: dict,
                        device: str, model_name: str
                        ) -> Tuple[torch.Tensor, Dict]:
    """Single training step for baseline models."""
    pde_name = config['pde_name']
    dt = get_batch_dt(batch, config)

    coords = batch['coords'].to(device)
    u_field = get_primary_field(pde_name, batch).to(device)
    times = batch['times'].to(device)

    B, T, N, D = u_field.shape

    total_loss = torch.tensor(0.0, device=device)

    if model_name == 'pinn':
        for b in range(B):
            loss, _ = model.compute_loss(coords[b], times, u_field[b])
            total_loss = total_loss + loss
    else:
        for b in range(B):
            trajectory = u_field[b]
            sample_coords = coords[b]
            if config.get('rotation_augmentation', False):
                if D % 3:
                    raise ValueError(
                        'Rotation augmentation requires packed 3D vector fields')
                rotation = random_rotation_matrix(device=trajectory.device)
                trajectory = (
                    trajectory.reshape(T, N, D // 3, 3) @ rotation.T
                ).reshape(T, N, D)
                sample_coords = sample_coords @ rotation.T
            n_windows = T - 1
            max_windows = config.get('n_train_windows', 4)
            if max_windows and n_windows > max_windows:
                starts = torch.randperm(n_windows, device=trajectory.device)[:max_windows]
            else:
                starts = range(n_windows)
            sample_loss = torch.tensor(0.0, device=device)
            for start in starts:
                start = int(start)
                sample_loss = sample_loss + model.compute_loss(
                    trajectory[start:start + 2], sample_coords, dt)
            total_loss = total_loss + sample_loss / len(starts)

    total_loss = total_loss / B
    return total_loss, {'total_loss': total_loss.item()}


def evaluate(model: nn.Module, test_loader, config: dict,
             device: str, model_name: str) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    pde_name = config['pde_name']

    metrics = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            dt = get_batch_dt(batch, config)
            coords = batch['coords'].to(device)
            u_field = get_primary_field(pde_name, batch).to(device)

            B, T, N, D = u_field.shape

            for b in range(B):
                u0 = u_field[b, 0]

                # One-step prediction
                if model_name in ('egno', 'egno_no_sindy'):
                    traj, info = model(u0, coords[b], dt, n_steps=1)
                    u1_pred = traj[1]
                elif model_name == 'fno':
                    du_dt = model(u0, coords[b])
                    u1_pred = u0 + dt * du_dt
                elif model_name == 'pinn':
                    u1_pred = model.predict(coords[b], dt)
                else:
                    du_dt = model(u0, coords[b])
                    u1_pred = u0 + dt * du_dt

                u1_true = u_field[b, 1] if T > 1 else u0

                # Rel-L2
                rel_l2 = torch.norm(u1_pred - u1_true) / (torch.norm(u1_true) + 1e-8)
                metrics['rel_l2'].append(rel_l2.item())

                # Accuracy under unseen SO(3) rotations (not just operator error).
                n_rotation_tests = int(config.get('n_rotation_tests', 3))
                if n_rotation_tests > 0 and D % 3 == 0 and model_name != 'pinn':
                    rotation_errors = []
                    n_vectors = D // 3
                    for _ in range(n_rotation_tests):
                        from models.egno.egno_model import random_rotation_matrix
                        rotation = random_rotation_matrix(device=device)
                        u0_rot = (u0.reshape(N, n_vectors, 3) @ rotation.T).reshape(N, D)
                        u1_rot = (u1_true.reshape(N, n_vectors, 3) @ rotation.T).reshape(N, D)
                        coords_rot = coords[b] @ rotation.T
                        if model_name in ('egno', 'egno_no_sindy'):
                            rotated_traj, _ = model(u0_rot, coords_rot, dt, n_steps=1)
                            pred_rot = rotated_traj[1]
                        else:
                            pred_rot = u0_rot + dt * model(u0_rot, coords_rot)
                        rotation_errors.append(float(
                            torch.norm(pred_rot - u1_rot) /
                            torch.norm(u1_rot).clamp(min=1e-8)))
                    metrics['rotation_rel_l2'].append(float(np.mean(rotation_errors)))

                # Multi-step rollout stability (at T=min(10, T-1))
                n_rollout_eval = min(10, T - 1)
                if model_name in ('egno', 'egno_no_sindy'):
                    traj_long, _ = model(u0, coords[b], dt, n_steps=n_rollout_eval)
                    u_final_pred = traj_long[-1]
                elif model_name in ('gns', 'egnn'):
                    traj_long = model.rollout(u0, coords[b], dt, n_rollout_eval)
                    u_final_pred = traj_long[-1]
                else:
                    u_final_pred = u0
                    for _ in range(n_rollout_eval):
                        if model_name == 'fno':
                            du = model(u_final_pred, coords[b])
                        elif model_name == 'pinn':
                            du = (model.predict(coords[b], dt) - u_final_pred) / dt
                        else:
                            du = model(u_final_pred, coords[b])
                        u_final_pred = u_final_pred + dt * du

                u_final_true = u_field[b, min(n_rollout_eval, T - 1)]
                stab = torch.norm(u_final_pred - u_final_true) / (
                    torch.norm(u_final_true) + 1e-8)
                metrics['stability'].append(stab.item())

                # Equivariance error (for equivariant models)
                if model_name in ('egno', 'egno_no_sindy', 'iso_kernel'):
                    eq_err = model.equivariance_error(u0, coords[b], n_rotations=5)
                    metrics['eq_error'].append(eq_err)

                # PDE recovery rate (for E-GNO)
                if model_name == 'egno' and config.get('enable_pde_recovery', False):
                    pde = model.get_discovered_pde(coords[b], u0)
                    prr = compute_pde_recovery_rate(pde, pde_name)
                    metrics['pde_recovery'].append(prr)

    # Aggregate
    results = {}
    for k, v in metrics.items():
        results[k] = np.mean(v)
        results[f'{k}_std'] = np.std(v)

    model.train()
    return results


def compute_pde_recovery_rate(discovered_pde: Dict, pde_name: str) -> float:
    """
    Check if discovered PDE matches the true PDE structure.

    Returns 1.0 if exact match, 0.0 otherwise.
    """
    true_terms = {
        # Coupled-state names encode both the operator and source/target field.
        'maxwell': {'curl_B_to_E', 'curl_E_to_B'},
        'navier_stokes': {'advection_u_to_u', 'laplacian_u_to_u'},
        'elasticity': {'identity_v_to_u', 'laplacian_u_to_v',
                       'div_grad_u_to_v'},
    }

    if pde_name not in true_terms:
        return 0.0

    discovered_set = set(discovered_pde.keys())
    target_set = true_terms[pde_name]

    if discovered_set == target_set:
        return 1.0
    # Partial credit: Jaccard similarity
    intersection = len(discovered_set & target_set)
    union = len(discovered_set | target_set)
    return intersection / max(union, 1)


def train(config: dict):
    """
    Main training loop.

    Args:
        config: dict with all training hyperparameters
    """
    # Setup
    device = config.get('device', get_best_device())
    model_name = config['model_name']
    pde_name = config['pde_name']
    n_epochs = config.get('n_epochs', 100)
    lr = config.get('lr', 1e-3)
    save_dir = config.get('save_dir', f'checkpoints/{model_name}_{pde_name}')
    seed = int(config.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if os.path.isdir(save_dir) and os.listdir(save_dir) and not config.get('allow_existing_output', False):
        raise FileExistsError(
            f'{save_dir} is not empty. Choose a new save_dir to avoid mixing or '
            'overwriting experimental artifacts.')
    os.makedirs(save_dir, exist_ok=True)
    run_metadata = {
        'config': config,
        'seed': seed,
        'python': platform.python_version(),
        'torch': torch.__version__,
        'device': str(device),
        'checkpoint_selection': 'validation_rel_l2',
    }

    # CSV logger for per-epoch metrics
    csv_path = os.path.join(save_dir, 'training_log.csv')
    csv_columns = [
        'epoch', 'model', 'pde', 'train_loss', 'traj_loss', 'sparse_loss',
        'sindy_loss', 'n_active_terms', 'val_rel_l2', 'val_rel_l2_std',
        'val_rotation_rel_l2', 'val_rotation_rel_l2_std',
        'val_stability', 'val_stability_std', 'val_eq_error', 'val_eq_error_std',
        'val_pde_recovery', 'val_pde_recovery_std', 'lr', 'time_sec',
    ]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, extrasaction='ignore')
    csv_writer.writeheader()

    # Data
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        data_dir=config.get('data_dir', 'dataset_generation/data'),
        pde_name=pde_name,
        batch_size=config.get('batch_size', 4),
        n_nodes_subsample=config.get('n_nodes_subsample', 1000),
        t_steps=config.get('t_steps', 20),
        num_workers=config.get('num_workers', 0),
        split_seed=int(config.get('split_seed', 0)),
        val_fraction=config.get('val_fraction', 0.2),
        strict_integrity=config.get('strict_integrity', True),
        max_train_trajectories=config.get('max_train_trajectories'),
    )
    run_metadata['splits'] = {
        'train': [os.path.basename(p) for p in train_loader.dataset.files],
        'val': [os.path.basename(p) for p in val_loader.dataset.files],
        'test': [os.path.basename(p) for p in test_loader.dataset.files],
    }
    with open(os.path.join(save_dir, 'run_metadata.json'), 'w') as f:
        json.dump(run_metadata, f, indent=2, default=str)

    # Model
    model = get_model(model_name, config, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} | PDE: {pde_name} | Params: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr,
                          weight_decay=config.get('weight_decay', 1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    best_rel_l2 = float('inf')
    history = defaultdict(list)

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = defaultdict(list)
        t_start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()

            if model_name in ('egno', 'egno_no_sindy'):
                loss, info = train_step_egno(model, batch, config, device)
            else:
                loss, info = train_step_baseline(
                    model, batch, config, device, model_name)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            for k, v in info.items():
                epoch_losses[k].append(v)

        scheduler.step()

        # IHT for E-SINDy (periodically)
        if model_name == 'egno' and (epoch + 1) % config.get('iht_every', 10) == 0:
            with torch.no_grad():
                for batch in train_loader:
                    coords = batch['coords'][0].to(device)
                    u_field = get_primary_field(pde_name, batch)[0].to(device)
                    model.run_iht(u_field, coords, get_batch_dt(batch, config))
                    break  # One batch is enough

        # Evaluate periodically
        if (epoch + 1) % config.get('eval_every', 5) == 0:
            eval_results = evaluate(model, val_loader, config, device, model_name)

            t_elapsed = time.time() - t_start
            avg_loss = np.mean(epoch_losses.get('total_loss',
                               epoch_losses.get('traj_loss', [0])))

            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Rel-L2: {eval_results.get('rel_l2', -1):.4f} | "
                  f"Stab: {eval_results.get('stability', -1):.4f} | "
                  f"EqErr: {eval_results.get('eq_error', -1):.5f} | "
                  f"PRR: {eval_results.get('pde_recovery', -1):.2f} | "
                  f"Time: {t_elapsed:.1f}s")

            # Write CSV row
            csv_row = {
                'epoch': epoch + 1,
                'model': model_name,
                'pde': pde_name,
                'train_loss': avg_loss,
                'traj_loss': np.mean(epoch_losses.get('traj_loss', [0])),
                'sparse_loss': np.mean(epoch_losses.get('sparse_loss', [0])),
                'sindy_loss': np.mean(epoch_losses.get('sindy_loss', [0])),
                'n_active_terms': np.mean(epoch_losses.get('n_active_terms', [0])),
                'val_rel_l2': eval_results.get('rel_l2', -1),
                'val_rel_l2_std': eval_results.get('rel_l2_std', 0),
                'val_rotation_rel_l2': eval_results.get('rotation_rel_l2', -1),
                'val_rotation_rel_l2_std': eval_results.get('rotation_rel_l2_std', 0),
                'val_stability': eval_results.get('stability', -1),
                'val_stability_std': eval_results.get('stability_std', 0),
                'val_eq_error': eval_results.get('eq_error', -1),
                'val_eq_error_std': eval_results.get('eq_error_std', 0),
                'val_pde_recovery': eval_results.get('pde_recovery', -1),
                'val_pde_recovery_std': eval_results.get('pde_recovery_std', 0),
                'lr': optimizer.param_groups[0]['lr'],
                'time_sec': t_elapsed,
            }
            csv_writer.writerow(csv_row)
            csv_file.flush()

            # Save best model
            if eval_results.get('rel_l2', float('inf')) < best_rel_l2:
                best_rel_l2 = eval_results['rel_l2']
                # Convert numpy values to native Python for safe serialization
                safe_results = {k: float(v) if hasattr(v, 'item') else v
                                for k, v in eval_results.items()}
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_results': safe_results,
                }, os.path.join(save_dir, 'best_model.pt'))

            # Update history
            for k, v in eval_results.items():
                history[k].append(v)
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_loss)

    # Close CSV logger
    csv_file.close()
    print(f"Training log CSV saved to: {csv_path}")

    # Save final model and history
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
    }, os.path.join(save_dir, 'final_model.pt'))

    with open(os.path.join(save_dir, 'validation_history.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)

    # Load the validation-selected checkpoint before the single final test pass.
    best_path = os.path.join(save_dir, 'best_model.pt')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    test_results = evaluate(model, test_loader, config, device, model_name)
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in test_results.items()}, f, indent=2)

    # Only print symbolic terms when an explicit, validated operator library is enabled.
    if model_name == 'egno' and config.get('enable_pde_recovery', False):
        with torch.no_grad():
            for batch in test_loader:
                coords = batch['coords'][0].to(device)
                u_field = get_primary_field(pde_name, batch)[0].to(device)
                pde = model.get_discovered_pde(coords, u_field[0])
                print(f"\nDiscovered PDE ({pde_name}):")
                for term, coeff in pde.items():
                    print(f"  {coeff:+.4f} * {term}")
                break

    print(f"\nBest Rel-L2: {best_rel_l2:.4f}")
    print(f"Final test Rel-L2: {test_results.get('rel_l2', -1):.4f}")
    print(f"Checkpoints saved to: {save_dir}")

    return model, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train E-GNO or baselines')
    parser.add_argument('--model', default='egno',
                        choices=['egno', 'egno_no_sindy', 'iso_kernel', 'fno',
                                 'pinn', 'gns', 'egnn', 'segnn', 'tfn'])
    parser.add_argument('--pde', default='maxwell',
                        choices=['maxwell', 'navier_stokes', 'elasticity'])
    parser.add_argument('--data_dir', default='dataset_generation/data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_nodes', type=int, default=1000,
                        help='Subsample to this many nodes')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--n_scales', type=int, default=3)
    parser.add_argument('--n_mp_layers', type=int, default=2)
    parser.add_argument('--l_max', type=int, default=1,
                        help='Maximum irrep order; l=1 is the verified Cartesian path')
    parser.add_argument('--r_max', type=float, default=0.15)
    parser.add_argument('--max_neighbors', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_train_trajectories', type=int, default=None)
    parser.add_argument('--increment_normalized_loss', action='store_true')
    parser.add_argument('--device', default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if omitted.')
    parser.add_argument('--save_dir', default=None)
    args = parser.parse_args()

    # PDE-specific configs
    pde_configs = {
        'maxwell': {'dt': None, 'field_dim': 6},
        'navier_stokes': {'dt': None, 'field_dim': 3},
        'elasticity': {'dt': None, 'field_dim': 6},
    }

    config = {
        'model_name': args.model,
        'pde_name': args.pde,
        'data_dir': args.data_dir,
        'n_epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'n_nodes_subsample': args.n_nodes,
        'device': args.device or get_best_device(),
        'save_dir': args.save_dir or f'checkpoints/{args.model}_{args.pde}',
        'hidden_channels': args.hidden_channels,
        'hidden_dim': args.hidden_dim,
        'n_scales': args.n_scales,
        'n_mp_layers': args.n_mp_layers,
        'l_max': args.l_max,
        'max_neighbors': args.max_neighbors,
        'n_library_terms': 10,
        't_steps': 20,
        'n_rollout_steps': 5,
        'iht_every': 10,
        'eval_every': 5,
        'r_max': args.r_max,
        'seed': args.seed,
        'max_train_trajectories': args.max_train_trajectories,
        'normalize_by_increment': args.increment_normalized_loss,
        'num_workers': 0,
        **pde_configs[args.pde],
    }

    train(config)
