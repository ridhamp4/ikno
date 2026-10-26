"""
Main experiment runner for the E-GNO paper.

Runs all experiments:
1. Main comparison (Table 3): All models on all PDEs
2. Ablation study (Table 4): E-GNO variants on Maxwell
3. Robustness study (Table 5): Noise and mesh resolution
4. Rotation generalization (Section 6.5)
"""

import os
import sys
import json
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train
from evaluate import run_evaluation, print_results_table
from utils import get_best_device
from collections import defaultdict


# ─── Default configs per PDE ──────────────────────────────────────────

PDE_CONFIGS = {
    'maxwell': {
        'field_dim': 6,
        'n_nodes_subsample': 500,  # Use 500 for fast runs; 2000+ for paper
    },
    'navier_stokes': {
        'field_dim': 3,
        'n_nodes_subsample': 500,
    },
    'elasticity': {
        'field_dim': 6,
        'n_nodes_subsample': 500,
    },
}


# ─── Model configs ────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'iso_kernel': {
        'hidden_dim': 64,
        'r_max': 0.4,
        'max_neighbors': 64,
    },
    'egno': {
        'hidden_channels': 32,
        'l_max': 1,
        'n_scales': 2,
        'n_mp_layers': 1,
        'n_rbf': 8,
        'n_library_terms': 10,
        'lambda_sparse': 0.1,
        'threshold': 0.05,
        'use_residual': True,
        'n_rollout_steps': 1,
        'max_neighbors': 32,
        'iht_every': 10,
        'ode_method': 'euler',
    },
    'egno_no_sindy': {
        'hidden_channels': 32,
        'l_max': 1,
        'n_scales': 2,
        'n_mp_layers': 1,
        'n_rbf': 8,
        'n_library_terms': 10,
        'lambda_sparse': 0.0,
        'n_rollout_steps': 1,
        'max_neighbors': 32,
        'ode_method': 'euler',
    },
    'fno': {
        'grid_size': 32,
        'fno_width': 20,
        'fno_modes': 8,
    },
    'pinn': {
        'hidden_dim': 128,
        'n_layers': 5,
    },
    'gns': {
        'hidden_dim': 128,
        'n_layers': 6,
        'r_max': 0.3,
    },
    'egnn': {
        'hidden_dim': 128,
        'n_layers': 4,
        'r_max': 0.3,
    },
}


def write_results_csv(filepath, rows, fieldnames=None):
    """Write a list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {filepath}")


def load_final_test_results(save_dir):
    """Load the single test pass performed after validation selection."""
    with open(os.path.join(save_dir, 'test_metrics.json')) as f:
        raw = json.load(f)
    results = {}
    for metric in ('rel_l2', 'rotation_rel_l2', 'stability', 'eq_error'):
        if metric in raw:
            results[f'{metric}_mean'] = raw[metric]
            results[f'{metric}_std'] = raw.get(f'{metric}_std', 0.0)
    if 'pde_recovery' in raw:
        results['prr_mean'] = raw['pde_recovery']
        results['prr_std'] = raw.get('pde_recovery_std', 0.0)
    return results


def aggregate_seed_results(per_seed):
    """Aggregate independent runs; reported std is across training seeds."""
    aggregate = {'n_seeds': len(per_seed), 'per_seed': per_seed}
    for metric in ('rel_l2', 'rotation_rel_l2', 'stability', 'eq_error', 'prr'):
        values = [result[f'{metric}_mean'] for result in per_seed.values()
                  if f'{metric}_mean' in result]
        if values:
            aggregate[f'{metric}_mean'] = float(np.mean(values))
            aggregate[f'{metric}_std'] = float(np.std(values))
    return aggregate


def run_main_comparison(args):
    """
    Table 3: Main comparison across all PDEs and models.
    """
    print("=" * 70)
    print("  EXPERIMENT 1: Main Comparison (Table 3)")
    print("=" * 70)

    all_results = {}
    models = ['persistence', 'two_frame_linear', 'sindy', 'fno', 'gns', 'egnn',
              'iso_kernel',
              'egno_no_sindy', 'egno']
    pdes = (['maxwell', 'navier_stokes', 'elasticity'] if args.pde == 'all'
            else [args.pde])

    for pde in pdes:
        print(f"\n{'─' * 50}")
        print(f"  PDE: {pde.upper()}")
        print(f"{'─' * 50}")
        pde_results = {}

        for model_name in models:
            if model_name in ('persistence', 'two_frame_linear'):
                baseline_config = {
                    'model_name': model_name, 'pde_name': pde,
                    'data_dir': args.data_dir, 'batch_size': args.batch_size,
                    'n_nodes_subsample': PDE_CONFIGS[pde]['n_nodes_subsample'],
                    't_steps': 20, 'save_dir': '', **PDE_CONFIGS[pde],
                }
                pde_results[model_name] = run_evaluation(baseline_config)
                continue
            if model_name == 'sindy':
                # SINDy is fit-based, not trained — run evaluation directly
                print(f"\n  Evaluating SINDy (fit-based)...")
                try:
                    sindy_config = {
                        'model_name': 'sindy',
                        'pde_name': pde,
                        'data_dir': args.data_dir,
                        'device': 'cpu',
                        'batch_size': args.batch_size,
                        'n_nodes_subsample': PDE_CONFIGS[pde]['n_nodes_subsample'],
                        't_steps': 20,
                        'save_dir': os.path.join(args.output_dir, f'sindy_{pde}'),
                        **PDE_CONFIGS[pde],
                    }
                    os.makedirs(sindy_config['save_dir'], exist_ok=True)
                    results = run_evaluation(sindy_config)
                    pde_results[model_name] = results
                except Exception as e:
                    print(f"    Error evaluating SINDy: {e}")
                    pde_results[model_name] = {}
                continue

            per_seed = {}
            for seed in args.seeds:
                config = {
                    'model_name': model_name, 'pde_name': pde,
                    'data_dir': args.data_dir, 'device': args.device,
                    'n_epochs': args.epochs, 'lr': 1e-3,
                    'batch_size': args.batch_size, 't_steps': 20,
                    'eval_every': max(1, args.epochs // 10),
                    'save_dir': os.path.join(args.output_dir, f'{model_name}_{pde}',
                                             f'seed_{seed}'),
                    'num_workers': 0, 'seed': seed, 'split_seed': 0,
                    **PDE_CONFIGS[pde], **MODEL_CONFIGS.get(model_name, {}),
                }
                print(f"\n  Training {model_name} (seed={seed})...")
                try:
                    train(config)
                    per_seed[str(seed)] = load_final_test_results(config['save_dir'])
                except Exception as e:
                    print(f"    Error training {model_name}, seed {seed}: {e}")
            pde_results[model_name] = (aggregate_seed_results(per_seed)
                                       if per_seed else {})

        all_results[pde] = pde_results

        print(f"\n  Results for {pde.upper()}:")
        print_results_table(pde_results)

    # Save combined results (JSON)
    output_path = os.path.join(args.output_dir, 'main_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save as CSV (Table 3 format: one row per model-PDE combination)
    csv_rows = []
    for pde, pde_results in all_results.items():
        for model_name, results in pde_results.items():
            row = {'model': model_name, 'pde': pde}
            row.update(results)
            csv_rows.append(row)
    csv_fields = ['model', 'pde', 'rel_l2_mean', 'rel_l2_std',
                  'rotation_rel_l2_mean', 'rotation_rel_l2_std',
                  'eq_error_mean', 'eq_error_std', 'stability_mean', 'stability_std',
                  'prr_mean', 'prr_std', 'n_seeds']
    write_results_csv(
        os.path.join(args.output_dir, 'main_comparison.csv'), csv_rows, csv_fields)


def run_ablation(args):
    """
    Table 4: Ablation study on Maxwell benchmark.
    """
    print("=" * 70)
    print("  EXPERIMENT 2: Ablation Study (Table 4)")
    print("=" * 70)

    pde = 'maxwell'
    ablation_configs = {
        'E-GNO (full)': MODEL_CONFIGS['egno'],
        'w/o Multi-Scale': {**MODEL_CONFIGS['egno'], 'n_scales': 1},
        'w/o CG products': {**MODEL_CONFIGS['egno'], 'l_max': 0},
        'w/o E-DOL': {**MODEL_CONFIGS['egno'], 'n_library_terms': 0},
        'w/o E-SINDy': {**MODEL_CONFIGS['egno'], 'lambda_sparse': 0.0},
        'w/o Neural ODE': {**MODEL_CONFIGS['egno'], 'use_residual': False,
                           'n_rollout_steps': 1},
    }

    results = {}
    for variant_name, variant_config in ablation_configs.items():
        safe_name = variant_name.replace(' ', '_').replace('/', '_')
        config = {
            'model_name': 'egno',
            'pde_name': pde,
            'data_dir': args.data_dir,
            'device': args.device,
            'n_epochs': args.epochs,
            'lr': 1e-3,
            'batch_size': args.batch_size,
            't_steps': 20,
            'eval_every': max(1, args.epochs // 10),
            'save_dir': os.path.join(args.output_dir, f'ablation_{safe_name}'),
            'num_workers': 0,
            **PDE_CONFIGS[pde],
            **variant_config,
        }

        print(f"\n  Training {variant_name}...")
        try:
            model, history = train(config)
            eval_results = load_final_test_results(config['save_dir'])
            n_params = sum(p.numel() for p in model.parameters())
            eval_results['n_params'] = n_params
            results[variant_name] = eval_results
        except Exception as e:
            print(f"    Error: {e}")
            results[variant_name] = {}

    print(f"\n  Ablation Results (Maxwell):")
    print_results_table(results)

    output_path = os.path.join(args.output_dir, 'ablation.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save as CSV (Table 4 format)
    csv_rows = []
    for variant_name, res in results.items():
        row = {'variant': variant_name}
        row.update(res)
        csv_rows.append(row)
    csv_fields = ['variant', 'rel_l2_mean', 'rel_l2_std', 'prr_mean', 'prr_std',
                  'eq_error_mean', 'eq_error_std', 'stability_mean', 'stability_std', 'n_params']
    write_results_csv(
        os.path.join(args.output_dir, 'ablation.csv'), csv_rows, csv_fields)


def run_robustness(args):
    """
    Table 5: Robustness to observed mesh resolution.

    Noise robustness requires separately generated, manifested datasets and is
    intentionally not inferred from a label.
    """
    print("=" * 70)
    print("  EXPERIMENT 3: Robustness Study (Table 5)")
    print("=" * 70)

    pde = 'maxwell'
    settings = [
        ('Observed nodes=125', {'n_nodes_subsample': 125}),
        ('Observed nodes=250', {'n_nodes_subsample': 250}),
        ('Observed nodes=500', {'n_nodes_subsample': 500}),
    ]

    results = {}
    for setting_name, setting_config in settings:
        safe_name = setting_name.replace(' ', '_').replace(',', '').replace('%', 'pct')
        config = {
            'model_name': 'egno',
            'pde_name': pde,
            'data_dir': args.data_dir,
            'device': args.device,
            'n_epochs': args.epochs,
            'lr': 1e-3,
            'batch_size': args.batch_size,
            't_steps': 20,
            'eval_every': max(1, args.epochs // 10),
            'save_dir': os.path.join(args.output_dir, f'robust_{safe_name}'),
            'num_workers': 0,
            **PDE_CONFIGS[pde],
            **MODEL_CONFIGS['egno'],
            **setting_config,
        }

        print(f"\n  Training E-GNO ({setting_name})...")
        try:
            model, history = train(config)
            eval_results = load_final_test_results(config['save_dir'])
            results[setting_name] = eval_results
        except Exception as e:
            print(f"    Error: {e}")
            results[setting_name] = {}

    print(f"\n  Robustness Results:")
    print_results_table(results)

    output_path = os.path.join(args.output_dir, 'robustness.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save as CSV (Table 5 format)
    csv_rows = []
    for setting_name, res in results.items():
        row = {'setting': setting_name}
        row.update(res)
        csv_rows.append(row)
    csv_fields = ['setting', 'rel_l2_mean', 'rel_l2_std', 'prr_mean', 'prr_std',
                  'eq_error_mean', 'eq_error_std', 'stability_mean', 'stability_std']
    write_results_csv(
        os.path.join(args.output_dir, 'robustness.csv'), csv_rows, csv_fields)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run E-GNO paper experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--experiment', default='all',
                        choices=['all', 'main', 'ablation', 'robustness'],
                        help='Which experiment to run')
    parser.add_argument('--data_dir', default='dataset_generation/data',
                        help='Root data directory')
    parser.add_argument('--pde', default='all',
                        choices=['all', 'maxwell', 'navier_stokes', 'elasticity'],
                        help='PDEs for the main comparison')
    parser.add_argument('--output_dir', default='results_corrected',
                        help='Output directory for results and checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (use 200+ for paper quality)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seeds', default='0,1,2',
                        help='Comma-separated independent training seeds')
    parser.add_argument('--device', default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if omitted.')
    args = parser.parse_args()
    args.seeds = [int(value.strip()) for value in args.seeds.split(',') if value.strip()]
    if not args.seeds:
        parser.error('--seeds must contain at least one integer')
    if args.experiment in ('ablation', 'robustness') and args.pde not in ('all', 'maxwell'):
        parser.error('ablation and robustness experiments currently target Maxwell')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        args.device = get_best_device()

    print(f"\nE-GNO Paper Experiments")
    print(f"Device: {args.device}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")

    if args.experiment in ('all', 'main'):
        run_main_comparison(args)

    if args.experiment in ('all', 'ablation'):
        run_ablation(args)

    if args.experiment in ('all', 'robustness'):
        run_robustness(args)

    print("\n✓ All experiments complete.")
