"""Run the targeted NeurReps controls on the development split.

The final post-design holdout is deliberately not accessed here. Variants are
screened using the existing train/validation/development-test data, after which
scientifically useful comparisons can be evaluated once on the frozen holdout.
"""

import argparse
import json
import os

from train import train


VARIANTS = {
    'gns_aug': {
        'model_name': 'gns', 'hidden_dim': 32, 'n_layers': 6,
        'rotation_augmentation': True, 'lr': 2e-3,
    },
    'egnn_aug': {
        'model_name': 'egnn', 'hidden_dim': 32, 'n_layers': 4,
        'rotation_augmentation': True, 'lr': 2e-3,
    },
    'gns_small': {
        'model_name': 'gns', 'hidden_dim': 9, 'n_layers': 6,
        'lr': 2e-3,
    },
    'tfn': {
        'model_name': 'tfn', 'hidden_channels': 8, 'n_layers': 3,
        'n_rbf': 8, 'radial_hidden': 32, 'lr': 2e-3,
    },
    'ikno_no_cross': {
        'model_name': 'iso_kernel',
        'basis_terms': ['difference', 'longitudinal', 'source'],
        'hidden_dim': 64, 'lr': 3e-3,
    },
    'ikno_no_source': {
        'model_name': 'iso_kernel',
        'basis_terms': ['difference', 'longitudinal', 'cross'],
        'hidden_dim': 64, 'lr': 3e-3,
    },
    'ikno_difference_only': {
        'model_name': 'iso_kernel',
        'basis_terms': ['difference', 'longitudinal'],
        'hidden_dim': 64, 'lr': 3e-3,
    },
    'ikno_delta_only': {
        'model_name': 'iso_kernel',
        'basis_terms': ['difference'],
        'hidden_dim': 64, 'lr': 3e-3,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_dir', default='dataset_generation/data_ns_clean')
    parser.add_argument('--output_dir', default='results_neurreps_extensions')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    variants = [value.strip() for value in args.variants.split(',') if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(',') if value.strip()]
    unknown = set(variants).difference(VARIANTS)
    if unknown:
        parser.error(f'Unknown variants: {sorted(unknown)}')

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'development_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as handle:
            summary = json.load(handle)
    else:
        summary = {}
    for variant in variants:
        summary.setdefault(variant, {})
        for seed in seeds:
            save_dir = os.path.join(args.output_dir, variant, f'seed_{seed}')
            config = {
                'pde_name': 'navier_stokes',
                'field_dim': 3,
                'data_dir': args.data_dir,
                'n_epochs': args.epochs,
                'batch_size': 4,
                'n_nodes_subsample': 128,
                't_steps': 20,
                'n_train_windows': 4,
                'eval_every': 5,
                'r_max': 0.4,
                'max_neighbors': 64,
                'weight_decay': 1e-5,
                'device': args.device,
                'seed': seed,
                'split_seed': 0,
                'num_workers': 0,
                'save_dir': save_dir,
                'n_rotation_tests': 3,
                **VARIANTS[variant],
            }
            print(f'\n=== {variant}, seed {seed} ===')
            model, _ = train(config)
            with open(os.path.join(save_dir, 'test_metrics.json')) as handle:
                metrics = json.load(handle)
            metrics['n_params'] = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad)
            summary[variant][str(seed)] = metrics
            with open(summary_path, 'w') as handle:
                json.dump(summary, handle, indent=2)


if __name__ == '__main__':
    main()
