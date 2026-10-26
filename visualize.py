"""
Visualization and analysis utilities for E-GNO experiments.

Generates:
1. Training curves (loss, metrics vs epoch)
2. Trajectory comparison plots
3. PDE recovery visualization
4. Equivariance error distribution
5. Ablation bar charts
"""

import numpy as np
import json
import csv
import os
import argparse
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, skipping visualization")


def plot_training_curves(history_path: str, save_dir: str):
    """Plot training loss and evaluation metrics over epochs."""
    if not HAS_MPL:
        return

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training loss
    if 'train_loss' in history:
        axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', lw=1.5)
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)

    # Rel-L2
    if 'rel_l2' in history:
        axes[0, 1].plot(history['epoch'], history['rel_l2'], 'r-', lw=1.5)
        axes[0, 1].set_ylabel('Rel-L2 Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_title('Relative L2 Error (Test)')
        axes[0, 1].grid(True, alpha=0.3)

    # PDE Recovery
    if 'pde_recovery' in history:
        axes[1, 0].plot(history['epoch'], history['pde_recovery'], 'g-o', ms=4, lw=1.5)
        axes[1, 0].set_ylabel('PDE Recovery Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_title('PDE Recovery Rate (Test)')
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].grid(True, alpha=0.3)

    # Equivariance Error
    if 'eq_error' in history:
        axes[1, 1].plot(history['epoch'], history['eq_error'], 'm-', lw=1.5)
        axes[1, 1].set_ylabel('Equivariance Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title('Equivariance Error (Test)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to {save_dir}/training_curves.pdf")


def plot_comparison_table(results_path: str, save_dir: str):
    """
    Plot the main comparison table (Table 3) as a grouped bar chart.
    """
    if not HAS_MPL:
        return

    with open(results_path) as f:
        all_results = json.load(f)

    metrics = ['rel_l2_mean', 'prr_mean', 'eq_error_mean', 'stability_mean']
    metric_labels = ['Rel-L2 ↓', 'PRR ↑', 'EqErr ↓', 'Stability ↓']

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    for pde_name, pde_results in all_results.items():
        models = list(pde_results.keys())
        for m_idx, metric in enumerate(metrics):
            values = [pde_results[m].get(metric, 0) for m in models]
            x = np.arange(len(models))
            width = 0.25
            offset = list(all_results.keys()).index(pde_name) * width

            axes[m_idx].bar(x + offset, values, width, label=pde_name,
                           alpha=0.8)
            axes[m_idx].set_xticks(x + width)
            axes[m_idx].set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            axes[m_idx].set_title(metric_labels[m_idx])
            axes[m_idx].grid(True, alpha=0.3, axis='y')

    axes[0].legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison plot saved to {save_dir}/comparison.pdf")


def plot_trajectory_comparison(traj_true: np.ndarray, traj_pred: np.ndarray,
                                coords: np.ndarray, save_path: str,
                                title: str = 'Trajectory Comparison',
                                node_idx: int = 0):
    """
    Plot true vs predicted trajectory for a single node.

    Args:
        traj_true: (T, N, 3) ground truth
        traj_pred: (T, N, 3) prediction
        coords: (N, 3) coordinates
        save_path: output file path
        node_idx: which node to plot
    """
    if not HAS_MPL:
        return

    T = traj_true.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['x', 'y', 'z']
    colors_true = ['tab:blue', 'tab:green', 'tab:red']
    colors_pred = ['tab:cyan', 'tab:olive', 'tab:orange']

    for d in range(3):
        axes[d].plot(t, traj_true[:, node_idx, d], '-', color=colors_true[d],
                     label=f'True u_{labels[d]}', lw=1.5)
        axes[d].plot(t, traj_pred[:, node_idx, d], '--', color=colors_pred[d],
                     label=f'Pred u_{labels[d]}', lw=1.5)
        axes[d].set_ylabel(f'u_{labels[d]}')
        axes[d].legend(loc='upper right', fontsize=9)
        axes[d].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Timestep')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_discovered_pde(discovered_pde: Dict[str, float], true_terms: set,
                        save_path: str, pde_name: str = ''):
    """
    Visualize discovered vs true PDE terms.
    """
    if not HAS_MPL:
        return

    all_terms = sorted(set(discovered_pde.keys()) | true_terms)
    coeffs = [discovered_pde.get(t, 0) for t in all_terms]
    is_true = [t in true_terms for t in all_terms]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(all_terms)), coeffs,
                  color=['#2ecc71' if t else '#e74c3c' for t in is_true],
                  edgecolor='black', alpha=0.8)

    ax.set_xticks(range(len(all_terms)))
    ax.set_xticklabels(all_terms, rotation=45, ha='right')
    ax.set_ylabel('Coefficient ξ')
    ax.set_title(f'Discovered PDE Terms ({pde_name})\nGreen = true term, Red = spurious')
    ax.axhline(y=0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_equivariance_histogram(eq_errors: List[float], save_path: str,
                                 model_name: str = ''):
    """Plot histogram of equivariance errors across rotations."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eq_errors, bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Equivariance Error')
    ax.set_ylabel('Count')
    ax.set_title(f'Equivariance Error Distribution ({model_name})')
    ax.axvline(x=np.mean(eq_errors), color='red', ls='--',
               label=f'Mean = {np.mean(eq_errors):.5f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation(ablation_results: Dict, save_path: str):
    """Plot ablation study results as grouped bar chart."""
    if not HAS_MPL:
        return

    variants = list(ablation_results.keys())
    metrics = ['rel_l2_mean', 'prr_mean']
    labels = ['Rel-L2 ↓', 'PRR ↑']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for m_idx, (metric, label) in enumerate(zip(metrics, labels)):
        values = [ablation_results[v].get(metric, 0) for v in variants]
        colors = ['#2ecc71' if v == 'E-GNO (full)' else '#3498db' for v in variants]

        axes[m_idx].barh(range(len(variants)), values, color=colors,
                         edgecolor='black', alpha=0.8)
        axes[m_idx].set_yticks(range(len(variants)))
        axes[m_idx].set_yticklabels(variants, fontsize=9)
        axes[m_idx].set_xlabel(label)
        axes[m_idx].set_title(label)
        axes[m_idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_3d_field(coords: np.ndarray, field: np.ndarray, save_path: str,
                  title: str = 'Vector Field', subsample: int = 500):
    """
    3D quiver plot of a vector field on the point cloud.
    """
    if not HAS_MPL:
        return

    N = coords.shape[0]
    idx = np.random.choice(N, min(subsample, N), replace=False)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(coords[idx, 0], coords[idx, 1], coords[idx, 2],
              field[idx, 0], field[idx, 1], field[idx, 2],
              length=0.02, normalize=True, alpha=0.6, color='steelblue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves_from_csv(csv_path: str, save_dir: str):
    """Plot training curves from per-epoch CSV log (training_log.csv)."""
    if not HAS_MPL:
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v and v != '' else 0.0 for k, v in row.items()
                         if k not in ('model', 'pde')})
            rows[-1]['model'] = row.get('model', '')
            rows[-1]['pde'] = row.get('pde', '')

    if not rows:
        return

    epochs = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_specs = [
        ('train_loss', 'Training Loss', True),
        ('rel_l2', 'Rel-L2 Error', False),
        ('stability', 'Rollout Stability', False),
        ('eq_error', 'Equivariance Error', True),
        ('pde_recovery', 'PDE Recovery Rate', False),
        ('sparse_loss', 'Sparsity Loss', True),
    ]

    for idx, (key, title, use_log) in enumerate(plot_specs):
        ax = axes[idx // 3, idx % 3]
        vals = [r.get(key, 0) for r in rows]
        ax.plot(epochs, vals, 'o-', lw=1.5, ms=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        if use_log and any(v > 0 for v in vals):
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Training: {rows[0].get('model', '')} on {rows[0].get('pde', '')}",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves_from_csv.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_from_csv(csv_path: str, save_dir: str):
    """Plot main comparison (Table 3) from main_comparison.csv."""
    if not HAS_MPL:
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    pdes = sorted(set(r['pde'] for r in rows))
    models = sorted(set(r['model'] for r in rows),
                    key=lambda m: ['sindy', 'fno', 'pinn', 'gns', 'egnn',
                                   'egno_no_sindy', 'egno'].index(m)
                    if m in ['sindy', 'fno', 'pinn', 'gns', 'egnn',
                             'egno_no_sindy', 'egno'] else 99)

    metrics = [('rel_l2_mean', 'rel_l2_std', 'Rel-L2 (lower is better)'),
               ('prr_mean', 'prr_std', 'PDE Recovery Rate (higher is better)'),
               ('eq_error_mean', 'eq_error_std', 'Equivariance Error (lower is better)'),
               ('stability_mean', 'stability_std', 'Stability (lower is better)')]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    width = 0.8 / len(pdes)

    for m_idx, (mean_key, std_key, title) in enumerate(metrics):
        ax = axes[m_idx]
        for p_idx, pde in enumerate(pdes):
            pde_rows = [r for r in rows if r['pde'] == pde]
            vals = []
            errs = []
            labels = []
            for model in models:
                mr = [r for r in pde_rows if r['model'] == model]
                if mr:
                    vals.append(float(mr[0].get(mean_key, 0)))
                    errs.append(float(mr[0].get(std_key, 0)))
                    labels.append(model)
                else:
                    vals.append(0)
                    errs.append(0)
                    labels.append(model)

            x = np.arange(len(labels))
            ax.bar(x + p_idx * width, vals, width, yerr=errs,
                   label=pde, alpha=0.8, capsize=2)

        ax.set_xticks(np.arange(len(models)) + width * (len(pdes) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'main_comparison.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Main comparison plot saved to {save_dir}/main_comparison.pdf")


def plot_ablation_from_csv(csv_path: str, save_dir: str):
    """Plot ablation study (Table 4) from ablation.csv."""
    if not HAS_MPL:
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    variants = [r['variant'] for r in rows]
    metrics = [('rel_l2_mean', 'Rel-L2'), ('prr_mean', 'PRR'),
               ('eq_error_mean', 'EqErr'), ('stability_mean', 'Stability')]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    for m_idx, (key, label) in enumerate(metrics):
        vals = [float(r.get(key, 0)) for r in rows]
        colors = ['#2ecc71' if 'full' in v.lower() else '#3498db' for v in variants]
        axes[m_idx].barh(range(len(variants)), vals, color=colors,
                         edgecolor='black', alpha=0.8)
        axes[m_idx].set_yticks(range(len(variants)))
        axes[m_idx].set_yticklabels(variants, fontsize=9)
        axes[m_idx].set_xlabel(label)
        axes[m_idx].set_title(label)
        axes[m_idx].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation.pdf'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Ablation plot saved to {save_dir}/ablation.pdf")


def generate_all_plots(results_dir: str):
    """Generate all plots from saved experiment results."""
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    plot_dir = os.path.join(results_dir, 'plots')

    # Training curves
    for subdir in os.listdir(results_dir):
        hist_path = os.path.join(results_dir, subdir, 'history.json')
        if os.path.exists(hist_path):
            print(f"Plotting training curves for {subdir}...")
            plot_training_curves(hist_path,
                                os.path.join(results_dir, subdir))

    # Training curves from CSV
    for subdir in os.listdir(results_dir):
        csv_train_path = os.path.join(results_dir, subdir, 'training_log.csv')
        if os.path.exists(csv_train_path):
            print(f"Plotting training curves (CSV) for {subdir}...")
            plot_training_curves_from_csv(csv_train_path,
                                          os.path.join(results_dir, subdir))

    # Main comparison (prefer CSV, fallback to JSON)
    comp_csv = os.path.join(results_dir, 'main_comparison.csv')
    comp_json = os.path.join(results_dir, 'main_comparison.json')
    if os.path.exists(comp_csv):
        print("Plotting main comparison from CSV...")
        plot_comparison_from_csv(comp_csv, plot_dir)
    elif os.path.exists(comp_json):
        print("Plotting main comparison from JSON...")
        plot_comparison_table(comp_json, plot_dir)

    # Ablation (prefer CSV, fallback to JSON)
    abl_csv = os.path.join(results_dir, 'ablation.csv')
    abl_json = os.path.join(results_dir, 'ablation.json')
    if os.path.exists(abl_csv):
        print("Plotting ablation from CSV...")
        plot_ablation_from_csv(abl_csv, plot_dir)
    elif os.path.exists(abl_json):
        print("Plotting ablation from JSON...")
        with open(abl_json) as f:
            abl = json.load(f)
        plot_ablation(abl, os.path.join(plot_dir, 'ablation.pdf'))

    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate E-GNO paper plots')
    parser.add_argument('--results_dir', default='results',
                        help='Directory containing experiment results')
    args = parser.parse_args()

    generate_all_plots(args.results_dir)
