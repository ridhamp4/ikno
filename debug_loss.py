"""Debug: trace FNO loss computation step by step."""
import torch
import sys, os
sys.path.insert(0, '.')
from data.dataset import create_dataloaders
from baselines.fno_baseline import FNOBaseline
from utils import get_best_device

device = get_best_device()
dt = 0.01

train_loader, _ = create_dataloaders(
    data_dir='dataset_generation/data', pde_name='maxwell',
    batch_size=2, n_nodes_subsample=500, num_workers=0)

fno = FNOBaseline(field_dim=3, grid_size=32, width=20, modes=8).to(device)

for batch in train_loader:
    u = batch['E'].to(device)  # (B, T, N, 3)
    coords = batch['coords'].to(device)  # (B, N, 3)
    times = batch['times']
    
    print(f"u shape: {u.shape}")
    print(f"times: {times[:5]}")
    print(f"u[0,0] stats: mean={u[0,0].mean():.4f}, std={u[0,0].std():.4f}")
    print(f"u[0,1] stats: mean={u[0,1].mean():.4f}, std={u[0,1].std():.4f}")
    
    u0 = u[0, 0]  # (N, 3)
    u1 = u[0, 1]
    du = u1 - u0
    du_dt = du / dt
    
    print(f"du = u1-u0: mean={du.mean():.6f}, std={du.std():.6f}, max={du.abs().max():.6f}")
    print(f"du/dt: mean={du_dt.mean():.4f}, std={du_dt.std():.4f}, max={du_dt.abs().max():.4f}")
    print(f"(du_dt**2).mean() = {(du_dt**2).mean():.4f}")
    
    # Forward pass
    du_dt_pred = fno(u0, coords[0])
    print(f"du_dt_pred: mean={du_dt_pred.mean():.6f}, std={du_dt_pred.std():.6f}")
    
    mse = ((du_dt_pred - du_dt)**2).mean()
    normalizer = (du_dt**2).mean().clamp(min=1e-6)
    print(f"MSE={mse:.4f}, normalizer={normalizer:.4f}, loss={mse/normalizer:.4f}")
    
    # Also check: what does compute_loss give?
    loss = fno.compute_loss(u[0], coords[0], dt)
    print(f"compute_loss = {loss.item():.6f}")
    break
