"""Quick convergence test: all baselines with output scale + r_max=0.3."""
if __name__ != '__main__':
    import pytest
    pytest.skip('manual 50-epoch convergence benchmark', allow_module_level=True)

import torch
from data.dataset import create_dataloaders
from baselines.fno_baseline import FNOBaseline
from baselines.gns_baseline import GNS
from baselines.egnn_baseline import EGNNBaseline
from baselines.segnn_baseline import SEGNN
from utils import get_best_device

device = get_best_device()
print(f"Device: {device}")
dt = 0.01

train_loader, _ = create_dataloaders(
    data_dir='dataset_generation/data', pde_name='maxwell',
    batch_size=2, n_nodes_subsample=500, num_workers=0)

for name, Model, kwargs in [
    ('FNO', FNOBaseline, dict(field_dim=3, grid_size=32, width=20, modes=8)),
    ('GNS', GNS, dict(field_dim=3, hidden_dim=128, n_layers=6, r_max=0.3)),
    ('EGNN', EGNNBaseline, dict(field_dim=3, hidden_dim=128, n_layers=4, r_max=0.3)),
    ('SEGNN', SEGNN, dict(field_dim=3, hidden_channels=64, l_max=1, n_layers=4, r_max=0.3)),
]:
    model = Model(**kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    for epoch in range(50):
        total = 0; n = 0
        for batch in train_loader:
            u = batch['E'].to(device)
            coords = batch['coords'].to(device)
            opt.zero_grad()
            loss_sum = 0
            for b in range(u.shape[0]):
                loss = model.compute_loss(u[b], coords[b], dt)
                loss_sum = loss_sum + loss
            loss_avg = loss_sum / u.shape[0]
            loss_avg.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            total += loss_avg.item(); n += 1
        sched.step()
        if (epoch+1) % 10 == 0:
            print(f"  {name:5s} epoch {epoch+1:3d}: loss={total/n:.4f}")
    print()
