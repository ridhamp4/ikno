"""
Smoke test: validates all E-GNO components work end-to-end with synthetic data.

Run: python test_smoke.py
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_spherical_harmonics():
    print("  [1/10] Spherical Harmonics...", end=" ")
    from models.egno.spherical_harmonics import SphericalHarmonics, GaussianRBF

    sh = SphericalHarmonics(l_max=2)
    r_hat = torch.randn(50, 3)
    r_hat = r_hat / r_hat.norm(dim=-1, keepdim=True)
    Y = sh(r_hat)
    assert 0 in Y and 1 in Y and 2 in Y
    assert Y[0].shape == (50, 1)
    assert Y[1].shape == (50, 3)
    assert Y[2].shape == (50, 5)

    rbf = GaussianRBF(n_rbf=16)
    r = torch.rand(50)
    out = rbf(r)
    assert out.shape == (50, 16)
    print("OK")


def test_cg_products():
    print("  [2/10] Clebsch-Gordan Products...", end=" ")
    from models.egno.spherical_harmonics import ClebschGordanProducts

    cg = ClebschGordanProducts(l_max=2)
    N = 20
    C = 8

    # type-1 x type-1 -> type-0, type-1, type-2
    f = torch.randn(N, 3 * C)
    g = torch.randn(N, 3)

    out_0 = cg.tensor_product(f, g, l1=1, l2=1, l3=0)
    assert out_0.shape == (N, 1 * C), f"Expected ({N},{C}), got {out_0.shape}"

    out_1 = cg.tensor_product(f, g, l1=1, l2=1, l3=1)
    assert out_1.shape == (N, 3 * C), f"Expected ({N},{3*C}), got {out_1.shape}"

    out_2 = cg.tensor_product(f, g, l1=1, l2=1, l3=2)
    assert out_2.shape == (N, 5 * C), f"Expected ({N},{5*C}), got {out_2.shape}"
    print("OK")


def test_graph_utils():
    print("  [3/10] Graph Construction...", end=" ")
    from models.graph_utils import radius_graph, build_multiscale_graphs

    coords = torch.rand(100, 3)
    edge_index, edge_vec = radius_graph(coords, 0.0, 0.3)
    assert edge_index.shape[0] == 2
    assert edge_vec.shape[1] == 3

    graphs = build_multiscale_graphs(coords, K=3)
    assert len(graphs) == 3
    for g in graphs:
        assert 'edge_index' in g
        assert 'edge_dist' in g
        assert 'edge_dir' in g
    print("OK")


def test_encoder():
    print("  [4/10] Equivariant Encoder...", end=" ")
    from models.egno.encoder import EquivariantEncoder

    enc = EquivariantEncoder(in_field_dim=3, hidden_channels=16, l_max=2)
    u = torch.randn(50, 3)
    features = enc(u)
    assert 0 in features and 1 in features and 2 in features
    assert features[0].shape == (50, 16)
    assert features[1].shape == (50, 3 * 16)
    assert features[2].shape == (50, 5 * 16)
    print("OK")


def test_ms_shmp():
    print("  [5/10] MS-SHMP...", end=" ")
    from models.egno.ms_shmp import MSSHMP
    from models.graph_utils import build_multiscale_graphs

    C = 16
    msshmp = MSSHMP(hidden_channels=C, l_max=1, n_scales=2, n_rbf=8, n_layers=1)

    N = 30
    coords = torch.rand(N, 3)
    graphs = build_multiscale_graphs(coords, K=2)

    h = {
        0: torch.randn(N, C),
        1: torch.randn(N, 3 * C),
    }

    h_new, scale_aggs = msshmp(h, graphs, N)
    assert 0 in h_new and 1 in h_new
    assert h_new[0].shape == (N, C)
    assert h_new[1].shape == (N, 3 * C)
    assert len(scale_aggs) == 2
    print("OK")


def test_e_dol():
    print("  [6/10] E-DOL (Library)...", end=" ")
    from models.egno.e_dol import EDOL

    C = 16
    N = 30
    edol = EDOL(hidden_channels=C, l_max=1, n_scales=2)

    h = {
        0: torch.randn(N, C),
        1: torch.randn(N, 3 * C),
    }
    scale_aggs = [
        {0: torch.randn(N, C), 1: torch.randn(N, 3 * C)},
        {0: torch.randn(N, C), 1: torch.randn(N, 3 * C)},
    ]
    u = torch.randn(N, 3)

    library = edol(h, scale_aggs, u)
    assert len(library) > 0

    theta, names = edol.get_type_consistent_library(library, target_type=1)
    assert theta is not None
    assert len(names) > 0
    assert theta.shape[0] == N
    print(f"OK ({len(names)} type-1 terms: {names})")


def test_e_sindy():
    print("  [7/10] E-SINDy...", end=" ")
    from models.egno.e_sindy import ESINDy

    N = 50
    P = 5
    C = 16
    D = 3 * C

    esindy = ESINDy(n_library_terms=P, hidden_channels=C)
    theta = torch.randn(N, P, D)
    du_dt = torch.randn(N, D)

    pred = esindy(theta)
    assert pred.shape == (N, D)

    loss, info = esindy.compute_loss(theta, du_dt)
    assert loss.item() > 0

    esindy.run_iht(theta, du_dt)
    pde = esindy.get_discovered_pde(['a', 'b', 'c', 'd', 'e'])
    print(f"OK (discovered {len(pde)} terms)")


def test_neural_ode():
    print("  [8/10] Neural ODE Rollout...", end=" ")
    from models.egno.neural_ode import NeuralODERollout

    ode = NeuralODERollout(field_dim=3, use_residual=True, hidden_channels=16)

    def simple_rhs(u):
        return -0.1 * u  # simple decay

    u0 = torch.randn(30, 3)
    traj = ode(simple_rhs, u0, dt=0.01, n_steps=5)
    assert traj.shape == (6, 30, 3)
    # Check that trajectory decays
    assert traj[-1].norm() < traj[0].norm() * 1.5  # shouldn't explode
    print("OK")


def test_full_egno():
    print("  [9/10] Full E-GNO Model...", end=" ")
    from models.egno import EGNO

    model = EGNO(
        field_dim=3,
        hidden_channels=16,
        l_max=1,
        n_scales=2,
        n_mp_layers=1,
        n_rbf=8,
        n_library_terms=8,
        lambda_sparse=0.01,
        use_residual=False,
        max_neighbors=16,
    )

    N = 30
    coords = torch.rand(N, 3)
    u0 = torch.randn(N, 3)

    # Single-step forward
    du_dt, info = model.compute_rhs(u0, coords)
    assert du_dt.shape == (N, 3)

    # Multi-step forward
    traj, info = model(u0, coords, dt=0.01, n_steps=2, method='euler')
    assert traj.shape == (3, N, 3)

    # Compute loss
    u_true = torch.randn(3, N, 3)
    loss, loss_info = model.compute_loss(u_true, coords, dt=0.01, n_rollout_steps=2)
    assert loss.item() > 0

    # Backprop
    loss.backward()
    n_params = sum(p.numel() for p in model.parameters())
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"OK ({n_params:,} params, {has_grad} with gradients)")


def test_baselines():
    print("  [10/10] Baselines...", end=" ")
    N = 30
    coords = torch.rand(N, 3)
    u0 = torch.randn(N, 3)
    u_traj = torch.randn(3, N, 3)

    # GNS
    from baselines.gns_baseline import GNS
    gns = GNS(field_dim=3, hidden_dim=32, n_layers=2, r_max=0.5)
    du_dt = gns(u0, coords)
    assert du_dt.shape == (N, 3)

    # FNO
    from baselines.fno_baseline import FNOBaseline
    fno = FNOBaseline(field_dim=3, grid_size=8, width=8, modes=4, n_layers=2)
    du_dt = fno(u0, coords)
    assert du_dt.shape == (N, 3)

    # SEGNN
    from baselines.segnn_baseline import SEGNN
    segnn = SEGNN(field_dim=3, hidden_channels=16, l_max=1, n_layers=2, r_max=0.5)
    du_dt = segnn(u0, coords)
    assert du_dt.shape == (N, 3)

    # PINN
    from baselines.pinn_baseline import PINNBaseline
    pinn = PINNBaseline(pde_type='maxwell', field_dim=3, hidden_dim=32, n_layers=2)
    u_pred = pinn.predict(coords, t=0.01)
    assert u_pred.shape == (N, 3)

    # SINDy (numpy-based)
    from baselines.sindy_baseline import SINDyBaseline
    sindy = SINDyBaseline(n_neighbors=10)
    coords_np = coords.numpy()
    u_np = u0.numpy()
    du_dt_np = np.random.randn(N, 3).astype(np.float32)
    sindy.fit(coords_np, u_np, du_dt_np)
    pred = sindy.predict(coords_np, u_np)
    assert pred.shape == (N, 3)

    print("OK (GNS, FNO, SEGNN, PINN, SINDy)")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  E-GNO Smoke Test")
    print("=" * 60 + "\n")

    tests = [
        test_spherical_harmonics,
        test_cg_products,
        test_graph_utils,
        test_encoder,
        test_ms_shmp,
        test_e_dol,
        test_e_sindy,
        test_neural_ode,
        test_full_egno,
        test_baselines,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")
