"""
Fourier Neural Operator (FNO) baseline.

Interpolates irregular mesh data onto a regular grid, applies FNO,
then interpolates back. No symmetry enforcement.

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class SpectralConv3d(nn.Module):
    """3D Fourier convolution layer.

    Stores weights as real float32 tensors (..., 2) for MPS/CUDA/CPU compatibility,
    and converts to complex via view_as_complex at forward time.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        # Store as real (..., 2) for device compatibility; last dim = [real, imag]
        for i in range(1, 5):
            w = scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
            setattr(self, f'weights{i}', nn.Parameter(w))

    def _get_complex_weight(self, idx: int) -> torch.Tensor:
        """Convert real (..., 2) parameter to complex tensor."""
        w = getattr(self, f'weights{idx}')
        return torch.view_as_complex(w.contiguous())

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        spatial_sizes = (x.size(-3), x.size(-2), x.size(-1))

        # Move to CPU for FFT if on MPS (MPS complex support varies by PyTorch version)
        orig_device = x.device
        use_cpu_fft = (orig_device.type == 'mps')
        if use_cpu_fft:
            x = x.cpu()

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(B, self.out_channels,
                              x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                              dtype=torch.cfloat, device=x.device)

        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        # Get complex weights (moved to same device as x_ft)
        w1 = self._get_complex_weight(1).to(x.device)
        w2 = self._get_complex_weight(2).to(x.device)
        w3 = self._get_complex_weight(3).to(x.device)
        w4 = self._get_complex_weight(4).to(x.device)

        out_ft[:, :, :m1, :m2, :m3] = \
            self.compl_mul3d(x_ft[:, :, :m1, :m2, :m3], w1)
        out_ft[:, :, -m1:, :m2, :m3] = \
            self.compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], w2)
        out_ft[:, :, :m1, -m2:, :m3] = \
            self.compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], w3)
        out_ft[:, :, -m1:, -m2:, :m3] = \
            self.compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], w4)

        result = torch.fft.irfftn(out_ft, s=spatial_sizes)

        if use_cpu_fft:
            result = result.to(orig_device)
        return result


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator.

    Operates on regular grids; irregular mesh data must be interpolated.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 width: int = 20, modes: int = 8, n_layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers

        # Input projection
        self.fc0 = nn.Linear(in_channels + 3, width)  # +3 for coordinates

        # Spectral conv layers
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.conv_layers.append(SpectralConv3d(width, width, modes, modes, modes))
            self.w_layers.append(nn.Conv3d(width, width, 1))

        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, Nx, Ny, Nz, in_channels) field on regular grid
            grid: (B, Nx, Ny, Nz, 3) coordinate grid

        Returns:
            out: (B, Nx, Ny, Nz, out_channels) predicted field
        """
        x = torch.cat([x, grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, Nx, Ny, Nz)

        for i in range(self.n_layers):
            x1 = self.conv_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)

        x = x.permute(0, 2, 3, 4, 1)  # (B, Nx, Ny, Nz, C)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class FNOBaseline(nn.Module):
    """
    FNO baseline wrapper that handles interpolation between
    irregular meshes and regular grids.
    """

    def __init__(self, field_dim: int = 3, grid_size: int = 32,
                 width: int = 20, modes: int = 8, n_layers: int = 4):
        super().__init__()
        self.field_dim = field_dim
        self.grid_size = grid_size

        self.fno = FNO3d(
            in_channels=field_dim,
            out_channels=field_dim,
            width=width,
            modes=modes,
            n_layers=n_layers,
        )

        # Create regular grid coordinates
        g = torch.linspace(0, 1, grid_size)
        gz, gy, gx = torch.meshgrid(g, g, g, indexing='ij')
        self.register_buffer('grid_coords',
                             torch.stack([gx, gy, gz], dim=-1))  # (N, N, N, 3)

    def interpolate_to_grid(self, coords: torch.Tensor, field: torch.Tensor
                           ) -> torch.Tensor:
        """
        Interpolate field from irregular mesh to regular grid using
        inverse-distance weighting.

        Args:
            coords: (N, 3) irregular mesh coordinates in [0, 1]³
            field: (N, d) field values

        Returns:
            grid_field: (Nx, Ny, Nz, d) field on regular grid
        """
        G = self.grid_size
        orig_device = coords.device

        # IDW interpolation: compute indices on CPU for MPS compatibility (no grad needed here)
        coords_cpu = coords.detach().cpu()
        field_dev = field  # keep on device for grad flow

        grid_flat = self.grid_coords.reshape(-1, 3).cpu()  # (G³, 3)
        dist = torch.cdist(grid_flat, coords_cpu)  # (G³, N)
        k = min(8, coords.shape[0])
        topk_dist, topk_idx = dist.topk(k, dim=1, largest=False)
        weights = 1.0 / (topk_dist + 1e-6)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Gather from field on original device (preserves grad flow)
        topk_idx_dev = topk_idx.to(orig_device)
        weights_dev = weights.to(orig_device)
        field_gathered = field_dev[topk_idx_dev]  # (G³, k, d)
        grid_field = (weights_dev.unsqueeze(-1) * field_gathered).sum(dim=1)  # (G³, d)
        return grid_field.reshape(G, G, G, -1)

    def interpolate_to_mesh(self, grid_field: torch.Tensor,
                            coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate from regular grid back to irregular mesh.

        Uses grid_sample on CPU when on MPS (backward not implemented on MPS).

        Args:
            grid_field: (Nx, Ny, Nz, d) field on grid
            coords: (N, 3) mesh coordinates

        Returns:
            field: (N, d) interpolated field
        """
        orig_device = grid_field.device
        # MPS doesn't support grid_sampler_3d_backward — run on CPU
        use_cpu = (orig_device.type == 'mps')
        if use_cpu:
            grid_field = grid_field.cpu()
            coords = coords.cpu()

        # grid_sample expects (B, C, D, H, W) input
        grid_in = grid_field.permute(3, 0, 1, 2).unsqueeze(0)  # (1, d, G, G, G)

        # Normalize coords to [-1, 1] for grid_sample
        grid_pts = 2.0 * coords - 1.0  # [0,1] -> [-1,1]
        # grid_sample expects (B, D_out, H_out, W_out, 3)
        grid_pts = grid_pts.view(1, 1, 1, -1, 3)

        sampled = F.grid_sample(grid_in, grid_pts, mode='bilinear',
                                 align_corners=True, padding_mode='border')
        result = sampled.squeeze(0).squeeze(1).squeeze(1).T  # (N, d)

        if use_cpu:
            result = result.to(orig_device)
        return result

    def forward(self, u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Predict du/dt using FNO.

        Args:
            u: (N, d) field on irregular mesh
            coords: (N, 3) mesh coordinates

        Returns:
            du_dt: (N, d) predicted derivative
        """
        # Interpolate to grid
        u_grid = self.interpolate_to_grid(coords, u)  # (G, G, G, d)
        u_grid = u_grid.unsqueeze(0)  # (1, G, G, G, d)

        grid = self.grid_coords.unsqueeze(0).to(u.device)  # (1, G, G, G, 3)

        # FNO forward
        du_dt_grid = self.fno(u_grid, grid)  # (1, G, G, G, d)

        # Interpolate back to mesh
        du_dt = self.interpolate_to_mesh(du_dt_grid.squeeze(0), coords)

        return du_dt

    def compute_loss(self, u_traj: torch.Tensor, coords: torch.Tensor,
                     dt: float) -> torch.Tensor:
        """One-step prediction loss (MSE on du/dt)."""
        u0 = u_traj[0]
        u1_true = u_traj[1]
        du_dt_true = (u1_true - u0) / dt
        du_dt_pred = self.forward(u0, coords)
        return ((du_dt_pred - du_dt_true) ** 2).mean()
