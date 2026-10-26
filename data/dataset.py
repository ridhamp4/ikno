"""
PyTorch Dataset and DataLoader for E-GNO experiments.
Loads trajectories from the generated .npz files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import warnings
import json
from typing import Dict, Optional, List, Tuple


class PDEDataset(Dataset):
    """
    Dataset for loading PDE trajectory data from .npz files.

    Each sample contains:
        - coords: (N_nodes, 3) spatial coordinates
        - fields: dict of field_name -> (T, N_nodes, d) field values
        - times: (T,) time values
        - metadata: dict of PDE parameters
    """

    def __init__(self, data_dir: str, pde_name: str, split: str = 'train',
                 n_nodes_subsample: Optional[int] = None,
                 t_steps: Optional[int] = None,
                 transform: Optional[callable] = None,
                 file_paths: Optional[List[str]] = None,
                 deterministic_subsample: bool = False,
                 subsample_seed: int = 0):
        """
        Args:
            data_dir: root data directory (e.g., 'dataset_generation/data')
            pde_name: 'maxwell', 'navier_stokes', or 'elasticity'
            split: 'train' or 'test'
            n_nodes_subsample: if set, randomly subsample to this many nodes
            t_steps: if set, use only this many timesteps
            transform: optional data transformation
        """
        self.data_dir = os.path.join(data_dir, pde_name, split)
        self.pde_name = pde_name
        self.split = split
        self.n_nodes_subsample = n_nodes_subsample
        self.t_steps = t_steps
        self.transform = transform
        self.deterministic_subsample = deterministic_subsample
        self.subsample_seed = subsample_seed

        # Discover trajectory files
        self.files = (sorted(file_paths) if file_paths is not None else sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('.npz')
        ]))

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No .npz files found in {self.data_dir}. "
                f"Run dataset_generation/generate_all.py first."
            )

        # Determine field names from the first file
        sample = np.load(self.files[0])
        self.field_names = [k for k in sample.keys()
                           if k not in ('coords', 'times', 'metadata_json')]
        sample.close()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx], allow_pickle=True)

        coords = torch.from_numpy(data['coords'].astype(np.float32))
        times = torch.from_numpy(data['times'].astype(np.float32))

        fields = {}
        for name in self.field_names:
            fields[name] = torch.from_numpy(data[name].astype(np.float32))

        # Parse metadata
        metadata = {}
        if 'metadata_json' in data:
            metadata = json.loads(str(data['metadata_json']))

        data.close()

        # Subsample nodes if requested
        if self.n_nodes_subsample is not None and self.n_nodes_subsample < coords.shape[0]:
            generator = None
            if self.deterministic_subsample:
                generator = torch.Generator().manual_seed(self.subsample_seed + idx)
            idx_nodes = torch.randperm(coords.shape[0], generator=generator)[:self.n_nodes_subsample]
            idx_nodes = idx_nodes.sort()[0]
            coords = coords[idx_nodes]
            for name in fields:
                fields[name] = fields[name][:, idx_nodes]

        # Subsample timesteps if requested
        if self.t_steps is not None and self.t_steps < times.shape[0]:
            fields = {name: fields[name][:self.t_steps] for name in fields}
            times = times[:self.t_steps]

        sample = {
            'coords': coords,
            'times': times,
            'metadata': metadata,
        }
        sample.update(fields)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_primary_field(self) -> str:
        """Return the primary field name for this PDE type."""
        if self.pde_name == 'maxwell':
            return 'E'  # Electric field (or could use both E, B)
        elif self.pde_name == 'navier_stokes':
            return 'u'  # Velocity
        elif self.pde_name == 'elasticity':
            return 'u'  # Displacement (also has 'v' = velocity)
        return self.field_names[0]

    def get_all_field_names(self) -> List[str]:
        """Return all vector field names."""
        return self.field_names


def collate_pde_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-size point clouds.

    Stacks trajectories and creates batch indices for graph construction.
    """
    # All trajectories should have the same number of nodes and timesteps
    # after subsampling, so we can stack them
    coords = torch.stack([b['coords'] for b in batch])  # (B, N, 3)
    times = batch[0]['times']  # (T,) - same for all

    result = {
        'coords': coords,
        'times': times,
        'batch_size': len(batch),
    }

    # Stack all fields
    field_keys = [k for k in batch[0].keys()
                  if k not in ('coords', 'times', 'metadata')]
    for key in field_keys:
        result[key] = torch.stack([b[key] for b in batch])  # (B, T, N, d)

    return result


def create_dataloaders(data_dir: str, pde_name: str,
                       batch_size: int = 4,
                       n_nodes_subsample: Optional[int] = None,
                       t_steps: Optional[int] = None,
                       num_workers: int = 4
                       ) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders for a PDE dataset.

    Args:
        data_dir: root data directory
        pde_name: 'maxwell', 'navier_stokes', or 'elasticity'
        batch_size: batch size
        n_nodes_subsample: optional node subsampling
        t_steps: optional timestep truncation
        num_workers: dataloader workers

    Returns:
        train_loader, test_loader
    """
    train_dataset = PDEDataset(
        data_dir, pde_name, split='train',
        n_nodes_subsample=n_nodes_subsample,
        t_steps=t_steps,
    )
    test_dataset = PDEDataset(
        data_dir, pde_name, split='test',
        n_nodes_subsample=n_nodes_subsample,
        t_steps=t_steps,
        deterministic_subsample=True,
        subsample_seed=30_000,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pde_batch, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pde_batch, num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def _trajectory_files(data_dir: str, pde_name: str, split: str) -> List[str]:
    split_dir = os.path.join(data_dir, pde_name, split)
    if not os.path.isdir(split_dir):
        return []
    return sorted(os.path.join(split_dir, f) for f in os.listdir(split_dir)
                  if f.endswith('.npz'))


def validate_dataset(data_dir: str, pde_name: str,
                     splits: Tuple[str, ...] = ('train', 'test'),
                     allow_seed_overlap: bool = False) -> Dict:
    """Validate trajectory metadata and reject scientifically unsafe splits."""
    required_fields = {
        'maxwell': {'E', 'B'},
        'navier_stokes': {'u'},
        'elasticity': {'u', 'v'},
    }[pde_name]
    report = {'pde': pde_name, 'splits': {}}
    seeds_by_split = {}

    for split in splits:
        files = _trajectory_files(data_dir, pde_name, split)
        if not files:
            raise FileNotFoundError(f'No trajectories found for {pde_name}/{split}')
        seeds = set()
        signatures = set()
        for path in files:
            with np.load(path, allow_pickle=True) as sample:
                fields = set(sample.files) - {'coords', 'times', 'metadata_json'}
                missing = required_fields - fields
                if missing:
                    raise ValueError(f'{path} is missing required fields: {sorted(missing)}')
                times = np.asarray(sample['times'], dtype=np.float64)
                if len(times) < 2 or not np.all(np.diff(times) > 0):
                    raise ValueError(f'{path} has invalid or non-increasing times')
                dt = float(np.diff(times).mean())
                if not np.allclose(np.diff(times), dt, rtol=1e-5, atol=1e-8):
                    raise ValueError(f'{path} has non-uniform saved time intervals')
                shapes = tuple((name, sample[name].shape[-1]) for name in sorted(required_fields))
                signatures.add((round(dt, 10), shapes))
                if 'metadata_json' in sample.files:
                    metadata = json.loads(str(sample['metadata_json']))
                    if metadata.get('seed') is not None:
                        seeds.add(int(metadata['seed']))
        if len(signatures) != 1:
            raise ValueError(f'{pde_name}/{split} mixes incompatible time steps or fields: '
                             f'{sorted(signatures)}')
        seeds_by_split[split] = seeds
        report['splits'][split] = {
            'n_trajectories': len(files),
            'seeds': sorted(seeds),
            'signature': list(signatures)[0],
        }

    if not allow_seed_overlap:
        names = list(seeds_by_split)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = seeds_by_split[left] & seeds_by_split[right]
                if overlap:
                    raise ValueError(
                        f'{pde_name} has seed leakage between {left} and {right}: '
                        f'{sorted(overlap)}')
    return report


def create_train_val_test_dataloaders(
        data_dir: str, pde_name: str, batch_size: int = 4,
        n_nodes_subsample: Optional[int] = None,
        t_steps: Optional[int] = None, num_workers: int = 4,
        split_seed: int = 0, val_fraction: float = 0.2,
        strict_integrity: bool = True,
        max_train_trajectories: Optional[int] = None):
    """Create loaders with a validation set that is never the test set."""
    if strict_integrity:
        validate_dataset(data_dir, pde_name, ('train', 'test'))

    train_files = _trajectory_files(data_dir, pde_name, 'train')
    val_files = _trajectory_files(data_dir, pde_name, 'val')
    if val_files:
        if strict_integrity:
            validate_dataset(data_dir, pde_name, ('train', 'val', 'test'))
    else:
        if len(train_files) < 2:
            raise ValueError('At least two training trajectories are required to create validation data')
        generator = torch.Generator().manual_seed(split_seed)
        order = torch.randperm(len(train_files), generator=generator).tolist()
        n_val = max(1, int(round(len(train_files) * val_fraction)))
        n_val = min(n_val, len(train_files) - 1)
        val_indices = set(order[:n_val])
        val_files = [f for i, f in enumerate(train_files) if i in val_indices]
        train_files = [f for i, f in enumerate(train_files) if i not in val_indices]
        warnings.warn('No val/ directory found; using a deterministic holdout from train/.',
                      stacklevel=2)

    if max_train_trajectories is not None:
        if max_train_trajectories < 1:
            raise ValueError('max_train_trajectories must be positive')
        # The holdout is deterministic and the remaining paths are sorted, so
        # the low-data subset is reproducible and shared by every method.
        train_files = train_files[:max_train_trajectories]

    common = dict(n_nodes_subsample=n_nodes_subsample, t_steps=t_steps)
    train_dataset = PDEDataset(data_dir, pde_name, 'train', file_paths=train_files,
                               deterministic_subsample=False, **common)
    val_dataset = PDEDataset(data_dir, pde_name, 'val', file_paths=val_files,
                             deterministic_subsample=True,
                             subsample_seed=20_000 + split_seed, **common)
    test_dataset = PDEDataset(data_dir, pde_name, 'test',
                              deterministic_subsample=True,
                              subsample_seed=30_000 + split_seed, **common)

    loader_args = dict(batch_size=batch_size, collate_fn=collate_pde_batch,
                       num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(split_seed),
                              **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader, test_loader


class RotationAugmentation:
    """Data augmentation: apply random SO(3) rotation to coordinates and fields."""

    def __call__(self, sample: Dict) -> Dict:
        # Random rotation matrix
        q, r = torch.linalg.qr(torch.randn(3, 3))
        d = torch.diag(torch.sign(torch.diag(r)))
        R = q @ d
        if torch.det(R) < 0:
            R[:, 0] = -R[:, 0]

        # Rotate coordinates
        sample['coords'] = sample['coords'] @ R.T

        # Rotate vector fields
        for key in list(sample.keys()):
            if key in ('coords', 'times', 'metadata'):
                continue
            val = sample[key]
            if val.dim() >= 2 and val.shape[-1] == 3:
                sample[key] = val @ R.T

        return sample
