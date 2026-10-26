"""
Graph construction utilities for E-GNO.
Multi-scale radius graph construction with edge features.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import math


def radius_graph(coords: torch.Tensor, r_min: float, r_max: float,
                 max_neighbors: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edges for a radius graph: connect nodes i,j if r_min <= ||x_i - x_j|| < r_max.

    Memory-efficient: processes in chunks for large N to avoid O(N²) memory.

    Args:
        coords: (N, 3) node coordinates
        r_min: minimum edge length
        r_max: maximum edge length
        max_neighbors: maximum neighbors per node

    Returns:
        edge_index: (2, E) edge indices [src, dst]
        edge_vec: (E, 3) edge displacement vectors (x_j - x_i)
    """
    N = coords.shape[0]
    device = coords.device

    # For small N, use the simple O(N²) approach
    if N <= 500:
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        mask = (dist >= r_min) & (dist < r_max)
        mask.fill_diagonal_(False)

        if max_neighbors < N:
            dist_masked = dist.clone()
            dist_masked[~mask] = float('inf')
            k = min(max_neighbors, mask.sum(dim=1).max().item())
            if k > 0:
                _, topk_idx = dist_masked.topk(k, dim=1, largest=False)
                new_mask = torch.zeros_like(mask)
                for i in range(N):
                    valid = topk_idx[i][dist_masked[i, topk_idx[i]] < float('inf')]
                    new_mask[i, valid] = True
                mask = new_mask

        src, dst = torch.where(mask)
        edge_index = torch.stack([src, dst], dim=0)
        edge_vec = diff[src, dst]
        return edge_index, edge_vec

    # Chunked approach for larger N — avoids allocating (N, N) matrices
    chunk_size = min(256, N)
    all_src = []
    all_dst = []
    all_vec = []

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        # (chunk, N, 3)
        diff_chunk = coords[i_start:i_end].unsqueeze(1) - coords.unsqueeze(0)
        dist_chunk = torch.norm(diff_chunk, dim=-1)  # (chunk, N)

        mask_chunk = (dist_chunk >= r_min) & (dist_chunk < r_max)
        # Remove self-loops
        for local_i in range(i_end - i_start):
            mask_chunk[local_i, i_start + local_i] = False

        # Limit neighbors per node in this chunk
        if max_neighbors < N:
            dist_masked = dist_chunk.clone()
            dist_masked[~mask_chunk] = float('inf')
            k = min(max_neighbors, N)
            _, topk_idx = dist_masked.topk(k, dim=1, largest=False)
            new_mask = torch.zeros_like(mask_chunk)
            for local_i in range(i_end - i_start):
                valid = topk_idx[local_i][dist_masked[local_i, topk_idx[local_i]] < float('inf')]
                new_mask[local_i, valid] = True
            mask_chunk = new_mask

        local_src, local_dst = torch.where(mask_chunk)
        if local_src.shape[0] > 0:
            all_src.append(local_src + i_start)
            all_dst.append(local_dst)
            all_vec.append(diff_chunk[local_src, local_dst])

    if len(all_src) == 0:
        return (torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, 3, device=device))

    src = torch.cat(all_src)
    dst = torch.cat(all_dst)
    edge_index = torch.stack([src, dst], dim=0)
    edge_vec = torch.cat(all_vec, dim=0)
    return edge_index, edge_vec


def build_multiscale_graphs(coords: torch.Tensor, K: int = 3,
                            r_scales: Optional[List[Tuple[float, float]]] = None,
                            max_neighbors: int = 64
                            ) -> List[Dict[str, torch.Tensor]]:
    """
    Build K resolution-stratified graphs for multi-scale message passing.

    If r_scales is not provided, automatically determines scale ranges based
    on the average nearest-neighbor distance.

    Args:
        coords: (N, 3) node coordinates
        K: number of scales
        r_scales: list of (r_min, r_max) per scale; if None, auto-computed
        max_neighbors: max neighbors per node per scale

    Returns:
        list of dicts, each containing:
            'edge_index': (2, E_k) edge indices
            'edge_vec': (E_k, 3) edge vectors
            'edge_dist': (E_k,) edge lengths
            'edge_dir': (E_k, 3) unit edge directions
    """
    if r_scales is None:
        # Auto-compute scales based on point cloud density (chunked for memory)
        with torch.no_grad():
            N = coords.shape[0]
            chunk = min(256, N)
            nn_dists = []
            for i in range(0, N, chunk):
                diff_c = coords[i:i+chunk].unsqueeze(1) - coords.unsqueeze(0)
                dist_c = torch.norm(diff_c, dim=-1)
                # mask self
                for j in range(min(chunk, N - i)):
                    dist_c[j, i + j] = float('inf')
                nn_dists.append(dist_c.min(dim=1)[0])
            nn_dist = torch.cat(nn_dists).mean().item()

        # Scale 1: nearest neighbors (gradient info)
        # Scale 2: medium range (Laplacian info)
        # Scale 3: longer range (higher-order info)
        r_scales = []
        for k in range(K):
            r_min = 0.0 if k == 0 else r_scales[-1][1]
            r_max = nn_dist * (2 ** (k + 1))
            r_scales.append((r_min, r_max))

    graphs = []
    for k, (r_min, r_max) in enumerate(r_scales):
        edge_index, edge_vec = radius_graph(coords, r_min, r_max, max_neighbors)

        if edge_index.shape[1] == 0:
            # Empty graph at this scale - create dummy
            graphs.append({
                'edge_index': torch.zeros(2, 0, dtype=torch.long, device=coords.device),
                'edge_vec': torch.zeros(0, 3, device=coords.device),
                'edge_dist': torch.zeros(0, device=coords.device),
                'edge_dir': torch.zeros(0, 3, device=coords.device),
                'r_min': r_min,
                'r_max': r_max,
                'scale_idx': k,
            })
            continue

        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_dir = edge_vec / edge_dist.unsqueeze(-1).clamp(min=1e-8)

        graphs.append({
            'edge_index': edge_index,
            'edge_vec': edge_vec,
            'edge_dist': edge_dist,
            'edge_dir': edge_dir,
            'r_min': r_min,
            'r_max': r_max,
            'scale_idx': k,
        })

    return graphs


def batch_graphs(graph_list: List[Dict[str, torch.Tensor]],
                 n_nodes_list: List[int]) -> Dict[str, torch.Tensor]:
    """
    Batch multiple graphs (same scale) into a single disconnected graph.

    Args:
        graph_list: list of graph dicts from build_multiscale_graphs
        n_nodes_list: number of nodes per graph

    Returns:
        batched graph dict with offset edge indices
    """
    edge_indices = []
    edge_vecs = []
    edge_dists = []
    edge_dirs = []
    batch_idx = []
    offset = 0

    for i, (graph, n_nodes) in enumerate(zip(graph_list, n_nodes_list)):
        if graph['edge_index'].shape[1] > 0:
            edge_indices.append(graph['edge_index'] + offset)
            edge_vecs.append(graph['edge_vec'])
            edge_dists.append(graph['edge_dist'])
            edge_dirs.append(graph['edge_dir'])
        batch_idx.extend([i] * n_nodes)
        offset += n_nodes

    if len(edge_indices) == 0:
        device = graph_list[0]['edge_index'].device
        return {
            'edge_index': torch.zeros(2, 0, dtype=torch.long, device=device),
            'edge_vec': torch.zeros(0, 3, device=device),
            'edge_dist': torch.zeros(0, device=device),
            'edge_dir': torch.zeros(0, 3, device=device),
            'batch': torch.tensor(batch_idx, dtype=torch.long, device=device),
        }

    return {
        'edge_index': torch.cat(edge_indices, dim=1),
        'edge_vec': torch.cat(edge_vecs, dim=0),
        'edge_dist': torch.cat(edge_dists, dim=0),
        'edge_dir': torch.cat(edge_dirs, dim=0),
        'batch': torch.tensor(batch_idx, dtype=torch.long,
                               device=edge_indices[0].device),
    }
