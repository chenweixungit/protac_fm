"""Collate functions for batching PROTAC data."""
import torch
from typing import Dict, List


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a batch of PROTAC samples.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Batched dict with all tensors properly batched
    """
    batch_size = len(batch)
    
    # Get dimensions
    N_per_sample = [b['x_data'].shape[0] for b in batch]
    N_total = sum(N_per_sample)
    
    E_per_sample = [b['edge_index'].shape[1] for b in batch]
    E_total = sum(E_per_sample)
    
    # Initialize batched tensors
    x_data = torch.zeros(N_total, 3)
    atom_type = torch.zeros(N_total, dtype=torch.long)
    node_kind = torch.zeros(N_total, dtype=torch.long)
    is_anchor = torch.zeros(N_total, dtype=torch.long)
    fragB_mask = torch.zeros(N_total, dtype=torch.bool)
    linker_mask = torch.zeros(N_total, dtype=torch.bool)
    batch_idx = torch.zeros(N_total, dtype=torch.long)
    
    edge_index = torch.zeros(2, E_total, dtype=torch.long)
    bond_type_gt = torch.zeros(E_total, dtype=torch.long)
    edge_batch = torch.zeros(E_total, dtype=torch.long)
    
    anchors = torch.zeros(batch_size, 2, dtype=torch.long)
    cond = torch.zeros(batch_size, batch[0]['cond'].shape[0])
    p_gt = torch.zeros(batch_size, 3)
    r_gt = torch.zeros(batch_size, 3)
    
    # Metadata
    N_fragA = torch.zeros(batch_size, dtype=torch.long)
    N_fragB = torch.zeros(batch_size, dtype=torch.long)
    N_linker = torch.zeros(batch_size, dtype=torch.long)
    
    # Fill tensors
    node_offset = 0
    edge_offset = 0
    
    for i, sample in enumerate(batch):
        n = sample['x_data'].shape[0]
        e = sample['edge_index'].shape[1]
        
        # Node features
        x_data[node_offset:node_offset + n] = sample['x_data']
        atom_type[node_offset:node_offset + n] = sample['atom_type']
        node_kind[node_offset:node_offset + n] = sample['node_kind']
        is_anchor[node_offset:node_offset + n] = sample['is_anchor']
        fragB_mask[node_offset:node_offset + n] = sample['fragB_mask']
        linker_mask[node_offset:node_offset + n] = sample['linker_mask']
        batch_idx[node_offset:node_offset + n] = i
        
        # Edge features (with offset)
        edge_index[:, edge_offset:edge_offset + e] = sample['edge_index'] + node_offset
        bond_type_gt[edge_offset:edge_offset + e] = sample['bond_type_gt']
        edge_batch[edge_offset:edge_offset + e] = i
        
        # Graph-level features
        anchors[i] = sample['anchors'] + node_offset
        cond[i] = sample['cond']
        p_gt[i] = sample['p_gt']
        r_gt[i] = sample['r_gt']
        
        # Metadata
        N_fragA[i] = sample['N_fragA']
        N_fragB[i] = sample['N_fragB']
        N_linker[i] = sample['N_linker']
        
        node_offset += n
        edge_offset += e
    
    # Compute ptr (pointer to start of each graph)
    ptr = torch.zeros(batch_size + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(torch.tensor(N_per_sample), dim=0)
    
    return {
        'x_data': x_data,
        'atom_type': atom_type,
        'node_kind': node_kind,
        'is_anchor': is_anchor,
        'fragB_mask': fragB_mask,
        'linker_mask': linker_mask,
        'batch': batch_idx,
        'ptr': ptr,
        'edge_index': edge_index,
        'bond_type_gt': bond_type_gt,
        'edge_batch': edge_batch,
        'anchors': anchors,
        'cond': cond,
        'p_gt': p_gt,
        'r_gt': r_gt,
        'N_fragA': N_fragA,
        'N_fragB': N_fragB,
        'N_linker': N_linker,
        'num_graphs': batch_size,
    }


def unbatch(batched: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """Unbatch a batched dict back to list of samples.
    
    Args:
        batched: Batched dict from collate_fn
        
    Returns:
        List of sample dicts
    """
    batch_size = batched['num_graphs']
    ptr = batched['ptr']
    
    samples = []
    
    for i in range(batch_size):
        start = ptr[i].item()
        end = ptr[i + 1].item()
        
        # Node mask for this sample
        node_mask = batched['batch'] == i
        edge_mask = batched['edge_batch'] == i
        
        sample = {
            'x_data': batched['x_data'][node_mask],
            'atom_type': batched['atom_type'][node_mask],
            'node_kind': batched['node_kind'][node_mask],
            'is_anchor': batched['is_anchor'][node_mask],
            'fragB_mask': batched['fragB_mask'][node_mask],
            'linker_mask': batched['linker_mask'][node_mask],
            'edge_index': batched['edge_index'][:, edge_mask] - start,
            'bond_type_gt': batched['bond_type_gt'][edge_mask],
            'anchors': batched['anchors'][i] - start,
            'cond': batched['cond'][i],
            'p_gt': batched['p_gt'][i],
            'r_gt': batched['r_gt'][i],
            'N_fragA': batched['N_fragA'][i].item(),
            'N_fragB': batched['N_fragB'][i].item(),
            'N_linker': batched['N_linker'][i].item(),
        }
        samples.append(sample)
    
    return samples
