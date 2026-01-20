"""Featurization utilities for building graph tensors."""
import numpy as np
import torch
from rdkit import Chem
from typing import Dict, List, Optional, Tuple


# Atom vocabulary
ATOM_VOCAB = ["MASK", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
ATOM_TO_IDX = {atom: idx for idx, atom in enumerate(ATOM_VOCAB)}

# Bond vocabulary
BOND_VOCAB = ["NONE", "SINGLE", "DOUBLE", "AROMATIC"]
BOND_TO_IDX = {bond: idx for idx, bond in enumerate(BOND_VOCAB)}

# RDKit bond type mapping
RDKIT_BOND_TO_IDX = {
    Chem.BondType.SINGLE: BOND_TO_IDX["SINGLE"],
    Chem.BondType.DOUBLE: BOND_TO_IDX["DOUBLE"],
    Chem.BondType.TRIPLE: BOND_TO_IDX["DOUBLE"],  # Map triple to double
    Chem.BondType.AROMATIC: BOND_TO_IDX["AROMATIC"],
}

# Maximum valence for each atom type
MAX_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 6, "F": 1, 
    "Cl": 1, "Br": 1, "I": 1, "P": 5, "MASK": 0
}


def get_atom_type_idx(atom: Chem.Atom) -> int:
    """Get atom type index from RDKit atom.
    
    Args:
        atom: RDKit Atom object
        
    Returns:
        Index in ATOM_VOCAB
    """
    symbol = atom.GetSymbol()
    if symbol in ATOM_TO_IDX:
        return ATOM_TO_IDX[symbol]
    else:
        return ATOM_TO_IDX["C"]  # Default to carbon


def get_bond_type_idx(bond: Chem.Bond) -> int:
    """Get bond type index from RDKit bond.
    
    Args:
        bond: RDKit Bond object
        
    Returns:
        Index in BOND_VOCAB
    """
    bond_type = bond.GetBondType()
    if bond_type in RDKIT_BOND_TO_IDX:
        return RDKIT_BOND_TO_IDX[bond_type]
    else:
        return BOND_TO_IDX["SINGLE"]  # Default to single


def build_graph_tensors(
    fragA_mol: Chem.Mol,
    fragB_mol: Chem.Mol,
    linker_mol: Chem.Mol,
    anchors: Dict,
    full_mol: Chem.Mol,
    N_L_max: int = 32,
    cond: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Build graph tensors for a single PROTAC sample.
    
    Args:
        fragA_mol: Fragment A molecule
        fragB_mol: Fragment B molecule
        linker_mol: Linker molecule
        anchors: Dict with anchor information
        full_mol: Full PROTAC molecule
        N_L_max: Maximum number of linker atoms
        cond: Optional condition vector
        
    Returns:
        sample_dict: Dict containing all tensors for the sample
    """
    conf = full_mol.GetConformer()
    
    # Get atom indices
    fragA_indices = anchors['fragA_atom_indices']
    fragB_indices = anchors['fragB_atom_indices']
    linker_indices = anchors['linker_atom_indices']
    
    N_fragA = len(fragA_indices)
    N_fragB = len(fragB_indices)
    N_linker = len(linker_indices)
    N_linker_padded = N_L_max
    
    # Total nodes: fragA + fragB + linker (padded)
    N = N_fragA + N_fragB + N_linker_padded
    
    # Initialize tensors
    x_data = torch.zeros(N, 3, dtype=torch.float32)
    atom_type = torch.zeros(N, dtype=torch.long)
    node_kind = torch.zeros(N, dtype=torch.long)  # 0=fragA, 1=fragB, 2=linker
    is_anchor = torch.zeros(N, dtype=torch.long)
    fragB_mask = torch.zeros(N, dtype=torch.bool)
    linker_mask = torch.zeros(N, dtype=torch.bool)
    
    # Index mapping: original -> new
    idx_map = {}
    
    # Fill fragA data
    for i, orig_idx in enumerate(fragA_indices):
        new_idx = i
        idx_map[orig_idx] = new_idx
        
        pos = conf.GetAtomPosition(orig_idx)
        x_data[new_idx] = torch.tensor([pos.x, pos.y, pos.z])
        
        atom = full_mol.GetAtomWithIdx(orig_idx)
        atom_type[new_idx] = get_atom_type_idx(atom)
        node_kind[new_idx] = 0  # fragA
        
        if orig_idx == anchors['anchorA_idx']:
            is_anchor[new_idx] = 1
    
    # Fill fragB data
    for i, orig_idx in enumerate(fragB_indices):
        new_idx = N_fragA + i
        idx_map[orig_idx] = new_idx
        
        pos = conf.GetAtomPosition(orig_idx)
        x_data[new_idx] = torch.tensor([pos.x, pos.y, pos.z])
        
        atom = full_mol.GetAtomWithIdx(orig_idx)
        atom_type[new_idx] = get_atom_type_idx(atom)
        node_kind[new_idx] = 1  # fragB
        fragB_mask[new_idx] = True
        
        if orig_idx == anchors['anchorB_idx']:
            is_anchor[new_idx] = 1
    
    # Fill linker data
    for i, orig_idx in enumerate(linker_indices):
        new_idx = N_fragA + N_fragB + i
        idx_map[orig_idx] = new_idx
        
        pos = conf.GetAtomPosition(orig_idx)
        x_data[new_idx] = torch.tensor([pos.x, pos.y, pos.z])
        
        atom = full_mol.GetAtomWithIdx(orig_idx)
        atom_type[new_idx] = get_atom_type_idx(atom)
        node_kind[new_idx] = 2  # linker
        linker_mask[new_idx] = True
    
    # Fill padding for linker (MASK tokens)
    for i in range(N_linker, N_linker_padded):
        new_idx = N_fragA + N_fragB + i
        atom_type[new_idx] = ATOM_TO_IDX["MASK"]
        node_kind[new_idx] = 2  # linker
        linker_mask[new_idx] = True
        # Padding coordinates: use mean of linker or zeros
        if N_linker > 0:
            linker_center = x_data[N_fragA + N_fragB:N_fragA + N_fragB + N_linker].mean(dim=0)
            x_data[new_idx] = linker_center
    
    # Build edge index (fully connected for now, can be optimized)
    edge_list = []
    edge_types = []
    
    # Add bonds from full molecule
    for bond in full_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        if begin_idx in idx_map and end_idx in idx_map:
            new_begin = idx_map[begin_idx]
            new_end = idx_map[end_idx]
            bond_type = get_bond_type_idx(bond)
            
            edge_list.append([new_begin, new_end])
            edge_list.append([new_end, new_begin])
            edge_types.append(bond_type)
            edge_types.append(bond_type)
    
    # Add fully connected edges for linker nodes (for message passing)
    linker_start = N_fragA + N_fragB
    linker_end = N_fragA + N_fragB + N_linker_padded
    
    for i in range(linker_start, linker_end):
        for j in range(linker_start, linker_end):
            if i != j:
                edge_list.append([i, j])
                edge_types.append(BOND_TO_IDX["NONE"])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        bond_type_gt = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        bond_type_gt = torch.zeros(0, dtype=torch.long)
    
    # Compute anchor indices in new ordering
    anchorA_new = idx_map.get(anchors['anchorA_idx'], 0)
    anchorB_new = idx_map.get(anchors['anchorB_idx'], N_fragA)
    anchors_tensor = torch.tensor([anchorA_new, anchorB_new], dtype=torch.long)
    
    # Compute pose_gt (relative pose of fragB w.r.t. fragA)
    fragA_coords = x_data[:N_fragA]
    fragB_coords = x_data[N_fragA:N_fragA + N_fragB]
    
    fragA_center = fragA_coords.mean(dim=0)
    fragB_center = fragB_coords.mean(dim=0)
    
    p_gt = fragB_center - fragA_center
    r_gt = torch.zeros(3)  # Identity rotation in axis-angle
    
    # Condition vector (default: zeros)
    if cond is None:
        cond = torch.zeros(16, dtype=torch.float32)
    
    sample_dict = {
        'x_data': x_data,
        'atom_type': atom_type,
        'node_kind': node_kind,
        'is_anchor': is_anchor,
        'fragB_mask': fragB_mask,
        'linker_mask': linker_mask,
        'edge_index': edge_index,
        'bond_type_gt': bond_type_gt,
        'anchors': anchors_tensor,
        'cond': cond,
        'p_gt': p_gt,
        'r_gt': r_gt,
        'N_fragA': N_fragA,
        'N_fragB': N_fragB,
        'N_linker': N_linker,
        'N_linker_padded': N_linker_padded,
    }
    
    return sample_dict


def compute_edge_attr(x: torch.Tensor, edge_index: torch.Tensor, 
                      node_kind: torch.Tensor) -> torch.Tensor:
    """Compute edge attributes.
    
    Args:
        x: (N, 3) node coordinates
        edge_index: (2, E) edge indices
        node_kind: (N,) node types
        
    Returns:
        edge_attr: (E, edge_dim) edge features
    """
    src, dst = edge_index
    
    # Distance
    dist = (x[src] - x[dst]).norm(dim=-1, keepdim=True)
    
    # Same fragment indicator
    same_frag = (node_kind[src] == node_kind[dst]).float().unsqueeze(-1)
    
    # Node kind features
    src_kind = torch.nn.functional.one_hot(node_kind[src], num_classes=3).float()
    dst_kind = torch.nn.functional.one_hot(node_kind[dst], num_classes=3).float()
    
    edge_attr = torch.cat([dist, same_frag, src_kind, dst_kind], dim=-1)
    
    return edge_attr
