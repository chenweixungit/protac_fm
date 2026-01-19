"""Chemical constraints for PROTAC generation."""
import torch
from rdkit import Chem
from typing import Dict, List, Tuple, Optional


# Maximum valence for each element
MAX_VALENCE = {
    6: 4,   # C
    7: 3,   # N
    8: 2,   # O
    16: 6,  # S
    9: 1,   # F
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
    15: 5,  # P
}

# Atom type index to element mapping
ATOM_IDX_TO_ELEMENT = {
    0: None,  # MASK
    1: 6,     # C
    2: 7,     # N
    3: 8,     # O
    4: 16,    # S
    5: 9,     # F
    6: 17,    # Cl
    7: 35,    # Br
    8: 53,    # I
    9: 15,    # P
}


def check_valence(
    atom_types: torch.Tensor,
    bond_types: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[bool, List[int]]:
    """Check if all atoms satisfy valence constraints.
    
    Args:
        atom_types: (N,) atom type indices
        bond_types: (E,) bond type indices (0=NONE, 1=SINGLE, 2=DOUBLE, 3=AROMATIC)
        edge_index: (2, E) edge indices
        
    Returns:
        valid: True if all atoms satisfy valence
        violations: List of atom indices with valence violations
    """
    N = atom_types.size(0)
    
    # Bond order mapping
    bond_order = {0: 0, 1: 1, 2: 2, 3: 1.5}
    
    # Compute valence for each atom
    valence = torch.zeros(N)
    src, dst = edge_index
    
    for i in range(edge_index.size(1)):
        order = bond_order.get(bond_types[i].item(), 0)
        valence[dst[i]] += order
    
    # Check violations
    violations = []
    for i in range(N):
        atom_idx = atom_types[i].item()
        if atom_idx == 0:  # MASK
            continue
        
        element = ATOM_IDX_TO_ELEMENT.get(atom_idx)
        if element is None:
            continue
        
        max_val = MAX_VALENCE.get(element, 4)
        if valence[i] > max_val:
            violations.append(i)
    
    return len(violations) == 0, violations


def check_anchor_connectivity(
    bond_types: torch.Tensor,
    edge_index: torch.Tensor,
    anchor_indices: torch.Tensor,
    linker_mask: torch.Tensor,
) -> Tuple[bool, Dict[str, int]]:
    """Check if anchors are properly connected to linker.
    
    Each anchor should have exactly one bond to linker (counting undirected edges once).
    
    Args:
        bond_types: (E,) bond type indices
        edge_index: (2, E) edge indices
        anchor_indices: (2,) anchor atom indices [anchorA, anchorB]
        linker_mask: (N,) linker node mask
        
    Returns:
        valid: True if anchors are properly connected
        info: Dict with connection info
    """
    src, dst = edge_index
    
    anchor_connections = {anchor_indices[0].item(): set(), anchor_indices[1].item(): set()}
    
    for i in range(edge_index.size(1)):
        if bond_types[i] == 0:  # NONE bond
            continue
        
        s, d = src[i].item(), dst[i].item()
        
        # Check if edge connects anchor to linker (only count once per pair)
        if s in anchor_connections and linker_mask[d]:
            anchor_connections[s].add(d)
        if d in anchor_connections and linker_mask[s]:
            anchor_connections[d].add(s)
    
    # Convert sets to counts
    anchor_counts = {k: len(v) for k, v in anchor_connections.items()}
    
    # Each anchor should have exactly 1 connection to linker
    valid = all(count == 1 for count in anchor_counts.values())
    
    return valid, anchor_counts


def check_connectivity(
    bond_types: torch.Tensor,
    edge_index: torch.Tensor,
    linker_mask: torch.Tensor,
    anchor_indices: torch.Tensor,
) -> Tuple[bool, int]:
    """Check if linker forms a connected component with anchors.
    
    Args:
        bond_types: (E,) bond type indices
        edge_index: (2, E) edge indices
        linker_mask: (N,) linker node mask
        anchor_indices: (2,) anchor atom indices
        
    Returns:
        valid: True if linker is connected
        num_components: Number of connected components
    """
    N = linker_mask.size(0)
    
    # Build adjacency list for non-NONE bonds
    adj = {i: [] for i in range(N)}
    src, dst = edge_index
    
    for i in range(edge_index.size(1)):
        if bond_types[i] == 0:  # NONE bond
            continue
        s, d = src[i].item(), dst[i].item()
        adj[s].append(d)
        adj[d].append(s)
    
    # Get linker nodes + anchors
    relevant_nodes = set(linker_mask.nonzero(as_tuple=True)[0].tolist())
    relevant_nodes.add(anchor_indices[0].item())
    relevant_nodes.add(anchor_indices[1].item())
    
    # BFS to find connected components
    visited = set()
    num_components = 0
    
    for start in relevant_nodes:
        if start in visited:
            continue
        
        # BFS
        queue = [start]
        visited.add(start)
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor in relevant_nodes and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        num_components += 1
    
    # Should be exactly 1 connected component
    valid = (num_components == 1)
    
    return valid, num_components


def check_no_fragA_fragB_bond(
    bond_types: torch.Tensor,
    edge_index: torch.Tensor,
    node_kind: torch.Tensor,
) -> bool:
    """Check that fragA and fragB are not directly bonded.
    
    Args:
        bond_types: (E,) bond type indices
        edge_index: (2, E) edge indices
        node_kind: (N,) node kind (0=fragA, 1=fragB, 2=linker)
        
    Returns:
        valid: True if no direct fragA-fragB bond
    """
    src, dst = edge_index
    
    for i in range(edge_index.size(1)):
        if bond_types[i] == 0:  # NONE bond
            continue
        
        s_kind = node_kind[src[i]].item()
        d_kind = node_kind[dst[i]].item()
        
        # Check for fragA-fragB bond
        if (s_kind == 0 and d_kind == 1) or (s_kind == 1 and d_kind == 0):
            return False
    
    return True


def check_illegal_bonds(
    atom_types: torch.Tensor,
    bond_types: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[bool, List[int]]:
    """Check for illegal bond types (e.g., halogen double bonds).
    
    Args:
        atom_types: (N,) atom type indices
        bond_types: (E,) bond type indices
        edge_index: (2, E) edge indices
        
    Returns:
        valid: True if no illegal bonds
        violations: List of edge indices with violations
    """
    # Halogens (F, Cl, Br, I) should only have single bonds
    halogen_types = {5, 6, 7, 8}  # F, Cl, Br, I indices
    
    violations = []
    src, dst = edge_index
    
    for i in range(edge_index.size(1)):
        if bond_types[i] in [2, 3]:  # DOUBLE or AROMATIC
            s_type = atom_types[src[i]].item()
            d_type = atom_types[dst[i]].item()
            
            if s_type in halogen_types or d_type in halogen_types:
                violations.append(i)
    
    return len(violations) == 0, violations


def validate_molecule(
    atom_types: torch.Tensor,
    bond_types: torch.Tensor,
    edge_index: torch.Tensor,
    node_kind: torch.Tensor,
    linker_mask: torch.Tensor,
    anchor_indices: torch.Tensor,
) -> Dict[str, bool]:
    """Run all validation checks on a generated molecule.
    
    Args:
        atom_types: (N,) atom type indices
        bond_types: (E,) bond type indices
        edge_index: (2, E) edge indices
        node_kind: (N,) node kind
        linker_mask: (N,) linker mask
        anchor_indices: (2,) anchor indices
        
    Returns:
        Dict with validation results
    """
    valence_ok, _ = check_valence(atom_types, bond_types, edge_index)
    anchor_ok, _ = check_anchor_connectivity(bond_types, edge_index, anchor_indices, linker_mask)
    connected_ok, _ = check_connectivity(bond_types, edge_index, linker_mask, anchor_indices)
    no_direct_bond = check_no_fragA_fragB_bond(bond_types, edge_index, node_kind)
    no_illegal, _ = check_illegal_bonds(atom_types, bond_types, edge_index)
    
    return {
        'valence': valence_ok,
        'anchor_connectivity': anchor_ok,
        'linker_connectivity': connected_ok,
        'no_fragA_fragB_bond': no_direct_bond,
        'no_illegal_bonds': no_illegal,
        'all_valid': valence_ok and anchor_ok and connected_ok and no_direct_bond and no_illegal,
    }
