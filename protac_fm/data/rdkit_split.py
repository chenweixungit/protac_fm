"""Split PROTAC molecules into fragments and linker."""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from typing import Tuple, List, Optional, Dict


def split_protac_into_fragments(full_mol: Chem.Mol) -> Tuple[Chem.Mol, Chem.Mol, Chem.Mol, Dict]:
    """Split a PROTAC molecule into fragA (warhead), fragB (E3 ligase), and linker.
    
    Args:
        full_mol: RDKit Mol object of the complete PROTAC with 3D conformer
        
    Returns:
        fragA_mol: Fragment A (POI ligand/warhead)
        fragB_mol: Fragment B (E3 ligand)
        linker_mol: Linker molecule
        anchors: Dict containing anchor information
            - anchorA_idx: anchor atom index in full_mol for fragA
            - anchorB_idx: anchor atom index in full_mol for fragB
            - linker_atom_indices: list of linker atom indices in full_mol
    """
    # Find ring systems
    ring_info = full_mol.GetRingInfo()
    ring_atoms = set()
    for ring in ring_info.AtomRings():
        ring_atoms.update(ring)
    
    # Find the two largest connected ring systems (likely warhead and E3 ligase)
    ring_systems = find_ring_systems(full_mol)
    
    if len(ring_systems) < 2:
        # Fallback: use simple heuristic based on molecular structure
        return split_by_heuristic(full_mol)
    
    # Sort ring systems by size (number of atoms)
    ring_systems = sorted(ring_systems, key=lambda x: len(x), reverse=True)
    
    # Take the two largest ring systems as fragments
    frag1_atoms = set(ring_systems[0])
    frag2_atoms = set(ring_systems[1])
    
    # Expand fragments to include attached non-ring atoms (except linker)
    frag1_atoms = expand_fragment(full_mol, frag1_atoms, frag2_atoms)
    frag2_atoms = expand_fragment(full_mol, frag2_atoms, frag1_atoms)
    
    # Remaining atoms are linker
    all_atoms = set(range(full_mol.GetNumAtoms()))
    linker_atoms = all_atoms - frag1_atoms - frag2_atoms
    
    # Find anchor atoms (atoms in fragments connected to linker)
    anchor1_idx, anchor2_idx = find_anchors(full_mol, frag1_atoms, frag2_atoms, linker_atoms)
    
    # Create fragment molecules
    fragA_mol = extract_fragment(full_mol, list(frag1_atoms))
    fragB_mol = extract_fragment(full_mol, list(frag2_atoms))
    linker_mol = extract_fragment(full_mol, list(linker_atoms))
    
    anchors = {
        'anchorA_idx': anchor1_idx,
        'anchorB_idx': anchor2_idx,
        'linker_atom_indices': list(linker_atoms),
        'fragA_atom_indices': list(frag1_atoms),
        'fragB_atom_indices': list(frag2_atoms),
    }
    
    return fragA_mol, fragB_mol, linker_mol, anchors


def find_ring_systems(mol: Chem.Mol) -> List[List[int]]:
    """Find connected ring systems in a molecule.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        List of ring systems, each is a list of atom indices
    """
    ring_info = mol.GetRingInfo()
    rings = list(ring_info.AtomRings())
    
    if not rings:
        return []
    
    # Build adjacency between rings (share atoms)
    n_rings = len(rings)
    ring_adj = [[False] * n_rings for _ in range(n_rings)]
    
    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            if set(rings[i]) & set(rings[j]):  # Shared atoms
                ring_adj[i][j] = True
                ring_adj[j][i] = True
    
    # Find connected components
    visited = [False] * n_rings
    ring_systems = []
    
    for start in range(n_rings):
        if visited[start]:
            continue
        
        # BFS
        system_atoms = set()
        queue = [start]
        visited[start] = True
        
        while queue:
            curr = queue.pop(0)
            system_atoms.update(rings[curr])
            
            for next_ring in range(n_rings):
                if ring_adj[curr][next_ring] and not visited[next_ring]:
                    visited[next_ring] = True
                    queue.append(next_ring)
        
        ring_systems.append(list(system_atoms))
    
    return ring_systems


def expand_fragment(mol: Chem.Mol, frag_atoms: set, other_frag_atoms: set) -> set:
    """Expand fragment to include attached non-ring atoms.
    
    Args:
        mol: RDKit Mol object
        frag_atoms: Current fragment atom indices
        other_frag_atoms: Other fragment atom indices (to avoid)
        
    Returns:
        Expanded fragment atom indices
    """
    expanded = set(frag_atoms)
    changed = True
    
    while changed:
        changed = False
        for atom_idx in list(expanded):
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                n_idx = neighbor.GetIdx()
                if n_idx not in expanded and n_idx not in other_frag_atoms:
                    # Check if this is a terminal group or small chain
                    if is_terminal_or_small_chain(mol, n_idx, expanded, other_frag_atoms):
                        expanded.add(n_idx)
                        changed = True
    
    return expanded


def is_terminal_or_small_chain(mol: Chem.Mol, atom_idx: int, frag_atoms: set, other_frag_atoms: set) -> bool:
    """Check if atom is terminal or part of a small chain attached to fragment.
    
    Args:
        mol: RDKit Mol object
        atom_idx: Atom index to check
        frag_atoms: Current fragment atoms
        other_frag_atoms: Other fragment atoms
        
    Returns:
        True if atom should be included in fragment
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # Terminal atoms (only one neighbor)
    if atom.GetDegree() == 1:
        return True
    
    # Check if atom is in a ring
    if atom.IsInRing():
        return False
    
    # Check neighbors
    neighbors_in_frag = sum(1 for n in atom.GetNeighbors() if n.GetIdx() in frag_atoms)
    neighbors_in_other = sum(1 for n in atom.GetNeighbors() if n.GetIdx() in other_frag_atoms)
    
    # If connected to other fragment, this is likely linker
    if neighbors_in_other > 0:
        return False
    
    # If mostly connected to current fragment, include it
    if neighbors_in_frag >= atom.GetDegree() - 1:
        return True
    
    return False


def find_anchors(mol: Chem.Mol, frag1_atoms: set, frag2_atoms: set, linker_atoms: set) -> Tuple[int, int]:
    """Find anchor atoms connecting fragments to linker.
    
    Args:
        mol: RDKit Mol object
        frag1_atoms: Fragment 1 atom indices
        frag2_atoms: Fragment 2 atom indices
        linker_atoms: Linker atom indices
        
    Returns:
        anchor1_idx: Anchor atom index in fragment 1
        anchor2_idx: Anchor atom index in fragment 2
    """
    anchor1_idx = None
    anchor2_idx = None
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # Check frag1-linker connection
        if begin_idx in frag1_atoms and end_idx in linker_atoms:
            anchor1_idx = begin_idx
        elif end_idx in frag1_atoms and begin_idx in linker_atoms:
            anchor1_idx = end_idx
        
        # Check frag2-linker connection
        if begin_idx in frag2_atoms and end_idx in linker_atoms:
            anchor2_idx = begin_idx
        elif end_idx in frag2_atoms and begin_idx in linker_atoms:
            anchor2_idx = end_idx
    
    return anchor1_idx, anchor2_idx


def extract_fragment(mol: Chem.Mol, atom_indices: List[int]) -> Chem.Mol:
    """Extract a fragment from a molecule.
    
    Args:
        mol: RDKit Mol object
        atom_indices: Atom indices to extract
        
    Returns:
        Fragment molecule with 3D coordinates
    """
    if not atom_indices:
        return None
    
    # Create editable molecule
    emol = Chem.RWMol(mol)
    
    # Get atoms to remove (in reverse order to maintain indices)
    atoms_to_remove = [i for i in range(mol.GetNumAtoms()) if i not in atom_indices]
    atoms_to_remove.sort(reverse=True)
    
    for atom_idx in atoms_to_remove:
        emol.RemoveAtom(atom_idx)
    
    frag_mol = emol.GetMol()
    
    try:
        Chem.SanitizeMol(frag_mol)
    except:
        pass  # Some fragments may not sanitize properly
    
    return frag_mol


def split_by_heuristic(full_mol: Chem.Mol) -> Tuple[Chem.Mol, Chem.Mol, Chem.Mol, Dict]:
    """Fallback splitting using simple heuristics.
    
    Assumes linker is the longest chain connecting two ring systems.
    
    Args:
        full_mol: RDKit Mol object
        
    Returns:
        Same as split_protac_into_fragments
    """
    # Find all atoms not in rings
    non_ring_atoms = []
    for atom in full_mol.GetAtoms():
        if not atom.IsInRing():
            non_ring_atoms.append(atom.GetIdx())
    
    # Find the longest path through non-ring atoms
    # This is a simplified heuristic
    
    # For now, return None if heuristic fails
    # In practice, you would implement a more sophisticated algorithm
    
    return None, None, None, {}


def get_fragment_info(mol: Chem.Mol, atom_indices: List[int]) -> Dict:
    """Get information about a fragment.
    
    Args:
        mol: Full molecule
        atom_indices: Atom indices of the fragment
        
    Returns:
        Dict with fragment information
    """
    if not atom_indices:
        return {}
    
    conf = mol.GetConformer()
    
    coords = []
    elements = []
    
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
        elements.append(atom.GetAtomicNum())
    
    return {
        'coords': np.array(coords, dtype=np.float32),
        'elements': np.array(elements, dtype=np.int32),
        'atom_indices': atom_indices,
    }
