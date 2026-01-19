"""Post-generation filtering for PROTAC molecules."""
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple, Optional
import numpy as np

from .properties import compute_properties, compute_linker_properties
from .constraints import validate_molecule


class MoleculeFilter:
    """Filter generated molecules based on property criteria."""
    
    def __init__(
        self,
        MW_min: float = 600,
        MW_max: float = 1200,
        logP_max: float = 6.0,
        TPSA_max: float = 180,
        HBD_max: int = 4,
        HBA_max: int = 12,
        RotB_min: int = 6,
        RotB_max: int = 20,
        QED_min: float = 0.2,
        SA_max: float = 6.0,
        linker_length_min: int = 3,
        linker_length_max: int = 30,
    ):
        """Initialize filter with criteria.
        
        Args:
            MW_min: Minimum molecular weight
            MW_max: Maximum molecular weight
            logP_max: Maximum logP
            TPSA_max: Maximum TPSA
            HBD_max: Maximum H-bond donors
            HBA_max: Maximum H-bond acceptors
            RotB_min: Minimum rotatable bonds
            RotB_max: Maximum rotatable bonds
            QED_min: Minimum QED score
            SA_max: Maximum SA score
            linker_length_min: Minimum linker atoms
            linker_length_max: Maximum linker atoms
        """
        self.criteria = {
            'MW_min': MW_min,
            'MW_max': MW_max,
            'logP_max': logP_max,
            'TPSA_max': TPSA_max,
            'HBD_max': HBD_max,
            'HBA_max': HBA_max,
            'RotB_min': RotB_min,
            'RotB_max': RotB_max,
            'QED_min': QED_min,
            'SA_max': SA_max,
            'linker_length_min': linker_length_min,
            'linker_length_max': linker_length_max,
        }
    
    def check_criteria(self, props: Dict[str, float]) -> Dict[str, bool]:
        """Check if properties meet all criteria.
        
        Args:
            props: Property dict from compute_properties
            
        Returns:
            Dict with pass/fail for each criterion
        """
        results = {}
        
        # MW range
        mw = props.get('MW', 0)
        results['MW'] = self.criteria['MW_min'] <= mw <= self.criteria['MW_max']
        
        # logP
        results['logP'] = props.get('logP', 10) <= self.criteria['logP_max']
        
        # TPSA
        results['TPSA'] = props.get('TPSA', 300) <= self.criteria['TPSA_max']
        
        # HBD
        results['HBD'] = props.get('HBD', 10) <= self.criteria['HBD_max']
        
        # HBA
        results['HBA'] = props.get('HBA', 20) <= self.criteria['HBA_max']
        
        # RotB range
        rotb = props.get('RotB', 0)
        results['RotB'] = self.criteria['RotB_min'] <= rotb <= self.criteria['RotB_max']
        
        # QED
        results['QED'] = props.get('QED', 0) >= self.criteria['QED_min']
        
        # SA
        results['SA'] = props.get('SA', 10) <= self.criteria['SA_max']
        
        # Linker length
        linker_len = props.get('linker_length', 0)
        results['linker_length'] = (
            self.criteria['linker_length_min'] <= linker_len <= self.criteria['linker_length_max']
        )
        
        return results
    
    def filter_single(
        self,
        mol: Chem.Mol,
        linker_atom_indices: Optional[List[int]] = None,
    ) -> Tuple[bool, Dict[str, float], Dict[str, bool]]:
        """Filter a single molecule.
        
        Args:
            mol: RDKit Mol object
            linker_atom_indices: Optional linker atom indices
            
        Returns:
            passed: True if molecule passes all filters
            props: Property dict
            criteria_results: Dict with pass/fail for each criterion
        """
        # Compute properties
        props = compute_properties(mol)
        
        if linker_atom_indices:
            linker_props = compute_linker_properties(mol, linker_atom_indices)
            props.update(linker_props)
        
        # Check criteria
        criteria_results = self.check_criteria(props)
        
        # Overall pass
        passed = all(criteria_results.values())
        
        return passed, props, criteria_results


def filter_molecules(
    molecules: List[Dict],
    filter_config: Dict,
) -> List[Dict]:
    """Filter a list of generated molecules.
    
    Args:
        molecules: List of molecule dicts with 'mol' and optionally 'linker_indices'
        filter_config: Filter configuration dict
        
    Returns:
        List of molecules that pass all filters, with properties added
    """
    mol_filter = MoleculeFilter(**filter_config)
    
    passed_molecules = []
    
    for mol_dict in molecules:
        mol = mol_dict.get('mol')
        if mol is None:
            continue
        
        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            mol_dict['sanitize_error'] = True
            continue
        
        linker_indices = mol_dict.get('linker_indices')
        passed, props, criteria = mol_filter.filter_single(mol, linker_indices)
        
        mol_dict['properties'] = props
        mol_dict['criteria'] = criteria
        mol_dict['passed_filter'] = passed
        
        if passed:
            passed_molecules.append(mol_dict)
    
    return passed_molecules


def rank_molecules(
    molecules: List[Dict],
    weights: Dict[str, float] = None,
) -> List[Dict]:
    """Rank filtered molecules by a weighted score.
    
    Args:
        molecules: List of molecule dicts with properties
        weights: Weights for each property (higher is better for score)
        
    Returns:
        Sorted list of molecules (best first)
    """
    if weights is None:
        weights = {
            'QED': 1.0,
            'SA': -0.5,  # Lower SA is better
            'logP': -0.1,  # Prefer lower logP
        }
    
    for mol_dict in molecules:
        props = mol_dict.get('properties', {})
        
        score = 0.0
        for prop, weight in weights.items():
            if prop in props:
                score += weight * props[prop]
        
        mol_dict['rank_score'] = score
    
    # Sort by score (descending)
    molecules.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
    
    return molecules


def reconstruct_molecule(
    fragA_mol: Chem.Mol,
    fragB_mol: Chem.Mol,
    linker_coords: np.ndarray,
    linker_atom_types: List[int],
    linker_bonds: List[Tuple[int, int, int]],
    anchor_A_idx: int,
    anchor_B_idx: int,
) -> Optional[Chem.Mol]:
    """Reconstruct full molecule from fragments and generated linker.
    
    Args:
        fragA_mol: Fragment A molecule
        fragB_mol: Fragment B molecule
        linker_coords: (N_linker, 3) linker coordinates
        linker_atom_types: List of atom type indices for linker
        linker_bonds: List of (atom1, atom2, bond_type) for linker internal bonds
        anchor_A_idx: Anchor atom index in fragA
        anchor_B_idx: Anchor atom index in fragB
        
    Returns:
        Reconstructed molecule or None if failed
    """
    # Atom type index to element symbol
    ATOM_IDX_TO_SYMBOL = {
        0: None,  # MASK
        1: 'C',
        2: 'N',
        3: 'O',
        4: 'S',
        5: 'F',
        6: 'Cl',
        7: 'Br',
        8: 'I',
        9: 'P',
    }
    
    # Bond type index to RDKit bond type
    BOND_IDX_TO_TYPE = {
        0: None,  # NONE
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.AROMATIC,
    }
    
    try:
        # Create editable molecule
        combo = Chem.RWMol(Chem.CombineMols(fragA_mol, fragB_mol))
        
        # Offset for fragB atoms
        fragA_num_atoms = fragA_mol.GetNumAtoms()
        fragB_anchor_new = fragA_num_atoms + anchor_B_idx
        
        # Add linker atoms
        linker_start_idx = combo.GetNumAtoms()
        linker_idx_map = {}
        
        for i, atom_type in enumerate(linker_atom_types):
            if atom_type == 0:  # MASK - skip
                continue
            
            symbol = ATOM_IDX_TO_SYMBOL.get(atom_type)
            if symbol is None:
                continue
            
            atom = Chem.Atom(symbol)
            new_idx = combo.AddAtom(atom)
            linker_idx_map[i] = new_idx
        
        # Add linker internal bonds
        for atom1, atom2, bond_type in linker_bonds:
            if atom1 not in linker_idx_map or atom2 not in linker_idx_map:
                continue
            
            rdkit_bond = BOND_IDX_TO_TYPE.get(bond_type)
            if rdkit_bond is None:
                continue
            
            combo.AddBond(linker_idx_map[atom1], linker_idx_map[atom2], rdkit_bond)
        
        # Connect linker to anchors
        # Find first and last linker atoms
        if linker_idx_map:
            linker_indices = sorted(linker_idx_map.keys())
            first_linker = linker_idx_map[linker_indices[0]]
            last_linker = linker_idx_map[linker_indices[-1]]
            
            # Connect to anchors
            combo.AddBond(anchor_A_idx, first_linker, Chem.BondType.SINGLE)
            combo.AddBond(fragB_anchor_new, last_linker, Chem.BondType.SINGLE)
        
        # Get molecule
        mol = combo.GetMol()
        
        # Add 3D coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        
        # Copy fragment coordinates
        if fragA_mol.GetNumConformers() > 0:
            fragA_conf = fragA_mol.GetConformer()
            for i in range(fragA_mol.GetNumAtoms()):
                pos = fragA_conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, pos)
        
        if fragB_mol.GetNumConformers() > 0:
            fragB_conf = fragB_mol.GetConformer()
            for i in range(fragB_mol.GetNumAtoms()):
                pos = fragB_conf.GetAtomPosition(i)
                conf.SetAtomPosition(fragA_num_atoms + i, pos)
        
        # Set linker coordinates
        for orig_idx, new_idx in linker_idx_map.items():
            if orig_idx < len(linker_coords):
                pos = linker_coords[orig_idx]
                conf.SetAtomPosition(new_idx, Chem.rdGeometry.Point3D(*pos))
        
        mol.AddConformer(conf, assignId=True)
        
        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        
        return mol
        
    except Exception as e:
        print(f"Error reconstructing molecule: {e}")
        return None
