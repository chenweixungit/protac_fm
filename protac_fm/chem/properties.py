"""RDKit property calculations for PROTAC molecules."""
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, rdMolDescriptors
from typing import Dict, Optional
import numpy as np


def compute_sa_score(mol: Chem.Mol) -> float:
    """Compute synthetic accessibility score.
    
    Based on: Ertl & Schuffenhauer, J. Cheminform. 2009
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        SA score (1-10, lower is better)
    """
    try:
        from rdkit.Chem import RDConfig
        import os
        import pickle
        import gzip
        
        # Try to load fragment scores
        _fscores = None
        for path in [
            os.path.join(RDConfig.RDDataDir, 'fpscores.pkl.gz'),
            'fpscores.pkl.gz',
        ]:
            if os.path.exists(path):
                with gzip.open(path, 'rb') as f:
                    _fscores = pickle.load(f)
                break
        
        if _fscores is None:
            # Fallback: use simple heuristic
            return compute_sa_score_simple(mol)
        
        # Compute fragment score
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        
        score1 = 0.
        nf = 0
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            if sfp in _fscores:
                score1 += _fscores[sfp] * v
            else:
                score1 -= 4 * v
        
        score1 /= nf
        
        # Compute complexity penalty
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for ring in ri.AtomRings():
            if len(ring) > 8:
                nMacrocycles += 1
        
        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = np.log10(nChiralCenters + 1)
        spiroPenalty = np.log10(nSpiro + 1)
        bridgePenalty = np.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        if nMacrocycles > 0:
            macrocyclePenalty = np.log10(2)
        
        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
        
        # Combine scores
        sascore = score1 + score2
        
        # Normalize to 1-10
        min_score = -4.0
        max_score = 2.5
        sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.
        
        if sascore > 10.:
            sascore = 10.
        if sascore < 1.:
            sascore = 1.
        
        return sascore
        
    except Exception as e:
        return compute_sa_score_simple(mol)


def compute_sa_score_simple(mol: Chem.Mol) -> float:
    """Simple SA score heuristic based on molecular complexity.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        SA score estimate (1-10)
    """
    nAtoms = mol.GetNumAtoms()
    nRings = rdMolDescriptors.CalcNumRings(mol)
    nRotBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    nChiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    
    # Simple heuristic
    score = 1.0
    score += nAtoms * 0.05
    score += nRings * 0.3
    score += nRotBonds * 0.1
    score += nChiral * 0.5
    
    score = min(max(score, 1.0), 10.0)
    return score


def compute_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Compute all relevant properties for a molecule.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dict with property values
    """
    if mol is None:
        return {}
    
    try:
        props = {}
        
        # Basic properties
        props['MW'] = Descriptors.MolWt(mol)
        props['logP'] = Descriptors.MolLogP(mol)
        props['TPSA'] = Descriptors.TPSA(mol)
        props['HBD'] = rdMolDescriptors.CalcNumHBD(mol)
        props['HBA'] = rdMolDescriptors.CalcNumHBA(mol)
        props['RotB'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        # QED (drug-likeness)
        try:
            props['QED'] = QED.qed(mol)
        except:
            props['QED'] = 0.0
        
        # SA score
        props['SA'] = compute_sa_score(mol)
        
        # Additional properties
        props['NumAtoms'] = mol.GetNumAtoms()
        props['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
        props['NumRings'] = rdMolDescriptors.CalcNumRings(mol)
        props['NumAromaticRings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        props['FractionCSP3'] = rdMolDescriptors.CalcFractionCSP3(mol)
        
        return props
        
    except Exception as e:
        print(f"Error computing properties: {e}")
        return {}


def compute_linker_properties(
    full_mol: Chem.Mol,
    linker_atom_indices: list,
) -> Dict[str, float]:
    """Compute properties specific to the linker.
    
    Args:
        full_mol: Full PROTAC molecule
        linker_atom_indices: Atom indices of linker in full molecule
        
    Returns:
        Dict with linker-specific properties
    """
    props = {}
    
    # Linker length (number of atoms)
    props['linker_length'] = len(linker_atom_indices)
    
    # Count rotatable bonds in linker
    linker_set = set(linker_atom_indices)
    rot_bonds = 0
    
    for bond in full_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        if begin_idx in linker_set and end_idx in linker_set:
            if bond.GetBondType() == Chem.BondType.SINGLE:
                begin_atom = full_mol.GetAtomWithIdx(begin_idx)
                end_atom = full_mol.GetAtomWithIdx(end_idx)
                
                if (begin_atom.GetDegree() > 1 and end_atom.GetDegree() > 1 and
                    not bond.IsInRing()):
                    rot_bonds += 1
    
    props['linker_rotatable_bonds'] = rot_bonds
    
    # Count rings in linker
    ri = full_mol.GetRingInfo()
    linker_rings = 0
    for ring in ri.AtomRings():
        if all(idx in linker_set for idx in ring):
            linker_rings += 1
    
    props['linker_rings'] = linker_rings
    
    # Maximum branching (max degree of linker atoms)
    max_degree = 0
    for idx in linker_atom_indices:
        atom = full_mol.GetAtomWithIdx(idx)
        # Count neighbors within linker
        linker_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetIdx() in linker_set)
        max_degree = max(max_degree, linker_neighbors)
    
    props['linker_max_branching'] = max_degree
    
    return props


def check_lipinski(props: Dict[str, float]) -> Dict[str, bool]:
    """Check Lipinski's Rule of Five (modified for PROTACs).
    
    Note: PROTACs typically violate standard Lipinski rules.
    This uses relaxed criteria.
    
    Args:
        props: Property dict from compute_properties
        
    Returns:
        Dict with rule compliance
    """
    # Relaxed criteria for PROTACs
    return {
        'MW_ok': props.get('MW', 0) <= 1200,  # Relaxed from 500
        'logP_ok': props.get('logP', 0) <= 7,  # Relaxed from 5
        'HBD_ok': props.get('HBD', 0) <= 6,   # Relaxed from 5
        'HBA_ok': props.get('HBA', 0) <= 15,  # Relaxed from 10
    }


def check_veber(props: Dict[str, float]) -> Dict[str, bool]:
    """Check Veber's rules for oral bioavailability.
    
    Args:
        props: Property dict from compute_properties
        
    Returns:
        Dict with rule compliance
    """
    return {
        'RotB_ok': props.get('RotB', 0) <= 20,  # Relaxed from 10
        'TPSA_ok': props.get('TPSA', 0) <= 200,  # Relaxed from 140
    }
