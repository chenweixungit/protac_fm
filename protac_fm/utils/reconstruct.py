"""Molecule reconstruction utilities."""
import numpy as np
import torch
from rdkit import Chem, Geometry
from scipy.spatial.distance import cdist
from copy import deepcopy
import re
import itertools
from typing import Optional


# Bond length constants (in pm / 100 = Angstrom)
BONDS_1 = {
    'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92, 'Cl': 127, 'Br': 141, 'I': 161, 'S': 134},
    'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135, 'Cl': 177, 'Br': 194, 'I': 214, 'S': 182},
    'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136, 'Cl': 175, 'Br': 214, 'S': 168},
    'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142, 'Cl': 164, 'Br': 172, 'S': 151},
    'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142, 'Cl': 166, 'Br': 178, 'S': 158},
    'Cl': {'H': 127, 'C': 177, 'N': 175, 'O': 164, 'F': 166, 'Cl': 199, 'Br': 214, 'S': 207},
    'Br': {'H': 141, 'C': 194, 'N': 214, 'O': 172, 'F': 178, 'Cl': 214, 'Br': 228, 'S': 225},
    'I': {'H': 161, 'C': 214, 'N': 222, 'O': 194, 'F': 187, 'Cl': 243, 'Br': 247, 'I': 266, 'S': 234},
    'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'F': 158, 'Cl': 207, 'Br': 225, 'I': 234, 'S': 204},
}

BONDS_2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'S': 150},
    'S': {'C': 160, 'O': 150, 'S': 189},
}

BONDS_3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
}

BOND_DICT = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}

MARGINS_EDM = [10, 5, 3]


def get_bond_order(atom1, atom2, distance, check_exists=True, margins=MARGINS_EDM):
    """Determine bond order from distance."""
    distance = 100 * distance  # Convert to pm
    
    if check_exists:
        if atom1 not in BONDS_1:
            return 0
        if atom2 not in BONDS_1[atom1]:
            return 0
    
    if distance < BONDS_1[atom1][atom2] + margins[0]:
        if atom1 in BONDS_2 and atom2 in BONDS_2[atom1]:
            thr_bond2 = BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in BONDS_3 and atom2 in BONDS_3[atom1]:
                    thr_bond3 = BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def create_conformer(coords):
    """Create RDKit conformer from coordinates."""
    conformer = Chem.Conformer()
    for i, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(float(x), float(y), float(z)))
    return conformer


def build_linker_mol(linker_x, linker_c, add_bonds=False):
    """Build linker molecule from coordinates and elements."""
    mol = Chem.RWMol()
    for atom in linker_c:
        a = Chem.Atom(int(atom))
        mol.AddAtom(a)
    
    if add_bonds:
        n = len(linker_x)
        ptable = Chem.GetPeriodicTable()
        dists = cdist(linker_x, linker_x, 'euclidean')
        
        for i in range(n):
            for j in range(i):
                atom_i = ptable.GetElementSymbol(int(linker_c[i]))
                atom_j = ptable.GetElementSymbol(int(linker_c[j]))
                order = get_bond_order(atom_i, atom_j, dists[i, j])
                if order > 0:
                    mol.AddBond(i, j, BOND_DICT[order])
    
    return mol


def reconstruct_mol_with_bond(frag_mol, frag_x, linker_x, linker_c,
                              linker_bond_index, linker_bond_type,
                              known_linker_bonds=True, check_validity=True):
    """Reconstruct full molecule from fragments and linker with bond info."""
    # Build linker mol
    linker_mol = build_linker_mol(linker_x, linker_c, add_bonds=known_linker_bonds)
    
    # Combine mol and assign conformer
    mol = Chem.CombineMols(frag_mol, linker_mol)
    linker_atom_idx = list(range(mol.GetNumAtoms() - linker_mol.GetNumAtoms(), mol.GetNumAtoms()))
    mol = Chem.RWMol(mol)
    
    all_x = np.concatenate([frag_x, linker_x], axis=0)
    mol.RemoveAllConformers()
    mol.AddConformer(create_conformer(all_x))
    
    linker_bond_index = linker_bond_index.tolist() if torch.is_tensor(linker_bond_index) else linker_bond_index
    linker_bond_type = linker_bond_type.tolist() if torch.is_tensor(linker_bond_type) else linker_bond_type
    
    anchor_indices = set()
    
    # Add bonds
    for i, type_this in enumerate(linker_bond_type):
        node_i, node_j = linker_bond_index[0][i], linker_bond_index[1][i]
        if node_i < node_j:
            if type_this == 0:
                continue
            
            if node_i in linker_atom_idx and node_j not in linker_atom_idx:
                anchor_indices.add(int(node_j))
            elif node_j in linker_atom_idx and node_i not in linker_atom_idx:
                anchor_indices.add(int(node_i))
            
            if type_this == 1:
                mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 4:
                mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
    
    mol = mol.GetMol()
    for anchor_idx in anchor_indices:
        atom = mol.GetAtomWithIdx(anchor_idx)
        atom.SetNumExplicitHs(0)
    
    if check_validity:
        mol = fix_validity(mol, linker_atom_idx)
    
    return mol


def fix_validity(mol, linker_atom_idx):
    """Try to fix molecule validity issues."""
    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        pass
    
    # Try fixing aromatic issues
    try:
        Chem.Kekulize(deepcopy(mol))
    except Chem.rdchem.KekulizeException:
        mol, fixed = fix_aromatic(mol)
        if fixed:
            return mol
    
    # Try fixing valence issues
    mol, fixed = fix_valence(mol, linker_atom_idx)
    if fixed:
        return mol
    
    # Try more aggressive aromatic fix
    mol, fixed = fix_aromatic(mol, strict=True, linker_atom_idx=linker_atom_idx)
    
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    
    return mol


def fix_valence(mol, linker_atom_idx=None):
    """Fix valence errors by adding formal charges."""
    mol = deepcopy(mol)
    fixed = False
    cnt_loop = 0
    
    while True:
        try:
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except:
            return mol, False
        
        cnt_loop += 1
        if cnt_loop > 100:
            break
        
        N4_valence = re.compile(r"Explicit valence for atom # ([0-9]+) N, 4, is greater than permitted")
        index = N4_valence.findall(err.args[0])
        if len(index) > 0:
            if linker_atom_idx is None or int(index[0]) in linker_atom_idx:
                mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
    
    return mol, fixed


def get_ring_sys(mol):
    """Get ring systems."""
    all_rings = Chem.GetSymmSSSR(mol)
    if len(all_rings) == 0:
        return []
    
    ring_sys_list = [list(all_rings[0])]
    for ring in all_rings[1:]:
        form_prev = False
        for prev_ring in ring_sys_list:
            if set(ring).intersection(set(prev_ring)):
                prev_ring.extend(ring)
                form_prev = True
                break
        if not form_prev:
            ring_sys_list.append(list(ring))
    
    return [list(set(x)) for x in ring_sys_list]


def get_all_subsets(ring_list):
    """Get all subsets of a list."""
    all_sub_list = []
    for n_sub in range(len(ring_list) + 1):
        all_sub_list.extend(itertools.combinations(ring_list, n_sub))
    return all_sub_list


def fix_aromatic(mol, strict=False, linker_atom_idx=None):
    """Fix aromatic ring issues."""
    mol_orig = mol
    aromatic_list = [a.GetIdx() for a in mol.GetAromaticAtoms()]
    N_ring_list = []
    S_ring_list = []
    
    for ring_sys in get_ring_sys(mol):
        if set(ring_sys).intersection(set(aromatic_list)):
            idx_N = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'N']
            if len(idx_N) > 0:
                idx_N.append(-1)
                N_ring_list.append(idx_N)
            idx_S = [atom for atom in ring_sys if mol.GetAtomWithIdx(atom).GetSymbol() == 'S']
            if len(idx_S) > 0:
                idx_S.append(-1)
                S_ring_list.append(idx_S)
    
    fixed = False
    
    # Try S permutations
    if strict:
        S_ring_list = [s for ring in S_ring_list for s in ring if s != -1]
        permutation = get_all_subsets(S_ring_list)
    else:
        permutation = list(itertools.product(*S_ring_list)) if S_ring_list else [()]
    
    for perm in permutation:
        mol = deepcopy(mol_orig)
        perm = [x for x in perm if x != -1]
        for idx in perm:
            if linker_atom_idx is None or idx in linker_atom_idx:
                mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        try:
            if strict:
                mol, fixed = fix_valence(mol, linker_atom_idx)
            Chem.SanitizeMol(mol)
            fixed = True
            break
        except:
            continue
    
    # Try N permutations
    if not fixed:
        if strict:
            N_ring_list = [s for ring in N_ring_list for s in ring if s != -1]
            permutation = get_all_subsets(N_ring_list)
        else:
            permutation = list(itertools.product(*N_ring_list)) if N_ring_list else [()]
        
        for perm in permutation:
            perm = [x for x in perm if x != -1]
            actions = itertools.product([0, 1], repeat=len(perm)) if perm else [()]
            
            for action in actions:
                mol = deepcopy(mol_orig)
                for idx, act_atom in zip(perm, action):
                    if linker_atom_idx is None or idx in linker_atom_idx:
                        if act_atom == 0:
                            mol.GetAtomWithIdx(idx).SetNumExplicitHs(1)
                        else:
                            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                try:
                    if strict:
                        mol, fixed = fix_valence(mol, linker_atom_idx)
                    Chem.SanitizeMol(mol)
                    fixed = True
                    break
                except:
                    continue
            if fixed:
                break
    
    return mol, fixed


def parse_sampling_result_with_bond(data_list, final_x, final_c, final_bond, atom_featurizer,
                                    known_linker_bonds=True, check_validity=False,
                                    prune_cross_bonds: bool = False,
                                    max_cross_per_frag: int = 1,
                                    max_total_cross: Optional[int] = 2):
    """Parse sampling results to molecules."""
    all_mols = []
    if not isinstance(data_list, list):
        data_list = [data_list] * len(final_x)
    
    for data, x_gen, c_gen, b_gen in zip(data_list, final_x, final_c, final_bond):
        x_gen_t = x_gen.detach() if torch.is_tensor(x_gen) else torch.as_tensor(x_gen)
        frag_pos = x_gen[data.fragment_mask > 0].cpu().numpy().astype(np.float64)
        linker_pos = x_gen[data.fragment_mask == 0].cpu().numpy().astype(np.float64)
        linker_ele = [atom_featurizer.get_element_from_index(int(c)) for c in c_gen[data.linker_mask]]
        linker_bond_type = b_gen[data.linker_bond_mask]
        linker_bond_index = data.edge_index[:, data.linker_bond_mask]

        if prune_cross_bonds:
            frag_mask = data.fragment_mask
            src = linker_bond_index[0]
            dst = linker_bond_index[1]

            frag_mask_t = frag_mask.detach() if torch.is_tensor(frag_mask) else torch.as_tensor(frag_mask)
            src_t = src.detach() if torch.is_tensor(src) else torch.as_tensor(src)
            dst_t = dst.detach() if torch.is_tensor(dst) else torch.as_tensor(dst)
            bt_t = linker_bond_type.detach() if torch.is_tensor(linker_bond_type) else torch.as_tensor(linker_bond_type)

            x_cpu = x_gen_t.to("cpu")
            src_cpu = src_t.to("cpu").long()
            dst_cpu = dst_t.to("cpu").long()
            frag_cpu = frag_mask_t.to("cpu").long()
            bt_cpu = bt_t.to("cpu").long().clone()

            is_linker = frag_cpu == 0
            cross = (is_linker[src_cpu] & (frag_cpu[dst_cpu] > 0)) | (is_linker[dst_cpu] & (frag_cpu[src_cpu] > 0))
            active = (bt_cpu > 0) & cross

            if active.any():
                dist = (x_cpu[src_cpu] - x_cpu[dst_cpu]).pow(2).sum(dim=-1)

                keep = torch.zeros_like(active, dtype=torch.bool)
                for frag_id in (1, 2):
                    sel = active & ((frag_cpu[src_cpu] == frag_id) | (frag_cpu[dst_cpu] == frag_id))
                    if not sel.any():
                        continue
                    idx = sel.nonzero(as_tuple=True)[0]
                    k = int(max_cross_per_frag) if max_cross_per_frag is not None else 0
                    if k <= 0:
                        continue
                    order = torch.argsort(dist[idx])
                    idx = idx[order[:k]]
                    keep[idx] = True

                if max_total_cross is not None:
                    max_total = int(max_total_cross)
                    keep_idx = (keep & active).nonzero(as_tuple=True)[0]
                    if keep_idx.numel() > max_total:
                        order = torch.argsort(dist[keep_idx])
                        keep_idx = keep_idx[order[:max_total]]
                        keep = torch.zeros_like(active, dtype=torch.bool)
                        keep[keep_idx] = True

                bt_cpu[active & (~keep)] = 0

            linker_bond_type = bt_cpu.to(linker_bond_type.device) if torch.is_tensor(linker_bond_type) else bt_cpu
        
        full_mol = reconstruct_mol_with_bond(
            data.frag_mol, frag_pos, linker_pos, linker_ele,
            linker_bond_index, linker_bond_type,
            known_linker_bonds, check_validity
        )
        all_mols.append(full_mol)
    
    return all_mols


def eval_success_rate(gen_mols_list):
    """Evaluate reconstruction and completeness rate."""
    n_recon, n_complete = 0, 0
    for mol in gen_mols_list:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        n_recon += 1
        smiles = Chem.MolToSmiles(mol)
        if '.' not in smiles:
            n_complete += 1
    
    total = len(gen_mols_list)
    return n_recon / total if total > 0 else 0, n_complete / total if total > 0 else 0
