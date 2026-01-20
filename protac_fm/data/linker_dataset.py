"""LinkerNet-compatible dataset for PROTAC data."""
import os
import pickle
import random

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .featurize import ATOM_VOCAB, BOND_VOCAB


class FragLinkerData(Data):
    """Data class for fragment-linker data, compatible with PyG."""
    
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.pos.size(0)
        elif key == 'bond_index':
            return self.pos.size(0)
        elif key == 'anchor_indices':
            return self.pos.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def clone(self):
        """Create a copy of this data object."""
        new_data = FragLinkerData()
        for key in self.keys:
            value = self[key]
            if isinstance(value, torch.Tensor):
                setattr(new_data, key, value.clone())
            else:
                setattr(new_data, key, value)
        return new_data


def torchify_dict(data_dict: Dict) -> Dict:
    """Convert numpy arrays to torch tensors."""
    result = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            if value.dtype in [np.float32, np.float64]:
                result[key] = torch.from_numpy(value).float()
            elif value.dtype in [np.int32, np.int64]:
                result[key] = torch.from_numpy(value).long()
            else:
                result[key] = torch.from_numpy(value)
        else:
            result[key] = value
    return result


def process_from_mol(rdmol: Chem.Mol) -> Dict:
    """Process RDKit mol to get basic features."""
    conf = rdmol.GetConformer()
    
    # Get positions
    pos = []
    for i in range(rdmol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pos.append([p.x, p.y, p.z])
    pos = np.array(pos, dtype=np.float32)
    
    # Get atom features
    element = []
    valence = []
    charge = []
    for atom in rdmol.GetAtoms():
        element.append(atom.GetAtomicNum())
        valence.append(atom.GetTotalValence())
        charge.append(atom.GetFormalCharge())
    
    # Get bonds
    bond_index = []
    bond_type = []
    for bond in rdmol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_index.append([i, j])
        bond_index.append([j, i])
        
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            bond_type.extend([1, 1])
        elif bt == Chem.BondType.DOUBLE:
            bond_type.extend([2, 2])
        elif bt == Chem.BondType.TRIPLE:
            bond_type.extend([3, 3])
        elif bt == Chem.BondType.AROMATIC:
            bond_type.extend([4, 4])
        else:
            bond_type.extend([1, 1])
    
    if bond_index:
        bond_index = np.array(bond_index, dtype=np.int64).T
        bond_type = np.array(bond_type, dtype=np.int64)
    else:
        bond_index = np.zeros((2, 0), dtype=np.int64)
        bond_type = np.zeros(0, dtype=np.int64)
    
    return {
        'rdmol': rdmol,
        'pos': pos,
        'element': np.array(element, dtype=np.int64),
        'valence': np.array(valence, dtype=np.int64),
        'charge': np.array(charge, dtype=np.int64),
        'bond_index': bond_index,
        'bond_type': bond_type,
    }


def find_axes(pos: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """Find principal axes using PCA."""
    if weights is None:
        weights = np.ones(len(pos))
    
    center = np.average(pos, axis=0, weights=weights)
    pos_centered = pos - center
    
    cov = np.zeros((3, 3))
    for i in range(len(pos)):
        cov += weights[i] * np.outer(pos_centered[i], pos_centered[i])
    cov /= weights.sum()
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, idx]
    
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    
    return axes.astype(np.float32)


def rotation_matrix_from_vectors(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Find rotation matrix that aligns vec1 to vec2."""
    a = vec1 / vec1.norm().clamp(min=1e-8)
    b = vec2 / vec2.norm().clamp(min=1e-8)
    
    v = torch.cross(a, b)
    c = torch.dot(a, b)
    s = v.norm()
    
    if s < 1e-8:
        if c > 0:
            return torch.eye(3, device=vec1.device, dtype=vec1.dtype)
        else:
            ortho = torch.tensor([1., 0., 0.], device=vec1.device, dtype=vec1.dtype)
            if torch.abs(torch.dot(a, ortho)) > 0.9:
                ortho = torch.tensor([0., 1., 0.], device=vec1.device, dtype=vec1.dtype)
            ortho = ortho - torch.dot(ortho, a) * a
            ortho = ortho / ortho.norm()
            return 2 * torch.outer(ortho, ortho) - torch.eye(3, device=vec1.device, dtype=vec1.dtype)
    
    kmat = torch.zeros(3, 3, device=vec1.device, dtype=vec1.dtype)
    kmat[0, 1] = -v[2]
    kmat[0, 2] = v[1]
    kmat[1, 0] = v[2]
    kmat[1, 2] = -v[0]
    kmat[2, 0] = -v[1]
    kmat[2, 1] = v[0]
    
    R = torch.eye(3, device=vec1.device, dtype=vec1.dtype) + kmat + kmat @ kmat * ((1 - c) / (s * s))
    return R


def global_to_local(R: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Transform global coordinates to local."""
    return (x - t) @ R


def local_to_global(R: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Transform local coordinates to global."""
    return x @ R.T + t


class LinkerDataset(Dataset):
    """Dataset compatible with LinkerNet format."""
    
    def __init__(
        self,
        raw_path: str,
        dset_name: str = 'protac',
        version: str = 'v1',
        split_mode: str = 'warhead',
        index_name: str = '3d_index.pkl',
        transform=None,
        reset: bool = False
    ):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.dset_name = dset_name
        self.index_path = os.path.join(self.raw_path, index_name)
        self.processed_path = os.path.join(
            os.path.dirname(self.raw_path),
            os.path.basename(self.raw_path) + f'_{version}.lmdb'
        )
        self.split_mode = split_mode
        self.split_path = os.path.join(
            os.path.dirname(self.raw_path),
            os.path.basename(self.raw_path) + f'_{version}_{split_mode}_split.pt'
        )
        
        self.transform = transform
        self.db = None
        self.keys = None
        self.split = None
        
        if not os.path.exists(self.processed_path) or reset:
            print(f'{self.processed_path} does not exist, processing data...')
            self._process()
        print(f'Loading dataset from {self.processed_path}')
        
        if not os.path.exists(self.split_path):
            print(f'{self.split_path} does not exist, creating split...')
            self._split_dataset()
        print(f'Loading split from {self.split_path}')
        self.split = torch.load(self.split_path)
    
    def _connect_db(self):
        """Establish database connection."""
        if self.db is not None:
            return
        
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
    
    def _close_db(self):
        """Close database connection."""
        if self.db is not None:
            self.db.close()
            self.db = None
            self.keys = None
    
    @staticmethod
    def process_data(index: Dict) -> Optional[FragLinkerData]:
        """Process a single data entry."""
        rdmol = index['mol']
        ligand_dict = process_from_mol(rdmol)
        
        # Align centers of fragments to x-axis
        frags_t = []
        for ind_key in ['atom_indices_f1', 'atom_indices_f2']:
            f_pos = ligand_dict['pos'][index[ind_key]]
            frags_t.append(f_pos.mean(-2))
        
        rot = rotation_matrix_from_vectors(
            torch.from_numpy(frags_t[1] - frags_t[0]),
            torch.tensor([1., 0., 0.])
        ).numpy()
        tr = -rot @ ((frags_t[0] + frags_t[1]) / 2)
        ligand_dict['pos'] = (rot @ ligand_dict['pos'].T).T + tr
        
        # Compute rotation and translation for each fragment
        frags_axes = []
        frags_t = []
        for ind_key in ['atom_indices_f1', 'atom_indices_f2']:
            f_pos = ligand_dict['pos'][index[ind_key]]
            f_charge = np.array(
                [rdmol.GetAtomWithIdx(i).GetAtomicNum() for i in index[ind_key]],
                dtype=np.float32
            )
            axes = find_axes(f_pos, f_charge)
            frags_axes.append(axes)
            frags_t.append(f_pos.mean(-2))
        
        ligand_dict['frags_R'] = np.stack(frags_axes)
        ligand_dict['frags_t'] = np.stack(frags_t)
        
        ligand_dict.update(index)
        data = FragLinkerData(**torchify_dict(ligand_dict))
        
        # Compute local positions
        f1_local_pos = global_to_local(
            data.frags_R[0], data.frags_t[0],
            data.pos[data.fragment_mask == 1]
        )
        f2_local_pos = global_to_local(
            data.frags_R[1], data.frags_t[1],
            data.pos[data.fragment_mask == 2]
        )
        data.frags_local_pos = torch.cat([f1_local_pos, f2_local_pos])
        data.frags_d = data.frags_t[1][0] - data.frags_t[0][0]
        
        # Find anchors
        link_indices = (data.linker_mask == 1).nonzero()[:, 0].tolist()
        frag_indices = (data.linker_mask == 0).nonzero()[:, 0].tolist()
        anchor_indices = [
            j for i, j in zip(*data.bond_index.tolist())
            if i in link_indices and j in frag_indices
        ]
        data.anchor_indices = anchor_indices
        anchor_mask = torch.zeros_like(data.fragment_mask)
        anchor_mask[anchor_indices] = 1
        data.anchor_mask = anchor_mask
        
        # Build neighbor list
        data.nbh_list = {
            i.item(): [
                j.item() for k, j in enumerate(data.bond_index[1])
                if data.bond_index[0, k].item() == i
            ]
            for i in data.bond_index[0]
        }
        
        return data
    
    def _process(self):
        """Process raw data and save to LMDB."""
        with open(self.index_path, 'rb') as f:
            all_index = pickle.load(f)
        
        data_list = []
        num_skipped = 0
        
        for idx, index in enumerate(tqdm(all_index, desc='Processing')):
            new_index = preprocess_protac_index(index)
            if new_index is None:
                num_skipped += 1
                continue
            
            try:
                data = self.process_data(new_index)
                if data is not None:
                    data_list.append(data)
                else:
                    num_skipped += 1
            except Exception as e:
                print(f'Error processing {idx}: {e}')
                num_skipped += 1
        
        # Save to LMDB
        db = lmdb.open(
            self.processed_path,
            map_size=2 * (1024 * 1024 * 1024),
            create=True,
            subdir=False,
            readonly=False,
        )
        
        with db.begin(write=True, buffers=True) as txn:
            for idx, data in enumerate(tqdm(data_list, desc='Writing to LMDB')):
                txn.put(
                    key=f'{idx:08d}'.encode(),
                    value=pickle.dumps(data)
                )
        db.close()
        print(f'Processed {len(data_list)} samples, skipped {num_skipped}')
    
    def _split_dataset(self):
        """Create train/val/test split."""
        if self.db is None:
            self._connect_db()
        
        warheads, ligases, linkers = [], [], []
        for idx in range(len(self.keys)):
            key = self.keys[idx]
            data = pickle.loads(self.db.begin().get(key))
            warheads.append(getattr(data, 'smi_warhead', ''))
            ligases.append(getattr(data, 'smi_ligase', ''))
            linkers.append(getattr(data, 'smi_linker', ''))
        
        split = {'train': [], 'val': [], 'test': []}
        
        if self.split_mode == 'warhead':
            unique_warheads = list(set(warheads))
            test_warheads = unique_warheads[-10:] if len(unique_warheads) >= 10 else unique_warheads[-2:]
            for idx in range(len(self.keys)):
                if warheads[idx] in test_warheads:
                    split['test'].append(idx)
                else:
                    split['train'].append(idx)
        elif self.split_mode == 'ligase':
            unique_ligases = list(set(ligases))
            test_ligases = unique_ligases[-5:] if len(unique_ligases) >= 5 else unique_ligases[-1:]
            for idx in range(len(self.keys)):
                if ligases[idx] in test_ligases:
                    split['test'].append(idx)
                else:
                    split['train'].append(idx)
        elif self.split_mode == 'random':
            all_ids = list(range(len(self.keys)))
            random.Random(2023).shuffle(all_ids)
            n_test = min(100, len(all_ids) // 10)
            split['train'] = all_ids[:-n_test]
            split['test'] = all_ids[-n_test:]
        
        split['val'] = split['test']
        torch.save(split, self.split_path)
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data


def preprocess_protac_index(index: Dict) -> Optional[Dict]:
    """Preprocess PROTAC index entry."""
    rdmol = index['mol']
    
    # Renumber atoms: warhead + ligase + linker
    new_mol = Chem.RenumberAtoms(
        rdmol,
        list(index['atom_indices_warhead']) +
        list(index['atom_indices_ligase']) +
        list(index['atom_indices_linker'])
    )
    
    num_atoms_f1 = len(index['atom_indices_warhead'])
    num_atoms_f2 = len(index['atom_indices_ligase'])
    atom_indices_f1 = list(range(num_atoms_f1))
    atom_indices_f2 = list(range(num_atoms_f1, num_atoms_f1 + num_atoms_f2))
    
    fragment_mask = torch.zeros(rdmol.GetNumAtoms()).long()
    fragment_mask[atom_indices_f1] = 1
    fragment_mask[atom_indices_f2] = 2
    linker_mask = (fragment_mask == 0)
    
    # Extract fragment and linker mols
    bond_ids = []
    for bond_idx, bond in enumerate(new_mol.GetBonds()):
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (fragment_mask[start] > 0) != (fragment_mask[end] > 0):
            bond_ids.append(bond_idx)
    
    try:
        break_mol = Chem.FragmentOnBonds(new_mol, bond_ids, addDummies=False)
        submols = Chem.GetMolFrags(break_mol, asMols=True)
        if len(submols) != 3:
            return None
        link_mol = submols[-1]
        frag_mol = Chem.CombineMols(submols[0], submols[1])
    except:
        return None
    
    new_index = {
        'smiles': Chem.MolToSmiles(new_mol),
        'smi_warhead': index.get('smi_warhead', ''),
        'smi_ligase': index.get('smi_ligase', ''),
        'smi_linker': index.get('smi_linker', ''),
        'mol': new_mol,
        'frag_mol': frag_mol,
        'link_mol': link_mol,
        'atom_indices_f1': atom_indices_f1,
        'atom_indices_f2': atom_indices_f2,
        'fragment_mask': fragment_mask,
        'linker_mask': linker_mask,
    }
    return new_index


def get_linker_dataset(cfg, transform_map=None):
    """Get linker dataset with splits."""
    dataset = LinkerDataset(
        raw_path=cfg.path,
        dset_name=cfg.name,
        version=cfg.version,
        split_mode=cfg.split_mode,
        index_name=cfg.index_name,
    )
    
    if dataset.split is not None:
        subsets = {k: Subset(dataset, indices=v) for k, v in dataset.split.items()}
        
        if transform_map is not None:
            t_subsets = {}
            for k, v in subsets.items():
                t_subsets[k] = TransformDataset(v, transform_map.get(k))
            return dataset, t_subsets
        return dataset, subsets
    
    return dataset


class TransformDataset(Dataset):
    """Dataset wrapper that applies transforms."""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        data = self.dataset[index]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def __len__(self):
        return len(self.dataset)
