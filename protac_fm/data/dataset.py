"""Dataset classes for PROTAC data."""
import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm
from typing import Dict, List, Optional

from .rdkit_split import split_protac_into_fragments, get_fragment_info
from .featurize import build_graph_tensors


class ProtacDataset(Dataset):
    """Dataset for PROTAC molecules."""
    
    def __init__(
        self,
        raw_path: str,
        processed_path: str,
        N_L_max: int = 32,
        transform=None,
        reset: bool = False
    ):
        """Initialize dataset.
        
        Args:
            raw_path: Path to raw data (index.pkl or similar)
            processed_path: Path to processed LMDB database
            N_L_max: Maximum number of linker atoms
            transform: Optional transform to apply
            reset: If True, reprocess data even if processed file exists
        """
        super().__init__()
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.N_L_max = N_L_max
        self.transform = transform
        
        self.db = None
        self.keys = None
        
        if not os.path.exists(processed_path) or reset:
            print(f"Processing data from {raw_path}...")
            self._process()
        
        print(f"Loading dataset from {processed_path}")
    
    def _connect_db(self):
        """Establish read-only database connection."""
        if self.db is not None:
            return
        
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
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
    
    def _process(self):
        """Process raw data and save to LMDB."""
        # Load raw data
        if self.raw_path.endswith('.pkl'):
            with open(self.raw_path, 'rb') as f:
                raw_data = pickle.load(f)
        else:
            # Assume it's a directory with SDF files
            raw_data = self._load_sdf_directory(self.raw_path)
        
        # Create LMDB database
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),
            create=True,
            subdir=False,
            readonly=False,
        )
        
        num_processed = 0
        num_skipped = 0
        
        with db.begin(write=True, buffers=True) as txn:
            for idx, entry in enumerate(tqdm(raw_data, desc="Processing")):
                try:
                    sample = self._process_entry(entry)
                    if sample is not None:
                        txn.put(
                            key=f'{num_processed:08d}'.encode(),
                            value=pickle.dumps(sample)
                        )
                        num_processed += 1
                    else:
                        num_skipped += 1
                except Exception as e:
                    print(f"Error processing entry {idx}: {e}")
                    num_skipped += 1
        
        db.close()
        print(f"Processed {num_processed} samples, skipped {num_skipped}")
    
    def _process_entry(self, entry: Dict) -> Optional[Dict]:
        """Process a single data entry.
        
        Args:
            entry: Dict containing molecule data
            
        Returns:
            Processed sample dict or None if processing fails
        """
        # Get molecule
        if 'mol' in entry:
            mol = entry['mol']
        elif 'smiles' in entry:
            mol = Chem.MolFromSmiles(entry['smiles'])
            if mol is not None:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
        else:
            return None
        
        if mol is None or mol.GetNumConformers() == 0:
            return None
        
        # Check if fragment info is provided
        if 'atom_indices_warhead' in entry and 'atom_indices_ligase' in entry:
            # Use provided fragment info
            fragA_indices = list(entry['atom_indices_warhead'])
            fragB_indices = list(entry['atom_indices_ligase'])
            linker_indices = list(entry.get('atom_indices_linker', []))
            
            if not linker_indices:
                all_indices = set(range(mol.GetNumAtoms()))
                linker_indices = list(all_indices - set(fragA_indices) - set(fragB_indices))
            
            # Find anchors
            anchorA_idx = self._find_anchor(mol, fragA_indices, linker_indices)
            anchorB_idx = self._find_anchor(mol, fragB_indices, linker_indices)
            
            anchors = {
                'anchorA_idx': anchorA_idx,
                'anchorB_idx': anchorB_idx,
                'linker_atom_indices': linker_indices,
                'fragA_atom_indices': fragA_indices,
                'fragB_atom_indices': fragB_indices,
            }
            
            fragA_mol = None
            fragB_mol = None
            linker_mol = None
        else:
            # Split molecule
            fragA_mol, fragB_mol, linker_mol, anchors = split_protac_into_fragments(mol)
            
            if fragA_mol is None:
                return None
        
        # Check linker size
        if len(anchors['linker_atom_indices']) > self.N_L_max:
            return None
        
        # Build graph tensors
        sample = build_graph_tensors(
            fragA_mol, fragB_mol, linker_mol, anchors, mol, self.N_L_max
        )
        
        # Add metadata
        if 'smiles' in entry:
            sample['smiles'] = entry['smiles']
        if 'smi_warhead' in entry:
            sample['smi_warhead'] = entry['smi_warhead']
        if 'smi_ligase' in entry:
            sample['smi_ligase'] = entry['smi_ligase']
        if 'smi_linker' in entry:
            sample['smi_linker'] = entry['smi_linker']
        
        return sample
    
    def _find_anchor(self, mol: Chem.Mol, frag_indices: List[int], 
                     linker_indices: List[int]) -> Optional[int]:
        """Find anchor atom connecting fragment to linker."""
        frag_set = set(frag_indices)
        linker_set = set(linker_indices)
        
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            if begin_idx in frag_set and end_idx in linker_set:
                return begin_idx
            if end_idx in frag_set and begin_idx in linker_set:
                return end_idx
        
        return None
    
    def _load_sdf_directory(self, path: str) -> List[Dict]:
        """Load molecules from SDF files in a directory."""
        data = []
        
        for filename in os.listdir(path):
            if filename.endswith('.sdf'):
                filepath = os.path.join(path, filename)
                suppl = Chem.SDMolSupplier(filepath, removeHs=False)
                
                for mol in suppl:
                    if mol is not None:
                        data.append({'mol': mol})
        
        return data
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data


class ProtacDataModule:
    """Data module for managing train/val/test splits."""
    
    def __init__(
        self,
        raw_path: str,
        processed_path: str,
        N_L_max: int = 32,
        batch_size: int = 16,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.N_L_max = N_L_max
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
        self.dataset = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
    
    def setup(self):
        """Setup datasets."""
        self.dataset = ProtacDataset(
            self.raw_path, self.processed_path, self.N_L_max
        )
        
        # Create splits
        n = len(self.dataset)
        indices = list(range(n))
        
        import random
        random.seed(self.seed)
        random.shuffle(indices)
        
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train + n_val]
        self.test_indices = indices[n_train + n_val:]
    
    def train_dataloader(self):
        from torch.utils.data import DataLoader, Subset
        from .collate import collate_fn
        
        train_set = Subset(self.dataset, self.train_indices)
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader, Subset
        from .collate import collate_fn
        
        val_set = Subset(self.dataset, self.val_indices)
        return DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self):
        from torch.utils.data import DataLoader, Subset
        from .collate import collate_fn
        
        test_set = Subset(self.dataset, self.test_indices)
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
