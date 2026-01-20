"""Data transforms compatible with LinkerNet."""
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem


def dataset_info(dataset):
    """Get dataset-specific information."""
    if dataset == 'protac':
        return {
            'atom_types': [
                'Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'
            ],
            'maximum_valence': {
                0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4,
                9: 1, 10: 2, 11: 2, 12: 4, 13: 6, 14: 3
            },
            'number_to_atom': {
                0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I',
                6: 'N', 7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'
            },
        }
    else:
        raise ValueError(f'Unknown dataset: {dataset}')


class FeaturizeAtom:
    """Featurize atoms for the model."""
    
    def __init__(self, dataset_name, known_anchor=False, add_atom_type=True, add_atom_feat=True):
        self.dataset_name = dataset_name
        self.dataset_info = dataset_info(dataset_name)
        self.known_anchor = known_anchor
        self.add_atom_type = add_atom_type
        self.add_atom_feat = add_atom_feat
    
    @property
    def num_classes(self):
        return len(self.dataset_info['atom_types'])
    
    @property
    def feature_dim(self):
        n_feat_dim = 2  # linker flag
        if self.known_anchor:
            n_feat_dim += 2  # anchor flag
        return n_feat_dim
    
    def get_index(self, atom_num, valence, charge):
        pt = Chem.GetPeriodicTable()
        atom_str = "%s%i(%i)" % (pt.GetElementSymbol(atom_num), valence, charge)
        if atom_str in self.dataset_info['atom_types']:
            return self.dataset_info['atom_types'].index(atom_str)
        return 1  # Default to C
    
    def get_element_from_index(self, index):
        pt = Chem.GetPeriodicTable()
        symb = self.dataset_info['number_to_atom'].get(index, 'C')
        return pt.GetAtomicNumber(symb)
    
    def __call__(self, data):
        if self.add_atom_type:
            x = [
                self.get_index(int(e), int(v), int(c))
                for e, v, c in zip(data.element, data.valence, data.charge)
            ]
            data.atom_type = torch.tensor(x)
        
        if self.add_atom_feat:
            linker_flag = F.one_hot((data.fragment_mask == 0).long(), 2)
            all_feats = [linker_flag]
            
            if self.known_anchor:
                anchor_flag = F.one_hot((data.anchor_mask == 1).long(), 2)
                all_feats.append(anchor_flag)
            
            data.atom_feat = torch.cat(all_feats, -1).float()
        
        return data


class BuildCompleteGraph:
    """Build complete graph with edge features."""
    
    def __init__(self, known_linker_bond=False, known_cand_anchors=False):
        self.known_linker_bond = known_linker_bond
        self.known_cand_anchors = known_cand_anchors
    
    @property
    def num_bond_classes(self):
        return 5  # NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC
    
    @property
    def bond_feature_dim(self):
        return 4  # (ll, lf, fl, ff) indicator
    
    @staticmethod
    def _get_interleave_edge_index(edge_index):
        edge_index_sym = torch.stack([edge_index[1], edge_index[0]])
        e = torch.zeros_like(torch.cat([edge_index, edge_index_sym], dim=-1))
        e[:, ::2] = edge_index
        e[:, 1::2] = edge_index_sym
        return e
    
    def _build_interleave_fc(self, n1_atoms, n2_atoms):
        eij = torch.triu_indices(n1_atoms, n2_atoms, offset=1)
        e = self._get_interleave_edge_index(eij)
        return e
    
    def __call__(self, data):
        num_nodes = len(data.pos)
        fc_edge_index = self._build_interleave_fc(num_nodes, num_nodes)
        data.edge_index = fc_edge_index
        
        src, dst = data.edge_index
        num_edges = len(fc_edge_index[0])
        
        # Edge type: (ll, lf, fl, ff)
        edge_type = torch.zeros(num_edges).long()
        l_ind_src = data.fragment_mask[src] == 0
        l_ind_dst = data.fragment_mask[dst] == 0
        edge_type[l_ind_src & l_ind_dst] = 0
        edge_type[l_ind_src & ~l_ind_dst] = 1
        edge_type[~l_ind_src & l_ind_dst] = 2
        edge_type[~l_ind_src & ~l_ind_dst] = 3
        data.edge_feat = F.one_hot(edge_type, num_classes=4).float()
        
        # Bond type
        bond_type = torch.zeros(num_edges).long()
        id_fc_edge = fc_edge_index[0] * num_nodes + fc_edge_index[1]
        id_frag_bond = data.bond_index[0] * num_nodes + data.bond_index[1]
        
        for idx, id_ in enumerate(id_frag_bond):
            matches = (id_fc_edge == id_).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                bond_type[matches[0]] = data.bond_type[idx]
        
        data.edge_type = bond_type
        
        # Linker bond mask
        if self.known_linker_bond:
            data.linker_bond_mask = (data.fragment_mask[src] == 0) ^ (data.fragment_mask[dst] == 0)
        else:
            data.linker_bond_mask = (data.fragment_mask[src] == 0) | (data.fragment_mask[dst] == 0)
        
        data.inner_edge_mask = (data.fragment_mask[src] == data.fragment_mask[dst])
        
        return data


class StackFragLocalPos:
    """Stack fragment local positions with padding."""
    
    def __init__(self, max_num_atoms=30):
        self.max_num_atoms = max_num_atoms
    
    def __call__(self, data):
        frag_idx_mask = data.fragment_mask[data.fragment_mask > 0]
        f1_pos = data.frags_local_pos[frag_idx_mask == 1]
        f2_pos = data.frags_local_pos[frag_idx_mask == 2]
        
        max_atoms = self.max_num_atoms
        
        # Pad fragment 1
        n1 = len(f1_pos)
        if n1 > max_atoms:
            f1_fill_pos = f1_pos[:max_atoms]
            f1_mask = torch.ones(max_atoms)
        else:
            f1_fill_pos = torch.cat([f1_pos, torch.zeros(max_atoms - n1, 3)], dim=0)
            f1_mask = torch.cat([torch.ones(n1), torch.zeros(max_atoms - n1)], dim=0)
        
        # Pad fragment 2
        n2 = len(f2_pos)
        if n2 > max_atoms:
            f2_fill_pos = f2_pos[:max_atoms]
            f2_mask = torch.ones(max_atoms)
        else:
            f2_fill_pos = torch.cat([f2_pos, torch.zeros(max_atoms - n2, 3)], dim=0)
            f2_mask = torch.cat([torch.ones(n2), torch.zeros(max_atoms - n2)], dim=0)
        
        data.frags_local_pos_filled = torch.stack([f1_fill_pos, f2_fill_pos], dim=0)
        data.frags_local_pos_mask = torch.stack([f1_mask, f2_mask], dim=0).bool()
        
        return data


class RelativeGeometry:
    """Compute relative geometry between fragments."""
    
    def __init__(self, mode='distance_and_two_rot'):
        self.mode = mode
    
    def __call__(self, data):
        if self.mode == 'relative_pos_and_rot':
            idx = torch.randint(0, 2, [1])[0]
            pos = (data.pos - data.frags_t[idx]) @ data.frags_R[idx]
            frags_R = data.frags_R[idx].T @ data.frags_R
            frags_t = (data.frags_t - data.frags_t[idx]) @ data.frags_R[idx]
            data.frags_rel_mask = torch.tensor([True, True])
            data.frags_rel_mask[idx] = False
            data.frags_atom_rel_mask = data.fragment_mask == (2 - idx)
        
        elif self.mode == 'two_pos_and_rot':
            rand_rot = get_random_rot()
            pos = data.pos @ rand_rot
            frags_R = rand_rot.T @ data.frags_R
            frags_t = data.frags_t @ rand_rot
            data.frags_rel_mask = torch.tensor([True, True])
        
        elif self.mode == 'distance_and_two_rot':
            frags_R = data.frags_R
            frags_t = data.frags_t
            pos = data.pos
            data.frags_rel_mask = torch.tensor([True, True])
        
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        
        data.frags_R = frags_R
        data.frags_t = frags_t
        data.pos = pos
        
        return data


class SelectCandAnchors:
    """Select candidate anchor atoms."""
    
    def __init__(self, mode='exact', k=2):
        self.mode = mode
        self.k = k
    
    @staticmethod
    def bfs(nbh_list, node, k=2, valid_list=None):
        if valid_list is None:
            valid_list = []
        visited = [node]
        queue = [node]
        level = [0]
        bfs_perm = []
        
        while len(queue) > 0:
            m = queue.pop(0)
            l = level.pop(0)
            if l > k:
                break
            bfs_perm.append(m)
            
            for neighbour in nbh_list.get(m, []):
                if neighbour not in visited and neighbour in valid_list:
                    visited.append(neighbour)
                    queue.append(neighbour)
                    level.append(l + 1)
        
        return bfs_perm
    
    def __call__(self, data):
        cand_anchors_mask = torch.zeros_like(data.fragment_mask).bool()
        
        if self.mode == 'exact':
            cand_anchors_mask[data.anchor_indices] = True
        elif self.mode == 'k-hop':
            for anchor in data.anchor_indices:
                a_frag_id = data.fragment_mask[anchor]
                a_valid_list = (data.fragment_mask == a_frag_id).nonzero(as_tuple=True)[0].tolist()
                a_cand = self.bfs(data.nbh_list, anchor, k=self.k, valid_list=a_valid_list)
                cand_anchors_mask[a_cand] = True
        
        data.cand_anchors_mask = cand_anchors_mask
        return data


def get_random_rot():
    """Generate random rotation matrix."""
    M = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(M)
    return torch.from_numpy(Q.astype(np.float32))
