"""Sampling script for PROTAC Flow Matching model."""
import argparse
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
import yaml

from .model import ProTACFlowModel
from .data import ProtacDataset, collate_fn
from .chem import validate_molecule, compute_properties, filter_molecules, rank_molecules
from .chem.filter import reconstruct_molecule, MoleculeFilter
from .diffusion.mask import DiscreteDenoiser
from .utils.seed import seed_all


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Sampler:
    """Sampler for PROTAC Flow Matching model."""
    
    def __init__(
        self,
        model: ProTACFlowModel,
        config: dict,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        
        # Sampling parameters
        self.n_steps = config['sample']['n_steps']
        self.dt = config['sample']['dt']
        self.max_resample_bond = config['sample']['max_resample_bond']
        
        # Discrete denoiser
        self.atom_denoiser = DiscreteDenoiser(
            num_classes=len(config['data']['atom_vocab']),
            mask_token=0,
        )
        self.bond_denoiser = DiscreteDenoiser(
            num_classes=len(config['data']['bond_vocab']),
            mask_token=0,
        )
        
        # Filter
        self.mol_filter = MoleculeFilter(**config['filter'])
    
    @torch.no_grad()
    def sample_batch(
        self,
        batch_data: dict,
        return_trajectory: bool = False,
    ) -> dict:
        """Sample from the model for a batch of inputs.
        
        Args:
            batch_data: Batched data dict (fragments, to generate linker)
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Dict with generated results
        """
        B = batch_data['num_graphs']
        
        # Move to device
        batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch_data.items()}
        
        # Get data
        x_data = batch_data['x_data']
        atom_type = batch_data['atom_type']
        node_kind = batch_data['node_kind']
        is_anchor = batch_data['is_anchor']
        edge_index = batch_data['edge_index']
        batch_idx = batch_data['batch']
        linker_mask = batch_data['linker_mask']
        fragB_mask = batch_data['fragB_mask']
        cond = batch_data['cond']
        anchors = batch_data['anchors']
        
        # Initialize linker coordinates from noise
        x_t = x_data.clone()
        x_t[linker_mask] = torch.randn_like(x_t[linker_mask])
        
        # Initialize atom types as MASK
        atom_type_t = atom_type.clone()
        atom_type_t[linker_mask] = 0  # MASK
        
        # Initialize bond types as NONE
        bond_type_t = torch.zeros_like(batch_data['bond_type_gt'])
        
        # Trajectory storage
        trajectory = [] if return_trajectory else None
        
        # ODE integration from t=1 to t=0
        for step in range(self.n_steps):
            t_val = 1.0 - step * self.dt
            t_next = max(0.0, t_val - self.dt)
            t = torch.full((B,), t_val, device=self.device)
            
            # Compute edge attributes
            edge_attr = self._compute_edge_attr(x_t, edge_index, node_kind)
            
            # Forward pass
            outputs = self.model.forward(
                x_t=x_t,
                atom_type_t=atom_type_t,
                node_kind=node_kind,
                is_anchor=is_anchor,
                edge_index=edge_index,
                edge_attr=edge_attr,
                t=t,
                cond=cond,
                batch=batch_idx,
                linker_mask=linker_mask,
                fragB_mask=fragB_mask,
            )
            
            # Update coordinates (Euler step)
            vel = outputs['vel']
            x_t = x_t + vel * self.dt
            
            # Update fragB pose
            dp = outputs['dp']
            dr = outputs['dr']
            
            # Apply pose update to fragB nodes
            for b in range(B):
                fragB_nodes = (batch_idx == b) & fragB_mask
                x_t[fragB_nodes] = x_t[fragB_nodes] + dp[b] * self.dt
            
            # Unmask atoms
            atom_type_t = self.atom_denoiser.sample_step(
                atom_type_t,
                outputs['atom_logits'],
                t_val,
                t_next,
            )
            
            # Store trajectory
            if return_trajectory:
                trajectory.append({
                    'x': x_t.cpu().clone(),
                    'atom_type': atom_type_t.cpu().clone(),
                    't': t_val,
                })
        
        # Final bond prediction with constraint checking
        edge_attr = self._compute_edge_attr(x_t, edge_index, node_kind)
        outputs = self.model.forward(
            x_t=x_t,
            atom_type_t=atom_type_t,
            node_kind=node_kind,
            is_anchor=is_anchor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            t=torch.zeros(B, device=self.device),
            cond=cond,
            batch=batch_idx,
            linker_mask=linker_mask,
            fragB_mask=fragB_mask,
        )
        
        # Sample bond types with constraint checking
        bond_type_final = self._sample_bonds_with_constraints(
            outputs['bond_logits'],
            edge_index,
            atom_type_t,
            node_kind,
            linker_mask,
            anchors,
            batch_idx,
            batch_data['edge_batch'],
        )
        
        results = {
            'x_final': x_t.cpu(),
            'atom_type_final': atom_type_t.cpu(),
            'bond_type_final': bond_type_final.cpu(),
            'batch': batch_idx.cpu(),
            'edge_index': edge_index.cpu(),
            'linker_mask': linker_mask.cpu(),
            'anchors': anchors.cpu(),
        }
        
        if return_trajectory:
            results['trajectory'] = trajectory
        
        return results
    
    def _compute_edge_attr(self, x, edge_index, node_kind):
        """Compute edge attributes."""
        import torch.nn.functional as F
        src, dst = edge_index
        
        same_kind = (node_kind[src] == node_kind[dst]).float().unsqueeze(-1)
        src_kind = F.one_hot(node_kind[src], num_classes=3).float()
        dst_kind = F.one_hot(node_kind[dst], num_classes=3).float()
        
        return torch.cat([same_kind, src_kind, dst_kind], dim=-1)
    
    def _sample_bonds_with_constraints(
        self,
        bond_logits,
        edge_index,
        atom_types,
        node_kind,
        linker_mask,
        anchors,
        batch_idx,
        edge_batch,
    ):
        """Sample bond types with constraint checking and resampling."""
        from .chem.constraints import (
            check_valence, check_anchor_connectivity, 
            check_connectivity, check_no_fragA_fragB_bond
        )
        
        B = anchors.size(0)
        bond_types = torch.zeros(bond_logits.size(0), dtype=torch.long, device=bond_logits.device)
        
        for b in range(B):
            edge_mask = edge_batch == b
            node_mask = batch_idx == b
            
            local_edge_index = edge_index[:, edge_mask]
            local_bond_logits = bond_logits[edge_mask]
            local_atom_types = atom_types[node_mask]
            local_node_kind = node_kind[node_mask]
            local_linker_mask = linker_mask[node_mask]
            local_anchors = anchors[b] - (batch_idx < b).sum()
            
            # Remap edge indices to local
            node_offset = node_mask.nonzero(as_tuple=True)[0][0]
            local_edge_index = local_edge_index - node_offset
            
            # Try sampling with constraints
            for attempt in range(self.max_resample_bond):
                # Sample bond types
                probs = torch.softmax(local_bond_logits, dim=-1)
                sampled_bonds = torch.multinomial(probs, 1).squeeze(-1)
                
                # Check constraints
                valid = True
                
                valence_ok, _ = check_valence(local_atom_types, sampled_bonds, local_edge_index)
                if not valence_ok:
                    valid = False
                
                anchor_ok, _ = check_anchor_connectivity(
                    sampled_bonds, local_edge_index, local_anchors, local_linker_mask
                )
                if not anchor_ok:
                    valid = False
                
                connected_ok, _ = check_connectivity(
                    sampled_bonds, local_edge_index, local_linker_mask, local_anchors
                )
                if not connected_ok:
                    valid = False
                
                no_direct = check_no_fragA_fragB_bond(sampled_bonds, local_edge_index, local_node_kind)
                if not no_direct:
                    valid = False
                
                if valid:
                    bond_types[edge_mask] = sampled_bonds
                    break
            else:
                # Use argmax if all attempts fail
                bond_types[edge_mask] = torch.argmax(local_bond_logits, dim=-1)
        
        return bond_types
    
    def validate_and_filter(self, results: dict) -> list:
        """Validate and filter generated molecules.
        
        Args:
            results: Results from sample_batch
            
        Returns:
            List of valid molecule dicts
        """
        from .chem.constraints import validate_molecule
        
        molecules = []
        
        ptr = [0]
        batch = results['batch']
        for i in range(batch.max().item() + 1):
            ptr.append((batch == i).sum().item() + ptr[-1])
        
        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i + 1]
            
            mol_dict = {
                'x': results['x_final'][start:end],
                'atom_type': results['atom_type_final'][start:end],
                'linker_mask': results['linker_mask'][start:end],
            }
            
            # Get edges for this molecule
            edge_mask = (results['edge_index'][0] >= start) & (results['edge_index'][0] < end)
            mol_dict['edge_index'] = results['edge_index'][:, edge_mask] - start
            mol_dict['bond_type'] = results['bond_type_final'][edge_mask]
            mol_dict['anchors'] = results['anchors'][i] - start
            
            # Validate
            validation = validate_molecule(
                mol_dict['atom_type'],
                mol_dict['bond_type'],
                mol_dict['edge_index'],
                torch.zeros_like(mol_dict['atom_type']),  # node_kind placeholder
                mol_dict['linker_mask'],
                mol_dict['anchors'],
            )
            
            mol_dict['validation'] = validation
            mol_dict['valid'] = validation['all_valid']
            
            molecules.append(mol_dict)
        
        return molecules


def main():
    parser = argparse.ArgumentParser(description='Sample from PROTAC Flow Matching model')
    parser.add_argument('--config', type=str, default='configs/sample.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per input')
    parser.add_argument('--data_idx', type=int, default=0,
                        help='Index of test data to use')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    seed_all(config['sample']['seed'])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = ProTACFlowModel(
        num_atom_types=len(config['data']['atom_vocab']),
        num_bond_types=len(config['data']['bond_vocab']),
        h_dim=config['model']['h_dim'],
        edge_dim=config['model']['edge_dim'],
        cond_dim=config['model']['cond_dim'],
        time_emb_dim=config['model']['time_emb_dim'],
        n_layers=config['model']['n_layers'],
        num_heads=config['model']['num_heads'],
        N_L_max=config['data']['N_L_max'],
    )
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create sampler
    sampler = Sampler(model, config, args.device)
    
    # Load test data
    dataset = ProtacDataset(
        config['data']['dataset_path'],
        config['data']['processed_path'],
        N_L_max=config['data']['N_L_max'],
    )
    
    # Get test sample
    test_data = dataset[args.data_idx]
    
    # Create batch by repeating
    data_list = [test_data.copy() for _ in range(args.num_samples)]
    batch_data = collate_fn(data_list)
    
    # Sample
    print(f"Sampling {args.num_samples} molecules...")
    results = sampler.sample_batch(batch_data, return_trajectory=False)
    
    # Validate and filter
    molecules = sampler.validate_and_filter(results)
    
    # Count valid
    num_valid = sum(1 for m in molecules if m['valid'])
    print(f"Valid molecules: {num_valid}/{len(molecules)} ({100*num_valid/len(molecules):.1f}%)")
    
    # Save results
    save_path = output_dir / f'samples_{args.data_idx}.pt'
    torch.save({
        'results': results,
        'molecules': molecules,
        'config': config,
    }, save_path)
    print(f"Saved results to {save_path}")


if __name__ == '__main__':
    main()
