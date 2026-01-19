"""Sampling script for PROTAC Flow Matching model."""
import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from tqdm.auto import tqdm
from rdkit import Chem, RDLogger

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protac_fm.utils.misc import load_config, seed_all
from protac_fm.data.linker_dataset import get_linker_dataset
from protac_fm.data.transforms import (
    FeaturizeAtom, BuildCompleteGraph, StackFragLocalPos, RelativeGeometry
)
from protac_fm.model.protac_flow import ProTACFlowModel
from protac_fm.utils.reconstruct import parse_sampling_result_with_bond, eval_success_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='./configs/protac.yml')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--test_idx', type=int, default=None, help='Test sample index (None for all)')
    args = parser.parse_args()
    
    RDLogger.DisableLog('rdApp.*')
    
    # Load configs
    config = load_config(args.config)
    seed_all(config.train.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transforms
    cfg_model = config.model
    atom_featurizer = FeaturizeAtom(
        config.dataset.name,
        known_anchor=cfg_model.known_anchor
    )
    graph_builder = BuildCompleteGraph(
        known_linker_bond=cfg_model.known_linker_bond
    )
    max_num_atoms = config.dataset.get('max_num_atoms', 70)
    
    test_transform = Compose([
        atom_featurizer,
        graph_builder,
        StackFragLocalPos(max_num_atoms=max_num_atoms),
        RelativeGeometry(mode=cfg_model.rel_geometry)
    ])
    
    # Load dataset
    print('Loading dataset...')
    dataset, subsets = get_linker_dataset(
        cfg=config.dataset,
        transform_map={'train': test_transform, 'val': test_transform, 'test': test_transform}
    )
    test_set = subsets['test']
    print(f'Test set size: {len(test_set)}')
    
    # Load model
    print('Loading model...')
    model = ProTACFlowModel(
        num_atom_types=atom_featurizer.num_classes,
        num_bond_types=graph_builder.num_bond_classes,
        h_dim=cfg_model.h_dim,
        edge_dim=cfg_model.edge_dim,
        cond_dim=cfg_model.cond_dim,
        time_emb_dim=cfg_model.time_emb_dim,
        n_layers=cfg_model.n_layers,
        num_heads=cfg_model.num_heads,
        N_L_max=cfg_model.N_L_max,
    ).to(args.device)
    
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint from {args.ckpt}')
    
    FOLLOW_BATCH = ['edge_type']
    COLLATE_EXCLUDE_KEYS = ['nbh_list']
    
    # Determine which samples to process
    if args.test_idx is not None:
        test_indices = [args.test_idx]
    else:
        test_indices = list(range(len(test_set)))
    
    all_results = []
    all_gen_mols = []
    
    for test_idx in tqdm(test_indices, desc='Sampling'):
        data = test_set[test_idx]
        data_list = [data.clone() for _ in range(args.num_samples)]
        batch = Batch.from_data_list(
            data_list, follow_batch=FOLLOW_BATCH, exclude_keys=COLLATE_EXCLUDE_KEYS
        ).to(args.device)
        
        with torch.no_grad():
            results = model.sample(batch, n_steps=config.sample.get('n_steps', 100))
            final_x = results['x_final']
            final_c = results['atom_type_final']
            final_bond = results['bond_type_final']
        
        gen_mols = parse_sampling_result_with_bond(
            data_list, final_x, final_c, final_bond, atom_featurizer,
            known_linker_bonds=cfg_model.known_linker_bond, check_validity=True
        )
        
        # Evaluate
        recon_rate, complete_rate = eval_success_rate(gen_mols)
        print(f'Test {test_idx}: Recon rate: {recon_rate:.4f}, Complete rate: {complete_rate:.4f}')
        
        result = {
            'test_idx': test_idx,
            'data': data,
            'final_x': final_x.cpu(),
            'final_c': final_c.cpu(),
            'final_bond': final_bond.cpu(),
            'gen_mols': gen_mols,
            'recon_rate': recon_rate,
            'complete_rate': complete_rate,
        }
        all_results.append(result)
        all_gen_mols.extend(gen_mols)
    
    # Save results
    output_path = os.path.join(args.output_dir, 'sampling_results.pt')
    torch.save(all_results, output_path)
    print(f'Saved results to {output_path}')
    
    # Save molecules as SDF
    sdf_path = os.path.join(args.output_dir, 'generated_molecules.sdf')
    writer = Chem.SDWriter(sdf_path)
    valid_count = 0
    for mol in all_gen_mols:
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                writer.write(mol)
                valid_count += 1
            except:
                pass
    writer.close()
    print(f'Saved {valid_count} valid molecules to {sdf_path}')
    
    # Print summary
    print('\n' + '=' * 50)
    print('SAMPLING SUMMARY')
    print('=' * 50)
    total_samples = len(test_indices) * args.num_samples
    avg_recon = np.mean([r['recon_rate'] for r in all_results])
    avg_complete = np.mean([r['complete_rate'] for r in all_results])
    print(f'Total samples: {total_samples}')
    print(f'Average recon rate: {avg_recon:.4f}')
    print(f'Average complete rate: {avg_complete:.4f}')
    print(f'Valid molecules saved: {valid_count}')
    print('=' * 50)


if __name__ == '__main__':
    main()
