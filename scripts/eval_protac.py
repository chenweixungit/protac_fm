"""Evaluation script for PROTAC Flow Matching model."""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit import DataStructs
from tqdm.auto import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protac_fm.utils.misc import load_config
from protac_fm.chem.properties import compute_properties, compute_sa_score


def eval_validity(gen_mols):
    """Evaluate validity metrics."""
    n_total = len(gen_mols)
    n_valid = 0
    n_complete = 0
    valid_mols = []
    
    for mol in gen_mols:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            n_valid += 1
            valid_mols.append(mol)
            
            smiles = Chem.MolToSmiles(mol)
            if '.' not in smiles:
                n_complete += 1
        except:
            continue
    
    return {
        'total': n_total,
        'valid': n_valid,
        'complete': n_complete,
        'validity_rate': n_valid / n_total if n_total > 0 else 0,
        'completeness_rate': n_complete / n_total if n_total > 0 else 0,
        'valid_mols': valid_mols,
    }


def eval_uniqueness(valid_mols):
    """Evaluate uniqueness of generated molecules."""
    smiles_set = set()
    for mol in valid_mols:
        try:
            smi = Chem.MolToSmiles(mol)
            smiles_set.add(smi)
        except:
            pass
    
    n_unique = len(smiles_set)
    n_total = len(valid_mols)
    
    return {
        'unique': n_unique,
        'uniqueness_rate': n_unique / n_total if n_total > 0 else 0,
    }


def eval_diversity(valid_mols, sample_size=1000):
    """Evaluate diversity using Tanimoto similarity."""
    if len(valid_mols) < 2:
        return {'diversity': 0.0, 'avg_similarity': 1.0}
    
    # Sample if too many molecules
    if len(valid_mols) > sample_size:
        import random
        valid_mols = random.sample(valid_mols, sample_size)
    
    # Compute fingerprints
    fps = []
    for mol in valid_mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        except:
            pass
    
    if len(fps) < 2:
        return {'diversity': 0.0, 'avg_similarity': 1.0}
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    diversity = 1.0 - avg_sim
    
    return {
        'diversity': diversity,
        'avg_similarity': avg_sim,
    }


def eval_properties(valid_mols):
    """Evaluate property distributions."""
    property_values = defaultdict(list)
    
    for mol in tqdm(valid_mols, desc='Computing properties'):
        try:
            props = compute_properties(mol)
            for key, value in props.items():
                if isinstance(value, (int, float)):
                    property_values[key].append(value)
        except:
            continue
    
    # Compute statistics
    stats = {}
    for key, values in property_values.items():
        if values:
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_std'] = np.std(values)
            stats[f'{key}_min'] = np.min(values)
            stats[f'{key}_max'] = np.max(values)
            stats[f'{key}_median'] = np.median(values)
    
    return stats, property_values


def eval_filter_pass_rate(valid_mols, filter_config):
    """Evaluate filter pass rates."""
    pass_counts = defaultdict(int)
    total = len(valid_mols)
    
    for mol in valid_mols:
        try:
            props = compute_properties(mol)
            
            # Check each criterion
            if filter_config.get('MW_min', 0) <= props.get('MW', 0) <= filter_config.get('MW_max', float('inf')):
                pass_counts['MW'] += 1
            if props.get('logP', 10) <= filter_config.get('logP_max', float('inf')):
                pass_counts['logP'] += 1
            if props.get('TPSA', 300) <= filter_config.get('TPSA_max', float('inf')):
                pass_counts['TPSA'] += 1
            if props.get('HBD', 10) <= filter_config.get('HBD_max', float('inf')):
                pass_counts['HBD'] += 1
            if props.get('HBA', 20) <= filter_config.get('HBA_max', float('inf')):
                pass_counts['HBA'] += 1
            if filter_config.get('RotB_min', 0) <= props.get('RotB', 0) <= filter_config.get('RotB_max', float('inf')):
                pass_counts['RotB'] += 1
            if props.get('QED', 0) >= filter_config.get('QED_min', 0):
                pass_counts['QED'] += 1
            if props.get('SA', 10) <= filter_config.get('SA_max', float('inf')):
                pass_counts['SA'] += 1
            
            # All filters
            all_pass = all([
                filter_config.get('MW_min', 0) <= props.get('MW', 0) <= filter_config.get('MW_max', float('inf')),
                props.get('logP', 10) <= filter_config.get('logP_max', float('inf')),
                props.get('TPSA', 300) <= filter_config.get('TPSA_max', float('inf')),
                props.get('HBD', 10) <= filter_config.get('HBD_max', float('inf')),
                props.get('HBA', 20) <= filter_config.get('HBA_max', float('inf')),
                filter_config.get('RotB_min', 0) <= props.get('RotB', 0) <= filter_config.get('RotB_max', float('inf')),
                props.get('QED', 0) >= filter_config.get('QED_min', 0),
                props.get('SA', 10) <= filter_config.get('SA_max', float('inf')),
            ])
            if all_pass:
                pass_counts['all_filters'] += 1
        except:
            continue
    
    # Compute rates
    rates = {}
    for key, count in pass_counts.items():
        rates[f'{key}_rate'] = count / total if total > 0 else 0
        rates[f'{key}_count'] = count
    rates['total'] = total
    
    return rates


def print_report(validity, uniqueness, diversity, property_stats, filter_rates):
    """Print evaluation report."""
    print('\n' + '=' * 60)
    print('EVALUATION REPORT')
    print('=' * 60)
    
    print('\n--- Validity Metrics ---')
    print(f"Total samples: {validity['total']}")
    print(f"Valid: {validity['valid']} ({100*validity['validity_rate']:.1f}%)")
    print(f"Complete: {validity['complete']} ({100*validity['completeness_rate']:.1f}%)")
    
    print('\n--- Uniqueness ---')
    print(f"Unique: {uniqueness['unique']} ({100*uniqueness['uniqueness_rate']:.1f}%)")
    
    print('\n--- Diversity ---')
    print(f"Diversity: {diversity['diversity']:.3f}")
    print(f"Avg similarity: {diversity['avg_similarity']:.3f}")
    
    print('\n--- Property Statistics ---')
    for key in ['MW', 'logP', 'TPSA', 'QED', 'SA', 'RotB']:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in property_stats:
            print(f"{key}: {property_stats[mean_key]:.2f} ± {property_stats.get(std_key, 0):.2f}")
    
    print('\n--- Filter Pass Rates ---')
    print(f"Total valid for filtering: {filter_rates.get('total', 0)}")
    for key in ['MW', 'logP', 'TPSA', 'HBD', 'HBA', 'RotB', 'QED', 'SA', 'all_filters']:
        rate_key = f'{key}_rate'
        count_key = f'{key}_count'
        if rate_key in filter_rates:
            print(f"  {key}: {filter_rates.get(count_key, 0)} ({100*filter_rates[rate_key]:.1f}%)")
    
    print('\n' + '=' * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to sampling results')
    parser.add_argument('--config', type=str, default='./configs/protac.yml', help='Config file for filter settings')
    parser.add_argument('--output', type=str, default=None, help='Output path for evaluation results')
    args = parser.parse_args()
    
    RDLogger.DisableLog('rdApp.*')
    
    # Load config
    config = load_config(args.config)
    filter_config = config.get('filter', {})
    
    # Load results
    print(f'Loading results from {args.results}...')
    results = torch.load(args.results)
    
    # Collect all generated molecules
    all_gen_mols = []
    if isinstance(results, list):
        for r in results:
            all_gen_mols.extend(r.get('gen_mols', []))
    elif isinstance(results, dict):
        all_gen_mols = results.get('gen_mols', [])
    
    print(f'Loaded {len(all_gen_mols)} generated molecules')
    
    # Evaluate validity
    print('Evaluating validity...')
    validity = eval_validity(all_gen_mols)
    valid_mols = validity['valid_mols']
    
    # Evaluate uniqueness
    print('Evaluating uniqueness...')
    uniqueness = eval_uniqueness(valid_mols)
    
    # Evaluate diversity
    print('Evaluating diversity...')
    diversity = eval_diversity(valid_mols)
    
    # Evaluate properties
    print('Evaluating properties...')
    property_stats, property_values = eval_properties(valid_mols)
    
    # Evaluate filter pass rates
    print('Evaluating filter pass rates...')
    filter_rates = eval_filter_pass_rate(valid_mols, filter_config)
    
    # Print report
    print_report(validity, uniqueness, diversity, property_stats, filter_rates)
    
    # Save results
    if args.output:
        eval_results = {
            'validity': validity,
            'uniqueness': uniqueness,
            'diversity': diversity,
            'property_stats': property_stats,
            'property_values': dict(property_values),
            'filter_rates': filter_rates,
        }
        # Remove non-serializable items
        eval_results['validity'] = {k: v for k, v in validity.items() if k != 'valid_mols'}
        
        torch.save(eval_results, args.output)
        print(f'\nSaved evaluation results to {args.output}')


if __name__ == '__main__':
    main()
