"""Evaluation script for PROTAC Flow Matching model."""
import argparse
import os
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import yaml

from rdkit import Chem

from .chem import compute_properties
from .chem.constraints import validate_molecule


def load_results(results_path: str) -> dict:
    """Load sampling results."""
    return torch.load(results_path)


def evaluate_validity(molecules: list) -> dict:
    """Evaluate chemical validity metrics.
    
    Args:
        molecules: List of molecule dicts from sampler
        
    Returns:
        Dict with validity metrics
    """
    metrics = {
        'total': len(molecules),
        'valence_valid': 0,
        'anchor_connected': 0,
        'linker_connected': 0,
        'no_direct_bond': 0,
        'no_illegal_bonds': 0,
        'all_valid': 0,
    }
    
    for mol in molecules:
        validation = mol.get('validation', {})
        
        if validation.get('valence', False):
            metrics['valence_valid'] += 1
        if validation.get('anchor_connectivity', False):
            metrics['anchor_connected'] += 1
        if validation.get('linker_connectivity', False):
            metrics['linker_connected'] += 1
        if validation.get('no_fragA_fragB_bond', False):
            metrics['no_direct_bond'] += 1
        if validation.get('no_illegal_bonds', False):
            metrics['no_illegal_bonds'] += 1
        if validation.get('all_valid', False):
            metrics['all_valid'] += 1
    
    # Compute rates
    total = metrics['total']
    if total > 0:
        metrics['valence_rate'] = metrics['valence_valid'] / total
        metrics['anchor_rate'] = metrics['anchor_connected'] / total
        metrics['connectivity_rate'] = metrics['linker_connected'] / total
        metrics['no_direct_rate'] = metrics['no_direct_bond'] / total
        metrics['no_illegal_rate'] = metrics['no_illegal_bonds'] / total
        metrics['validity_rate'] = metrics['all_valid'] / total
    
    return metrics


def evaluate_properties(molecules: list, rdkit_mols: list = None) -> dict:
    """Evaluate property distributions.
    
    Args:
        molecules: List of molecule dicts
        rdkit_mols: Optional list of RDKit Mol objects
        
    Returns:
        Dict with property statistics
    """
    property_values = defaultdict(list)
    
    for i, mol in enumerate(molecules):
        if not mol.get('valid', False):
            continue
        
        # If RDKit mol is available, compute properties
        if rdkit_mols and i < len(rdkit_mols) and rdkit_mols[i] is not None:
            props = compute_properties(rdkit_mols[i])
            for key, value in props.items():
                if isinstance(value, (int, float)):
                    property_values[key].append(value)
    
    # Compute statistics
    stats = {}
    for key, values in property_values.items():
        if values:
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_std'] = np.std(values)
            stats[f'{key}_min'] = np.min(values)
            stats[f'{key}_max'] = np.max(values)
            stats[f'{key}_median'] = np.median(values)
    
    return stats


def evaluate_diversity(molecules: list, rdkit_mols: list = None) -> dict:
    """Evaluate diversity metrics.
    
    Args:
        molecules: List of molecule dicts
        rdkit_mols: Optional list of RDKit Mol objects
        
    Returns:
        Dict with diversity metrics
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    metrics = {}
    
    # Filter valid molecules
    valid_mols = []
    if rdkit_mols:
        for mol, mol_dict in zip(rdkit_mols, molecules):
            if mol is not None and mol_dict.get('valid', False):
                valid_mols.append(mol)
    
    if len(valid_mols) < 2:
        return {'diversity': 0.0, 'uniqueness': 0.0}
    
    # Compute fingerprints
    fps = []
    for mol in valid_mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
        except:
            pass
    
    if len(fps) < 2:
        return {'diversity': 0.0, 'uniqueness': 0.0}
    
    # Compute pairwise Tanimoto similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    # Diversity = 1 - average similarity
    metrics['diversity'] = 1.0 - np.mean(similarities)
    metrics['avg_similarity'] = np.mean(similarities)
    
    # Uniqueness (based on SMILES)
    smiles_set = set()
    for mol in valid_mols:
        try:
            smi = Chem.MolToSmiles(mol)
            smiles_set.add(smi)
        except:
            pass
    
    metrics['uniqueness'] = len(smiles_set) / len(valid_mols) if valid_mols else 0.0
    metrics['unique_count'] = len(smiles_set)
    
    return metrics


def evaluate_filter_pass_rate(molecules: list, filter_config: dict) -> dict:
    """Evaluate filter pass rates.
    
    Args:
        molecules: List of molecule dicts with properties
        filter_config: Filter configuration
        
    Returns:
        Dict with filter pass rates
    """
    from .chem.filter import MoleculeFilter
    
    mol_filter = MoleculeFilter(**filter_config)
    
    pass_counts = defaultdict(int)
    total_valid = 0
    
    for mol in molecules:
        if not mol.get('valid', False):
            continue
        
        total_valid += 1
        props = mol.get('properties', {})
        
        if props:
            criteria = mol_filter.check_criteria(props)
            for key, passed in criteria.items():
                if passed:
                    pass_counts[key] += 1
            
            if all(criteria.values()):
                pass_counts['all_filters'] += 1
    
    # Compute rates
    metrics = {'total_valid': total_valid}
    for key, count in pass_counts.items():
        metrics[f'{key}_rate'] = count / total_valid if total_valid > 0 else 0.0
        metrics[f'{key}_count'] = count
    
    return metrics


def print_evaluation_report(
    validity_metrics: dict,
    property_stats: dict,
    diversity_metrics: dict,
    filter_metrics: dict,
):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    print("\n--- Validity Metrics ---")
    print(f"Total samples: {validity_metrics['total']}")
    print(f"Valence valid: {validity_metrics['valence_valid']} ({100*validity_metrics.get('valence_rate', 0):.1f}%)")
    print(f"Anchor connected: {validity_metrics['anchor_connected']} ({100*validity_metrics.get('anchor_rate', 0):.1f}%)")
    print(f"Linker connected: {validity_metrics['linker_connected']} ({100*validity_metrics.get('connectivity_rate', 0):.1f}%)")
    print(f"All valid: {validity_metrics['all_valid']} ({100*validity_metrics.get('validity_rate', 0):.1f}%)")
    
    print("\n--- Property Statistics ---")
    for key in ['MW', 'logP', 'TPSA', 'QED', 'SA', 'RotB']:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in property_stats:
            print(f"{key}: {property_stats[mean_key]:.2f} ± {property_stats.get(std_key, 0):.2f}")
    
    print("\n--- Diversity Metrics ---")
    print(f"Diversity: {diversity_metrics.get('diversity', 0):.3f}")
    print(f"Uniqueness: {100*diversity_metrics.get('uniqueness', 0):.1f}%")
    print(f"Unique count: {diversity_metrics.get('unique_count', 0)}")
    
    print("\n--- Filter Pass Rates ---")
    print(f"Total valid for filtering: {filter_metrics.get('total_valid', 0)}")
    print(f"All filters passed: {filter_metrics.get('all_filters_count', 0)} ({100*filter_metrics.get('all_filters_rate', 0):.1f}%)")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate PROTAC Flow Matching results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to sampling results file')
    parser.add_argument('--config', type=str, default='configs/sample.yaml',
                        help='Path to config file (for filter settings)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save evaluation results')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load results
    print(f"Loading results from {args.results}...")
    data = load_results(args.results)
    molecules = data.get('molecules', [])
    
    print(f"Loaded {len(molecules)} molecules")
    
    # Evaluate validity
    print("Evaluating validity...")
    validity_metrics = evaluate_validity(molecules)
    
    # Evaluate properties (if RDKit mols available)
    print("Evaluating properties...")
    rdkit_mols = data.get('rdkit_mols', None)
    property_stats = evaluate_properties(molecules, rdkit_mols)
    
    # Evaluate diversity
    print("Evaluating diversity...")
    diversity_metrics = evaluate_diversity(molecules, rdkit_mols)
    
    # Evaluate filter pass rates
    print("Evaluating filter pass rates...")
    filter_metrics = evaluate_filter_pass_rate(molecules, config.get('filter', {}))
    
    # Print report
    print_evaluation_report(
        validity_metrics,
        property_stats,
        diversity_metrics,
        filter_metrics,
    )
    
    # Save results
    if args.output:
        results = {
            'validity': validity_metrics,
            'properties': property_stats,
            'diversity': diversity_metrics,
            'filters': filter_metrics,
        }
        torch.save(results, args.output)
        print(f"\nSaved evaluation results to {args.output}")


if __name__ == '__main__':
    main()
