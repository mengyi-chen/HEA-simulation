"""
Step 1: Compute energies for local environments using ML potentials

This script supports multiple ML models via energy_models.py:
- CHGNet: Uses native batch computation
- MACE: Uses torch-sim for GPU batched computation

Usage:
    python 1_compute_energies.py --model chgnet --batch_size 32 --gpu_idx 4
    python 1_compute_energies.py --model mace --mace_model medium-omat-0 --batch_size 16 --gpu_idx 5
"""

import os
import sys
import argparse

# Parse arguments FIRST, before any CUDA/torch imports
parser = argparse.ArgumentParser(description='Compute energies with ML potentials')
parser.add_argument('--model', type=str, default='chgnet', choices=['chgnet', 'mace'],
                    help='ML model to use (default: chgnet)')
parser.add_argument('--mace_model', type=str, default='medium-omat-0',
                    help='MACE foundation model name (only used when --model mace)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for GPU computation (default: 32)')
parser.add_argument('--gpu_idx', type=int, default=5,
                    help='GPU index to use (default: 5)')
args = parser.parse_args()

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)

# Add parent directory to path for energy_models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import torch and other dependencies
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import torch
from ase import Atoms

from energy_models import create_energy_model


def load_structures_from_npz(npz_path, dataset_name="dataset"):
    """Load structures from NPZ file created in Step 0.

    This file contains:
    - atomic_numbers: array of shape (n_structures, n_atoms) with atomic numbers
    - positions: template positions array (shared across all structures)
    - cell: template cell array (shared across all structures)

    Args:
        npz_path: Path to NPZ file
        dataset_name: Name for progress bar (e.g., "train", "test")

    Returns:
        atoms_list: List of ASE Atoms objects
        metadata_list: List of metadata dicts with structure_idx and n_atoms
    """
    npz_path = Path(npz_path)

    print(f"Loading {dataset_name} structures from: {npz_path}")
    data = np.load(npz_path)

    atomic_numbers_bulk = data['atomic_numbers']  # Shape: (n_structures, n_atoms)
    template_positions = data['positions']         # Shape: (n_atoms, 3)
    template_cell = data['cell']                   # Shape: (3, 3)

    print(f"  Atomic numbers shape: {atomic_numbers_bulk.shape}")
    print(f"  Template positions shape: {template_positions.shape}")
    print(f"  Template cell shape: {template_cell.shape}")

    atoms_list = []
    metadata_list = []

    for idx in tqdm(range(len(atomic_numbers_bulk)), desc=f"Creating {dataset_name} ASE Atoms"):
        atomic_numbers = atomic_numbers_bulk[idx]

        # Skip empty structures (all zeros)
        if np.all(atomic_numbers == 0):
            continue

        # Filter out Li (3) and F (9) - ghost atoms used for vacancy markers
        mask = (atomic_numbers != 3) & (atomic_numbers != 9)
        filtered_atomic_numbers = atomic_numbers[mask]
        filtered_positions = template_positions[mask]

        # Skip if no atoms remain after filtering
        if len(filtered_atomic_numbers) == 0:
            continue

        # Convert atomic numbers to Python ints to avoid numpy int32 issues
        atoms = Atoms(
            numbers=[int(z) for z in filtered_atomic_numbers],
            positions=filtered_positions.copy(),
            cell=template_cell.copy(),
            pbc=[True, True, False]
        )
        atoms_list.append(atoms)

        metadata_list.append({
            'structure_idx': idx,  # Index within this dataset (train or test)
            'n_atoms': int(len(filtered_atomic_numbers)),
        })

    return atoms_list, metadata_list


def compute_energies_batched(atoms_list, energy_model, batch_size=32):
    """Compute energies using the energy model with batched computation.

    Args:
        atoms_list: List of ASE Atoms objects
        energy_model: EnergyModel instance from energy_models.py
        batch_size: Batch size for computation

    Returns:
        List of energies (eV)
    """
    print(f"Computing energies for {len(atoms_list)} structures in batches of {batch_size}...")

    energies = []
    n_batches = (len(atoms_list) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Computing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(atoms_list))
        batch = atoms_list[start_idx:end_idx]

        batch_energies = energy_model.predict_energy(batch, batch_size=batch_size)

        for e in batch_energies:
            if np.isnan(e):
                energies.append(None)
            else:
                energies.append(float(e))

    return energies


def process_dataset(npz_file, output_file, dataset_name, energy_model, batch_size):
    """Process a single dataset (train or test) and save energies.

    Args:
        npz_file: Path to input NPZ file
        output_file: Path to output JSON file
        dataset_name: Name of dataset ("train" or "test")
        energy_model: EnergyModel instance
        batch_size: Batch size for GPU computation

    Returns:
        dict with statistics
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")

    atoms_list, metadata_list = load_structures_from_npz(npz_file, dataset_name)
    print(f"Loaded {len(atoms_list)} {dataset_name} structures")

    if len(atoms_list) == 0:
        print(f"No structures to process for {dataset_name}")
        return None

    model_name = energy_model.get_model_name()

    energies = compute_energies_batched(
        atoms_list,
        energy_model,
        batch_size=batch_size
    )

    results = []
    valid_count = 0
    for meta, energy in zip(metadata_list, energies):
        meta['energy'] = energy
        meta['model'] = model_name
        results.append(meta)
        if energy is not None:
            valid_count += 1

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    valid_energies = [r['energy'] for r in results if r['energy'] is not None]

    stats = {
        'dataset': dataset_name,
        'total': len(results),
        'valid': valid_count,
        'output_file': str(output_file)
    }

    if valid_energies:
        stats['energy_min'] = float(np.min(valid_energies))
        stats['energy_max'] = float(np.max(valid_energies))
        stats['energy_mean'] = float(np.mean(valid_energies))
        stats['energy_std'] = float(np.std(valid_energies))

    print(f"\n{dataset_name.upper()} results:")
    print(f"  Total structures: {len(results)}")
    print(f"  Successfully computed: {valid_count}")
    if valid_energies:
        print(f"  Energy range: [{np.min(valid_energies):.4f}, {np.max(valid_energies):.4f}] eV")
        print(f"  Energy mean: {np.mean(valid_energies):.4f} ± {np.std(valid_energies):.4f} eV")
    print(f"  Saved to: {output_file}")

    return stats


def main():
    local_struct_dir = Path('./local_structures')

    # After setting CUDA_VISIBLE_DEVICES, use cuda:0 since only one GPU is visible
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("Computing Energies with ML Potentials")
    print("="*60)
    print(f"Model type: {args.model}")
    if args.model == 'mace':
        print(f"MACE model: {args.mace_model}")
    print(f"Using GPU: {args.gpu_idx} (via CUDA_VISIBLE_DEVICES)")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("="*60)

    # Create energy model
    print(f"\nInitializing {args.model} model...")
    if args.model == 'chgnet':
        energy_model = create_energy_model('chgnet', device=device)
        model_suffix = 'chgnet'
        output_dir = Path('./chgnet_energies')
    else:  # mace
        energy_model = create_energy_model('mace', device=device, model_name=args.mace_model)
        model_suffix = args.mace_model.replace('-', '_').replace('.', '_')
        output_dir = Path('./mace_energies')

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = energy_model.get_model_name()
    print(f"Model initialized: {model_name}")

    all_stats = []

    # Process train dataset
    train_file = local_struct_dir / 'train_atomic_numbers.npz'
    if train_file.exists():
        train_output = output_dir / f'train_energies_{model_suffix}.json'
        train_stats = process_dataset(
            train_file, train_output, "train",
            energy_model, args.batch_size
        )
        if train_stats:
            all_stats.append(train_stats)
    else:
        print(f"\nWarning: Train file not found: {train_file}")

    # Process test dataset
    test_file = local_struct_dir / 'test_atomic_numbers.npz'
    if test_file.exists():
        test_output = output_dir / f'test_energies_{model_suffix}.json'
        test_stats = process_dataset(
            test_file, test_output, "test",
            energy_model, args.batch_size
        )
        if test_stats:
            all_stats.append(test_stats)
    else:
        print(f"\nWarning: Test file not found: {test_file}")

    # Save summary statistics
    summary_file = output_dir / f'energy_stats_{model_suffix}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'model': model_name,
            'model_type': args.model,
            'mace_model': args.mace_model if args.model == 'mace' else None,
            'batch_size': args.batch_size,
            'datasets': all_stats
        }, f, indent=2)

    print()
    print("="*60)
    print("Energy Computation Complete!")
    print("="*60)
    print(f"Model: {model_name}")
    for stats in all_stats:
        print(f"  {stats['dataset']}: {stats['valid']}/{stats['total']} structures")
    print(f"\nOutput directory: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    print("="*60)


if __name__ == '__main__':
    main()