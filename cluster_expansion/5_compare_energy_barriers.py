"""
Step 5: Compare energy barriers between CE and ML potentials (MACE/CHGNet)

This script:
1. Loads template structure (1512 atoms) and builds nearest neighbor list
2. Loads training AND test symbols from NPZ files separately
3. For each structure, randomly samples hopping events
4. Computes energy barriers using:
   - MACE or CHGNet (via energy_models.py)
   - Cluster Expansion (using processor.compute_property_change)
5. Saves comparison results for BOTH train and test sets

Usage:
    python 5_compare_energy_barriers.py --model chgnet --n_samples 100 --gpu_idx 5
    python 5_compare_energy_barriers.py --model mace --mace_model medium-omat-0 --n_samples 100 --gpu_idx 5
"""

import os
import sys
import argparse

# Add parent directory to path for energy_models import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Parse arguments FIRST, before any CUDA/torch imports
parser = argparse.ArgumentParser(description='Compare energy barriers between CE and ML potentials')
parser.add_argument('--model', type=str, default='chgnet', choices=['chgnet', 'mace'],
                    help='ML model to use (default: chgnet)')
parser.add_argument('--mace_model', type=str, default='medium-omat-0',
                    help='MACE foundation model name (only used when --model mace)')
parser.add_argument('--n_samples', type=int, default=100,
                    help='Number of structures to sample per dataset (default: 100)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for ML model (default: 32)')
parser.add_argument('--gpu_idx', type=int, default=5,
                    help='GPU index to use (default: 5)')
parser.add_argument('--ce_data_dir', type=str, default='./ce_data',
                    help='Directory containing CE data files')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')
parser.add_argument('--output_dir', type=str, default='./comparison_results',
                    help='Output directory (default: ./comparison_results)')
args = parser.parse_args()

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)

import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch
from ase import Atoms
from matscipy.neighbours import neighbour_list
from scipy.stats import spearmanr

from energy_models import create_energy_model


# =============================================================================
# Constants
# =============================================================================

ATOMIC_NUMBER_TO_SYMBOL = {
    3: 'Li', 9: 'F', 8: 'O', 13: 'Al', 24: 'Cr', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu'
}

SYMBOL_TO_ATOMIC_NUMBER = {v: k for k, v in ATOMIC_NUMBER_TO_SYMBOL.items()}

# Cation species (excluding Li which is vacancy)
CATION_SYMBOLS = ['Al', 'Cr', 'Fe', 'Co', 'Ni', 'Cu']
CATION_ATOMIC_NUMBERS = [13, 24, 26, 27, 28, 29]

# Spinel structure parameters
UNIT_CELL_SIZE = 8.5333518982

# Nearest neighbor factors (from utils.py)
NN_FACTOR_A = 0.454663336987   # (√3/4) * 1.05 for A-A / A-B neighbors
NN_FACTOR_B = 0.435307003734   # (√11/8) * 1.05 for B-B / B-A neighbors
NN_FACTOR_O = 0.371231060123   # (√2/4) * 1.05 for O-O neighbors


# =============================================================================
# Nearest Neighbor Builder
# =============================================================================

def build_nearest_neighbors(positions, cell, symbols):
    """
    Build nearest neighbor list for hopping events.

    For cations: vacancy (Li) can hop with neighboring real cations
    For oxygen: vacancy (F) can hop with neighboring O atoms

    Args:
        positions: Nx3 array of Cartesian positions
        cell: 3x3 cell matrix
        symbols: list of element symbols

    Returns:
        neighbors_dict: dict mapping atom_idx -> list of neighbor indices
    """
    n_atoms = len(positions)
    symbols = np.array(symbols)

    # Identify site types
    cation_mask = np.isin(symbols, ['Li'] + CATION_SYMBOLS)
    anion_mask = np.isin(symbols, ['O', 'F'])

    # Use the larger cutoff to capture all neighbors
    nn_distance_cation = max(NN_FACTOR_A, NN_FACTOR_B) * UNIT_CELL_SIZE
    nn_distance_oxygen = NN_FACTOR_O * UNIT_CELL_SIZE
    max_cutoff = max(nn_distance_cation, nn_distance_oxygen)

    # Build neighbor list
    i_list, j_list, d_list = neighbour_list(
        'ijd',
        positions=positions,
        cutoff=max_cutoff,
        cell=cell,
        pbc=[True, True, False]
    )

    # Filter valid pairs (i != j)
    valid = i_list != j_list
    i_list = i_list[valid]
    j_list = j_list[valid]
    d_list = d_list[valid]

    neighbors_dict = defaultdict(list)

    for i, j, d in zip(i_list, j_list, d_list):
        # Cation-cation neighbors
        if cation_mask[i] and cation_mask[j] and d <= nn_distance_cation:
            neighbors_dict[i].append(j)
        # Anion-anion neighbors
        elif anion_mask[i] and anion_mask[j] and d <= nn_distance_oxygen:
            neighbors_dict[i].append(j)

    return dict(neighbors_dict)


def find_hopping_candidates(symbols, neighbors_dict):
    """
    Find valid hopping candidates (vacancy-atom pairs).

    Returns:
        list of (vacancy_idx, atom_idx, atom_symbol) tuples
    """
    candidates = []
    symbols = np.array(symbols)

    for vac_idx, neighbors in neighbors_dict.items():
        vac_symbol = symbols[vac_idx]

        if vac_symbol == 'Li':  # Cation vacancy
            for neighbor_idx in neighbors:
                neighbor_symbol = symbols[neighbor_idx]
                if neighbor_symbol in CATION_SYMBOLS:
                    candidates.append((vac_idx, neighbor_idx, neighbor_symbol))

        elif vac_symbol == 'F':  # Oxygen vacancy
            for neighbor_idx in neighbors:
                neighbor_symbol = symbols[neighbor_idx]
                if neighbor_symbol == 'O':
                    candidates.append((vac_idx, neighbor_idx, neighbor_symbol))

    return candidates


# =============================================================================
# CE Calculator
# =============================================================================

class CEBarrierCalculator:
    """Calculate energy barriers using Cluster Expansion."""

    def __init__(self, ce_data_dir='./ce_data'):
        self.ce_data_dir = Path(ce_data_dir)
        self._load_data()

    def _load_data(self):
        """Load processor, ECIs, and mappings."""
        print("Loading CE data...")

        # Load processor
        with open(self.ce_data_dir / 'processor.pkl', 'rb') as f:
            self.processor = pickle.load(f)

        # Load ECIs
        self.ecis = np.load(self.ce_data_dir / 'ecis_L1.npy')
        self.processor.coefs = self.ecis

        # Load site mapping
        with open(self.ce_data_dir / 'struct_idx_for_proc_site.json', 'r') as f:
            self.struct_idx_for_proc_site = np.array(json.load(f))

        # Load species to code mapping
        with open(self.ce_data_dir / 'species_to_code.json', 'r') as f:
            self.species_to_code = json.load(f)

        # Build reverse mapping: struct_idx -> proc_idx
        self.proc_idx_for_struct_site = np.zeros(len(self.struct_idx_for_proc_site), dtype=np.int32)
        for proc_idx, struct_idx in enumerate(self.struct_idx_for_proc_site):
            self.proc_idx_for_struct_site[struct_idx] = proc_idx

        print(f"  ECIs: {len(self.ecis)} coefficients")
        print(f"  Sites: {len(self.struct_idx_for_proc_site)}")
        print(f"  Species to code: {self.species_to_code}")

    def symbols_to_occupancy(self, symbols):
        """Convert symbols array to occupancy array (processor order)."""
        symbols = np.array(symbols)
        reordered_symbols = symbols[self.struct_idx_for_proc_site]
        return np.array([self.species_to_code[s] for s in reordered_symbols], dtype=np.int32)

    def compute_energy_change(self, occupancy, vacancy_struct_idx, atom_struct_idx, atom_symbol):
        """
        Compute energy change for a hop using CE.

        Args:
            occupancy: Current occupancy array (processor order)
            vacancy_struct_idx: Vacancy index in structure order
            atom_struct_idx: Atom index in structure order
            atom_symbol: Symbol of the hopping atom

        Returns:
            float: Energy change (E_final - E_initial) in eV
        """
        # Convert structure indices to processor indices
        vac_proc_idx = self.proc_idx_for_struct_site[vacancy_struct_idx]
        atom_proc_idx = self.proc_idx_for_struct_site[atom_struct_idx]

        # Determine vacancy symbol based on atom type
        vac_symbol = 'Li' if atom_symbol in CATION_SYMBOLS else 'F'

        # Create flips: vacancy gets atom, atom gets vacancy
        flips = [
            (vac_proc_idx, self.species_to_code[atom_symbol]),   # vacancy -> atom
            (atom_proc_idx, self.species_to_code[vac_symbol])    # atom -> vacancy
        ]

        return self.processor.compute_property_change(occupancy, flips)


# =============================================================================
# ML Calculator
# =============================================================================

class MLBarrierCalculator:
    """Calculate energy barriers using ML potentials (MACE or CHGNet)."""

    def __init__(self, model_type='chgnet', mace_model='medium-omat-0', device='cuda:0'):
        self.model_type = model_type
        self.device = device

        print(f"Initializing {model_type} model...")
        if model_type == 'mace':
            self.model = create_energy_model('mace', device=device, model_name=mace_model)
        else:
            self.model = create_energy_model('chgnet', device=device)

        self.model_name = self.model.get_model_name()
        print(f"  Model: {self.model_name}")

    def create_atoms(self, positions, cell, symbols):
        """Create ASE Atoms object, filtering out vacancies."""
        symbols = np.array(symbols)

        # Filter out vacancies (Li and F)
        mask = ~np.isin(symbols, ['Li', 'F'])
        filtered_positions = positions[mask]
        filtered_symbols = symbols[mask]

        return Atoms(
            symbols=list(filtered_symbols),
            positions=filtered_positions,
            cell=cell,
            pbc=[True, True, False]
        )

    def compute_energy_change(self, positions, cell, symbols_initial, symbols_final, batch_size=32):
        """
        Compute energy change for a hop using ML potential.

        Args:
            positions: Nx3 array of positions
            cell: 3x3 cell matrix
            symbols_initial: Initial symbols list
            symbols_final: Final symbols list (after hop)
            batch_size: Batch size for energy computation

        Returns:
            float: Energy change (E_final - E_initial) in eV
        """
        atoms_initial = self.create_atoms(positions, cell, symbols_initial)
        atoms_final = self.create_atoms(positions, cell, symbols_final)

        # Compute energies in batch
        energies = self.model.predict_energy([atoms_initial, atoms_final], batch_size=batch_size)

        e_initial = energies[0]
        e_final = energies[1]

        if np.isnan(e_initial) or np.isnan(e_final):
            return None

        return e_final - e_initial


# =============================================================================
# Helper Functions
# =============================================================================

def compute_statistics(ce_barriers, ml_barriers):
    """Compute statistics for barrier comparison."""
    errors = ce_barriers - ml_barriers

    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))
    pearson_r = np.corrcoef(ce_barriers, ml_barriers)[0, 1]
    spearman_r, _ = spearmanr(ce_barriers, ml_barriers)

    return {
        'n_barriers': len(errors),
        'rmse_eV': float(rmse),
        'mae_eV': float(mae),
        'max_error_eV': float(max_err),
        'pearson_correlation': float(pearson_r),
        'spearman_correlation': float(spearman_r),
        'ml_barrier_mean': float(ml_barriers.mean()),
        'ml_barrier_std': float(ml_barriers.std()),
        'ce_barrier_mean': float(ce_barriers.mean()),
        'ce_barrier_std': float(ce_barriers.std())
    }


def print_statistics(stats, dataset_name):
    """Print statistics for a dataset."""
    print(f"\n{dataset_name} Results:")
    print(f"  Barriers computed: {stats['n_barriers']}")
    print(f"  ML barriers:  mean={stats['ml_barrier_mean']:.4f}, std={stats['ml_barrier_std']:.4f} eV")
    print(f"  CE barriers:  mean={stats['ce_barrier_mean']:.4f}, std={stats['ce_barrier_std']:.4f} eV")
    print(f"  RMSE:         {stats['rmse_eV']:.4f} eV ({stats['rmse_eV']*1000:.2f} meV)")
    print(f"  MAE:          {stats['mae_eV']:.4f} eV ({stats['mae_eV']*1000:.2f} meV)")
    print(f"  Max error:    {stats['max_error_eV']:.4f} eV")
    print(f"  Pearson R:    {stats['pearson_correlation']:.4f}")
    print(f"  Spearman R:   {stats['spearman_correlation']:.4f}")


def process_dataset(all_atomic_numbers, template_positions, template_cell,
                    neighbors_dict, ce_calc, ml_calc, n_samples, batch_size,
                    dataset_name, seed_offset=0):
    """
    Process a dataset (train or test) and compute barriers.

    Args:
        all_atomic_numbers: (N_structures, N_atoms) array
        template_positions: (N_atoms, 3) array
        template_cell: (3, 3) array
        neighbors_dict: dict of neighbor lists
        ce_calc: CEBarrierCalculator instance
        ml_calc: MLBarrierCalculator instance
        n_samples: number of structures to sample
        batch_size: batch size for ML computation
        dataset_name: 'train' or 'test'
        seed_offset: offset for random seed

    Returns:
        results: list of result dicts
        ce_barriers: numpy array
        ml_barriers: numpy array
    """
    np.random.seed(args.seed + seed_offset)

    n_structures = len(all_atomic_numbers)
    sample_indices = np.random.choice(
        n_structures,
        size=min(n_samples, n_structures),
        replace=False
    )

    results = []
    ce_barriers = []
    ml_barriers = []

    for struct_idx in tqdm(sample_indices, desc=f"Processing {dataset_name}"):
        atomic_numbers = all_atomic_numbers[struct_idx]
        symbols = [ATOMIC_NUMBER_TO_SYMBOL[z] for z in atomic_numbers]

        # Get occupancy for CE
        occupancy = ce_calc.symbols_to_occupancy(symbols)

        # Find hopping candidates for this structure
        candidates = find_hopping_candidates(symbols, neighbors_dict)

        if len(candidates) == 0:
            continue

        # Randomly sample one hopping event
        vac_idx, atom_idx, atom_symbol = candidates[np.random.randint(len(candidates))]

        # Compute CE barrier
        ce_barrier = ce_calc.compute_energy_change(
            occupancy, vac_idx, atom_idx, atom_symbol
        )

        # Create final symbols (after hop)
        symbols_final = symbols.copy()
        symbols_final[vac_idx] = atom_symbol
        symbols_final[atom_idx] = 'Li' if atom_symbol in CATION_SYMBOLS else 'F'

        # Compute ML barrier
        ml_barrier = ml_calc.compute_energy_change(
            template_positions, template_cell,
            symbols, symbols_final,
            batch_size=batch_size
        )

        if ml_barrier is not None:
            ce_barriers.append(ce_barrier)
            ml_barriers.append(ml_barrier)

            results.append({
                'structure_idx': int(struct_idx),
                'vacancy_idx': int(vac_idx),
                'atom_idx': int(atom_idx),
                'atom_symbol': atom_symbol,
                'ce_barrier': float(ce_barrier),
                'ml_barrier': float(ml_barrier),
                'error': float(ce_barrier - ml_barrier),
                'dataset': dataset_name
            })

    return results, np.array(ce_barriers), np.array(ml_barriers)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Energy Barrier Comparison: CE vs ML Potential")
    print("=" * 60)
    print(f"ML Model: {args.model}")
    if args.model == 'mace':
        print(f"MACE Model: {args.mace_model}")
    print(f"Samples per dataset: {args.n_samples}")
    print(f"GPU: {args.gpu_idx}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # =========================================================================
    # 1. Load template structure and both datasets
    # =========================================================================
    print("\n[1] Loading structures...")

    local_struct_dir = Path('./local_structures')

    # Load train data
    train_file = local_struct_dir / 'train_atomic_numbers.npz'
    train_data = np.load(train_file)
    train_atomic_numbers = train_data['atomic_numbers']
    template_positions = train_data['positions']
    template_cell = train_data['cell']

    # Load test data
    test_file = local_struct_dir / 'test_atomic_numbers.npz'
    test_data = np.load(test_file)
    test_atomic_numbers = test_data['atomic_numbers']

    n_atoms = len(template_positions)

    print(f"  Train structures: {len(train_atomic_numbers)}")
    print(f"  Test structures:  {len(test_atomic_numbers)}")
    print(f"  Atoms per structure: {n_atoms}")
    print(f"  Cell: {template_cell[0,0]:.2f} x {template_cell[1,1]:.2f} x {template_cell[2,2]:.2f} Å")

    # =========================================================================
    # 2. Build nearest neighbors using first structure as reference
    # =========================================================================
    print("\n[2] Building nearest neighbor list...")

    ref_atomic_numbers = train_atomic_numbers[0]
    ref_symbols = [ATOMIC_NUMBER_TO_SYMBOL[z] for z in ref_atomic_numbers]

    neighbors_dict = build_nearest_neighbors(template_positions, template_cell, ref_symbols)

    total_neighbors = sum(len(v) for v in neighbors_dict.values())
    print(f"  Sites with neighbors: {len(neighbors_dict)}")
    print(f"  Total neighbor pairs: {total_neighbors}")

    # =========================================================================
    # 3. Initialize calculators
    # =========================================================================
    print("\n[3] Initializing calculators...")

    ce_calc = CEBarrierCalculator(ce_data_dir=args.ce_data_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.model == 'mace':
        ml_calc = MLBarrierCalculator(model_type='mace', mace_model=args.mace_model, device=device)
    else:
        ml_calc = MLBarrierCalculator(model_type='chgnet', device=device)

    # =========================================================================
    # 4. Process TRAIN dataset
    # =========================================================================
    print("\n[4] Computing energy barriers on TRAIN set...")

    train_results, train_ce_barriers, train_ml_barriers = process_dataset(
        train_atomic_numbers, template_positions, template_cell,
        neighbors_dict, ce_calc, ml_calc,
        args.n_samples, args.batch_size, 'train', seed_offset=0
    )

    train_stats = compute_statistics(train_ce_barriers, train_ml_barriers)
    print_statistics(train_stats, "TRAIN")

    # =========================================================================
    # 5. Process TEST dataset
    # =========================================================================
    print("\n[5] Computing energy barriers on TEST set...")

    test_results, test_ce_barriers, test_ml_barriers = process_dataset(
        test_atomic_numbers, template_positions, template_cell,
        neighbors_dict, ce_calc, ml_calc,
        args.n_samples, args.batch_size, 'test', seed_offset=1000
    )

    test_stats = compute_statistics(test_ce_barriers, test_ml_barriers)
    print_statistics(test_stats, "TEST")

    # =========================================================================
    # 6. Print summary comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: TRAIN vs TEST")
    print("=" * 60)
    print(f"{'Metric':<20} {'TRAIN':<15} {'TEST':<15}")
    print("-" * 50)
    print(f"{'N barriers':<20} {train_stats['n_barriers']:<15} {test_stats['n_barriers']:<15}")
    print(f"{'RMSE (meV)':<20} {train_stats['rmse_eV']*1000:<15.2f} {test_stats['rmse_eV']*1000:<15.2f}")
    print(f"{'MAE (meV)':<20} {train_stats['mae_eV']*1000:<15.2f} {test_stats['mae_eV']*1000:<15.2f}")
    print(f"{'Max error (meV)':<20} {train_stats['max_error_eV']*1000:<15.2f} {test_stats['max_error_eV']*1000:<15.2f}")
    print(f"{'Pearson R':<20} {train_stats['pearson_correlation']:<15.4f} {test_stats['pearson_correlation']:<15.4f}")
    print(f"{'Spearman R':<20} {train_stats['spearman_correlation']:<15.4f} {test_stats['spearman_correlation']:<15.4f}")

    # =========================================================================
    # 7. Save results
    # =========================================================================
    print("\n[6] Saving results...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_suffix = args.model if args.model == 'chgnet' else f"mace_{args.mace_model.replace('-', '_')}"

    # Combine all results
    all_results = train_results + test_results

    # Save JSON results
    output_data = {
        'config': {
            'ml_model': args.model,
            'mace_model': args.mace_model if args.model == 'mace' else None,
            'n_samples': args.n_samples,
            'seed': args.seed,
            'n_atoms': n_atoms
        },
        'train_statistics': train_stats,
        'test_statistics': test_stats,
        'barriers': all_results
    }

    json_file = output_dir / f'barrier_comparison_{model_suffix}.json'
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  JSON: {json_file}")

    # Save numpy arrays (separate for train/test)
    npz_file = output_dir / f'barrier_data_{model_suffix}.npz'
    np.savez(
        npz_file,
        train_ce_barriers=train_ce_barriers,
        train_ml_barriers=train_ml_barriers,
        train_errors=train_ce_barriers - train_ml_barriers,
        test_ce_barriers=test_ce_barriers,
        test_ml_barriers=test_ml_barriers,
        test_errors=test_ce_barriers - test_ml_barriers
    )
    print(f"  NPZ:  {npz_file}")

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
