"""
Step 0: Extract local cubic environments centered at vacancies from 24x24x24 supercells
PARALLELIZED VERSION - Uses multiprocessing for significant speedup

This script:
1. Reads VASP files from configs_2025_11_26_14_20_16/
2. Identifies vacancies (X for cation vacancy, XO for oxygen vacancy)
3. For each vacancy, extracts local cubic environment with proper wrapping
4. Saves structures in NPZ format for fast loading with torch-sim:
   - X (cation vacancy) -> Li (placeholder for cluster expansion)
   - XO (oxygen vacancy) -> F (placeholder for cluster expansion)
5. Outputs local structures ready for MACE energy calculation

All structures are saved to a single folder in NPZ format for efficient loading.

Usage:
    python 0_extract_local_envs_parallel.py --n_workers 16
"""

import numpy as np
from pymatgen.core import Structure, Lattice
from pathlib import Path
import json
from tqdm import tqdm
import os
from ase.io import read
from io import StringIO
import multiprocessing as mp
from functools import partial
import argparse

# Atomic number mapping
ATOMIC_NUMBERS = {
    'X': 0, 'Li': 3, 'Al': 13, 'Cr': 24, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'O': 8, 'F': 9, 'XO': 0
}

# Atomic numbers for random structure generation
CATION_ATOMIC_NUMBERS = [3, 13, 24, 26, 27, 28, 29]  # Li(vacancy), Al, Cr, Fe, Co, Ni, Cu
OXYGEN_ATOMIC_NUMBERS = [8, 9]  # O, F(vacancy)

# Parameters matching utils.py
N_CUTOFF_CELL = 1  # Number of unit cells for cutoff radius
UNIT_CELL_SIZE = 8.5333518982  # Unit cell lattice constant in Angstrom
CUTOFF = N_CUTOFF_CELL * UNIT_CELL_SIZE  # ~8.533 Å


def read_poscar_with_custom_symbols(poscar_path, custom_symbol_map=None):
    """Read POSCAR file with support for custom element symbols"""
    if custom_symbol_map is None:
        custom_symbol_map = {'XO': 'He'}

    with open(poscar_path, 'r') as f:
        lines = f.readlines()

    reverse_map = {v: k for k, v in custom_symbol_map.items()}
    modified_lines = lines.copy()

    for custom_sym, placeholder in custom_symbol_map.items():
        if len(lines) > 5:
            modified_lines[5] = modified_lines[5].replace(custom_sym, placeholder)

    poscar_str = ''.join(modified_lines)
    atoms = read(StringIO(poscar_str), format='vasp')

    cell = atoms.cell.array
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

    for placeholder, custom_sym in reverse_map.items():
        symbols = [custom_sym if s == placeholder else s for s in symbols]

    return cell, positions, symbols


def load_structure_with_vacancies_replaced(poscar_path):
    """Load structure and immediately replace X->Li, XO->F for pymatgen compatibility"""
    cell, positions, symbols = read_poscar_with_custom_symbols(poscar_path)
    
    vacancy_map = {}
    for i, sym in enumerate(symbols):
        if sym == 'X':
            vacancy_map[i] = 'X'
        elif sym == 'XO':
            vacancy_map[i] = 'XO'
    
    symbols_replaced = []
    for sym in symbols:
        if sym == 'X':
            symbols_replaced.append('Li')
        elif sym == 'XO':
            symbols_replaced.append('F')
        else:
            symbols_replaced.append(sym)
    
    structure = Structure(
        lattice=Lattice(cell),
        species=symbols_replaced,
        coords=positions,
        coords_are_cartesian=False
    )
    
    return structure, vacancy_map


def get_vacancy_indices(vacancy_map):
    """Get vacancy indices from vacancy map."""
    cation_vacancies = [idx for idx, vtype in vacancy_map.items() if vtype == 'X']
    oxygen_vacancies = [idx for idx, vtype in vacancy_map.items() if vtype == 'XO']
    return cation_vacancies, oxygen_vacancies


def get_local_cubic_environment_centered_at_vacancy(structure, vac_idx, vac_symbol, cutoff=CUTOFF):
    """Extract a local unit-cell-based environment centered at a vacancy.

    Uses 3x3x3 unit cells. Z direction: if vac_cell_idx[2]==0, use [0,1,2]; otherwise use [-1,0,1] relative to center.

    Returns:
        positions: Nx3 Cartesian positions (vacancy at origin)
        symbols: List of element symbols
        cell: 3x3 cell matrix (always 3x3x3 unit cells)
        n_real_atoms: Number of non-vacancy atoms
        vac_cluster_idx: Index of vacancy in the local cluster
    """
    pos_cart = structure.cart_coords
    pos_frac = structure.frac_coords
    vac_pos_cart = pos_cart[vac_idx]
    vac_pos_frac = pos_frac[vac_idx]

    cell = structure.lattice.matrix

    # Detect supercell size
    supercell_size = np.array([
        int(round(cell[0, 0] / UNIT_CELL_SIZE)),
        int(round(cell[1, 1] / UNIT_CELL_SIZE)),
        int(round(cell[2, 2] / UNIT_CELL_SIZE))
    ])

    # Determine which unit cell the vacancy belongs to
    vac_cell_idx = np.floor(vac_pos_frac * supercell_size + 1e-6).astype(int)

    # Pre-compute unit cell indices for all atoms
    all_cell_indices = np.floor(pos_frac * supercell_size + 1e-6).astype(int)

    # XY direction: 3x3 grid centered on vacancy cell (with PBC)
    x_cells = [(vac_cell_idx[0] + offset) % supercell_size[0] for offset in [-1, 0, 1]]
    y_cells = [(vac_cell_idx[1] + offset) % supercell_size[1] for offset in [-1, 0, 1]]

    # Z direction: always 3 cells (3x3x3 local environment)
    # If vac_cell_idx[2] == 0 (bottom), use [0, 1, 2]
    # If vac_cell_idx[2] == supercell_size[2] - 1 (top), use [-2, -1, 0]
    # Otherwise use [-1, 0, 1] relative to center
    if vac_cell_idx[2] == 0:
        z_offset = [0, 1, 2]
    elif vac_cell_idx[2] == supercell_size[2] - 1:
        z_offset = [-2, -1, 0]
    else:
        z_offset = [-1, 0, 1]
    z_cells = [vac_cell_idx[2] + offset for offset in z_offset]

    # Filter atoms in selected cells
    in_x_cells = np.isin(all_cell_indices[:, 0], x_cells)
    in_y_cells = np.isin(all_cell_indices[:, 1], y_cells)
    in_z_cells = np.isin(all_cell_indices[:, 2], z_cells)
    in_cells = in_x_cells & in_y_cells & in_z_cells

    indices_in_cube = np.where(in_cells)[0]

    cluster_pos = pos_cart[indices_in_cube]
    cluster_symbols = [str(structure[i].specie) for i in indices_in_cube]

    # Unwrap positions in XY direction (handle PBC) and shift to vacancy-centered coordinates
    wrapped_pos = cluster_pos.copy()
    for i in range(len(wrapped_pos)):
        delta = wrapped_pos[i] - vac_pos_cart
        # Unwrap in x and y directions only (apply PBC)
        delta[0] -= np.round(delta[0] / cell[0, 0]) * cell[0, 0]
        delta[1] -= np.round(delta[1] / cell[1, 1]) * cell[1, 1]
        wrapped_pos[i] = delta  # Position relative to vacancy (vacancy at origin)

    # Always 3x3x3 local cell
    n_cells = 3
    local_cell = np.array([
        [n_cells * UNIT_CELL_SIZE, 0.0, 0.0],
        [0.0, n_cells * UNIT_CELL_SIZE, 0.0],
        [0.0, 0.0, n_cells * UNIT_CELL_SIZE]
    ])

    # Shift positions so all atoms are inside the local cell
    # Simply shift by -min to make minimum coordinate = 0
    min_pos = wrapped_pos.min(axis=0)
    wrapped_pos = wrapped_pos - min_pos

    # CRITICAL: Sort atoms by position (x, y, z) to match ClusterExpansionProcessor ordering
    # Processor uses XYZ ordering: x slowest, z fastest (same as C-order for 3D array)
    # np.lexsort sorts by LAST key first, so we pass (z, y, x) to get x-primary sorting
    sort_keys = np.round(wrapped_pos / 1e-4).astype(int)  # Round to 1e-4 angstrom precision
    sort_indices = np.lexsort((sort_keys[:, 2], sort_keys[:, 1], sort_keys[:, 0]))

    # Apply sorting
    wrapped_pos = wrapped_pos[sort_indices]
    cluster_symbols = [cluster_symbols[i] for i in sort_indices]

    # Update vacancy index after sorting
    old_vac_cluster_idx = np.where(indices_in_cube == vac_idx)[0][0]
    vac_cluster_idx = np.where(sort_indices == old_vac_cluster_idx)[0][0]

    # Count non-vacancy atoms
    n_real_atoms = sum(1 for s in cluster_symbols if s not in ['Li', 'F'])

    return wrapped_pos, cluster_symbols, local_cell, n_real_atoms, vac_cluster_idx


def process_single_file(vasp_file, output_dir, cutoff, max_vac_per_file=50, min_atoms=0,
                       template_positions=None, template_cell=None, save_template=False, n_templates=10):
    """Process a single VASP file and extract local environments.

    Args:
        vasp_file: Path to VASP file
        output_dir: Output directory for NPZ files
        cutoff: Cutoff radius for local environment extraction
        max_vac_per_file: Maximum vacancies to process per file
        min_atoms: Skip structures with <= this many atoms (after removing Li/F)
        template_positions: Template positions (if already extracted)
        template_cell: Template cell (if already extracted)
        save_template: Whether to save individual template files
        n_templates: Number of template structures to save
    """
    vasp_file = Path(vasp_file)
    output_dir = Path(output_dir)
    step_num = vasp_file.stem.split('_')[-1]
    
    structure, vacancy_map = load_structure_with_vacancies_replaced(str(vasp_file))
    
    cation_vac_indices, oxygen_vac_indices = get_vacancy_indices(vacancy_map)
    n_cation_vac = len(cation_vac_indices)
    n_oxygen_vac = len(oxygen_vac_indices)
    
    all_vac_indices = cation_vac_indices + oxygen_vac_indices
    
    if len(all_vac_indices) > max_vac_per_file:
        np.random.seed(int(step_num) if step_num.isdigit() else hash(step_num) % (2**31))
        all_vac_indices = np.random.choice(all_vac_indices, size=max_vac_per_file, replace=False).tolist()
    
    file_metadata = []
    all_atomic_numbers = []
    first_positions = None
    first_cell = None
    
    for idx, vac_idx in enumerate(all_vac_indices):
        vac_symbol = vacancy_map[vac_idx]
        
        
        positions, symbols, cell, n_real_atoms, vac_local_idx = get_local_cubic_environment_centered_at_vacancy(
            structure, vac_idx, vac_symbol, cutoff=cutoff
        )
        
        # Convert symbols to atomic numbers
        atomic_numbers = np.array([ATOMIC_NUMBERS.get(s, 0) for s in symbols], dtype=np.int32)
        
        # Store first positions and cell as template
        if first_positions is None:
            first_positions = positions.astype(np.float32)
            first_cell = cell.astype(np.float32)
        
        # Verify that positions and cell are the same for all structures
        if not np.allclose(positions, first_positions, atol=1e-6):
            print(f"Warning: positions differ for step{step_num}_vac{vac_idx}_{vac_symbol}")
        if not np.allclose(cell, first_cell, atol=1e-6):
            print(f"Warning: cell differs for step{step_num}_vac{vac_idx}_{vac_symbol}")
        
        # Count non-vacancy atoms
        n_real_atoms = sum(1 for s in symbols if s not in ['Li', 'F'])
        
        # Skip if structure has too few atoms
        if n_real_atoms <= min_atoms:
            continue

        # Save first n_templates structures as individual templates for testing
        if save_template and idx < n_templates:
            output_file = output_dir / f"template_{idx}_step{step_num}_vac{vac_idx}_{vac_symbol}.npz"
            np.savez_compressed(
                output_file,
                # Full structure (with Li/F for cluster expansion)
                positions=positions.astype(np.float32),
                atomic_numbers=atomic_numbers,
                cell=cell.astype(np.float32),
                symbols=np.array(symbols, dtype='U2'),
                # Metadata
                vacancy_type=vac_symbol,
                vacancy_local_idx=vac_local_idx,
                n_atoms_total=len(positions),
                n_real_atoms=n_real_atoms
            )
        
        # Store atomic numbers for bulk storage
        all_atomic_numbers.append(atomic_numbers)
        
        file_metadata.append({
            'step': step_num,
            'vacancy_idx': int(vac_idx),
            'vacancy_type': vac_symbol,
            'vacancy_local_idx': int(vac_local_idx),
            'n_atoms_total': len(positions),
            'n_real_atoms': n_real_atoms
        })
            

    
    return file_metadata, n_cation_vac, n_oxygen_vac, all_atomic_numbers, (first_positions, first_cell)


def process_file_wrapper(args):
    """Wrapper for multiprocessing - unpacks arguments"""
    vasp_file, output_dir, cutoff, max_vac_per_file, min_atoms, save_template, n_templates = args
    return process_single_file(vasp_file, output_dir, cutoff, max_vac_per_file, min_atoms,
                               save_template=save_template, n_templates=n_templates)


def main():
    parser = argparse.ArgumentParser(description='Extract local environments (parallel version)')
    parser.add_argument('--n_workers', type=int, default=64,
                        help='Number of parallel workers (default: number of CPUs)')
    parser.add_argument('--max_vac_per_file', type=int, default=50,
                        help='Maximum vacancies to sample per file (default: 50)')
    parser.add_argument('--min_atoms', type=int, default=100,
                        help='Skip structures with <= this many atoms (default: 100)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split (default: 42)')
    parser.add_argument('--n_random', type=int, default=5000,
                        help='Number of random structures to generate (default: 0, disabled)')
    parser.add_argument('--vac_pct_min', type=float, default=0.1,
                        help='Minimum vacancy percentage for random structures (default: 0.1)')
    parser.add_argument('--vac_pct_max', type=float, default=0.8,
                        help='Maximum vacancy percentage for random structures (default: 0.8)')
    parser.add_argument('--n_templates', type=int, default=10,
                        help='Number of template structures to save (default: 10)')
    args = parser.parse_args()
    
    configs_dir = Path('../kmc_result/configs_2025_11_26_14_20_16')
    output_dir = Path('./local_structures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cutoff = CUTOFF
    max_vac_per_file = args.max_vac_per_file
    min_atoms = args.min_atoms
    n_workers = args.n_workers or mp.cpu_count()
    test_ratio = args.test_ratio
    seed = args.seed
    n_random = args.n_random
    vac_pct_min = args.vac_pct_min
    vac_pct_max = args.vac_pct_max
    n_templates = args.n_templates

    print("="*60)
    print("Extracting Local Environments (PARALLEL VERSION - OPTIMIZED)")
    print("="*60)
    print(f"Number of workers: {n_workers}")
    print(f"Cutoff: {cutoff:.4f} Å")
    print(f"Max vacancies per file: {max_vac_per_file}")
    print(f"Min atoms filter: skip structures with <= {min_atoms} atoms")
    print(f"Vacancy replacement: X -> Li, XO -> F")
    print(f"Train/test split: {1-test_ratio:.0%} train, {test_ratio:.0%} test (seed={seed})")
    if n_random > 0:
        print(f"Random structures: {n_random} (vacancy {vac_pct_min*100:.0f}%-{vac_pct_max*100:.0f}%)")
    print(f"Output: Bulk atomic_numbers + {n_templates} template structures")
    print("="*60)
    print()

    vasp_files = sorted(configs_dir.glob('POSCAR_step_*.vasp'))
    print(f"Found {len(vasp_files)} VASP files")
    print()

    # Prepare arguments for parallel processing
    # Only save templates for first file
    task_args = [(str(f), str(output_dir), cutoff, max_vac_per_file, min_atoms, i == 0, n_templates)
                 for i, f in enumerate(vasp_files)]
    
    all_metadata = []
    all_atomic_numbers_list = []
    total_cation_vac = 0
    total_oxygen_vac = 0
    errors = []
    template_positions = None
    template_cell = None
    
    # Use multiprocessing pool
    with mp.Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_file_wrapper, task_args),
            total=len(vasp_files),
            desc="Processing VASP files"
        ))
    
    # Collect results
    for file_metadata, n_cation, n_oxygen, atomic_numbers_list, template_info in results:
        all_metadata.extend(file_metadata)
        all_atomic_numbers_list.extend(atomic_numbers_list)
        total_cation_vac += n_cation
        total_oxygen_vac += n_oxygen
        
        # Store template from first successful file
        if template_positions is None and template_info[0] is not None:
            template_positions, template_cell = template_info
    
    # Generate random structures if requested
    n_extracted = len(all_atomic_numbers_list)
    print(f"\nStructures extracted from VASP files: {n_extracted}")

    n_random_generated = 0
    if n_random > 0 and len(all_atomic_numbers_list) > 0:
        print(f"\nGenerating {n_random} random structures...")
        random_structures = []
        np.random.seed(seed + 1000)
        template = all_atomic_numbers_list[0]
        n_atoms = len(template)

        cation_site_mask = np.isin(template, CATION_ATOMIC_NUMBERS)
        oxygen_site_mask = np.isin(template, OXYGEN_ATOMIC_NUMBERS)
        cation_indices = np.where(cation_site_mask)[0]
        oxygen_indices = np.where(oxygen_site_mask)[0]
        n_cation_sites = len(cation_indices)
        n_oxygen_sites = len(oxygen_indices)
        real_cations = [13, 24, 26, 27, 28, 29]  # Al, Cr, Fe, Co, Ni, Cu

        print(f"  Cation sites: {n_cation_sites}, Oxygen sites: {n_oxygen_sites}")

        random_n_real_atoms = []  # Track n_real_atoms for each random structure
        for i in range(n_random):
            new_structure = np.zeros(n_atoms, dtype=np.int32)

            # Generate random vacancy percentage
            cation_vac_pct = np.random.uniform(vac_pct_min, vac_pct_max)
            oxygen_vac_pct = np.random.uniform(vac_pct_min, vac_pct_max)

            # Cation assignments
            n_cation_vacancies = int(n_cation_sites * cation_vac_pct)
            n_real_cations = n_cation_sites - n_cation_vacancies
            cation_types = np.random.choice(real_cations, size=n_real_cations, replace=True)
            cation_assignments = np.concatenate([
                cation_types,
                np.full(n_cation_vacancies, 3, dtype=np.int32)
            ])
            np.random.shuffle(cation_assignments)

            # Oxygen assignments
            n_oxygen_vacancies = int(n_oxygen_sites * oxygen_vac_pct)
            n_real_oxygens = n_oxygen_sites - n_oxygen_vacancies
            oxygen_assignments = np.concatenate([
                np.full(n_real_oxygens, 8, dtype=np.int32),
                np.full(n_oxygen_vacancies, 9, dtype=np.int32)
            ])
            np.random.shuffle(oxygen_assignments)

            new_structure[cation_indices] = cation_assignments
            new_structure[oxygen_indices] = oxygen_assignments
            random_structures.append(new_structure)

            # Compute n_real_atoms (non-vacancy atoms: not Li=3, not F=9)
            n_real = n_real_cations + n_real_oxygens
            random_n_real_atoms.append(n_real)

        # Add metadata for random structures
        for i in range(n_random):
            all_metadata.append({
                'step': 'random',
                'vacancy_idx': -1,
                'vacancy_type': 'random',
                'vacancy_local_idx': -1,
                'n_atoms_total': n_atoms,
                'n_real_atoms': random_n_real_atoms[i],
                'is_random': True
            })

        all_atomic_numbers_list.extend(random_structures)
        print(f"  Generated {len(random_structures)} random structures")

    # Save bulk atomic numbers
    all_atomic_numbers_array = np.array(all_atomic_numbers_list, dtype=np.int32)
    n_total = len(all_atomic_numbers_array)
    print(f"Total structures (extracted + random): {n_total}")

    # Train/test split
    np.random.seed(seed)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Sort indices for consistent ordering
    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    print(f"Train/test split: {n_train} train, {n_test} test")

    # Split data
    train_atomic_numbers = all_atomic_numbers_array[train_indices]
    test_atomic_numbers = all_atomic_numbers_array[test_indices]

    train_metadata = [all_metadata[i] for i in train_indices]
    test_metadata = [all_metadata[i] for i in test_indices]

    # Add original index to metadata for reference
    for i, idx in enumerate(train_indices):
        train_metadata[i]['original_idx'] = int(idx)
    for i, idx in enumerate(test_indices):
        test_metadata[i]['original_idx'] = int(idx)

    # Save train data
    train_file = output_dir / 'train_atomic_numbers.npz'
    np.savez_compressed(
        train_file,
        atomic_numbers=train_atomic_numbers,
        positions=template_positions,
        cell=template_cell
    )
    print(f"✓ Saved train file: {train_file}")
    print(f"  - atomic_numbers shape: {train_atomic_numbers.shape}")

    # Save test data
    test_file = output_dir / 'test_atomic_numbers.npz'
    np.savez_compressed(
        test_file,
        atomic_numbers=test_atomic_numbers,
        positions=template_positions,
        cell=template_cell
    )
    print(f"✓ Saved test file: {test_file}")
    print(f"  - atomic_numbers shape: {test_atomic_numbers.shape}")

    # Also save combined file for backward compatibility
    bulk_file = output_dir / 'all_structures_atomic_numbers.npz'
    np.savez_compressed(
        bulk_file,
        atomic_numbers=all_atomic_numbers_array,
        positions=template_positions,
        cell=template_cell
    )
    print(f"✓ Saved bulk file (all data): {bulk_file}")
    print(f"  - atomic_numbers shape: {all_atomic_numbers_array.shape}")
    print(f"  - positions shape: {template_positions.shape}")
    print(f"  - cell shape: {template_cell.shape}")

    # Save train metadata
    train_metadata_file = output_dir / 'train_metadata.json'
    with open(train_metadata_file, 'w') as f:
        json.dump(train_metadata, f, indent=2)

    # Save test metadata
    test_metadata_file = output_dir / 'test_metadata.json'
    with open(test_metadata_file, 'w') as f:
        json.dump(test_metadata, f, indent=2)

    # Save combined metadata for backward compatibility
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    # Save split indices
    split_file = output_dir / 'train_test_split.json'
    with open(split_file, 'w') as f:
        json.dump({
            'train_indices': train_indices.tolist(),
            'test_indices': test_indices.tolist(),
            'n_train': n_train,
            'n_test': n_test,
            'test_ratio': test_ratio,
            'seed': seed
        }, f, indent=2)
    
    # Save parameters
    n_random_generated = n_total - n_extracted
    params_file = output_dir / 'extraction_params.json'
    with open(params_file, 'w') as f:
        json.dump({
            'n_cutoff_cell': N_CUTOFF_CELL,
            'unit_cell_size': UNIT_CELL_SIZE,
            'cutoff': cutoff,
            'cube_size': 2.0 * cutoff,
            'n_files_processed': len(vasp_files),
            'n_workers': n_workers,
            'total_cation_vacancies': total_cation_vac,
            'total_oxygen_vacancies': total_oxygen_vac,
            'n_structures_extracted': n_extracted,
            'n_random_structures': n_random_generated,
            'n_total_structures': n_total,
            'n_train': n_train,
            'n_test': n_test,
            'test_ratio': test_ratio,
            'split_seed': seed,
            'sampling_strategy': 'centered_at_vacancies',
            'max_vacancies_per_file': max_vac_per_file,
            'vacancy_replacement': 'X->Li, XO->F',
            'output_format': 'bulk_npz',
            'n_template_structures': n_templates,
            'train_file': 'train_atomic_numbers.npz',
            'test_file': 'test_atomic_numbers.npz',
            'bulk_file': 'all_structures_atomic_numbers.npz',
            'random_structures': {
                'enabled': n_random > 0,
                'n_generated': n_random_generated,
                'vacancy_pct_min': vac_pct_min,
                'vacancy_pct_max': vac_pct_max
            },
            'n_errors': len(errors)
        }, f, indent=2)
    
    print()
    print("="*60)
    print("Extraction Complete!")
    print("="*60)
    print(f"Workers used: {n_workers}")
    print(f"Files processed: {len(vasp_files)}")
    print(f"Total cation vacancies (X): {total_cation_vac}")
    print(f"Total oxygen vacancies (XO): {total_oxygen_vac}")
    print(f"Structures from VASP files: {n_extracted}")
    if n_random_generated > 0:
        print(f"Random structures generated: {n_random_generated}")
        print(f"  - Vacancy range: {vac_pct_min*100:.0f}%-{vac_pct_max*100:.0f}%")
    print(f"Total structures: {n_total}")
    print(f"  - Train set: {n_train} ({100*n_train/n_total:.1f}%)")
    print(f"  - Test set:  {n_test} ({100*n_test/n_total:.1f}%)")
    print(f"Template structures saved: {n_templates}")
    print(f"Random seed for split: {seed}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
