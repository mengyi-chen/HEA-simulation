"""
Step 2: Process structures and save for Cluster Expansion Training
MULTI-CPU PARALLEL VERSION

This script computes correlation vectors using ClusterExpansionProcessor.
Key approach:
  - Uses subspace.structure_site_mapping() to get the correct site mapping
  - Uses processor.allowed_species[i].index(spec) to get the correct occupancy code
  - Parallelizes correlation computation across multiple CPUs

Usage:
    python 2_process_save_wrangler_parallel.py [--n_workers N]
"""

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import os
import time
import pickle
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

from pymatgen.core import Structure
from smol.cofe import ClusterSubspace
from smol.moca import ClusterExpansionProcessor


# Atomic number to symbol mapping (from 0_extract_local_envs_parallel.py)
ATOMIC_NUMBER_TO_SYMBOL = {
    3: 'Li',   # cation vacancy placeholder
    9: 'F',    # oxygen vacancy placeholder
    8: 'O',
    13: 'Al',
    24: 'Cr',
    26: 'Fe',
    27: 'Co',
    28: 'Ni',
    29: 'Cu'
}


# Global variables for worker processes
_processor = None
_struct_idx_for_proc_site = None
_species_to_code = None
_all_atomic_numbers = None
_metadata = None


def init_worker(processor_pkl, struct_idx_for_proc_site, species_to_code, all_atomic_numbers, metadata):
    """Initialize worker process with shared data."""
    global _processor, _struct_idx_for_proc_site, _species_to_code, _all_atomic_numbers, _metadata
    _processor = pickle.loads(processor_pkl)
    _struct_idx_for_proc_site = struct_idx_for_proc_site
    _species_to_code = species_to_code
    _all_atomic_numbers = all_atomic_numbers
    _metadata = metadata


def atomic_numbers_to_symbols(atomic_numbers):
    """Convert atomic numbers array to symbols array."""
    return np.array([ATOMIC_NUMBER_TO_SYMBOL[z] for z in atomic_numbers])


def process_single_structure(task):
    """
    Process a single structure (worker function).

    Args:
        task: tuple of (struct_idx, energy)

    Returns:
        dict with structure_idx, corr_vector, energy, n_atoms, n_real_atoms
    """
    struct_idx, energy = task

    # Get atomic numbers and convert to symbols
    atomic_numbers = _all_atomic_numbers[struct_idx]
    symbols = atomic_numbers_to_symbols(atomic_numbers)

    # Convert to occupancy
    reordered_symbols = symbols[_struct_idx_for_proc_site]
    occupancy = np.array([_species_to_code[s] for s in reordered_symbols], dtype=np.int32)

    # Compute correlation vector
    corr_vector = _processor.compute_feature_vector(occupancy)

    # Get n_real_atoms from metadata
    n_real_atoms = _metadata[struct_idx]['n_real_atoms']

    return {
        'structure_idx': struct_idx,
        'corr_vector': corr_vector,
        'energy': energy,
        'n_atoms': len(atomic_numbers),
        'n_real_atoms': n_real_atoms
    }


def load_structure_from_npz(path):
    """Load structure from NPZ file."""
    data = np.load(path, allow_pickle=True)
    lattice = data['cell']
    coords = data['positions']
    species = data['symbols']
    return Structure(lattice, species, coords, coords_are_cartesian=True)


def create_primitive_structure(ref_path='../generate_config/POSCAR.POSCAR.vasp'):
    """Create the primitive structure for the HEA spinel oxide."""
    unit = Structure.from_file(ref_path)
    print("Using CONVENTIONAL cubic cell")
    prim = unit.copy()
    prim.replace_species({
        'Fe': {'Cu': 1/7, 'Ni': 1/7, 'Fe': 1/7, 'Al': 1/7,'Cr': 1/7, 'Co': 1/7, 'Li': 1/7},
        'O': {'O': 1/2, 'F': 1/2}
    })
    return prim


def load_energies(energy_file):
    """Load MACE energies from JSON file.

    Returns:
        energy_map: dict mapping structure_idx (int) -> energy
    """
    with open(energy_file, 'r') as f:
        energy_data = json.load(f)

    energy_map = {}
    for entry in energy_data:
        structure_idx = entry['structure_idx']
        energy = entry.get('energy')
        if energy is not None:
            energy_map[structure_idx] = energy

    return energy_map


def build_species_to_code(processor):
    """
    Build a unified species -> code mapping from processor.allowed_species.
    """
    species_to_code = {}

    for site_idx, allowed_species in enumerate(processor.allowed_species):
        for code, sp in enumerate(allowed_species):
            sp_str = str(sp)
            if sp_str in species_to_code:
                if species_to_code[sp_str] != code:
                    raise RuntimeError(f"Conflict: {sp_str} has code {species_to_code[sp_str]} but site {site_idx} wants code {code}")
            else:
                species_to_code[sp_str] = code

    print(f"✅ Species to code mapping: {species_to_code}")
    return species_to_code


def symbols_to_occupancy(symbols, struct_idx_for_proc_site, species_to_code):
    """Convert symbols array to occupancy array."""
    reordered_symbols = symbols[struct_idx_for_proc_site]
    return np.array([species_to_code[s] for s in reordered_symbols], dtype=np.int32)


def build_site_mapping(subspace, sc_matrix, template_structure):
    """Build site mapping using subspace.structure_site_mapping."""
    print("\n[3c] Building site mapping using subspace.structure_site_mapping...")
    t0 = time.time()

    supercell = subspace.structure.copy()
    supercell.make_supercell(sc_matrix)
    site_mapping = subspace.structure_site_mapping(supercell, template_structure)

    print(f"✓ Site mapping built: {len(site_mapping)} sites")
    print(f"  Time: {time.time()-t0:.2f} seconds")
    print(f"site_mapping: {site_mapping[:20]} ...")

    return site_mapping


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process structures for CE training (parallel version)')
    parser.add_argument('--n_workers', type=int, default=None, help='Number of worker processes (default: cpu_count)')
    args = parser.parse_args()

    n_workers = args.n_workers or cpu_count()

    # =========================================================================
    # Configuration
    # =========================================================================
    local_struct_dir = Path('./local_structures')
    train_energy_file = Path('./mace_energies/train_energies_medium_omat_0.json')
    output_dir = Path('./ce_data')
    os.makedirs(str(output_dir), exist_ok=True)

    print("="*60)
    print(f"Processing TRAIN Structures for CE (PARALLEL - {n_workers} workers)")
    print("="*60)

    # =========================================================================
    # 1. Create ClusterSubspace
    # =========================================================================
    print("\n[1] Creating ClusterSubspace...")
    prim = create_primitive_structure()
    print(f"Primitive cell atoms: {len(prim)}")

    subspace = ClusterSubspace.from_cutoffs(
        prim,
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        cutoffs={2: 6.0, 3: 4.0},
        basis='indicator',
        supercell_size='volume'
    )

    print(f"Correlation functions: {subspace.num_corr_functions}")
    print(f"Orbits: {len(subspace.orbits)}")

    # =========================================================================
    # 2. Load TRAIN data and create Processor
    # =========================================================================
    print("\n[2] Loading TRAIN data and creating Processor...")
    t0 = time.time()

    # Load train atomic numbers file
    train_file = local_struct_dir / 'train_atomic_numbers.npz'
    train_data = np.load(train_file)
    all_atomic_numbers = train_data['atomic_numbers']  # Shape: (N_structures, N_atoms)
    template_positions = train_data['positions']       # Shape: (N_atoms, 3)
    template_cell = train_data['cell']                 # Shape: (3, 3)

    print(f"Loaded train data: {all_atomic_numbers.shape[0]} structures, {all_atomic_numbers.shape[1]} atoms each")

    # Load train metadata
    metadata_file = local_struct_dir / 'train_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} train structures")

    # Load template structure from saved template file
    template_files = sorted(local_struct_dir.glob('template_*.npz'))
    template_path = str(template_files[0])
    template_structure = load_structure_from_npz(template_path)
    print(f"Template: {template_path}")

    sc_matrix = subspace.scmatrix_from_structure(template_structure)
    print(f"Supercell matrix:\n{sc_matrix}")

    dummy_coefficients = np.zeros(subspace.num_corr_functions)
    processor = ClusterExpansionProcessor(subspace, sc_matrix, dummy_coefficients)

    print(f"Processor built in {time.time()-t0:.1f} seconds")
    print(f"Supercell size: {processor.size} sites")
    print(f"Processor Structure Atoms: {len(processor.structure)}")

    # =========================================================================
    # 3. Build site mapping and species encoding
    # =========================================================================
    print("\n[3] Building mappings...")

    species_to_code = build_species_to_code(processor)
    site_mapping = build_site_mapping(subspace, sc_matrix, template_structure)

    n_sites = len(processor.allowed_species)
    print(f"Number of sites in occupancy: {n_sites}")

    struct_idx_for_proc_site = np.array([site_mapping.index(i) for i in range(n_sites)])

    # =========================================================================
    # 4. Verify occupancy mapping
    # =========================================================================
    # Use the saved template files for verification
    print("\n[4] Verifying occupancy mapping (using template structures)...")

    for i, template_file in enumerate(template_files[:3]):
        test_data = np.load(str(template_file), allow_pickle=True)
        test_symbols = test_data['symbols']
        test_structure = load_structure_from_npz(str(template_file))

        manual_occu = symbols_to_occupancy(test_symbols, struct_idx_for_proc_site, species_to_code)
        correct_occu = processor.occupancy_from_structure(test_structure)

        if np.array_equal(manual_occu, correct_occu):
            print(f"  ✓ Test {i+1}: {template_file.name} - PASS")
        else:
            mismatch_count = np.sum(manual_occu != correct_occu)
            print(f"  ✗ Test {i+1}: {template_file.name} - FAIL ({mismatch_count} mismatches)")
            raise RuntimeError("Occupancy mapping verification failed!")

    # =========================================================================
    # 5. Load train energies
    # =========================================================================
    print("\n[5] Loading train energies...")
    energy_map = load_energies(train_energy_file)
    print(f"Loaded {len(energy_map)} train energies")

    # =========================================================================
    # 6. Process all train structures in parallel
    # =========================================================================
    print("\n[6] Processing train structures in parallel...")

    # Find structures with energies
    n_total_structures = all_atomic_numbers.shape[0]

    # Prepare task list: (struct_idx, energy)
    tasks = [(idx, energy_map[idx]) for idx in range(n_total_structures) if idx in energy_map]
    print(f"Train structures with energies: {len(tasks)} / {n_total_structures}")

    # Serialize processor for worker initialization
    processor_pkl = pickle.dumps(processor)

    # Process in parallel
    start_time = time.time()

    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(processor_pkl, struct_idx_for_proc_site, species_to_code, all_atomic_numbers, metadata)
    ) as pool:
        results = list(tqdm(
            pool.imap(process_single_structure, tasks),
            total=len(tasks),
            desc="Computing correlations"
        ))

    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.1f} seconds")
    print(f"Average: {elapsed_time/len(tasks)*1000:.2f} ms per structure")
    print(f"Speedup vs sequential estimate: {len(tasks) * 3.4 / 1000 / elapsed_time:.1f}x")

    # =========================================================================
    # 7. Build feature matrix
    # =========================================================================
    print("\n[7] Building feature matrix...")
    n_structures = len(results)
    n_corr = subspace.num_corr_functions

    feature_matrix = np.zeros((n_structures, n_corr))
    energies = np.zeros(n_structures)
    structure_indices = []
    n_atoms_list = []
    n_real_atoms_list = []

    for i, result in enumerate(results):
        feature_matrix[i] = result['corr_vector']
        energies[i] = result['energy']
        structure_indices.append(result['structure_idx'])
        n_atoms_list.append(result['n_atoms'])
        n_real_atoms_list.append(result['n_real_atoms'])

    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Feature matrix rank: {np.linalg.matrix_rank(feature_matrix)}")

    # =========================================================================
    # 8. Process TEST data (if available)
    # =========================================================================
    test_file = local_struct_dir / 'test_atomic_numbers.npz'
    test_energy_file = Path('./mace_energies/test_energies_medium_omat_0.json')
    test_results = None

    if test_file.exists() and test_energy_file.exists():
        print("\n[8] Processing TEST data...")

        # Load test data
        test_data = np.load(test_file)
        test_atomic_numbers = test_data['atomic_numbers']

        # Load test metadata
        test_metadata_file = local_struct_dir / 'test_metadata.json'
        with open(test_metadata_file, 'r') as f:
            test_metadata = json.load(f)

        # Load test energies
        test_energy_map = load_energies(test_energy_file)
        print(f"Loaded {len(test_energy_map)} test energies")

        n_test_structures = test_atomic_numbers.shape[0]
        test_tasks = [(idx, test_energy_map[idx]) for idx in range(n_test_structures) if idx in test_energy_map]
        print(f"Test structures with energies: {len(test_tasks)} / {n_test_structures}")

        # Process test in parallel
        test_start_time = time.time()

        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(processor_pkl, struct_idx_for_proc_site, species_to_code, test_atomic_numbers, test_metadata)
        ) as pool:
            test_results = list(tqdm(
                pool.imap(process_single_structure, test_tasks),
                total=len(test_tasks),
                desc="Computing test correlations"
            ))

        test_elapsed = time.time() - test_start_time
        print(f"Test processing completed in {test_elapsed:.1f} seconds")

        # Build test feature matrix
        n_test = len(test_results)
        test_feature_matrix = np.zeros((n_test, n_corr))
        test_energies = np.zeros(n_test)
        test_structure_indices = []
        test_n_atoms_list = []
        test_n_real_atoms_list = []

        for i, result in enumerate(test_results):
            test_feature_matrix[i] = result['corr_vector']
            test_energies[i] = result['energy']
            test_structure_indices.append(result['structure_idx'])
            test_n_atoms_list.append(result['n_atoms'])
            test_n_real_atoms_list.append(result['n_real_atoms'])

        print(f"Test feature matrix shape: {test_feature_matrix.shape}")

        # Save test data
        np.savez_compressed(
            output_dir / 'test_ce_data.npz',
            feature_matrix=test_feature_matrix,
            energies=test_energies,
            structure_indices=np.array(test_structure_indices),
            n_atoms=np.array(test_n_atoms_list),
            n_real_atoms=np.array(test_n_real_atoms_list)
        )
        print(f"Saved test data to: {output_dir / 'test_ce_data.npz'}")
    else:
        print("\n[8] No TEST data found - skipping test processing")

    # =========================================================================
    # 9. Save TRAIN results
    # =========================================================================
    print("\n[9] Saving TRAIN results...")

    np.savez_compressed(
        output_dir / 'ce_data.npz',
        feature_matrix=feature_matrix,
        energies=energies,
        structure_indices=np.array(structure_indices),
        n_atoms=np.array(n_atoms_list),
        n_real_atoms=np.array(n_real_atoms_list)
    )

    with open(output_dir / 'subspace.pkl', 'wb') as f:
        pickle.dump(subspace, f)

    with open(output_dir / 'processor.pkl', 'wb') as f:
        pickle.dump(processor, f)

    with open(output_dir / 'site_mapping.json', 'w') as f:
        json.dump(site_mapping, f)

    with open(output_dir / 'struct_idx_for_proc_site.json', 'w') as f:
        json.dump(struct_idx_for_proc_site.tolist(), f)

    with open(output_dir / 'species_to_code.json', 'w') as f:
        json.dump(species_to_code, f, indent=2)

    stats = {
        'n_structures': n_structures,
        'n_corr_functions': n_corr,
        'n_orbits': len(subspace.orbits),
        'n_sites': processor.size,
        'energy_min': float(np.min(energies)),
        'energy_max': float(np.max(energies)),
        'energy_mean': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
        'cutoffs': {'pair': 6.0, 'triplet': 4.0},
        'basis': 'indicator',
        'processing_time_seconds': elapsed_time,
        'n_workers': n_workers
    }

    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved:")
    print(f"  - {output_dir / 'ce_data.npz'}")
    print(f"  - {output_dir / 'subspace.pkl'}")
    print(f"  - {output_dir / 'processor.pkl'}")
    print(f"  - {output_dir / 'site_mapping.json'}")
    print(f"  - {output_dir / 'struct_idx_for_proc_site.json'}")
    print(f"  - {output_dir / 'species_to_code.json'}")
    print(f"  - {output_dir / 'stats.json'}")

    print(f"\nStatistics:")
    print(f"  Structures: {stats['n_structures']}")
    print(f"  Correlation functions: {stats['n_corr_functions']}")
    print(f"  Sites per structure: {stats['n_sites']}")
    print(f"  Energy range: [{stats['energy_min']:.2f}, {stats['energy_max']:.2f}] eV")
    print(f"  Workers used: {n_workers}")

    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
