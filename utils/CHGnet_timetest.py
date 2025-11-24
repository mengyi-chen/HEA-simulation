#!/usr/bin/env python3
"""
Cavity Healing kMC for HEA Spinel
Concise implementation with CHGNet barrier calculation
"""
# TODO: maybe add soft boundary on z direction later
# TODO: consider oxygen vacancy diffusion
# TODO: multiple event per kMC step (parallel)
import numpy as np
from ase.io import read
from chgnet.model import CHGNet
import pickle
from collections import defaultdict
import time
from utils import *
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
import tempfile
import torch
from matscipy.neighbours import neighbour_list
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
import os
import argparse

class CavityHealingKMC:
    def __init__(self, poscar_path, device, temperature=1200, cutoff=6.0, batch_size=64):
        """Initialize cavity healing kMC

        Args:
            poscar_path: Path to POSCAR file
            temperature: Temperature in Kelvin
            cutoff: Cutoff radius for neighbor list (Angstrom)
            device: device for CHGNet ('cpu' or CUDA device index)
            batch_size: Batch size for CHGNet barrier calculations (default: 64)
        """ 
        print("="*60)
        print("Cavity Healing kMC Initialization")
        print("="*60)

        self.T = temperature
        self.cutoff = cutoff
        self.kb = 8.617e-5  # eV/K
        self.nu0 = 1e13  # Hz
        self.device = device
        self.batch_size = batch_size
        self.adaptor = AseAtomsAdaptor()

        # Read structure
        print(f"Reading: {poscar_path}")
        
        # Handle custom element symbols (X for cation vacancy, XO for oxygen vacancy)
        # ASE does not recognize XO, so we need to preprocess
        with open(poscar_path, "r") as f:
            poscar_content = f.read()
        
        # Replace XO with He temporarily (He is unlikely in HEA spinels)
        poscar_modified = poscar_content.replace("  XO  ", "  He  ")
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vasp", delete=False) as tmp:
            tmp.write(poscar_modified)
            tmp_path = tmp.name
        
        atoms = read(tmp_path)
        os.unlink(tmp_path)
        
        self.cell = atoms.cell.array
        self.positions = atoms.get_scaled_positions()
        self.symbols = atoms.get_chemical_symbols()
        
        # Restore XO symbols
        self.symbols = ["XO" if s == "He" else s for s in self.symbols]
        
        # Determine supercell size from lattice constants
        # Unit cell is ~8.5 Angstrom, so supercell_size = cell / 8.5
        self.unit_cell_size = 8.5333518982  # From POSCAR.POSCAR.vasp
        self.supercell_size = np.array([
            int(round(self.cell[0, 0] / self.unit_cell_size)),
            int(round(self.cell[1, 1] / self.unit_cell_size)),
            int(round(self.cell[2, 2] / self.unit_cell_size))
        ])
        print(f"  Detected {self.supercell_size[0]}x{self.supercell_size[1]}x{self.supercell_size[2]} supercell")

        # Identify atom types
        self._parse_structure()

        # Build all neighbor lists in one pass
        print("Building neighbor lists...")
        self._build_all_neighbors()

        # Initialize CHGNet
        print("Loading CHGNet...")
        self.chgnet = CHGNet.load(use_device=self.device)

        # ALERT: set back 
        # self.chgnet.graph_converter.atom_graph_cutoff = 8.0

        # kMC state
        self.time = 0
        self.step = 0

        # Timing statistics
        self.timing_stats = {
            'times': [],           # Wall-clock time for each call
            'batch_sizes': [],     # Batch size for each call
            'n_structures': [],    # Number of structures for each call
            'gpu_memory_used': [], # GPU memory used (MB)
            'gpu_utilization': []  # GPU utilization (%)
        }

         # Build initial catalog
        self.build_event_catalog()

        print("Initialization complete!")
        print("="*60)

        

    def _parse_structure(self):
        """Parse structure into A-sites, B-sites, O-sites"""
        # Count atoms by type
        # Cation vacancy mask (only 'X')
        self.vacancy_mask = np.array([s == 'X' for s in self.symbols])

        # Cation mask (exclude 'X', 'XO', and 'O')
        self.cation_mask = np.array([s not in ['X', 'XO', 'O'] for s in self.symbols])

        # Oxygen mask (only 'O', not 'XO')
        self.oxygen_mask = np.array([s == 'O' for s in self.symbols])

        # Cation or vacancy mask (exclude 'O' and 'XO')
        self.cation_vacancy_mask = np.array([s not in ['O', 'XO'] for s in self.symbols])

        

        print(f"  Cation vacancies (X): {self.vacancy_mask.sum()}")
        print(f"  Oxygen vacancies (XO): {sum(1 for s in self.symbols if s == 'XO')}")
        print(f"  Cations: {self.cation_mask.sum()}")
        print(f"  Oxygens (O): {self.oxygen_mask.sum()}")

        # For spinel structure AB2O4, identify A-sites and B-sites by position:
        # A-sites (tetrahedral): coordinates in unit cell at multiples of 0.25 (0, 0.25, 0.5, 0.75)
        # B-sites (octahedral): coordinates in unit cell at odd multiples of 0.125 (0.125, 0.375, 0.625, 0.875)
        
        # Initialize A-site and B-site masks
        self.A_mask = np.zeros(len(self.symbols), dtype=bool)
        self.B_mask = np.zeros(len(self.symbols), dtype=bool)
        
        # Get indices of all cations and vacancies (they occupy cation sites)
        cation_and_vacancy_indices = np.where(self.cation_mask | self.vacancy_mask)[0]
        
        # Tolerance for coordinate comparison
        tol = 0.02
        
        for atom_idx in cation_and_vacancy_indices:
            pos = self.positions[atom_idx]
            
            # Wrap coordinates to [0, 1)
            pos_wrapped = pos % 1.0
            
            # Map position back to unit cell coordinates
            coords_scaled = (pos_wrapped * self.supercell_size) % 1.0
            
            # A-sites: all coordinates at 0, 0.25, 0.5, or 0.75 in unit cell
            is_A_site = True
            for coord in coords_scaled:
                # Find nearest multiple of 0.25
                nearest_quarter = np.round(coord * 4) / 4
                if abs(coord - nearest_quarter) > tol:
                    is_A_site = False
                    break
            
            if is_A_site:
                self.A_mask[atom_idx] = True
            else:
                self.B_mask[atom_idx] = True
        
        print('  A-sites:', self.A_mask.sum())
        print('  B-sites:', self.B_mask.sum())
         
    def _build_all_neighbors(self):
        """Build neighbor lists for A-sites and B-sites (including O neighbors)"""
        # Get cartesian positions
        pos_cart = self.positions @ self.cell

        # Build neighbor list once with the general cutoff (includes distances)
        i_list, j_list, d_list = neighbour_list('ijd', positions=pos_cart,
                                                cutoff=self.cutoff, cell=self.cell,
                                                pbc=[True, True, False])

        # Store neighbor list only for A-sites and B-sites (includes all neighbors: cations and O)
        self.neighbors = defaultdict(list)
        for i, j in zip(i_list, j_list):
            if i != j:  # Exclude self
                # Only add neighbors for A-sites and B-sites (not for O atoms)
                if self.A_mask[i] or self.B_mask[i]:
                    self.neighbors[i].append(j)

        # # Statistics for general neighbor list
        A_site_indices = np.where(self.A_mask)[0]
        B_site_indices = np.where(self.B_mask)[0]
        
        n_A_with_neighbors = sum(1 for i in self.neighbors.keys() if self.A_mask[i])
        n_B_with_neighbors = sum(1 for i in self.neighbors.keys() if self.B_mask[i])
        
        avg_A_all_neighbors = np.mean([len(self.neighbors[i]) for i in A_site_indices if i in self.neighbors]) if n_A_with_neighbors > 0 else 0
        avg_B_all_neighbors = np.mean([len(self.neighbors[i]) for i in B_site_indices if i in self.neighbors]) if n_B_with_neighbors > 0 else 0
        
        print(f"  General neighbor list: {n_A_with_neighbors} A-sites (avg {avg_A_all_neighbors:.1f} neighbors), {n_B_with_neighbors} B-sites (avg {avg_B_all_neighbors:.1f} neighbors)")
        
        # ============================================================
        # Now build nearest-neighbor list for diffusion (A-A and B-B)
        # ============================================================
        # Nearest-neighbor distances in ideal cubic spinel structure:
        #   A-B (1st nearest metal): 12 neighbors at (√11/8)a ≈ 0.415a  
        #   A-A (2nd nearest metal): 4 neighbors at (√3/4)a ≈ 0.433a
        #   B-B (1st nearest metal): 6 neighbors at (√2/4)a ≈ 0.354a
        #   B-A (2nd nearest metal): 12 neighbors at (√11/8)a ≈ 0.415a
        
        # Calculate nearest-neighbor distances with tolerance
        nn_distance_A = (np.sqrt(3) / 4.0) * self.unit_cell_size * 1.05  # A-A: 4 neighbors
        # nn_distance_B = (np.sqrt(2) / 4.0) * self.unit_cell_size * 1.05  # B-B: 6 neighbors        
        nn_distance_B = (np.sqrt(11) / 8.0) * self.unit_cell_size * 1.05  # B-B: 6 neighbors        
        print(f"  NN cutoffs: A-A = {nn_distance_A:.3f} Å, B-B = {nn_distance_B:.3f} Å")
        
        # Initialize unified nearest-neighbor dictionary
        self.nearest_neighbors = defaultdict(list)
        
        # Filter for nearest cation-cation connections
        for i, j, dist in zip(i_list, j_list, d_list):
            if i == j:
                continue
            
            # A-B connections (12 neighbors) + A-A connections (4 neighbors)
            if self.A_mask[i] and self.cation_vacancy_mask[j] and dist <= nn_distance_A:
                self.nearest_neighbors[i].append((j, dist))
            # B-B connections (6 neighbors) + B-A connections (6 neighbors)
            elif self.B_mask[i] and self.cation_vacancy_mask[j] and dist <= nn_distance_B:
                self.nearest_neighbors[i].append((j, dist))
    
        # Statistics
        n_A_with_nn = sum(1 for i in self.nearest_neighbors.keys() if self.A_mask[i])
        n_B_with_nn = sum(1 for i in self.nearest_neighbors.keys() if self.B_mask[i])
        avg_A_neighbors = np.mean([len(self.nearest_neighbors[i]) for i in A_site_indices if i in self.nearest_neighbors]) if n_A_with_nn > 0 else 0
        avg_B_neighbors = np.mean([len(self.nearest_neighbors[i]) for i in B_site_indices if i in self.nearest_neighbors]) if n_B_with_nn > 0 else 0
        print(f"  Nearest neighbors for diffusion: {n_A_with_nn} A-sites (avg {avg_A_neighbors:.1f}), {n_B_with_nn} B-sites (avg {avg_B_neighbors:.1f})")

    def build_event_catalog(self):
        """Build initial event catalog with nearest-neighbor A<->A and B<->B hops only"""
        print("\nBuilding event catalog...")

        # Find all vacancies
        vacancy_indices = np.where(self.vacancy_mask)[0]

        # Collect all hop pairs first (instead of computing barriers one by one)
        hop_pairs = []
        for vac in vacancy_indices:
            # Get nearest cation neighbors from unified list
            nearest_neighbors = [n for n, dist in self.nearest_neighbors.get(vac, [])
                                if self.cation_mask[n]]

            # Create events for nearest neighbors
            for cat in nearest_neighbors:
                hop_pairs.append((vac, cat))

        print(f"  Collected {len(hop_pairs)} potential events")
        print(f"  Computing barriers in batch mode (batch_size={self.batch_size})...")

        # Batch compute all barriers (GPU parallelism!)
        barriers_array = self.compute_barriers_batch(hop_pairs, batch_size=self.batch_size)

        # Build events, barriers, and rates arrays
        events_list = []
        barriers_list = []
        rates_list = []

        for i, (vac, cat) in enumerate(hop_pairs):
            barrier = barriers_array[i]
            rate = self.nu0 * np.exp(-barrier / (self.kb * self.T))

            events_list.append([vac, cat])
            barriers_list.append(barrier)
            rates_list.append(rate)

        # Convert to numpy arrays (more efficient for large catalogs)
        self.events = np.array(events_list, dtype=np.int32)  # (N, 2) array
        self.barriers = np.array(barriers_list, dtype=np.float32)
        self.rates = np.array(rates_list, dtype=np.float32)

        # Print timing statistics
        self.print_timing_statistics()


    def compute_barriers_batch(self, hop_pairs, batch_size=64):
        """Batch compute barriers for multiple hops using GPU parallelism

        Args:
            hop_pairs: List of (vac_idx, cat_idx) tuples
            batch_size: Batch size for CHGNet prediction (default: 64)

        Returns:
            barriers: numpy array of barriers (in eV)
        """
        if len(hop_pairs) == 0:
            return np.array([])

        # Build all initial and final structures
        pmg_init_list = []
        pmg_final_list = []

        for vac_idx, cat_idx in hop_pairs:
            struct_init, struct_final = self._build_structures_for_hop(vac_idx, cat_idx)
            pmg_init_list.append(self.adaptor.get_structure(struct_init))
            pmg_final_list.append(self.adaptor.get_structure(struct_final))

        # Batch predict energies with specified batch size
        # CHGNet returns different formats for single vs batch:
        # - Single structure: {'e': value}
        # - Batch (list): [{'e': value1}, {'e': value2}, ...]

        # Print structure information
        total_atoms = sum(len(struct) for struct in pmg_init_list)
        avg_atoms = total_atoms / len(pmg_init_list) if pmg_init_list else 0
        print(f"  Total number of structures: {len(pmg_init_list)}, Average atoms per structure: {avg_atoms:.1f}")

        for _ in range(5):
            # Get initial GPU stats if using CUDA
            if isinstance(self.device, torch.device):
                torch.cuda.synchronize(self.device)
                gpu_mem_before = torch.cuda.memory_allocated(self.device) / 1024**2  # MB

            start_time = time.perf_counter()
            E_init_results = self.chgnet.predict_structure(pmg_init_list, task='e', batch_size=batch_size)
            elapsed_time = time.perf_counter() - start_time

            E_final_results = self.chgnet.predict_structure(pmg_final_list, task='e', batch_size=batch_size)

            # Get final GPU stats if using CUDA
            if isinstance(self.device, torch.device):
                torch.cuda.synchronize(self.device)
                gpu_mem_after = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
                gpu_mem_used = gpu_mem_after - gpu_mem_before

                # Get GPU utilization using nvidia-smi
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits',
                        '-i', str(self.device.index if self.device.index is not None else 0)],
                        capture_output=True, text=True, timeout=1
                    )
                    gpu_util = float(result.stdout.strip())
                except:
                    gpu_util = -1  # Unable to query
            else:
                gpu_mem_used = 0
                gpu_util = 0

            self.timing_stats['times'].append(elapsed_time)
            self.timing_stats['batch_sizes'].append(batch_size)
            self.timing_stats['n_structures'].append(len(hop_pairs))
            self.timing_stats['gpu_memory_used'].append(gpu_mem_used)
            self.timing_stats['gpu_utilization'].append(gpu_util)

        # Extract energies
        if isinstance(E_init_results, list):
            # Batch prediction returns list of dicts
            E_init_array = np.array([result['e'] for result in E_init_results])
            E_final_array = np.array([result['e'] for result in E_final_results])
        else:
            # Single structure (shouldn't happen with list input, but handle it)
            E_init_array = np.array([E_init_results['e']])
            E_final_array = np.array([E_final_results['e']])

        # Compute barriers
        barriers = np.maximum(0, E_final_array - E_init_array)

        return barriers

    def _build_structures_for_hop(self, vac_idx, cat_idx):
        """Build initial and final structures for a hop with consistent neighborhood

        Args:
            vac_idx: Index of vacancy site (destination)
            cat_idx: Index of cation (source)

        Returns:
            struct_init: ASE Atoms with cation at cat_idx
            struct_final: ASE Atoms with cation moved to vac_idx (same atoms)
        """

        # Build combined cluster including neighbors of both sites
        cluster_indices = set([cat_idx, vac_idx])
        cluster_indices.update(self.neighbors.get(cat_idx, []))
        cluster_indices.update(self.neighbors.get(vac_idx, []))
        cluster_indices = list(cluster_indices)

        # Get positions and symbols
        cluster_pos = self.positions[cluster_indices] @ self.cell
        cluster_symbols = [self.symbols[i] for i in cluster_indices]

        # Remove both cation vacancies (X) and oxygen vacancies (XO)
        non_vac_mask = [s not in ['X', 'XO'] for s in cluster_symbols]
        non_vac_indices = [idx for idx, m in zip(cluster_indices, non_vac_mask) if m]
        non_vac_pos = cluster_pos[non_vac_mask]
        non_vac_symbols = [s for s, m in zip(cluster_symbols, non_vac_mask) if m]

        # Find the hopping cation index in the cluster
        cation_cluster_idx = non_vac_indices.index(cat_idx)

        # Check if cluster is empty
        if len(non_vac_symbols) == 0:
            raise ValueError(f"Empty cluster for hop {cat_idx}->{vac_idx}")

        # Initial structure: all atoms at current positions
        struct_init = Atoms(
            symbols=non_vac_symbols,
            positions=non_vac_pos,
            cell=self.cell,
            pbc=[True, True, False]
        )

        # Final structure: same atoms, but move the cation to vacancy position
        final_pos = non_vac_pos.copy()
        final_pos[cation_cluster_idx] = self.positions[vac_idx] @ self.cell

        struct_final = Atoms(
            symbols=non_vac_symbols,
            positions=final_pos,
            cell=self.cell,
            pbc=[True, True, False]
        )

        return struct_init, struct_final

    def print_timing_statistics(self):
        """Print timing statistics for CHGNet predictions"""
        if not self.timing_stats['times']:
            print("No timing statistics available.")
            return

        times = np.array(self.timing_stats['times'])
        n_structures = np.array(self.timing_stats['n_structures'])
        batch_sizes = np.array(self.timing_stats['batch_sizes'])
        gpu_memory = np.array(self.timing_stats['gpu_memory_used'])
        gpu_util = np.array(self.timing_stats['gpu_utilization'])

        print("\n" + "="*60)
        print("CHGNet Prediction Timing Statistics")
        print("="*60)

        # Calculate average time per structure
        total_structures = n_structures.sum()
        total_time = times.sum()
        if total_structures > 0:
            # Calculate number of batches per call
            # Each call processes n_structures with given batch_size
            num_batches_per_call = np.ceil(n_structures / batch_sizes).astype(int)
            total_batches = num_batches_per_call.sum()
            avg_time_per_batch = total_time / total_batches
            print(f"  Batch size: {batch_sizes.mean():.1f}")
            print(f"  Average time per batch: {avg_time_per_batch*1000:.4f}ms (Total batches: {total_batches})")

            avg_time_per_structure = total_time / total_structures
            print(f"  Average time per structure: {avg_time_per_structure*1000:.4f}ms")


        # GPU statistics (if available)
        if torch.cuda.is_available() and isinstance(self.device, torch.device) and self.device.type == 'cuda':
            print(f"\n  GPU Statistics:")
            print(f"    Device: {torch.cuda.get_device_name(self.device)}")
            print(f"    Peak memory allocated: {torch.cuda.max_memory_allocated(self.device) / 1024**2:.1f}MB")

            # Average GPU utilization (excluding -1 values which mean query failed)
            valid_util = gpu_util[gpu_util >= 0]
            if len(valid_util) > 0:
                print(f"    Average GPU utilization: {valid_util.mean():.1f}%")

        print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cavity Healing kMC')
    parser.add_argument('--device', type=str, default='4',
                       help='CUDA device (0, 1, 2, ...) or "cpu"')
    parser.add_argument('--temp', type=float, default=1200,
                       help='Temperature in Kelvin')
    parser.add_argument('--cutoff', type=float, default=6.0,
                       help='Cutoff radius in Angstrom')
    parser.add_argument('--steps', type=int, default=int(1e6),
                       help='Number of kMC steps')
    parser.add_argument('--interval', type=int, default=1000,
                       help='Save interval')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--log_file', type=str, default='kmc_steps.csv',
                       help='CSV file to log each kMC step (use "none" to disable)')
    parser.add_argument('--sro_interval', type=int, default=1000,
                       help='Interval for calculating and logging SRO (use 0 to disable)')
    parser.add_argument('--sro_log_file', type=str, default='sro_log.csv',
                       help='CSV file to log Warren-Cowley parameters')
    parser.add_argument('--poscar_path', type=str, default='../generate_config/POSCAR_6x6x6_with_cavity.vasp',
                       help='Path to POSCAR file')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for CHGNet barrier calculations (default: 64)')
    args = parser.parse_args()

    # Parse device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Run cavity healing simulation
    kmc = CavityHealingKMC(
        poscar_path=args.poscar_path,
        device=device,
        temperature=args.temp,
        cutoff=args.cutoff,
        batch_size=args.batch_size,
    )
    