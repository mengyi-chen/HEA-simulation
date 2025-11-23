"""
Energy barrier calculation for atomic hops
"""
import numpy as np
from typing import List, Tuple
from ase import Atoms
from structure import SpinelStructure
from neighbors import NeighborManager


def extract_energies(results) -> np.ndarray:
    """Extract energies from CHGNet prediction results
    
    Args:
        results: CHGNet prediction results (dict or list of dicts)
    
    Returns:
        energies: numpy array of energies (in eV)
    """
    if isinstance(results, list):
        return np.array([r['e'] for r in results])
    else:
        return np.array([results['e']])


class BarrierCalculator:
    """Calculates energy barriers for atomic hops
    
    Responsibilities:
    - Build initial and final structures for hops
    - Batch compute barriers using CHGNet
    - Handle structure conversions (ASE <-> pymatgen)
    """
    
    def __init__(self, energy_model, adaptor):
        """Initialize barrier calculator
        
        Args:
            energy_model: CHGNet model for energy prediction
            adaptor: AseAtomsAdaptor for ASE <-> pymatgen conversion
        """
        self.energy_model = energy_model
        self.adaptor = adaptor
    
    def build_hop_structures(self, structure: SpinelStructure, 
                           neighbor_manager: NeighborManager,
                           vac_idx: int, cat_idx: int) -> Tuple[Atoms, Atoms]:
        """Build initial and final structures for a hop
        
        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            vac_idx: Vacancy index (destination)
            cat_idx: Cation index (source)
        
        Returns:
            struct_init: Initial structure (cation at cat_idx)
            struct_final: Final structure (cation moved to vac_idx)
        """
        # Build combined cluster including neighbors of both sites
        cluster_indices = set([cat_idx, vac_idx])
        
        # Get neighbors from CSR
        neighbors_csr = neighbor_manager.neighbors_csr
        if neighbors_csr.has_neighbors(cat_idx):
            cluster_indices.update(neighbors_csr.get_neighbors(cat_idx))
        if neighbors_csr.has_neighbors(vac_idx):
            cluster_indices.update(neighbors_csr.get_neighbors(vac_idx))
        
        cluster_indices = list(cluster_indices)
        
        # Get positions and symbols
        cluster_pos = structure.positions[cluster_indices] @ structure.cell
        cluster_symbols = [structure.symbols[i] for i in cluster_indices]
        
        # Remove vacancies (both X and XO)
        non_vac_mask = [s not in ['X', 'XO'] for s in cluster_symbols]
        non_vac_indices = [idx for idx, m in zip(cluster_indices, non_vac_mask) if m]
        non_vac_pos = cluster_pos[non_vac_mask]
        non_vac_symbols = [s for s, m in zip(cluster_symbols, non_vac_mask) if m]
        
        if len(non_vac_symbols) == 0:
            raise ValueError(f"Empty cluster for hop {cat_idx}->{vac_idx}")
        
        # Find cation index in cluster
        cation_cluster_idx = non_vac_indices.index(cat_idx)
        
        # Initial structure
        struct_init = Atoms(
            symbols=non_vac_symbols,
            positions=non_vac_pos,
            cell=structure.cell,
            pbc=[True, True, False]
        )
        
        # Final structure (move cation to vacancy position)
        final_pos = non_vac_pos.copy()
        final_pos[cation_cluster_idx] = structure.positions[vac_idx] @ structure.cell
        
        struct_final = Atoms(
            symbols=non_vac_symbols,
            positions=final_pos,
            cell=structure.cell,
            pbc=[True, True, False]
        )
        
        return struct_init, struct_final
    
    def compute_barriers_batch(self, structure: SpinelStructure,
                              neighbor_manager: NeighborManager,
                              hop_pairs: List[Tuple[int, int]],
                              batch_size: int = 64) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Batch compute barriers for multiple hops

        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            hop_pairs: List of (vac_idx, cat_idx) tuples
            batch_size: Batch size for CHGNet prediction

        Returns:
            barriers: Array of energy barriers (in eV) for valid hops
            valid_hop_pairs: List of valid (vac_idx, cat_idx) tuples (filtered)
        """
        if len(hop_pairs) == 0:
            return np.array([]), []

        # Build all structures and filter out invalid ones
        valid_structures = []
        valid_hop_pairs = []

        for vac, cat in hop_pairs:
            init_struct, final_struct = self.build_hop_structures(
                structure, neighbor_manager, vac, cat
            )
            # Filter out structures with only 1 atom (isolated atom case)
            if len(init_struct) > 1 and len(final_struct) > 1:
                valid_structures.append((init_struct, final_struct))
                valid_hop_pairs.append((vac, cat))

        if len(valid_structures) == 0:
            return np.array([]), []

        init_structs, final_structs = zip(*valid_structures)

        # Convert to pymatgen
        pmg_init = [self.adaptor.get_structure(s) for s in init_structs]
        pmg_final = [self.adaptor.get_structure(s) for s in final_structs]

        # Batch predict energies
        E_init = extract_energies(
            self.energy_model.predict_structure(pmg_init, task='e', batch_size=batch_size)
        )
        E_final = extract_energies(
            self.energy_model.predict_structure(pmg_final, task='e', batch_size=batch_size)
        )

        # Compute barriers
        barriers = np.maximum(0, E_final - E_init)

        return barriers, valid_hop_pairs
