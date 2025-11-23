"""
Neighbor list management for KMC simulation
"""
import numpy as np
import logging
from collections import defaultdict
from matscipy.neighbours import neighbour_list
from optimized_structures import CSRNeighborList
from structure import SpinelStructure
from utils import KMCParams

logger = logging.getLogger(__name__)


class NeighborManager:
    """Manages neighbor lists for structure
    
    Responsibilities:
    - Build general neighbor list for all atoms (for energy calculations)
    - Build nearest-neighbor list for diffusion (A-A and B-B hops)
    - Provide efficient access to neighbors via CSR format
    - Maintain dict format for SRO calculation compatibility
    """
    
    def __init__(self, structure: SpinelStructure, params: KMCParams):
        """Initialize neighbor manager
        
        Args:
            structure: SpinelStructure instance
            params: KMCParams instance
        """
        self.structure = structure
        self.params = params
        
        # Neighbor lists (will be populated)
        self.neighbors_csr = None  # General neighbors (CSR)
        self.neighbors_dict = None  # General neighbors (dict, for SRO)
        self.nearest_neighbors_csr = None  # Nearest neighbors for diffusion (CSR)
        
        logger.info("Building neighbor lists...")
        self._build_all_neighbors()
    
    def _build_all_neighbors(self) -> None:
        """Build both general and nearest-neighbor lists"""
        pos_cart = self.structure.positions @ self.structure.cell
        n_atoms = self.structure.n_atoms
        
        # Build general neighbor list
        i_list, j_list, d_list = neighbour_list(
            'ijd',
            positions=pos_cart,
            cutoff=self.params.cutoff,
            cell=self.structure.cell,
            pbc=[True, True, False]
        )
        
        # Store general neighbors (for A-sites and B-sites only)
        neighbors_dict = defaultdict(list)
        for i, j in zip(i_list, j_list):
            if i != j:
                if self.structure.A_mask[i] or self.structure.B_mask[i]:
                    neighbors_dict[i].append(j)
        
        # Convert to CSR and keep dict
        self.neighbors_csr = CSRNeighborList.from_dict(neighbors_dict, n_atoms, with_distances=False)
        self.neighbors_dict = neighbors_dict
        
        # Log statistics
        self._log_general_neighbors()
        
        # Build nearest-neighbor list for diffusion
        self._build_nearest_neighbors(i_list, j_list, d_list, n_atoms)
    
    def _log_general_neighbors(self) -> None:
        """Log statistics for general neighbor list"""
        A_indices = np.where(self.structure.A_mask)[0]
        B_indices = np.where(self.structure.B_mask)[0]
        
        n_A = sum(1 for i in A_indices if self.neighbors_csr.has_neighbors(i))
        n_B = sum(1 for i in B_indices if self.neighbors_csr.has_neighbors(i))
        
        avg_A = np.mean([self.neighbors_csr.n_neighbors(i) for i in A_indices 
                        if self.neighbors_csr.has_neighbors(i)]) if n_A > 0 else 0
        avg_B = np.mean([self.neighbors_csr.n_neighbors(i) for i in B_indices 
                        if self.neighbors_csr.has_neighbors(i)]) if n_B > 0 else 0
        
        logger.info(f"General neighbor list (CSR): {n_A} A-sites (avg {avg_A:.1f} neighbors), "
                   f"{n_B} B-sites (avg {avg_B:.1f} neighbors)")
        logger.info(f"Memory usage: {self.neighbors_csr.memory_usage() / 1024:.1f} KB")
    
    def _build_nearest_neighbors(self, i_list, j_list, d_list, n_atoms) -> None:
        """Build nearest-neighbor list for A-A and B-B diffusion"""
        nn_distance_A = self.params.nn_distance_A
        nn_distance_B = self.params.nn_distance_B
        
        logger.info(f"NN cutoffs: A-A = {nn_distance_A:.3f} Å, B-B = {nn_distance_B:.3f} Å")
        
        # Filter for nearest cation-cation connections
        nearest_neighbors_dict = defaultdict(list)
        
        for i, j, dist in zip(i_list, j_list, d_list):
            if i == j:
                continue
            
            # A-site connections
            if self.structure.A_mask[i] and self.structure.cation_vacancy_mask[j] and dist <= nn_distance_A:
                nearest_neighbors_dict[i].append((j, dist))
            # B-site connections
            elif self.structure.B_mask[i] and self.structure.cation_vacancy_mask[j] and dist <= nn_distance_B:
                nearest_neighbors_dict[i].append((j, dist))
        
        # Convert to CSR with distances
        self.nearest_neighbors_csr = CSRNeighborList.from_dict(
            nearest_neighbors_dict, n_atoms, with_distances=True
        )
        
        # Log statistics
        self._log_nearest_neighbors()
    
    def _log_nearest_neighbors(self) -> None:
        """Log statistics for nearest-neighbor list"""
        A_indices = np.where(self.structure.A_mask)[0]
        B_indices = np.where(self.structure.B_mask)[0]
        
        n_A = sum(1 for i in A_indices if self.nearest_neighbors_csr.has_neighbors(i))
        n_B = sum(1 for i in B_indices if self.nearest_neighbors_csr.has_neighbors(i))
        
        avg_A = np.mean([self.nearest_neighbors_csr.n_neighbors(i) for i in A_indices 
                        if self.nearest_neighbors_csr.has_neighbors(i)]) if n_A > 0 else 0
        avg_B = np.mean([self.nearest_neighbors_csr.n_neighbors(i) for i in B_indices 
                        if self.nearest_neighbors_csr.has_neighbors(i)]) if n_B > 0 else 0
        
        logger.info(f"Nearest neighbors for diffusion (CSR): {n_A} A-sites (avg {avg_A:.1f}), "
                   f"{n_B} B-sites (avg {avg_B:.1f})")
        logger.info(f"Memory usage: {self.nearest_neighbors_csr.memory_usage() / 1024:.1f} KB")
