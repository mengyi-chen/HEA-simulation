"""
Neighbor list management for KMC simulation
"""
import numpy as np
import logging
from collections import defaultdict
from matscipy.neighbours import neighbour_list
from optimized_structures import CSRNeighborList
from structure import SpinelStructure
from utils import KMCParams, AtomType

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
        self.nearest_neighbors_csr = None  # Nearest neighbors for cation diffusion (CSR)
        self.oxygen_neighbors_csr = None  # Nearest neighbors for oxygen diffusion (CSR)
        
        logger.info("Building neighbor lists...")
        # Build general neighbor list for all atoms
        self._build_all_neighbors()

        # Build nearest-neighbor list for cation diffusion (separate call with tighter cutoff)
        self._build_nearest_neighbors()

        # Build oxygen neighbor list (separate call with tighter cutoff)
        self._build_oxygen_neighbors()

    
    def _build_all_neighbors(self) -> None:
        """Build both general and nearest-neighbor lists"""
        pos_cart = self.structure.positions @ self.structure.cell
        n_atoms = self.structure.n_atoms

        # Build general neighbor list
        i_list, j_list = neighbour_list(
            'ij',
            positions=pos_cart,
            cutoff=self.params.cutoff,
            cell=self.structure.cell,
            pbc=[True, True, False]
        )

        # Store general neighbors (for all atoms) - vectorized
        neighbors_dict = defaultdict(list)
        valid_pairs = (i_list != j_list)
        i_valid = i_list[valid_pairs]
        j_valid = j_list[valid_pairs]

        for i, j in zip(i_valid, j_valid):
            neighbors_dict[i].append(j)

        # Convert to CSR and keep dict
        self.neighbors_csr = CSRNeighborList.from_dict(neighbors_dict, n_atoms, with_distances=False)
        self.neighbors_dict = neighbors_dict

        # Log statistics
        self._log_general_neighbors()
         
    def _log_general_neighbors(self) -> None:
        """Log statistics for general neighbor list (for energy calculations)"""
        A_indices = np.where(self.structure.A_mask)[0]
        B_indices = np.where(self.structure.B_mask)[0]
        O_indices = np.where(self.structure.O_mask)[0]

        total_bonds_A = sum(self.neighbors_csr.n_neighbors(i) for i in A_indices)
        total_bonds_B = sum(self.neighbors_csr.n_neighbors(i) for i in B_indices)
        total_bonds_O = sum(self.neighbors_csr.n_neighbors(i) for i in O_indices)

        avg_A = total_bonds_A / len(A_indices) if len(A_indices) > 0 else 0
        avg_B = total_bonds_B / len(B_indices) if len(B_indices) > 0 else 0
        avg_O = total_bonds_O / len(O_indices) if len(O_indices) > 0 else 0

        logger.info(f"General neighbor list for energy calculations (all atoms within {self.params.cutoff:.1f} Å):")
        logger.info(f"  A-sites: {total_bonds_A} bonds (avg {avg_A:.1f} per atom)")
        logger.info(f"  B-sites: {total_bonds_B} bonds (avg {avg_B:.1f} per atom)")
        logger.info(f"  O-sites: {total_bonds_O} bonds (avg {avg_O:.1f} per atom)")
        logger.info(f"  Memory usage: {self.neighbors_csr.memory_usage() / 1024:.1f} KB")
    
    def _build_nearest_neighbors(self) -> None:
        """Build nearest-neighbor lists for cation diffusion (vectorized)"""

        pos_cart = self.structure.positions @ self.structure.cell
        n_atoms = self.structure.n_atoms

        nn_distance_A = self.params.nn_distance_A
        nn_distance_B = self.params.nn_distance_B

        logger.info(f"NN cutoffs: A-A = {nn_distance_A:.3f} Å, B-B = {nn_distance_B:.3f} Å")

        # Use tighter cutoff for cation-cation pairs
        max_cation_cutoff = max(nn_distance_A, nn_distance_B)
        i_list, j_list, d_list = neighbour_list(
            'ijd',
            positions=pos_cart,
            cutoff=max_cation_cutoff,
            cell=self.structure.cell,
            pbc=[True, True, False]
        )

        # Vectorized filtering for cation-cation connections
        valid_pairs = (i_list != j_list)

        # Pre-compute masks for all pairs
        A_mask_i = self.structure.A_mask[i_list[valid_pairs]]
        B_mask_i = self.structure.B_mask[i_list[valid_pairs]]
        AB_mask_j = self.structure.AB_mask[j_list[valid_pairs]]
        d_valid = d_list[valid_pairs]
        i_valid = i_list[valid_pairs]
        j_valid = j_list[valid_pairs]

        # A-site connections
        A_pairs = A_mask_i & AB_mask_j & (d_valid <= nn_distance_A)
        # B-site connections
        B_pairs = B_mask_i & AB_mask_j & (d_valid <= nn_distance_B)
        # Combine
        cation_pairs = A_pairs | B_pairs

        # Build dict from filtered pairs
        nearest_neighbors_dict = defaultdict(list)
        i_cation = i_valid[cation_pairs]
        j_cation = j_valid[cation_pairs]
        d_cation = d_valid[cation_pairs]

        for i, j, dist in zip(i_cation, j_cation, d_cation):
            nearest_neighbors_dict[i].append((j, dist))

        # Convert to CSR with distances
        self.nearest_neighbors_csr = CSRNeighborList.from_dict(
            nearest_neighbors_dict, n_atoms, with_distances=True
        )

        # Log statistics for cation neighbors
        self._log_nearest_neighbors()

    
    def _log_nearest_neighbors(self) -> None:
        """Log statistics for nearest-neighbor list (for cation diffusion)"""
        A_indices = np.where(self.structure.A_mask)[0]
        B_indices = np.where(self.structure.B_mask)[0]

        total_bonds_A = sum(self.nearest_neighbors_csr.n_neighbors(i) for i in A_indices)
        total_bonds_B = sum(self.nearest_neighbors_csr.n_neighbors(i) for i in B_indices)

        avg_A = total_bonds_A / len(A_indices) if len(A_indices) > 0 else 0
        avg_B = total_bonds_B / len(B_indices) if len(B_indices) > 0 else 0

        nn_distance_A = self.params.nn_distance_A
        nn_distance_B = self.params.nn_distance_B
        logger.info(f"Nearest neighbors for cation diffusion (cation-cation only):")
        logger.info(f"  A-sites: {total_bonds_A} bonds (avg {avg_A:.1f} per atom within {nn_distance_A:.2f} Å)")
        logger.info(f"  B-sites: {total_bonds_B} bonds (avg {avg_B:.1f} per atom within {nn_distance_B:.2f} Å)")
        logger.info(f"  Memory usage: {self.nearest_neighbors_csr.memory_usage() / 1024:.1f} KB")

    def _build_oxygen_neighbors(self) -> None:
        """Build nearest-neighbor list for oxygen diffusion (vectorized)

        Connect atom i to atom j IF both are oxygen species (O or XO) AND distance < nn_distance_O
        """

        pos_cart = self.structure.positions @ self.structure.cell
        n_atoms = self.structure.n_atoms

        nn_distance_O = self.params.nn_distance_O

        # Use oxygen-specific cutoff
        i_list, j_list, d_list = neighbour_list(
            'ijd',
            positions=pos_cart,
            cutoff=nn_distance_O,
            cell=self.structure.cell,
            pbc=[True, True, False]
        )

        # Vectorized filtering for oxygen-oxygen connections
        valid_pairs = (i_list != j_list)

        # Pre-compute masks for all pairs
        O_mask_i = self.structure.O_mask[i_list[valid_pairs]]
        O_mask_j = self.structure.O_mask[j_list[valid_pairs]]
        d_valid = d_list[valid_pairs]
        i_valid = i_list[valid_pairs]
        j_valid = j_list[valid_pairs]

        # Oxygen-oxygen connections (within cutoff, already filtered by neighbour_list)
        oxygen_pairs = O_mask_i & O_mask_j

        # Build dict from filtered pairs
        oxygen_neighbors_dict = defaultdict(list)
        i_oxygen = i_valid[oxygen_pairs]
        j_oxygen = j_valid[oxygen_pairs]
        d_oxygen = d_valid[oxygen_pairs]

        for i, j, dist in zip(i_oxygen, j_oxygen, d_oxygen):
            oxygen_neighbors_dict[i].append((j, dist))

        # Convert to CSR with distances
        self.oxygen_neighbors_csr = CSRNeighborList.from_dict(
            oxygen_neighbors_dict, n_atoms, with_distances=True
        )

        # Log statistics
        self._log_oxygen_neighbors()

    def _log_oxygen_neighbors(self) -> None:
        """Log statistics for oxygen neighbor list (for oxygen diffusion)"""
        oxygen_indices = np.where(self.structure.O_mask)[0]

        total_bonds_O = sum(self.oxygen_neighbors_csr.n_neighbors(i) for i in oxygen_indices)

        avg_oxygen = total_bonds_O / len(oxygen_indices) if len(oxygen_indices) > 0 else 0

        nn_distance_O = self.params.nn_distance_O
        logger.info(f"Nearest neighbors for oxygen diffusion (oxygen-oxygen only):")
        logger.info(f"  O-sites: {total_bonds_O} bonds (avg {avg_oxygen:.1f} per atom within {nn_distance_O:.2f} Å)")
        logger.info(f"  Memory usage: {self.oxygen_neighbors_csr.memory_usage() / 1024:.1f} KB")
