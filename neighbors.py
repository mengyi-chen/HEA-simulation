"""
Neighbor list management for KMC simulation
"""
import numpy as np
import logging
import pickle
from pathlib import Path
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
    def __init__(self, structure: SpinelStructure, params: KMCParams,
                 neighbor_file: str = None, auto_save: bool = True):
        """Initialize neighbor manager

        Args:
            structure: SpinelStructure instance
            params: KMCParams instance
            neighbor_file: Optional path to load/save neighbor lists. If None and auto_save=True,
                          saves to 'generate_config/neighbor_{size}.pkl'
            auto_save: If True and neighbor_file is None, automatically saves neighbor lists
                      after building them
        """
        self.structure = structure
        self.params = params

        # Neighbor lists (will be populated)
        self.neighbors_csr = None  # General neighbors (CSR)
        self.neighbors_dict = None  # General neighbors (dict, for SRO)
        self.nearest_neighbors_csr = None  # Nearest neighbors for cation diffusion (CSR)
        self.nearest_neighbors_dict = None  # Nearest neighbors (dict, for saving/loading)
        self.oxygen_neighbors_csr = None  # Nearest neighbors for oxygen diffusion (CSR)
        self.oxygen_neighbors_dict = None  # Oxygen neighbors (dict, for saving/loading)

        # Load from file if provided
        if neighbor_file is not None:
            logger.info(f"Loading neighbor lists from {neighbor_file}...")
            try:
                self.load_neighbors(neighbor_file)
                return  # Successfully loaded, no need to build
            except Exception as e:
                logger.warning(f"Failed to load neighbor lists: {e}")
                logger.info("Will build neighbor lists from scratch...")

        # Build neighbor lists
        logger.info("Building neighbor lists...")
        self._build_all_neighbors()
        self._build_nearest_neighbors()
        self._build_oxygen_neighbors()

        # Auto-save if requested
        if auto_save:
            size_str = f"{structure.supercell_size[0]}x{structure.supercell_size[1]}x{structure.supercell_size[2]}"
            save_file = f'generate_config/neighbor_{size_str}.pkl'
            logger.info(f"Auto-saving neighbor lists to {save_file}...")
            self.save_neighbors(save_file)

    # def _build_all_neighbors(self) -> None:
    #     """Build both general and nearest-neighbor lists"""
    #     pos_cart = self.structure.positions @ self.structure.cell
    #     n_atoms = self.structure.n_atoms

    #     # Build general neighbor list
    #     i_list, j_list = neighbour_list(
    #         'ij',
    #         positions=pos_cart,
    #         cutoff=6.0,
    #         cell=self.structure.cell,
    #         pbc=[True, True, False]
    #     )

    #     # Store general neighbors (for all atoms) - vectorized
    #     neighbors_dict = defaultdict(list)
    #     valid_pairs = (i_list != j_list)
    #     i_valid = i_list[valid_pairs]
    #     j_valid = j_list[valid_pairs]

    #     for i, j in zip(i_valid, j_valid):
    #         neighbors_dict[i].append(j)
        
    #     # Convert to CSR and keep dict
    #     self.neighbors_csr = CSRNeighborList.from_dict(neighbors_dict, n_atoms, with_distances=False)
    #     self.neighbors_dict = neighbors_dict

    #     # Log statistics
    #     self._log_general_neighbors()

    def _build_all_neighbors(self) -> None:
        """Build general neighbor list using cubic boundaries"""
        pos_cart = self.structure.positions @ self.structure.cell
        n_atoms = self.structure.n_atoms

        # Get actual cutoff
        actual_cutoff = self.params.cutoff

        # Build cubic neighbor list for all atoms
        neighbors_dict = defaultdict(list)

        # For each atom, find neighbors in cubic region
        for center_idx in range(n_atoms):
            center_pos = pos_cart[center_idx]

            # Calculate displacements from center to all other atoms
            displacements = pos_cart - center_pos

            # Apply periodic boundary conditions for x,y (not z)
            displacements[:, 0] -= np.round(displacements[:, 0] / self.structure.cell[0, 0]) * self.structure.cell[0, 0]
            displacements[:, 1] -= np.round(displacements[:, 1] / self.structure.cell[1, 1]) * self.structure.cell[1, 1]

            # For x and y: use PBC, select atoms within [-actual_cutoff, actual_cutoff)
            # Left boundary >= -actual_cutoff (include), right boundary < actual_cutoff (exclude)
            # Add tolerance to strictly include left boundary and exclude right boundary
            tol = 0.1
            in_xy = (displacements[:, 0] >= -actual_cutoff - tol) & (displacements[:, 0] < actual_cutoff - tol) & \
                    (displacements[:, 1] >= -actual_cutoff - tol) & (displacements[:, 1] < actual_cutoff - tol)

            # For z direction: calculate distance to top and bottom surfaces
            # Supercell height in z
            z_height = self.structure.cell[2, 2]

            # Center atom z position
            center_z = center_pos[2]

            # Distance to bottom surface (z=0)
            dist_to_bottom = center_z
            # Distance to top surface (z=z_height)
            dist_to_top = z_height - center_z

            # Handle asymmetric boundaries:
            # If distance to one side < actual_cutoff, include all to that side
            # and adjust the other side to maintain total cutoff range
            target_total_cutoff = 2.0 * actual_cutoff

            if dist_to_bottom < actual_cutoff:
                # Bottom side too close: include all to bottom, adjust top
                actual_cutoff_bottom = dist_to_bottom
                actual_cutoff_top = target_total_cutoff - dist_to_bottom
            elif dist_to_top < actual_cutoff:
                # Top side too close: include all to top, adjust bottom
                actual_cutoff_top = dist_to_top
                actual_cutoff_bottom = target_total_cutoff - dist_to_top
            else:
                # Both sides have enough space: use symmetric cutoff
                actual_cutoff_bottom = actual_cutoff
                actual_cutoff_top = actual_cutoff

            # For z: select atoms with asymmetric cutoffs
            in_z = (displacements[:, 2] >= -actual_cutoff_bottom - tol) & \
                   (displacements[:, 2] < actual_cutoff_top - tol)

            # Combine all conditions
            in_cube = in_xy & in_z

            # Exclude the center atom itself
            in_cube[center_idx] = False

            # Find neighbor indices
            neighbor_indices = np.where(in_cube)[0]
            neighbors_dict[center_idx] = list(neighbor_indices)

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

        # Convert to CSR with distances and store dict
        self.nearest_neighbors_csr = CSRNeighborList.from_dict(
            nearest_neighbors_dict, n_atoms, with_distances=True
        )
        self.nearest_neighbors_dict = nearest_neighbors_dict

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

        # Convert to CSR with distances and store dict
        self.oxygen_neighbors_csr = CSRNeighborList.from_dict(
            oxygen_neighbors_dict, n_atoms, with_distances=True
        )
        self.oxygen_neighbors_dict = oxygen_neighbors_dict

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

    def save_neighbors(self, filename: str) -> None:
        """Save neighbor dictionaries to a file

        Args:
            filename: Path to save the neighbor dictionaries
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'neighbors_dict': dict(self.neighbors_dict),
            'nearest_neighbors_dict': dict(self.nearest_neighbors_dict),
            'oxygen_neighbors_dict': dict(self.oxygen_neighbors_dict),
            'n_atoms': self.structure.n_atoms,
            'params': {
                'cutoff': self.params.cutoff,
                'nn_distance_A': self.params.nn_distance_A,
                'nn_distance_B': self.params.nn_distance_B,
                'nn_distance_O': self.params.nn_distance_O,
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Neighbor lists saved to {filename}")

    def load_neighbors(self, filename: str) -> None:
        """Load neighbor dictionaries from a file

        Args:
            filename: Path to load the neighbor dictionaries from
        """
        filepath = Path(filename)

        if not filepath.exists():
            raise FileNotFoundError(f"Neighbor file not found: {filename}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Validate parameters match
        saved_params = data['params']
        if (abs(saved_params['cutoff'] - self.params.cutoff) > 1e-6 or
            abs(saved_params['nn_distance_A'] - self.params.nn_distance_A) > 1e-6 or
            abs(saved_params['nn_distance_B'] - self.params.nn_distance_B) > 1e-6 or
            abs(saved_params['nn_distance_O'] - self.params.nn_distance_O) > 1e-6):
            logger.warning("Loaded neighbor list parameters differ from current params!")
            logger.warning(f"  Saved: cutoff={saved_params['cutoff']:.3f}, "
                         f"nn_A={saved_params['nn_distance_A']:.3f}, "
                         f"nn_B={saved_params['nn_distance_B']:.3f}, "
                         f"nn_O={saved_params['nn_distance_O']:.3f}")
            logger.warning(f"  Current: cutoff={self.params.cutoff:.3f}, "
                         f"nn_A={self.params.nn_distance_A:.3f}, "
                         f"nn_B={self.params.nn_distance_B:.3f}, "
                         f"nn_O={self.params.nn_distance_O:.3f}")

        # Validate number of atoms matches
        if data['n_atoms'] != self.structure.n_atoms:
            raise ValueError(
                f"Number of atoms mismatch: loaded {data['n_atoms']}, "
                f"structure has {self.structure.n_atoms}"
            )

        # Load dictionaries
        self.neighbors_dict = defaultdict(list, data['neighbors_dict'])
        self.nearest_neighbors_dict = defaultdict(list, data['nearest_neighbors_dict'])
        self.oxygen_neighbors_dict = defaultdict(list, data['oxygen_neighbors_dict'])

        # Convert to CSR format
        n_atoms = self.structure.n_atoms
        self.neighbors_csr = CSRNeighborList.from_dict(
            self.neighbors_dict, n_atoms, with_distances=False
        )
        self.nearest_neighbors_csr = CSRNeighborList.from_dict(
            self.nearest_neighbors_dict, n_atoms, with_distances=True
        )
        self.oxygen_neighbors_csr = CSRNeighborList.from_dict(
            self.oxygen_neighbors_dict, n_atoms, with_distances=True
        )

        # Log statistics
        logger.info("Neighbor lists loaded successfully")
        self._log_general_neighbors()
        self._log_nearest_neighbors()
        self._log_oxygen_neighbors()
