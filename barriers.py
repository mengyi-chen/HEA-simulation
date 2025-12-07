"""
Energy barrier calculation for atomic hops
"""
import numpy as np
from typing import List, Tuple
from ase import Atoms
from structure import SpinelStructure
from neighbors import NeighborManager
from energy_models import EnergyModel
from tqdm import tqdm 

class BarrierCalculator:
    """Calculates energy barriers for atomic hops

    Responsibilities:
    - Build initial and final structures for hops
    - Batch compute barriers using energy models
    - Handle structure conversions (ASE <-> pymatgen)
    - Apply element-dependent base barriers
    """

    def __init__(self, energy_model: EnergyModel, base_barriers: dict = None):
        """Initialize barrier calculator

        Args:
            energy_model: EnergyModel instance for energy prediction
            base_barriers: Dictionary of base barriers (in eV) for different elements.
                          If None, uses default values.
        """
        self.energy_model = energy_model

        # Set base barriers with default values if not provided
        if base_barriers is None:
            self.base_barriers = {
                'Cu': 0.6,   # Fastest
                'Fe': 0.9,
                'Co': 1.1,
                'Ni': 1.4,
                'Al': 2.0,
                'Cr': 2.4,   # Slowest, anchors the lattice
                'O': 2.5,    # Oxygen diffusion baseline
            }
        else:
            self.base_barriers = base_barriers
    
    # def build_hop_structures(self, structure: SpinelStructure,
    #                     neighbor_manager: NeighborManager,
    #                     vac_idx: int, atom_idx: int) -> Tuple[Atoms, Atoms]:
    #     """Build initial and final structures for a hop

    #     Args:
    #         structure: SpinelStructure instance
    #         neighbor_manager: NeighborManager instance
    #         vac_idx: Vacancy index (destination)
    #         atom_idx: Atom index (source, can be cation or oxygen)

    #     Returns:
    #         struct_init: Initial structure (atom at atom_idx)
    #         struct_final: Final structure (atom moved to vac_idx)
    #     """
    #     # Build combined cluster including neighbors of both sites
    #     cluster_indices = set([atom_idx, vac_idx])

    #     # Get neighbors from CSR
    #     neighbors_csr = neighbor_manager.neighbors_csr
    #     cluster_indices.update(neighbors_csr.get_neighbors(atom_idx))
    #     cluster_indices.update(neighbors_csr.get_neighbors(vac_idx))
    #     cluster_indices = list(cluster_indices)

    #     # Get positions and symbols
    #     cluster_pos = structure.positions[cluster_indices] @ structure.cell
    #     cluster_symbols = [structure.symbols[i] for i in cluster_indices]

    #     # Remove vacancies (both X and XO)
    #     non_vac_mask = [s not in ['X', 'XO'] for s in cluster_symbols]
    #     non_vac_indices = [idx for idx, m in zip(cluster_indices, non_vac_mask) if m]
    #     non_vac_pos = cluster_pos[non_vac_mask]
    #     non_vac_symbols = [s for s, m in zip(cluster_symbols, non_vac_mask) if m]

    #     if len(non_vac_symbols) == 0:
    #         raise ValueError(f"Empty cluster for hop {atom_idx}->{vac_idx}")

    #     # Find atom index in cluster
    #     atom_cluster_idx = non_vac_indices.index(atom_idx)

    #     # Initial structure
    #     struct_init = Atoms(
    #         symbols=non_vac_symbols,
    #         positions=non_vac_pos,
    #         cell=structure.cell,
    #         pbc=[True, True, False]
    #     )

    #     # Final structure (move atom to vacancy position)
    #     final_pos = non_vac_pos.copy()
    #     final_pos[atom_cluster_idx] = structure.positions[vac_idx] @ structure.cell

    #     struct_final = Atoms(
    #         symbols=non_vac_symbols,
    #         positions=final_pos,
    #         cell=structure.cell,
    #         pbc=[True, True, False]
    #     )

    #     return struct_init, struct_final
    
    # def build_hop_structures(self, structure: SpinelStructure,
    #                        neighbor_manager: NeighborManager,
    #                        vac_idx: int, atom_idx: int) -> Tuple[Atoms, Atoms]:
    #     """Build initial and final structures for a hop using cubic local environment
    #
    #     Args:
    #         structure: SpinelStructure instance
    #         neighbor_manager: NeighborManager instance
    #         vac_idx: Vacancy index (destination)
    #         atom_idx: Atom index (source, can be cation or oxygen)
    #
    #     Returns:
    #         struct_init: Initial structure (atom at atom_idx) in cubic cell
    #         struct_final: Final structure (atom moved to vac_idx) in cubic cell
    #     """
    #     # Build combined cluster including neighbors of vacancy
    #     cluster_indices = set([atom_idx, vac_idx])
    #
    #     # Get neighbors from CSR (already built with cubic boundaries)
    #     neighbors_csr = neighbor_manager.neighbors_csr
    #     cluster_indices.update(neighbors_csr.get_neighbors(vac_idx))
    #     cluster_indices = list(cluster_indices)
    #
    #     # Get positions and symbols
    #     cluster_pos = structure.positions[cluster_indices] @ structure.cell
    #     cluster_symbols = [structure.symbols[i] for i in cluster_indices]
    #
    #     # Remove vacancies (both X and XO)
    #     non_vac_mask = [s not in ['X', 'XO'] for s in cluster_symbols]
    #     non_vac_indices = [idx for idx, m in zip(cluster_indices, non_vac_mask) if m]
    #     non_vac_pos = cluster_pos[non_vac_mask]
    #     non_vac_symbols = [s for s, m in zip(cluster_symbols, non_vac_mask) if m]
    #
    #     # Find atom index in cluster
    #     atom_cluster_idx = non_vac_indices.index(atom_idx)
    #
    #     # Prepare cubic cell for the cluster
    #     actual_cutoff = neighbor_manager.params.cutoff
    #     cube_size = 2.0 * actual_cutoff
    #     cubic_cell = np.array([
    #         [cube_size, 0.0, 0.0],
    #         [0.0, cube_size, 0.0],
    #         [0.0, 0.0, cube_size]
    #     ])
    #
    #     # Unwrap positions relative to vacancy (handle PBC)
    #     vac_pos_cart = structure.positions[vac_idx] @ structure.cell
    #     unwrapped_pos = non_vac_pos.copy()
    #     for i in range(len(unwrapped_pos)):
    #         delta = unwrapped_pos[i] - vac_pos_cart
    #         # Unwrap in x and y directions (apply PBC)
    #         delta[0] -= np.round(delta[0] / structure.cell[0, 0]) * structure.cell[0, 0]
    #         delta[1] -= np.round(delta[1] / structure.cell[1, 1]) * structure.cell[1, 1]
    #         # Z direction: no PBC wrapping needed
    #         unwrapped_pos[i] = vac_pos_cart + delta
    #
    #     # Center atoms in cubic cell (vacancy at center)
    #     center_shift = np.array([cube_size/2.0, cube_size/2.0, cube_size/2.0])
    #     init_pos = unwrapped_pos - vac_pos_cart + center_shift
    #
    #     # Final structure: move atom to vacancy position
    #     final_pos = init_pos.copy()
    #     final_pos[atom_cluster_idx] = center_shift  # Atom moves to vacancy (center)
    #
    #     # Create structures with cubic cell and correct PBC
    #     # PBC in x,y directions (periodic in plane), but not in z (surface normal)
    #     struct_init = Atoms(
    #         symbols=non_vac_symbols,
    #         positions=init_pos,
    #         cell=cubic_cell,
    #         pbc=[True, True, False]
    #     )
    #
    #     struct_final = Atoms(
    #         symbols=non_vac_symbols,
    #         positions=final_pos,
    #         cell=cubic_cell,
    #         pbc=[True, True, False]
    #     )
    #
    #     return struct_init, struct_final

    def build_hop_structures(self, structure: SpinelStructure,
                           neighbor_manager: NeighborManager,
                           vac_idx: int, atom_idx: int) -> Tuple[Atoms, Atoms]:
        """Build initial and final structures for a hop using unit-cell-based local environment

        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            vac_idx: Vacancy index (destination)
            atom_idx: Atom index (source, can be cation or oxygen)

        Returns:
            struct_init: Initial structure (atom at atom_idx) with unit-cell-based environment
            struct_final: Final structure (atom moved to vac_idx) with unit-cell-based environment
        """
        # Build combined cluster including neighbors of vacancy
        cluster_indices = set([atom_idx, vac_idx])

        # Get neighbors from CSR (now built with unit-cell-based boundaries)
        neighbors_csr = neighbor_manager.neighbors_csr
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

        # Find atom index in cluster
        atom_cluster_idx = non_vac_indices.index(atom_idx)

        # Determine the unit-cell-based local environment dimensions
        # Get vacancy's unit cell position
        vac_pos_frac = structure.positions[vac_idx]
        supercell_size = structure.supercell_size
        vac_cell_idx = np.floor(vac_pos_frac * supercell_size + 1e-6).astype(int)

        # Always 3x3x3 local environment
        n_cells = 3

        # Calculate local cell dimensions
        unit_cell_size = neighbor_manager.params.unit_cell_size
        local_cell = np.array([
            [n_cells * unit_cell_size, 0.0, 0.0],
            [0.0, n_cells * unit_cell_size, 0.0],
            [0.0, 0.0, n_cells * unit_cell_size]
        ])

        # Unwrap positions in XY direction (handle PBC at supercell boundaries)
        vac_pos_cart = structure.positions[vac_idx] @ structure.cell
        init_pos = non_vac_pos.copy()
        for i in range(len(init_pos)):
            delta = init_pos[i] - vac_pos_cart
            # Unwrap in x and y directions only (apply PBC)
            delta[0] -= np.round(delta[0] / structure.cell[0, 0]) * structure.cell[0, 0]
            delta[1] -= np.round(delta[1] / structure.cell[1, 1]) * structure.cell[1, 1]
            init_pos[i] = delta  # Position relative to vacancy

        # Shift positions so all atoms are inside the local cell
        # Simply shift by -min to make minimum coordinate = 0
        min_pos = init_pos.min(axis=0)
        init_pos = init_pos - min_pos

        # Vacancy position after shift (was at origin, now shifted by -min_pos)
        vac_local_pos = -min_pos

        # CRITICAL: Sort atoms by position (x, y, z) to match ClusterExpansionProcessor ordering
        # Processor uses XYZ ordering: x slowest, z fastest
        # np.lexsort sorts by LAST key first, so we pass (z, y, x) to get x-primary sorting
        sort_keys = np.round(init_pos / 1e-4).astype(int)  # Round to 1e-4 angstrom precision
        sort_indices = np.lexsort((sort_keys[:, 2], sort_keys[:, 1], sort_keys[:, 0]))

        # Apply sorting to initial structure
        init_pos_sorted = init_pos[sort_indices]
        init_symbols_sorted = [non_vac_symbols[i] for i in sort_indices]

        # Update atom_cluster_idx after sorting
        atom_cluster_idx_sorted = np.where(sort_indices == atom_cluster_idx)[0][0]

        # Final structure: move atom to vacancy position
        final_pos = init_pos_sorted.copy()
        final_pos[atom_cluster_idx_sorted] = vac_local_pos  # Atom moves to vacancy position

        # Sort final structure as well (positions changed, need to re-sort)
        # Same XYZ ordering as initial structure
        sort_keys_final = np.round(final_pos / 1e-4).astype(int)
        sort_indices_final = np.lexsort((sort_keys_final[:, 2], sort_keys_final[:, 1], sort_keys_final[:, 0]))

        final_pos_sorted = final_pos[sort_indices_final]
        final_symbols_sorted = [init_symbols_sorted[i] for i in sort_indices_final]

        # Create structures with local cell and correct PBC
        # PBC in x,y directions (periodic in plane), but not in z (surface normal)
        struct_init = Atoms(
            symbols=init_symbols_sorted,
            positions=init_pos_sorted,
            cell=local_cell,
            pbc=[True, True, False]
        )

        struct_final = Atoms(
            symbols=final_symbols_sorted,
            positions=final_pos_sorted,
            cell=local_cell,
            pbc=[True, True, False]
        )

        return struct_init, struct_final
    
    def compute_barriers_batch(self, structure: SpinelStructure,
                              neighbor_manager: NeighborManager,
                              hop_pairs: List[Tuple[int, int]],
                              batch_size: int = 64) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Batch compute barriers for multiple hops

        Uses formula: E_barrier = E_base + max(0, ΔE_MLP)

        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            hop_pairs: List of (vac_idx, atom_idx) tuples
            batch_size: Batch size for energy model prediction

        Returns:
            barriers: Array of energy barriers (in eV) for valid hops
            valid_hop_pairs: List of valid (vac_idx, atom_idx) tuples (filtered)
        """
        if len(hop_pairs) == 0:
            return np.array([]), []

        # Process in batches to reduce memory usage
        all_barriers = []
        all_valid_pairs = []

        for batch_start in range(0, len(hop_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(hop_pairs))
            batch_hop_pairs = hop_pairs[batch_start:batch_end]

            # Build structures for this batch only
            batch_init_structs = []
            batch_final_structs = []
            batch_valid_pairs = []
            batch_hopping_elements = []

            for vac, atom in batch_hop_pairs:
                init_struct, final_struct = self.build_hop_structures(
                    structure, neighbor_manager, vac, atom
                )
                # Filter out structures with only 1 atom (isolated atom case)
                if len(init_struct) > 1 and len(final_struct) > 1:
                    batch_init_structs.append(init_struct)
                    batch_final_structs.append(final_struct)
                    batch_valid_pairs.append((vac, atom))
                    # Get the element that is hopping (at atom_idx)
                    batch_hopping_elements.append(structure.symbols[atom])

            if len(batch_init_structs) == 0:
                continue

            # Predict energies for this batch
            E_init = self.energy_model.predict_energy(batch_init_structs, batch_size=batch_size)
            E_final = self.energy_model.predict_energy(batch_final_structs, batch_size=batch_size)

            # Compute thermodynamic contribution
            delta_E_mlp = E_final - E_init

            # Apply base barrier + thermodynamic correction formula
            batch_barriers = np.zeros(len(batch_valid_pairs))
            for i, element in enumerate(batch_hopping_elements):
                base_barrier = self.base_barriers.get(element, 1.5)  # Default to 1.5 eV if not found
                batch_barriers[i] = base_barrier + 0.5 * delta_E_mlp[i]
                if batch_barriers[i] < 0.0:
                    batch_barriers[i] = 0.0  # Ensure non-negative barriers

            all_barriers.append(batch_barriers)
            all_valid_pairs.extend(batch_valid_pairs)

        if len(all_barriers) == 0:
            return np.array([]), []

        # Concatenate all batch results
        barriers = np.concatenate(all_barriers)

        return barriers, all_valid_pairs
