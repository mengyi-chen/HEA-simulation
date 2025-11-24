"""
Energy barrier calculation for atomic hops
"""
import numpy as np
from typing import List, Tuple
from ase import Atoms
from structure import SpinelStructure
from neighbors import NeighborManager
from energy_models import EnergyModel


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
    
    def build_hop_structures(self, structure: SpinelStructure,
                           neighbor_manager: NeighborManager,
                           vac_idx: int, atom_idx: int) -> Tuple[Atoms, Atoms]:
        """Build initial and final structures for a hop

        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            vac_idx: Vacancy index (destination)
            atom_idx: Atom index (source, can be cation or oxygen)

        Returns:
            struct_init: Initial structure (atom at atom_idx)
            struct_final: Final structure (atom moved to vac_idx)
        """
        # Build combined cluster including neighbors of both sites
        cluster_indices = set([atom_idx, vac_idx])

        # Get neighbors from CSR
        neighbors_csr = neighbor_manager.neighbors_csr
        cluster_indices.update(neighbors_csr.get_neighbors(atom_idx))
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
            raise ValueError(f"Empty cluster for hop {atom_idx}->{vac_idx}")

        # Find atom index in cluster
        atom_cluster_idx = non_vac_indices.index(atom_idx)

        # Initial structure
        struct_init = Atoms(
            symbols=non_vac_symbols,
            positions=non_vac_pos,
            cell=structure.cell,
            pbc=[True, True, False]
        )

        # Final structure (move atom to vacancy position)
        final_pos = non_vac_pos.copy()
        final_pos[atom_cluster_idx] = structure.positions[vac_idx] @ structure.cell

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

        # Build all structures and filter out invalid ones
        valid_structures = []
        valid_hop_pairs = []
        hopping_elements = []

        for vac, atom in hop_pairs:
            init_struct, final_struct = self.build_hop_structures(
                structure, neighbor_manager, vac, atom
            )
            # Filter out structures with only 1 atom (isolated atom case)
            if len(init_struct) > 1 and len(final_struct) > 1:
                valid_structures.append((init_struct, final_struct))
                valid_hop_pairs.append((vac, atom))
                # Get the element that is hopping (at atom_idx)
                hopping_elements.append(structure.symbols[atom])

        if len(valid_structures) == 0:
            return np.array([]), []

        init_structs, final_structs = zip(*valid_structures)

        # Batch predict energies using the energy model (directly with ASE Atoms)
        E_init = self.energy_model.predict_energy(list(init_structs), batch_size=batch_size)
        E_final = self.energy_model.predict_energy(list(final_structs), batch_size=batch_size)

        # Compute thermodynamic contribution
        delta_E_mlp = E_final - E_init

        # Apply base barrier + thermodynamic correction formula
        barriers = np.zeros(len(valid_hop_pairs))
        for i, element in enumerate(hopping_elements):
            base_barrier = self.base_barriers.get(element, 1.5)  # Default to 1.5 eV if not found
            barriers[i] = base_barrier + max(0, delta_E_mlp[i])

        return barriers, valid_hop_pairs
