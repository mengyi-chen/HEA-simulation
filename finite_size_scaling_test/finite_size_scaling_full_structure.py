#!/usr/bin/env python3
"""
Cavity Healing kMC for HEA Spinel (Refactored)

Simplified implementation with clear separation of concerns:
- SpinelStructure: Manages atomic structure
- NeighborManager: Handles neighbor lists
- BarrierCalculator: Computes energy barriers
- EventManager: Manages KMC events
- SROCalculator: Calculates short-range order
- SimulationLogger: Handles logging
""" 
import numpy as np
import time
import os, sys
sys.path.append('../')
import argparse
import logging
import torch
import json
from tqdm import tqdm
from typing import Optional, Tuple, List

from utils import *
from structure import SpinelStructure
from neighbors import NeighborManager
from barriers import BarrierCalculator
from energy_models import create_energy_model
from energy_models import EnergyModel

logger = logging.getLogger(__name__)

class BarrierCalculatorFull(BarrierCalculator):
    """Extended BarrierCalculator to handle both cation and oxygen vacancy hops"""
    def __init__(self, energy_model: EnergyModel, base_barriers: dict = None):
        super().__init__(energy_model, base_barriers)
    
    def build_hop_structures(self, structure: SpinelStructure,
                           neighbor_manager: NeighborManager,
                           vac_idx: int, atom_idx: int) -> Tuple[Atoms, Atoms]:
        """Build initial and final structures for a hop using cubic local environment

        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            vac_idx: Vacancy index (destination)
            atom_idx: Atom index (source, can be cation or oxygen)

        Returns:
            struct_init: Initial structure (atom at atom_idx) in cubic cell
            struct_final: Final structure (atom moved to vac_idx) in cubic cell
        """
        # Build combined cluster including neighbors of vacancy

        cluster_pos = structure.positions @ structure.cell
        cluster_symbols = structure.symbols

        # Remove vacancies (both X and XO)
        non_vac_mask = [s not in ['X', 'XO'] for s in cluster_symbols]
        # Create list of indices for all atoms in the full structure
        all_indices = list(range(len(cluster_symbols)))
        non_vac_indices = [idx for idx, m in zip(all_indices, non_vac_mask) if m]
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
    

class EventManager:
    """Manages KMC event catalog
    
    Responsibilities:
    - Build initial event catalog
    - Select events based on rates (KMC algorithm)
    - Update events after hops (remove old, add new)
    - Execute atomic hops
    """
    
    def __init__(self, structure: SpinelStructure, neighbor_manager: NeighborManager,
                 barrier_calculator: BarrierCalculatorFull, params: KMCParams):
        """Initialize event manager
        
        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            barrier_calculator: BarrierCalculatorFull instance
            params: KMCParams instance
        """
        self.structure = structure
        self.neighbors = neighbor_manager
        self.barrier_calc = barrier_calculator
        self.params = params
        
        # Event catalog (will be populated)
        self.catalog = None
    
    def build_catalog(self) -> None:
        """Build initial event catalog for both cation and oxygen diffusion"""
        logger.info("Building event catalog...")

        # Type A: Cation vacancy hopping with neighboring cations
        cation_vacancy_indices = np.where(self.structure.vacancy_mask)[0]
        hop_pairs = []

        for vac in cation_vacancy_indices:
            neighbors_idx = self.neighbors.nearest_neighbors_csr.get_neighbors(vac)
            for neighbor_idx in neighbors_idx:
                if self.structure.cation_mask[neighbor_idx]:
                    hop_pairs.append((vac, neighbor_idx))

        logger.info(f"Collected {len(hop_pairs)} potential cation events")

        # Type B: Oxygen vacancy hopping with neighboring oxygens
        oxygen_vacancy_indices = np.where(self.structure.oxygen_vacancy_mask)[0]
        oxygen_hop_count = 0

        for vac in oxygen_vacancy_indices:
            neighbors_idx = self.neighbors.oxygen_neighbors_csr.get_neighbors(vac)
            for neighbor_idx in neighbors_idx:
                if self.structure.oxygen_mask[neighbor_idx]:
                    hop_pairs.append((vac, neighbor_idx))
                    oxygen_hop_count += 1

        logger.info(f"Collected {oxygen_hop_count} potential oxygen events")
        logger.info(f"Total potential events: {len(hop_pairs)}")

        logger.info(f"Choose 200 hops to compute barriers...")
        idx = np.random.choice(len(hop_pairs), size=200, replace=False)
        hop_pairs = [hop_pairs[i] for i in idx]

        # Batch compute barriers (filters out invalid hops)
        barriers_array, valid_hop_pairs = self.barrier_calc.compute_barriers_batch(
            self.structure, self.neighbors, hop_pairs,
            batch_size=self.params.batch_size
        )

        # Build event catalog
        events_arr = np.array(valid_hop_pairs, dtype=np.int32)
        barriers_arr = barriers_array.astype(np.float32)

        return events_arr, barriers_arr
            

class CavityHealingKMC:
    """Main KMC simulator - coordinates all components
    
    This class is now much simpler, delegating responsibilities to:
    - structure: Atom management
    - neighbors: Neighbor list management
    - barrier_calc: Energy calculations
    - events: Event catalog management
    - sro_calc: SRO calculations
    - sim_logger: Logging
    """
    
    def __init__(self, poscar_path: str, device, params: Optional[KMCParams] = None,
                 energy_model_type: str = 'chgnet', mace_model_name: str = 'medium-omat-0',
                 neighbor_file: str = None,  configs_folder: str = 'kmc_configs'):
        """Initialize KMC simulator

        Args:
            poscar_path: Path to POSCAR file
            device: Device for energy model
            params: KMCParams instance
            energy_model_type: Type of energy model ('chgnet', 'm3gnet', 'mace')
            mace_model_path: Path to MACE model file (only for MACE)
            neighbor_file: Optional path to load/save neighbor lists
        """
        logger.info("="*60)
        logger.info("Cavity Healing kMC Initialization")
        logger.info("="*60)

        self.params = params if params is not None else KMCParams()
        self.device = device
        self.configs_folder = configs_folder

        # Read structure
        logger.info(f"Reading: {poscar_path}")
        cell, positions, symbols = read_poscar_with_custom_symbols(poscar_path)

        # Initialize components
        self.structure = SpinelStructure(cell, positions, symbols, self.params)
        self.neighbors = NeighborManager(self.structure, self.params, neighbor_file=neighbor_file)

        # Initialize energy model
        logger.info(f"Loading {energy_model_type} model...")
        if energy_model_type == 'mace':
            energy_model = create_energy_model(energy_model_type, device=device, model_name=mace_model_name)
        else:
            energy_model = create_energy_model(energy_model_type, device=device)
        logger.info(f"Using energy model: {energy_model.get_model_name()}")

        # Log base barriers
        logger.info("Base energy barriers (eV):")
        for element, barrier in sorted(self.params.base_barriers.items()):
            logger.info(f"  {element}: {barrier:.2f}")

        self.barrier_calc = BarrierCalculatorFull(energy_model, base_barriers=self.params.base_barriers)
        
        # Initialize event manager
        self.events = EventManager(self.structure, self.neighbors, self.barrier_calc, self.params)
        events_arr, barriers_arr = self.events.build_catalog()
        np.save(f'{self.configs_folder}/events.npy', events_arr)
        np.save(f'{self.configs_folder}/barriers.npy', barriers_arr)
        
       
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cavity Healing kMC')
    parser.add_argument('--device', type=str, default='5', help='CUDA device or cpu')
    parser.add_argument('--temp', type=float, default=1000, help='Temperature in Kelvin')
    # parser.add_argument('--n_cutoff_cell', type=int, default=1, help='DEPRECATED: Number of unit cells for cutoff radius (no longer used)')
    parser.add_argument('--steps', type=int, default=int(1e6), help='Number of kMC steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=1000, help='Save interval for configurations')
    parser.add_argument('--log_file', type=str, default='kmc_steps.csv', help='CSV file for kMC steps')
    parser.add_argument('--sro_interval', type=int, default=1000, help='Interval for SRO calculation')
    parser.add_argument('--sro_log_file', type=str, default='sro_log.csv', help='CSV file for SRO')
    parser.add_argument('--poscar_path', type=str, default='./POSCAR_step_79000.vasp')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for energy model')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Checkpoint interval')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume')

    # Neighbor list management
    parser.add_argument('--neighbor_file', type=str, default=None, help='Path to load/save neighbor lists (if None, auto-saves to generate_config/neighbor_{size}.pkl)')

    # Energy model selection
    parser.add_argument('--energy_model', type=str, default='chgnet', choices=['chgnet', 'mace'], help='Energy model to use')
    parser.add_argument('--mace_model_name', type=str, default='medium-omat-0',
                        help='MACE foundation model name (e.g., medium-omat-0, small-omat-0, medium, large)')

    # Configuration file for barriers
    parser.add_argument('--barriers_config', type=str, default='../config_uniform.json', help='Path to JSON config file for barriers')

    args = parser.parse_args()

    # Load barriers from config file
    with open(args.barriers_config, 'r') as f:
        config = json.load(f)
        base_barriers = config.get('base_barriers', None)

    # Parse device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Start fresh
    configs_folder = f"full_structure_{args.energy_model}"
    os.makedirs(configs_folder, exist_ok=True)

    log_file = os.path.join(configs_folder, args.log_file)
    sro_log_file = os.path.join(configs_folder, args.sro_log_file)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(configs_folder, 'kmc.log')),
            logging.StreamHandler()
        ]
    )
    
    # Create parameters
    params = KMCParams(
        temperature=args.temp,
        # n_cutoff_cell=args.n_cutoff_cell,
        batch_size=args.batch_size,
        base_barriers=base_barriers
    )
    
    # Create KMC instance
    kmc = CavityHealingKMC(
        poscar_path=args.poscar_path,
        device=device,
        params=params,
        energy_model_type=args.energy_model,
        mace_model_name=args.mace_model_name,
        neighbor_file=args.neighbor_file,
        configs_folder=configs_folder
    )
