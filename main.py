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
# TODO: CHGNet efficiency improvements
# TODO: add oxygen dynamics 
# TODO: add energy barrier: the current setting is not totally correct 
import numpy as np
import time
import os
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
from events import EventManager
from sro import SROCalculator
from logging_utils import SimulationLogger
from checkpoint import save_checkpoint, load_checkpoint
from energy_models import create_energy_model

logger = logging.getLogger(__name__)


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
                 energy_model_type: str = 'chgnet', mace_model_path: str = None):
        """Initialize KMC simulator

        Args:
            poscar_path: Path to POSCAR file
            device: Device for energy model
            params: KMCParams instance
            energy_model_type: Type of energy model ('chgnet', 'm3gnet', 'mace')
            mace_model_path: Path to MACE model file (only for MACE)
        """
        logger.info("="*60)
        logger.info("Cavity Healing kMC Initialization (Refactored)")
        logger.info("="*60)

        self.params = params if params is not None else KMCParams()
        self.device = device

        # Read structure
        logger.info(f"Reading: {poscar_path}")
        cell, positions, symbols = read_poscar_with_custom_symbols(poscar_path)

        # Initialize components
        self.structure = SpinelStructure(cell, positions, symbols, self.params)
        self.neighbors = NeighborManager(self.structure, self.params)

        # Initialize energy model
        logger.info(f"Loading {energy_model_type} model...")
        if energy_model_type == 'mace':
            energy_model = create_energy_model(energy_model_type, device=device, model_path=mace_model_path)
        else:
            energy_model = create_energy_model(energy_model_type, device=device)
        logger.info(f"Using energy model: {energy_model.get_model_name()}")

        # Log base barriers
        logger.info("Base energy barriers (eV):")
        for element, barrier in sorted(self.params.base_barriers.items()):
            logger.info(f"  {element}: {barrier:.2f}")

        self.barrier_calc = BarrierCalculator(energy_model, base_barriers=self.params.base_barriers)
        
        # Initialize event manager
        self.events = EventManager(self.structure, self.neighbors, self.barrier_calc, self.params)
        self.events.build_catalog()
        
        # Initialize SRO calculator
        self.sro_calc = SROCalculator(self.params.elements)
        
        # KMC state
        self.time = 0
        self.step = 0
        
        logger.info("Initialization complete!")
        logger.info("="*60)
    
    def run_kmc(self, configs_folder: str, n_steps: int = 1000,
               log_interval: int = 100, log_file: str = 'kmc_steps.csv',
               sro_interval: int = 100, sro_log_file: str = 'sro_log.csv',
               checkpoint_interval: int = 10) -> None:
        """Run KMC simulation
        
        Args:
            configs_folder: Directory to save configurations
            n_steps: Number of KMC steps
            log_interval: Save interval for configurations
            log_file: CSV file for step logging
            sro_interval: Interval for SRO calculation
            sro_log_file: CSV file for SRO logging
            checkpoint_interval: Interval for checkpointing
        """
        logger.info(f"Running kMC for {n_steps} steps...")
        logger.info("="*60)
        
        # Setup logger
        sim_logger = SimulationLogger(configs_folder, log_file, sro_log_file)
        
        # Determine if resuming
        is_resuming = self.step > 0
        file_mode = 'a' if is_resuming else 'w'
        
        if is_resuming:
            logger.info(f"Resuming from step {self.step}, time {self.time:.6e} s")
            logger.info("Checking and cleaning log files...")
            SimulationLogger.cleanup_csv(log_file, self.step)
            SimulationLogger.cleanup_csv(sro_log_file, self.step)
            logger.info(f"Log files cleaned. Appending from step {self.step}")
        
        checkpoint_path = os.path.join(configs_folder, 'checkpoint.npz')

        try:
            with open(log_file, file_mode) as log_f, open(sro_log_file, file_mode) as sro_f:
                # Write headers if starting fresh
                if not is_resuming:
                    log_f.write("step,time,dt,vac_idx,cat_idx,barrier,rate,"
                              "n_cation_vacancies,n_oxygen_vacancies,n_events\n")
                    header = "step,time," + ",".join(self.sro_calc.element_pairs)
                    sro_f.write(header + "\n")
                    sro_f.flush()
                
                logger.info(f"Logging to: {log_file}")
                logger.info(f"SRO logging to: {sro_log_file} (every {sro_interval} steps)")
                logger.info(f"Checkpointing to: {checkpoint_path} (every {checkpoint_interval} steps)")
                
                start_step = self.step
                end_step = start_step + n_steps

                for step in tqdm(range(start_step, end_step)):
                    # Select event
                    event_idx, dt, vac, cat, barrier, rate = self.events.select_event()
                    if event_idx is None:
                        logger.warning("No more events!")
                        break

                    # Update time
                    self.time += dt

                    # Log step
                    n_cation_vac = self.structure.vacancy_mask.sum()
                    n_oxygen_vac = (self.structure.atom_types == AtomType.OXYGEN_VACANCY).sum()
                    n_events = self.events.catalog.size

                    sim_logger.write_step(log_f, self.step, self.time, dt, vac, cat,
                                         barrier, rate, n_cation_vac, n_oxygen_vac, n_events)

                    # Calculate and log SRO
                    if sro_interval > 0 and self.step % sro_interval == 0:
                        sro_values = self.sro_calc.calculate(self.structure.symbols,
                                                            self.neighbors.neighbors_dict)
                        sim_logger.write_sro(sro_f, self.step, self.time, sro_values,
                                           self.sro_calc.element_pairs)

                    # Execute event and update
                    self.events.update_after_hop(vac, cat)
                    self.step += 1
                    
                    # Save configuration
                    if log_interval > 0 and self.step % log_interval == 0:
                        sim_logger.save_configuration(self.step, self.structure.positions,
                                                     self.structure.cell, self.structure.symbols)
                    
                    # Save checkpoint
                    if self.step % checkpoint_interval == 0:
                        save_checkpoint(self, checkpoint_path)

        except Exception as e:
            # Log error
            logger.error("="*60)
            logger.error(f"Error during simulation: {type(e).__name__}: {e}")
            logger.error("="*60)
            raise

        finally:
            # Save checkpoint on any exit (normal, interrupt, or error)
            logger.info("Saving checkpoint before exit...")
            save_checkpoint(self, checkpoint_path)
            
        logger.info("="*60)
        logger.info(f"Simulation complete! Total time: {self.time:.2e} s")
        logger.info(f"Step log saved to: {log_file}")
        logger.info(f"SRO log saved to: {sro_log_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cavity Healing kMC (Refactored)')
    parser.add_argument('--device', type=str, default='2', help='CUDA device or cpu')
    parser.add_argument('--temp', type=float, default=1000, help='Temperature in Kelvin')
    parser.add_argument('--cutoff', type=float, default=6.0, help='Cutoff radius in Angstrom')
    parser.add_argument('--steps', type=int, default=int(1e6), help='Number of kMC steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=1000, help='Save interval for configurations')
    parser.add_argument('--log_file', type=str, default='kmc_steps.csv', help='CSV file for kMC steps')
    parser.add_argument('--sro_interval', type=int, default=1000, help='Interval for SRO calculation')
    parser.add_argument('--sro_log_file', type=str, default='sro_log.csv', help='CSV file for SRO')
    parser.add_argument('--poscar_path', type=str, default='./generate_config/POSCAR_6x6x6_with_cavity.vasp')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for energy model')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Checkpoint interval')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume')

    # Energy model selection
    parser.add_argument('--energy_model', type=str, default='chgnet', choices=['chgnet', 'm3gnet', 'mace'], help='Energy model to use')
    parser.add_argument('--mace_model_path', type=str, default='../MACE/mace-mh-1.model', help='Path to MACE model file')

    # Configuration file for barriers
    parser.add_argument('--barriers_config', type=str, default='config_uniform.json', help='Path to JSON config file for barriers')

    args = parser.parse_args()

    # Load barriers from config file
    with open(args.barriers_config, 'r') as f:
        config = json.load(f)
        base_barriers = config.get('base_barriers', None)

    # Parse device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # Check if resuming
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from}")
        
        configs_folder = os.path.dirname(args.resume_from)
        
        # Setup logging (append mode)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(configs_folder, 'kmc.log'), mode='a'),
                logging.StreamHandler()
            ]
        )
        
        logger.info("\n" + "="*60)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("="*60)

        # Load checkpoint with energy model parameters
        kmc = load_checkpoint(
            args.resume_from,
            device,
            CavityHealingKMC,
            energy_model_type=args.energy_model,
            mace_model_path=args.mace_model_path
        )

        log_file = os.path.join(configs_folder, args.log_file)
        sro_log_file = os.path.join(configs_folder, args.sro_log_file)
    
    else:
        # Start fresh
        run_time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        configs_folder = f"configs_{run_time_str}"
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
            cutoff=args.cutoff,
            batch_size=args.batch_size,
            base_barriers=base_barriers
        )
        
        # Create KMC instance
        kmc = CavityHealingKMC(
            poscar_path=args.poscar_path,
            device=device,
            params=params,
            energy_model_type=args.energy_model,
            mace_model_path=args.mace_model_path
        )
    
    # Run simulation
    kmc.run_kmc(
        configs_folder=configs_folder,
        n_steps=args.steps,
        log_interval=args.log_interval,
        log_file=log_file,
        sro_interval=args.sro_interval,
        sro_log_file=sro_log_file,
        checkpoint_interval=args.checkpoint_interval
    )
    
    logger.info("Finish running!")
