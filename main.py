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
import os
import argparse
import logging
import torch
from chgnet.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor
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
    
    def __init__(self, poscar_path: str, device, params: Optional[KMCParams] = None):
        """Initialize KMC simulator
        
        Args:
            poscar_path: Path to POSCAR file
            device: Device for CHGNet
            params: KMCParams instance
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
        
        # Initialize CHGNet
        logger.info("Loading CHGNet...")
        chgnet = CHGNet.load(use_device=device)
        chgnet.graph_converter.atom_graph_cutoff = 10.0
        
        adaptor = AseAtomsAdaptor()
        self.barrier_calc = BarrierCalculator(chgnet, adaptor)
        
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

                # Time statistics
                time_select = 0.0
                time_sro = 0.0
                time_update = 0.0
                time_update_remove = 0.0
                time_update_execute = 0.0
                time_update_collect = 0.0
                time_update_add = 0.0
                time_update_add_barriers = 0.0
                time_update_add_arrays = 0.0
                time_update_add_catalog = 0.0
                stats_interval = 50

                for step in tqdm(range(start_step, end_step)):
                    # Select event
                    t0 = time.perf_counter()
                    event_idx, dt, vac, cat, barrier, rate = self.events.select_event()
                    time_select += time.perf_counter() - t0

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
                        t0 = time.perf_counter()
                        sro_values = self.sro_calc.calculate(self.structure.symbols,
                                                            self.neighbors.neighbors_dict)
                        sim_logger.write_sro(sro_f, self.step, self.time, sro_values,
                                           self.sro_calc.element_pairs)
                        time_sro += time.perf_counter() - t0

                    # Execute event and update
                    t0 = time.perf_counter()
                    timings = self.events.update_after_hop(vac, cat)
                    time_update += time.perf_counter() - t0

                    # Accumulate detailed timings
                    time_update_remove += timings['remove']
                    time_update_execute += timings['execute']
                    time_update_collect += timings['collect']
                    time_update_add += timings['add_events']
                    time_update_add_barriers += timings['add_compute_barriers']
                    time_update_add_arrays += timings['add_create_arrays']
                    time_update_add_catalog += timings['add_catalog_add']

                    self.step += 1

                    # Print time statistics every stats_interval steps
                    if self.step % stats_interval == 0:
                        steps_done = self.step - start_step
                        total_time = time_select + time_update + time_sro

                        logger.info(f"\n{'='*60}")
                        logger.info(f"Step {self.step} - Time stats (avg over {steps_done} steps):")
                        logger.info(f"{'='*60}")

                        # Main components with percentages
                        select_pct = (time_select / total_time * 100) if total_time > 0 else 0
                        update_pct = (time_update / total_time * 100) if total_time > 0 else 0
                        sro_pct = (time_sro / total_time * 100) if total_time > 0 else 0

                        logger.info(f"  Select:        {time_select/steps_done*1000:6.3f} ms/step ({select_pct:5.1f}%)")
                        logger.info(f"  Update (total):{time_update/steps_done*1000:6.3f} ms/step ({update_pct:5.1f}%)")

                        # Update breakdown with percentages relative to total update time
                        if time_update > 0:
                            remove_pct = time_update_remove / time_update * 100
                            execute_pct = time_update_execute / time_update * 100
                            collect_pct = time_update_collect / time_update * 100
                            add_pct = time_update_add / time_update * 100

                            logger.info(f"    - Remove:    {time_update_remove/steps_done*1000:6.3f} ms/step ({remove_pct:5.1f}%)")
                            logger.info(f"    - Execute:   {time_update_execute/steps_done*1000:6.3f} ms/step ({execute_pct:5.1f}%)")
                            logger.info(f"    - Collect:   {time_update_collect/steps_done*1000:6.3f} ms/step ({collect_pct:5.1f}%)")
                            logger.info(f"    - Add events:{time_update_add/steps_done*1000:6.3f} ms/step ({add_pct:5.1f}%)")

                            # Add events breakdown with percentages relative to add_events time
                            if time_update_add > 0:
                                barriers_pct = time_update_add_barriers / time_update_add * 100
                                arrays_pct = time_update_add_arrays / time_update_add * 100
                                catalog_pct = time_update_add_catalog / time_update_add * 100

                                logger.info(f"      * Barriers: {time_update_add_barriers/steps_done*1000:6.3f} ms/step ({barriers_pct:5.1f}%)")
                                logger.info(f"      * Arrays:   {time_update_add_arrays/steps_done*1000:6.3f} ms/step ({arrays_pct:5.1f}%)")
                                logger.info(f"      * Catalog:  {time_update_add_catalog/steps_done*1000:6.3f} ms/step ({catalog_pct:5.1f}%)")

                        if time_sro > 0:
                            logger.info(f"  SRO:           {time_sro:.3f} s total ({sro_pct:5.1f}%)")

                        logger.info(f"{'='*60}")
                    
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
    parser.add_argument('--device', type=str, default='4', help='CUDA device or cpu')
    parser.add_argument('--temp', type=float, default=1200, help='Temperature in Kelvin')
    parser.add_argument('--cutoff', type=float, default=6.0, help='Cutoff radius in Angstrom')
    parser.add_argument('--steps', type=int, default=int(1e6), help='Number of kMC steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=1000, help='Save interval for configurations')
    parser.add_argument('--log_file', type=str, default='kmc_steps.csv', help='CSV file for kMC steps')
    parser.add_argument('--sro_interval', type=int, default=1000, help='Interval for SRO calculation')
    parser.add_argument('--sro_log_file', type=str, default='sro_log.csv', help='CSV file for SRO')
    parser.add_argument('--poscar_path', type=str, default='../generate_config/POSCAR_6x6x6_with_cavity.vasp')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for CHGNet')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Checkpoint interval')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
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
        
        # Load checkpoint
        kmc = load_checkpoint(args.resume_from, device, CavityHealingKMC)
        
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
            batch_size=args.batch_size
        )
        
        # Create KMC instance
        kmc = CavityHealingKMC(
            poscar_path=args.poscar_path,
            device=device,
            params=params
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
