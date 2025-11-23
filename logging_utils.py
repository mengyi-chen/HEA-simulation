"""
Logging utilities for KMC simulation
"""
import os
import logging
from typing import Dict
from utils import write_poscar, AtomType

logger = logging.getLogger(__name__)


class SimulationLogger:
    """Handles logging for KMC simulation
    
    Responsibilities:
    - Log each KMC step
    - Log SRO parameters
    - Save configurations periodically
    - Clean up CSV files after resume
    """
    
    def __init__(self, configs_folder: str, log_file: str, sro_log_file: str):
        """Initialize simulation logger
        
        Args:
            configs_folder: Directory for saving configurations
            log_file: CSV file for step logging
            sro_log_file: CSV file for SRO logging
        """
        self.configs_folder = configs_folder
        self.log_file = log_file
        self.sro_log_file = sro_log_file
    
    def write_step(self, log_f, step: int, time: float, dt: float, vac: int, cat: int,
                  barrier: float, rate: float, n_cation_vac: int, n_oxygen_vac: int,
                  n_events: int) -> None:
        """Write a KMC step to log file
        
        Args:
            log_f: File handle
            step: Step number
            time: Simulation time
            dt: Time increment
            vac: Vacancy index
            cat: Cation index
            barrier: Energy barrier
            rate: Event rate
            n_cation_vac: Number of cation vacancies
            n_oxygen_vac: Number of oxygen vacancies
            n_events: Number of events in catalog
        """
        log_f.write(f"{step},{time:.6e},{dt:.6e},{vac},{cat},{barrier:.6f},{rate:.6e},"
                   f"{n_cation_vac},{n_oxygen_vac},{n_events}\n")
        log_f.flush()
    
    def write_sro(self, sro_f, step: int, time: float, sro_values: Dict[str, float],
                 element_pairs: list) -> None:
        """Write SRO parameters to log file
        
        Args:
            sro_f: File handle
            step: Step number
            time: Simulation time
            sro_values: Dict of SRO parameters
            element_pairs: List of element pair names
        """
        sro_line = f"{step},{time:.6e}"
        for pair_name in element_pairs:
            value = sro_values.get(pair_name, float('nan'))
            sro_line += f",{value:.6f}" if not (value != value) else ",nan"
        sro_f.write(sro_line + "\n")
        sro_f.flush()
    
    def save_configuration(self, step: int, positions, cell, symbols) -> None:
        """Save current configuration to POSCAR file
        
        Args:
            step: Step number
            positions: Fractional positions
            cell: Cell matrix
            symbols: Atomic symbols
        """
        poscar_file = f'{self.configs_folder}/POSCAR_step_{step}.vasp'
        write_poscar(positions, cell, symbols, poscar_file, pbc=[True, True, False])
    
    @staticmethod
    def cleanup_csv(csv_path: str, checkpoint_step: int) -> None:
        """Clean up CSV file to remove steps beyond checkpoint
        
        Args:
            csv_path: Path to CSV file
            checkpoint_step: Step number from checkpoint
        """
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            return
        
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) <= 1:
            return
        
        header = lines[0]
        data_lines = []
        removed_count = 0
        
        for line in lines[1:]:
            try:
                step = int(line.split(',')[0])
                if step < checkpoint_step:
                    data_lines.append(line)
                else:
                    removed_count += 1
            except (ValueError, IndexError):
                data_lines.append(line)
        
        if removed_count > 0:
            with open(csv_path, 'w') as f:
                f.write(header)
                f.writelines(data_lines)
            logger.info(f"Removed {removed_count} duplicate steps from {os.path.basename(csv_path)}")
        else:
            logger.info(f"{os.path.basename(csv_path)} is clean (no duplicates)")
