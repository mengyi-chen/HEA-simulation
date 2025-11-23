"""
Checkpoint utilities for kMC simulation (adapted for new structure)
"""
import numpy as np
import pickle
import logging
from collections import defaultdict
from chgnet.model import CHGNet
from pymatgen.io.ase import AseAtomsAdaptor
from typing import Any

from structure import SpinelStructure
from neighbors import NeighborManager
from barriers import BarrierCalculator
from events import EventManager
from sro import SROCalculator
from optimized_structures import CSRNeighborList, EventCatalog

logger = logging.getLogger(__name__)


def save_checkpoint(kmc_instance, checkpoint_path: str) -> None:
    """Save complete simulation state to checkpoint file
    
    Args:
        kmc_instance: CavityHealingKMC instance to save
        checkpoint_path: Path to save checkpoint file
    """
    logger.info(f"Saving checkpoint to {checkpoint_path}...")
    
    # Get random state
    np_random_state = np.random.get_state()
    
    # Extract event catalog
    events, barriers, rates = kmc_instance.events.catalog.to_arrays()
    
    # Extract neighbor lists
    neighbors_dict = kmc_instance.neighbors.neighbors_dict
    nearest_neighbors_dict = kmc_instance.neighbors.nearest_neighbors_csr.to_dict(with_distances=True)
    
    # Pickle complex objects
    neighbors_bytes = pickle.dumps(dict(neighbors_dict))
    nearest_neighbors_bytes = pickle.dumps(nearest_neighbors_dict)
    random_state_bytes = pickle.dumps(np_random_state)
    params_bytes = pickle.dumps(kmc_instance.params)
    sro_pairs_bytes = pickle.dumps(kmc_instance.sro_calc.element_pairs)
    
    # Prepare checkpoint data
    checkpoint_data = {
        # Simulation state
        'step': kmc_instance.step,
        'time': kmc_instance.time,
        
        # Structure data
        'cell': kmc_instance.structure.cell,
        'positions': kmc_instance.structure.positions,
        'symbols': np.array(kmc_instance.structure.symbols, dtype=object),
        'atom_types': kmc_instance.structure.atom_types,
        
        # Event catalog
        'events': events,
        'barriers': barriers,
        'rates': rates,
        'event_capacity': kmc_instance.events.catalog.capacity,
        
        # Neighbor lists (pickled)
        'neighbors_bytes': np.frombuffer(neighbors_bytes, dtype=np.uint8),
        'nearest_neighbors_bytes': np.frombuffer(nearest_neighbors_bytes, dtype=np.uint8),
        
        # Random state (pickled)
        'random_state_bytes': np.frombuffer(random_state_bytes, dtype=np.uint8),
        
        # Parameters (pickled)
        'params_bytes': np.frombuffer(params_bytes, dtype=np.uint8),
        'supercell_size': kmc_instance.structure.supercell_size,
        
        # SRO tracking (pickled)
        'sro_pairs_bytes': np.frombuffer(sro_pairs_bytes, dtype=np.uint8),
    }
    
    # Save
    np.savez_compressed(checkpoint_path, **checkpoint_data)
    logger.info(f"Checkpoint saved successfully at step {kmc_instance.step}, time {kmc_instance.time:.6e} s")
    logger.info(f"Event catalog: {kmc_instance.events.catalog.size} events")


def load_checkpoint(checkpoint_path: str, device, kmc_class) -> Any:
    """Load a CavityHealingKMC instance from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device for CHGNet
        kmc_class: The CavityHealingKMC class
    
    Returns:
        Restored CavityHealingKMC instance
    """
    logger.info("="*60)
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    logger.info("="*60)
    
    # Load checkpoint
    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    
    # Unpickle parameters
    params_bytes = checkpoint['params_bytes'].tobytes()
    params = pickle.loads(params_bytes)
    
    # Create instance (bypass __init__)
    instance = kmc_class.__new__(kmc_class)
    
    # Set basic attributes
    instance.params = params
    instance.device = device
    instance.step = int(checkpoint['step'])
    instance.time = float(checkpoint['time'])
    
    # Restore structure
    instance.structure = SpinelStructure.__new__(SpinelStructure)
    instance.structure.cell = checkpoint['cell']
    instance.structure.positions = checkpoint['positions']
    instance.structure.symbols = checkpoint['symbols'].tolist()
    instance.structure.atom_types = checkpoint['atom_types']
    instance.structure.params = params
    instance.structure.supercell_size = checkpoint['supercell_size']
    
    # Load CHGNet
    logger.info("Loading CHGNet...")
    chgnet = CHGNet.load(use_device=device)
    chgnet.graph_converter.atom_graph_cutoff = 8.0
    
    # Restore neighbor lists
    n_atoms = len(instance.structure.positions)
    
    neighbors_bytes = checkpoint['neighbors_bytes'].tobytes()
    neighbors_dict = pickle.loads(neighbors_bytes)
    
    nearest_neighbors_bytes = checkpoint['nearest_neighbors_bytes'].tobytes()
    nearest_neighbors_dict = pickle.loads(nearest_neighbors_bytes)
    
    instance.neighbors = NeighborManager.__new__(NeighborManager)
    instance.neighbors.structure = instance.structure
    instance.neighbors.params = params
    instance.neighbors.neighbors_dict = defaultdict(list, neighbors_dict)
    instance.neighbors.neighbors_csr = CSRNeighborList.from_dict(neighbors_dict, n_atoms, with_distances=False)
    instance.neighbors.nearest_neighbors_csr = CSRNeighborList.from_dict(nearest_neighbors_dict, n_atoms, with_distances=True)
    
    # Restore barrier calculator
    adaptor = AseAtomsAdaptor()
    instance.barrier_calc = BarrierCalculator(chgnet, adaptor)
    
    # Restore event manager
    events = checkpoint['events']
    barriers = checkpoint['barriers']
    rates = checkpoint['rates']
    capacity = checkpoint.get('event_capacity', max(len(events) * 2, 10000))
    
    catalog = EventCatalog(initial_capacity=int(capacity))
    catalog._events[:len(events)] = events
    catalog._barriers[:len(barriers)] = barriers
    catalog._rates[:len(rates)] = rates
    catalog._size = len(events)
    
    instance.events = EventManager.__new__(EventManager)
    instance.events.structure = instance.structure
    instance.events.neighbors = instance.neighbors
    instance.events.barrier_calc = instance.barrier_calc
    instance.events.params = params
    instance.events.catalog = catalog
    
    # Restore SRO calculator
    sro_pairs_bytes = checkpoint['sro_pairs_bytes'].tobytes()
    element_pairs = pickle.loads(sro_pairs_bytes)
    
    instance.sro_calc = SROCalculator(params.elements)
    if element_pairs is not None:
        instance.sro_calc.element_pairs = element_pairs
    
    # Restore random state
    random_state_bytes = checkpoint['random_state_bytes'].tobytes()
    np_random_state = pickle.loads(random_state_bytes)
    np.random.set_state(np_random_state)
    
    logger.info(f"Resumed from step {instance.step}, time {instance.time:.6e} s")
    logger.info(f"Event catalog: {instance.events.catalog.size} events, {instance.events.catalog.capacity} capacity")
    logger.info(f"Utilization: {instance.events.catalog.utilization():.1%}")
    logger.info("="*60)
    
    return instance
