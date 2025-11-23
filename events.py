"""
Event catalog management for KMC simulation
"""
import numpy as np
import logging
from typing import Optional, Tuple, List
from optimized_structures import EventCatalog
from structure import SpinelStructure
from neighbors import NeighborManager
from barriers import BarrierCalculator
from utils import KMCParams, AtomType

logger = logging.getLogger(__name__)


class EventManager:
    """Manages KMC event catalog
    
    Responsibilities:
    - Build initial event catalog
    - Select events based on rates (KMC algorithm)
    - Update events after hops (remove old, add new)
    - Execute atomic hops
    """
    
    def __init__(self, structure: SpinelStructure, neighbor_manager: NeighborManager,
                 barrier_calculator: BarrierCalculator, params: KMCParams):
        """Initialize event manager
        
        Args:
            structure: SpinelStructure instance
            neighbor_manager: NeighborManager instance
            barrier_calculator: BarrierCalculator instance
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

        # Batch compute barriers (filters out invalid hops)
        barriers_array, valid_hop_pairs = self.barrier_calc.compute_barriers_batch(
            self.structure, self.neighbors, hop_pairs,
            batch_size=self.params.batch_size
        )

        # Build event catalog
        events_arr = np.array(valid_hop_pairs, dtype=np.int32)
        barriers_arr = barriers_array.astype(np.float32)
        rates_arr = (self.params.nu0 * np.exp(-barriers_arr / self.params.kbt)).astype(np.float32)
        
        self.catalog = EventCatalog.from_arrays(events_arr, barriers_arr, rates_arr)
        
        logger.info(f"Event catalog created: {self.catalog.size} events, {self.catalog.capacity} capacity")
        logger.info(f"Memory usage: {self.catalog.memory_usage() / 1024:.1f} KB")
        logger.info(f"Utilization: {self.catalog.utilization():.1%}")
    
    def select_event(self) -> Tuple[Optional[int], Optional[float], Optional[int], 
                                   Optional[int], Optional[float], Optional[float]]:
        """Select a random event based on rates (KMC algorithm)
        
        Returns:
            Tuple of (event_idx, dt, vac, cat, barrier, rate)
            Returns (None, None, None, None, None, None) if no events
        """
        if self.catalog.size == 0:
            return None, None, None, None, None, None
        
        # Get rates
        rates = self.catalog.get_rates()
        total_rate = rates.sum()
        
        # Select event
        normalized_rates = rates / total_rate
        cumsum = np.cumsum(normalized_rates)
        r = np.random.random()
        event_idx = np.searchsorted(cumsum, r)
        
        # Time increment
        dt = -np.log(np.random.random()) / total_rate
        
        # Get event info
        events = self.catalog.get_events()
        barriers = self.catalog.get_barriers()
        
        vac, cat = events[event_idx]
        barrier = barriers[event_idx]
        rate = rates[event_idx]
        
        return event_idx, dt, vac, cat, barrier, rate
    
    def execute_hop(self, vac_idx: int, cat_idx: int) -> Tuple[int, int]:
        """Execute the hop by swapping atoms
        
        Args:
            vac_idx: Vacancy index (before hop)
            cat_idx: Cation index (before hop)
        
        Returns:
            Tuple of (new_vac_idx, new_cat_idx) after hop
        """
        self.structure.swap_atoms(vac_idx, cat_idx)
        
        # Return new indices
        new_vac_idx = cat_idx  # Was cation, now vacancy
        new_cat_idx = vac_idx  # Was vacancy, now cation
        
        return new_vac_idx, new_cat_idx
    
    def update_after_hop(self, vac_idx: int, cat_idx: int) -> None:
        """Update event catalog after a hop

        Args:
            vac_idx: Vacancy position BEFORE hop
            cat_idx: Cation position BEFORE hop
        """
        # Step 1: Remove old events
        self._remove_old_events(vac_idx, cat_idx)

        # Step 2: Execute hop
        new_vac_idx, new_cat_idx = self.execute_hop(vac_idx, cat_idx)

        # Step 3: Add new events
        hop_pairs = self._collect_new_hop_pairs(new_vac_idx, new_cat_idx)
        self._add_new_events(hop_pairs)
    
    def _remove_old_events(self, vac_idx: int, cat_idx: int) -> None:
        """Remove events involving the vacancy and cation"""
        events = self.catalog.get_events()
        keep_mask = (events[:, 0] != vac_idx) & (events[:, 1] != cat_idx)
        self.catalog.remove_by_mask(keep_mask)
    
    def _collect_new_hop_pairs(self, new_vac_idx: int, new_cat_idx: int) -> List[Tuple[int, int]]:
        """Collect new hop pairs after a hop

        Intelligently detects vacancy type:
        - If new vacancy is a cation vacancy (X), scan for cation neighbors
        - If new vacancy is an oxygen vacancy (XO), scan for oxygen neighbors
        """
        hop_pairs = []

        # Detect the type of the new vacancy
        is_cation_vacancy = self.structure.vacancy_mask[new_vac_idx]
        is_oxygen_vacancy = self.structure.oxygen_vacancy_mask[new_vac_idx]

        # New vacancy - add events based on vacancy type
        if is_cation_vacancy:
            # Cation vacancy: scan for cation neighbors
            neighbors_idx = self.neighbors.nearest_neighbors_csr.get_neighbors(new_vac_idx)
            for neighbor_idx in neighbors_idx:
                if self.structure.cation_mask[neighbor_idx]:
                    hop_pairs.append((new_vac_idx, neighbor_idx))

        elif is_oxygen_vacancy:
            # Oxygen vacancy: scan for oxygen neighbors
            neighbors_idx = self.neighbors.oxygen_neighbors_csr.get_neighbors(new_vac_idx)
            for neighbor_idx in neighbors_idx:
                if self.structure.oxygen_mask[neighbor_idx]:
                    hop_pairs.append((new_vac_idx, neighbor_idx))

        # Detect the type of the new atom (what was the vacancy before)
        is_cation = self.structure.cation_mask[new_cat_idx]
        is_oxygen = self.structure.oxygen_mask[new_cat_idx]

        # New atom - add events from its vacancy neighbors
        if is_cation:
            # Cation: scan for cation vacancy neighbors
            neighbors_idx = self.neighbors.nearest_neighbors_csr.get_neighbors(new_cat_idx)
            for neighbor_idx in neighbors_idx:
                if self.structure.vacancy_mask[neighbor_idx]:
                    hop_pairs.append((neighbor_idx, new_cat_idx))

        elif is_oxygen:
            # Oxygen: scan for oxygen vacancy neighbors
            neighbors_idx = self.neighbors.oxygen_neighbors_csr.get_neighbors(new_cat_idx)
            for neighbor_idx in neighbors_idx:
                if self.structure.oxygen_vacancy_mask[neighbor_idx]:
                    hop_pairs.append((neighbor_idx, new_cat_idx))

        return hop_pairs
    
    def _add_new_events(self, hop_pairs: List[Tuple[int, int]]) -> None:
        """Compute barriers and add new events to catalog"""
        if len(hop_pairs) == 0:
            return

        # Batch compute barriers (filters out invalid hops)
        barriers_array, valid_hop_pairs = self.barrier_calc.compute_barriers_batch(
            self.structure, self.neighbors, hop_pairs,
            batch_size=self.params.batch_size
        )

        if len(valid_hop_pairs) == 0:
            return

        # Create events
        new_events = np.array(valid_hop_pairs, dtype=np.int32)
        new_barriers = barriers_array.astype(np.float32)
        new_rates = (self.params.nu0 * np.exp(-new_barriers / self.params.kbt)).astype(np.float32)

        # Add to catalog
        self.catalog.add_events(new_events, new_barriers, new_rates)
