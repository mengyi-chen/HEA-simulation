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
from utils import KMCParams

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
        """Build initial event catalog"""
        logger.info("Building event catalog...")
        
        # Find all vacancies
        vacancy_indices = np.where(self.structure.vacancy_mask)[0]
        
        # Collect all possible hop pairs
        hop_pairs = []
        for vac in vacancy_indices:
            if self.neighbors.nearest_neighbors_csr.has_neighbors(vac):
                neighbors_idx, _ = self.neighbors.nearest_neighbors_csr.get_neighbors_with_distances(vac)
                for neighbor_idx in neighbors_idx:
                    if self.structure.cation_mask[neighbor_idx]:
                        hop_pairs.append((vac, neighbor_idx))
        
        logger.info(f"Collected {len(hop_pairs)} potential events")

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
    
    def update_after_hop(self, vac_idx: int, cat_idx: int) -> dict:
        """Update event catalog after a hop

        Args:
            vac_idx: Vacancy position BEFORE hop
            cat_idx: Cation position BEFORE hop

        Returns:
            dict with timing statistics for each step
        """
        import time

        timings = {}

        # Step 1: Remove old events
        t0 = time.perf_counter()
        self._remove_old_events(vac_idx, cat_idx)
        timings['remove'] = time.perf_counter() - t0

        # Step 2: Execute hop
        t0 = time.perf_counter()
        new_vac_idx, new_cat_idx = self.execute_hop(vac_idx, cat_idx)
        timings['execute'] = time.perf_counter() - t0

        # Step 3: Add new events
        t0 = time.perf_counter()
        hop_pairs = self._collect_new_hop_pairs(new_vac_idx, new_cat_idx)
        timings['collect'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        add_timings = self._add_new_events(hop_pairs)
        timings['add_events'] = time.perf_counter() - t0

        # Add detailed breakdown of add_events
        timings['add_compute_barriers'] = add_timings['compute_barriers']
        timings['add_create_arrays'] = add_timings['create_arrays']
        timings['add_catalog_add'] = add_timings['catalog_add']

        return timings
    
    def _remove_old_events(self, vac_idx: int, cat_idx: int) -> None:
        """Remove events involving the vacancy and cation"""
        events = self.catalog.get_events()
        keep_mask = (events[:, 0] != vac_idx) & (events[:, 1] != cat_idx)
        self.catalog.remove_by_mask(keep_mask)
    
    def _collect_new_hop_pairs(self, new_vac_idx: int, new_cat_idx: int) -> List[Tuple[int, int]]:
        """Collect new hop pairs after a hop"""
        hop_pairs = []
        
        # New vacancy - add events to its cation neighbors
        if self.neighbors.nearest_neighbors_csr.has_neighbors(new_vac_idx):
            neighbors_idx, _ = self.neighbors.nearest_neighbors_csr.get_neighbors_with_distances(new_vac_idx)
            for neighbor_idx in neighbors_idx:
                if self.structure.cation_mask[neighbor_idx]:
                    hop_pairs.append((new_vac_idx, neighbor_idx))
        
        # New cation - add events from its vacancy neighbors
        if self.neighbors.nearest_neighbors_csr.has_neighbors(new_cat_idx):
            neighbors_idx, _ = self.neighbors.nearest_neighbors_csr.get_neighbors_with_distances(new_cat_idx)
            for neighbor_idx in neighbors_idx:
                if self.structure.vacancy_mask[neighbor_idx]:
                    hop_pairs.append((neighbor_idx, new_cat_idx))
        
        return hop_pairs
    
    def _add_new_events(self, hop_pairs: List[Tuple[int, int]]) -> dict:
        """Compute barriers and add new events to catalog

        Returns:
            dict with timing statistics for barrier computation and catalog update
        """
        import time

        timings = {}

        if len(hop_pairs) == 0:
            return {'compute_barriers': 0.0, 'create_arrays': 0.0, 'catalog_add': 0.0}

        # Batch compute barriers (filters out invalid hops)
        t0 = time.perf_counter()
        barriers_array, valid_hop_pairs = self.barrier_calc.compute_barriers_batch(
            self.structure, self.neighbors, hop_pairs,
            batch_size=self.params.batch_size
        )
        timings['compute_barriers'] = time.perf_counter() - t0

        if len(valid_hop_pairs) == 0:
            return {'compute_barriers': timings['compute_barriers'], 'create_arrays': 0.0, 'catalog_add': 0.0}

        # Create events
        t0 = time.perf_counter()
        new_events = np.array(valid_hop_pairs, dtype=np.int32)
        new_barriers = barriers_array.astype(np.float32)
        new_rates = (self.params.nu0 * np.exp(-new_barriers / self.params.kbt)).astype(np.float32)
        timings['create_arrays'] = time.perf_counter() - t0

        # Add to catalog
        t0 = time.perf_counter()
        self.catalog.add_events(new_events, new_barriers, new_rates)
        timings['catalog_add'] = time.perf_counter() - t0

        return timings
