"""
Optimized data structures for kMC simulation
- CSRNeighborList: Memory-efficient neighbor storage
- EventCatalog: Dynamic event management with pre-allocation
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class CSRNeighborList:
    """Compressed Sparse Row format for neighbor lists
    
    Memory-efficient storage for neighbor connectivity.
    """
    
    def __init__(self, n_atoms: int):
        self.n_atoms = n_atoms
        self.indices = np.array([], dtype=np.int32)
        self.distances = np.array([], dtype=np.float32)
        self.indptr = np.zeros(n_atoms + 1, dtype=np.int32)
    
    @classmethod
    def from_dict(cls, neighbor_dict: Dict[int, List], n_atoms: int, 
                  with_distances: bool = False) -> 'CSRNeighborList':
        """Build CSR from dictionary of lists"""
        csr = cls(n_atoms)
        
        total_neighbors = sum(len(neighbors) for neighbors in neighbor_dict.values())
        
        csr.indices = np.empty(total_neighbors, dtype=np.int32)
        if with_distances:
            csr.distances = np.empty(total_neighbors, dtype=np.float32)
        
        offset = 0
        for atom_i in range(n_atoms):
            csr.indptr[atom_i] = offset
            
            if atom_i in neighbor_dict:
                neighbors = neighbor_dict[atom_i]
                n_neighbors = len(neighbors)
                
                if with_distances:
                    for idx, (neighbor_j, dist) in enumerate(neighbors):
                        csr.indices[offset + idx] = neighbor_j
                        csr.distances[offset + idx] = dist
                else:
                    csr.indices[offset:offset + n_neighbors] = neighbors
                
                offset += n_neighbors
        
        csr.indptr[n_atoms] = offset
        return csr
    
    def get_neighbors(self, atom_i: int) -> np.ndarray:
        """Get neighbor indices for atom i"""
        start = self.indptr[atom_i]
        end = self.indptr[atom_i + 1]
        return self.indices[start:end]
    
    def get_neighbors_with_distances(self, atom_i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get neighbors and distances for atom i"""
        start = self.indptr[atom_i]
        end = self.indptr[atom_i + 1]
        return self.indices[start:end], self.distances[start:end]
    
    def has_neighbors(self, atom_i: int) -> bool:
        """Check if atom i has any neighbors"""
        return self.indptr[atom_i + 1] > self.indptr[atom_i]
    
    def n_neighbors(self, atom_i: int) -> int:
        """Get number of neighbors for atom i"""
        return self.indptr[atom_i + 1] - self.indptr[atom_i]
    
    def to_dict(self, with_distances: bool = False) -> Dict[int, List]:
        """Convert CSR back to dictionary (for compatibility)"""
        result = {}
        for atom_i in range(self.n_atoms):
            if self.has_neighbors(atom_i):
                if with_distances:
                    neighbors, dists = self.get_neighbors_with_distances(atom_i)
                    result[atom_i] = [(n, d) for n, d in zip(neighbors, dists)]
                else:
                    result[atom_i] = self.get_neighbors(atom_i).tolist()
        return result
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        return (self.indices.nbytes + self.indptr.nbytes + 
                (self.distances.nbytes if len(self.distances) > 0 else 0))


class EventCatalog:
    """Dynamic array for events with pre-allocation
    
    Efficiently manages add/remove operations during kMC simulation.
    """
    
    def __init__(self, initial_capacity: int = 10000):
        self._events = np.empty((initial_capacity, 2), dtype=np.int32)
        self._barriers = np.empty(initial_capacity, dtype=np.float32)
        self._rates = np.empty(initial_capacity, dtype=np.float32)
        self._size = 0
        self._capacity = initial_capacity
    
    @classmethod
    def from_arrays(cls, events: np.ndarray, barriers: np.ndarray, 
                   rates: np.ndarray) -> 'EventCatalog':
        """Create catalog from existing arrays"""
        n_events = len(events)
        capacity = max(n_events * 2, 10000)
        
        catalog = cls(initial_capacity=capacity)
        catalog._events[:n_events] = events
        catalog._barriers[:n_events] = barriers
        catalog._rates[:n_events] = rates
        catalog._size = n_events
        
        return catalog
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def capacity(self) -> int:
        return self._capacity
    
    def get_events(self) -> np.ndarray:
        return self._events[:self._size]
    
    def get_barriers(self) -> np.ndarray:
        return self._barriers[:self._size]
    
    def get_rates(self) -> np.ndarray:
        return self._rates[:self._size]
    
    def add_events(self, new_events: np.ndarray, new_barriers: np.ndarray, 
                  new_rates: np.ndarray) -> None:
        """Add new events (batch operation)"""
        n_new = len(new_events)
        if n_new == 0:
            return
        
        while self._size + n_new > self._capacity:
            self._grow()
        
        self._events[self._size:self._size + n_new] = new_events
        self._barriers[self._size:self._size + n_new] = new_barriers
        self._rates[self._size:self._size + n_new] = new_rates
        self._size += n_new
    
    def remove_by_mask(self, keep_mask: np.ndarray) -> None:
        """Remove events where keep_mask is False"""
        n_keep = keep_mask.sum()
        
        self._events[:n_keep] = self._events[:self._size][keep_mask]
        self._barriers[:n_keep] = self._barriers[:self._size][keep_mask]
        self._rates[:n_keep] = self._rates[:self._size][keep_mask]
        self._size = n_keep
    
    def _grow(self) -> None:
        """Double capacity"""
        new_capacity = self._capacity * 2
        
        new_events = np.empty((new_capacity, 2), dtype=np.int32)
        new_barriers = np.empty(new_capacity, dtype=np.float32)
        new_rates = np.empty(new_capacity, dtype=np.float32)
        
        new_events[:self._size] = self._events[:self._size]
        new_barriers[:self._size] = self._barriers[:self._size]
        new_rates[:self._size] = self._rates[:self._size]
        
        self._events = new_events
        self._barriers = new_barriers
        self._rates = new_rates
        self._capacity = new_capacity
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Export as compact arrays (for checkpointing)"""
        return (
            self._events[:self._size].copy(),
            self._barriers[:self._size].copy(),
            self._rates[:self._size].copy()
        )
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        return self._events.nbytes + self._barriers.nbytes + self._rates.nbytes
    
    def utilization(self) -> float:
        """Get memory utilization (fraction of capacity used)"""
        return self._size / self._capacity if self._capacity > 0 else 0.0
