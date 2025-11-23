"""
Short-range order (SRO) calculation
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


def build_sro_element_pairs(elements: List[str]) -> List[str]:
    """Build list of element pairs for SRO calculation
    
    Args:
        elements: List of element symbols
    
    Returns:
        List of element pair strings (e.g., ['X-X', 'X-Ni', ...])
    """
    cations = [e for e in elements if e != 'X']
    
    sro_pairs = []
    sro_pairs.append('X-X')  # Vacancy clustering
    
    for cation in cations:
        sro_pairs.append(f'X-{cation}')
    
    for i, elem_A in enumerate(cations):
        for j, elem_B in enumerate(cations):
            if j >= i:
                sro_pairs.append(f"{elem_A}-{elem_B}")
    
    return sro_pairs


class SROCalculator:
    """Calculates Warren-Cowley short-range order parameters
    
    Responsibilities:
    - Calculate symmetric Warren-Cowley parameters
    - Track element pairs
    - Format results for logging
    """
    
    def __init__(self, elements: List[str]):
        """Initialize SRO calculator
        
        Args:
            elements: List of elements to track
        """
        self.elements = list(elements)
        self.element_pairs = build_sro_element_pairs(self.elements)
    
    def calculate(self, symbols: List[str], neighbors: Dict) -> Dict[str, float]:
        """Calculate symmetric Warren-Cowley SRO parameters
        
        Args:
            symbols: List of atomic symbols
            neighbors: Dict mapping atom_i -> [neighbor_indices]
        
        Returns:
            Dict with keys like 'Ni-Co' and symmetric α values
        """
        # Count atoms
        symbol_counts = Counter(symbols)
        n_total = len(symbols)
        
        # Calculate concentrations
        concentrations = {}
        for elem in self.elements:
            concentrations[elem] = symbol_counts.get(elem, 0) / n_total
        
        # Count neighbor pairs
        pair_counts = {}
        type_neighbor_count = {}
        
        for atom_idx, symbol_i in enumerate(symbols):
            if symbol_i not in self.elements:
                continue
            
            neighbor_list = neighbors.get(atom_idx, [])
            
            # Handle both formats (list of indices or list of tuples)
            if len(neighbor_list) > 0 and isinstance(neighbor_list[0], tuple):
                neighbor_indices = [n[0] for n in neighbor_list]
            else:
                neighbor_indices = neighbor_list
            
            if len(neighbor_indices) == 0:
                continue
            
            # Count neighbors
            for neighbor_idx in neighbor_indices:
                symbol_j = symbols[neighbor_idx]
                
                if symbol_j not in self.elements:
                    continue
                
                pair_key = (symbol_i, symbol_j)
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
                type_neighbor_count[symbol_i] = type_neighbor_count.get(symbol_i, 0) + 1
        
        # Calculate symmetric Warren-Cowley parameters
        alpha_dict = {}
        
        for i, elem_A in enumerate(self.elements):
            for j, elem_B in enumerate(self.elements):
                if j < i:  # Skip lower triangle
                    continue
                
                pair_name = f"{elem_A}-{elem_B}"
                
                # Check if we have data
                if (concentrations[elem_A] == 0 or concentrations[elem_B] == 0 or
                    elem_A not in type_neighbor_count or elem_B not in type_neighbor_count):
                    alpha_dict[pair_name] = float('nan')
                    continue
                
                # Get pair counts
                n_AB = pair_counts.get((elem_A, elem_B), 0)
                n_BA = pair_counts.get((elem_B, elem_A), 0)
                
                # Calculate probabilities
                P_AB = n_AB / type_neighbor_count[elem_A] if type_neighbor_count[elem_A] > 0 else 0
                P_BA = n_BA / type_neighbor_count[elem_B] if type_neighbor_count[elem_B] > 0 else 0
                
                # Calculate directional alphas
                C_A = concentrations[elem_A]
                C_B = concentrations[elem_B]
                
                if C_B > 0 and C_A > 0:
                    alpha_AB = 1.0 - P_AB / C_B
                    alpha_BA = 1.0 - P_BA / C_A
                    alpha_sym = (alpha_AB + alpha_BA) / 2.0
                    alpha_dict[pair_name] = alpha_sym
                else:
                    alpha_dict[pair_name] = float('nan')
        
        return alpha_dict
