"""
Spinel structure management and atom classification
"""
import numpy as np
import logging
from utils import AtomType, KMCParams

logger = logging.getLogger(__name__)


class SpinelStructure:
    """Manages atomic structure and atom type classification for spinel
    
    Responsibilities:
    - Store cell, positions, and symbols
    - Classify atoms by type (A-site, B-site, vacancies, oxygen)
    - Provide convenient masks for different atom types
    - Handle structure modifications (swapping atoms)
    """
    
    def __init__(self, cell: np.ndarray, positions: np.ndarray, 
                 symbols: list, params: KMCParams):
        """Initialize spinel structure
        
        Args:
            cell: 3x3 cell matrix
            positions: Nx3 fractional positions
            symbols: List of atomic symbols
            params: KMCParams instance
        """
        self.cell = cell
        self.positions = positions
        self.symbols = symbols
        self.params = params
        
        # Determine supercell size
        self.supercell_size = self._detect_supercell_size()
        
        # Classify atoms
        self.atom_types = np.zeros(len(symbols), dtype=np.int8)
        self._classify_atoms()
        
        logger.info(f"Spinel structure initialized:")
        logger.info(f"Supercell: {self.supercell_size[0]}x{self.supercell_size[1]}x{self.supercell_size[2]}")
        self._log_atom_statistics()
    
    def _detect_supercell_size(self) -> np.ndarray:
        """Detect supercell dimensions from lattice constants"""
        return np.array([
            int(round(self.cell[0, 0] / self.params.unit_cell_size)),
            int(round(self.cell[1, 1] / self.params.unit_cell_size)),
            int(round(self.cell[2, 2] / self.params.unit_cell_size))
        ])
    
    def _classify_atoms(self) -> None:
        """Classify all atoms by type using spinel structure rules"""
        tol = self.params.coord_tolerance
        
        # First pass: oxygen and oxygen vacancies
        for i, s in enumerate(self.symbols):
            if s == 'O':
                self.atom_types[i] = AtomType.OXYGEN
            elif s == 'XO':
                self.atom_types[i] = AtomType.OXYGEN_VACANCY
        
        # Second pass: classify cation sites
        cation_vacancy_indices = np.array([
            i for i, s in enumerate(self.symbols)
            if s not in ['O', 'XO']
        ])
        
        for atom_idx in cation_vacancy_indices:
            pos = self.positions[atom_idx]
            pos_wrapped = pos % 1.0
            coords_scaled = (pos_wrapped * self.supercell_size) % 1.0
            
            # A-sites: all coordinates at 0, 0.25, 0.5, or 0.75
            is_A_site = all(
                abs(coord - np.round(coord * 4) / 4) <= tol
                for coord in coords_scaled
            )
            
            if self.symbols[atom_idx] == 'X':
                self.atom_types[atom_idx] = AtomType.VACANCY_A if is_A_site else AtomType.VACANCY_B
            else:
                self.atom_types[atom_idx] = AtomType.CATION_A if is_A_site else AtomType.CATION_B
    
    def _log_atom_statistics(self) -> None:
        """Log atom type distribution"""
        logger.info(f"Cation vacancies (X): {self.vacancy_mask.sum()}")
        logger.info(f"Oxygen vacancies (XO): {(self.atom_types == AtomType.OXYGEN_VACANCY).sum()}")
        logger.info(f"Cations: {self.cation_mask.sum()}")
        logger.info(f"Oxygens (O): {(self.atom_types == AtomType.OXYGEN).sum()}")
        logger.info(f"A-sites: {self.A_mask.sum()}")
        logger.info(f"B-sites: {self.B_mask.sum()}")
    
    @property
    def n_atoms(self) -> int:
        """Total number of atoms"""
        return len(self.positions)
    
    @property
    def vacancy_mask(self) -> np.ndarray:
        """Mask for all cation vacancies (X)"""
        return (self.atom_types == AtomType.VACANCY_A) | (self.atom_types == AtomType.VACANCY_B)
    
    @property
    def cation_mask(self) -> np.ndarray:
        """Mask for all cations (excluding vacancies and oxygen)"""
        return (self.atom_types == AtomType.CATION_A) | (self.atom_types == AtomType.CATION_B)
    
    @property
    def oxygen_mask(self) -> np.ndarray:
        """Mask for regular oxygen atoms (O, not XO)"""
        return self.atom_types == AtomType.OXYGEN
    
    @property
    def cation_vacancy_mask(self) -> np.ndarray:
        """Mask for cations and vacancies (excluding oxygen)"""
        return (
            (self.atom_types == AtomType.CATION_A) |
            (self.atom_types == AtomType.CATION_B) |
            (self.atom_types == AtomType.VACANCY_A) |
            (self.atom_types == AtomType.VACANCY_B)
        )
    
    @property
    def A_mask(self) -> np.ndarray:
        """Mask for all A-sites (tetrahedral)"""
        return (self.atom_types == AtomType.CATION_A) | (self.atom_types == AtomType.VACANCY_A)
    
    @property
    def B_mask(self) -> np.ndarray:
        """Mask for all B-sites (octahedral)"""
        return (self.atom_types == AtomType.CATION_B) | (self.atom_types == AtomType.VACANCY_B)
    
    def swap_atoms(self, idx1: int, idx2: int) -> None:
        """Swap two atoms (for executing hops)
        
        Args:
            idx1: First atom index
            idx2: Second atom index
        """
        # Swap symbols
        self.symbols[idx1], self.symbols[idx2] = self.symbols[idx2], self.symbols[idx1]
        
        # Swap atom types
        self.atom_types[idx1], self.atom_types[idx2] = self.atom_types[idx2], self.atom_types[idx1]
