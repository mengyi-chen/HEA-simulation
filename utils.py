"""
Base utilities and data structures for kMC simulation
"""
import numpy as np
import random
import torch
from ase import Atoms
from ase.io import write
from dataclasses import dataclass
from enum import IntEnum
from io import StringIO
from ase.io import read

class AtomType(IntEnum):
    """Enum for atom types in spinel structure"""
    OXYGEN = 0              # Regular oxygen (O)
    OXYGEN_VACANCY = 1      # Oxygen vacancy (XO)
    CATION_A = 2           # Cation on A-site (tetrahedral)
    CATION_B = 3           # Cation on B-site (octahedral)
    VACANCY_A = 4          # Cation vacancy on A-site (X)
    VACANCY_B = 5          # Cation vacancy on B-site (X)


@dataclass
class KMCParams:
    """All parameters for kinetic Monte Carlo simulation"""
    # Simulation parameters
    temperature: float = 1200.0  # Temperature in Kelvin
    n_cutoff_cell: int = 1  # Number of unit cells for cutoff radius (half-length of local cube)
    batch_size: int = 64  # Batch size for CHGNet barrier calculations
    nu0: float = 1e13  # Attempt frequency in Hz
    kb: float = 8.617e-5  # Boltzmann constant in eV/K

    # Spinel structure parameters
    unit_cell_size: float = 8.5333518982  # Unit cell lattice constant in Angstrom
    coord_tolerance: float = 0.02  # Tolerance for coordinate comparison

    # Nearest-neighbor distances in ideal cubic spinel structure:
    #   A-B (1st nearest metal): 12 neighbors at (√11/8)a ≈ 0.415a
    #   A-A (2nd nearest metal): 4 neighbors at (√3/4)a ≈ 0.433a
    #   B-B (1st nearest metal): 6 neighbors at (√2/4)a ≈ 0.354a
    #   B-A (2nd nearest metal): 12 neighbors at (√11/8)a ≈ 0.415a
    #   O-O (32e sites): 12 neighbors at (√2/4)a ≈ 0.354a

    # Nearest-neighbor distances in ideal cubic spinel structure
    nn_factor_A: float = 0.454663336987  # (√3/4) * 1.05 for A-A neighbors
    nn_factor_B: float = 0.435307003734  # (√11/8) * 1.05 for B-B neighbors
    nn_factor_O: float = 0.371231060123  # (√2/4) * 1.05 for O-O neighbors

    # Element list for SRO calculation
    elements: tuple = ('X', 'Ni', 'Co', 'Cu', 'Fe', 'Cr', 'Al')

    # Base energy barriers (in eV) for different elements
    base_barriers: dict = None

    def __post_init__(self):
        """Initialize base_barriers with default values if not provided"""
        if self.base_barriers is None:
            self.base_barriers = {
                'Cu': 0.6,   # Fastest
                'Fe': 0.9,
                'Co': 1.1,
                'Ni': 1.4,
                'Al': 2.0,
                'Cr': 2.4,   # Slowest, anchors the lattice
                'O': 2.5,    # Oxygen diffusion baseline
            }

    @property
    def kbt(self):
        """Return kB*T for convenience"""
        return self.kb * self.temperature

    @property
    def cutoff(self):
        """Calculate cutoff radius from n_cutoff_cell"""
        return self.n_cutoff_cell * self.unit_cell_size

    @property
    def nn_distance_A(self):
        """first and second Nearest-neighbor cutoff for A-site diffusion"""
        return self.nn_factor_A * self.unit_cell_size

    @property
    def nn_distance_B(self):
        """first and second Nearest-neighbor cutoff for B-site diffusion"""
        return self.nn_factor_B * self.unit_cell_size

    @property
    def nn_distance_O(self):
        """Nearest-neighbor cutoff for O-site diffusion"""
        return self.nn_factor_O * self.unit_cell_size


def set_seed(seed):
    """Set the seed for reproducibility across NumPy, Python, and PyTorch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_poscar_with_custom_symbols(poscar_path, custom_symbol_map=None):
    """Read POSCAR file with support for custom element symbols

    Args:
        poscar_path: Path to POSCAR file
        custom_symbol_map: Dict mapping custom symbols to placeholder elements

    Returns:
        cell: 3x3 cell matrix
        positions: Nx3 scaled positions
        symbols: List of element symbols (with custom symbols restored)
    """
    if custom_symbol_map is None:
        custom_symbol_map = {'XO': 'He'}

    with open(poscar_path, 'r') as f:
        lines = f.readlines()

    reverse_map = {v: k for k, v in custom_symbol_map.items()}
    modified_lines = lines.copy()

    for custom_sym, placeholder in custom_symbol_map.items():
        if len(lines) > 5:
            modified_lines[5] = modified_lines[5].replace(custom_sym, placeholder)

    poscar_str = ''.join(modified_lines)

    atoms = read(StringIO(poscar_str), format='vasp')

    cell = atoms.cell.array
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

    for placeholder, custom_sym in reverse_map.items():
        symbols = [custom_sym if s == placeholder else s for s in symbols]

    return cell, positions, symbols


def write_poscar(positions, cell, symbols, filename, pbc=[True, True, False]):
    """Write configuration as POSCAR file with proper element sorting"""
    symbols_for_ase = ['He' if s == 'XO' else s for s in symbols]
    positions_cart = positions @ cell

    unique_elements = []
    for s in symbols_for_ase:
        if s not in unique_elements:
            unique_elements.append(s)

    element_to_idx = {elem: i for i, elem in enumerate(unique_elements)}
    sort_indices = sorted(range(len(symbols_for_ase)),
                          key=lambda i: element_to_idx[symbols_for_ase[i]])

    sorted_positions = positions_cart[sort_indices]
    sorted_symbols = [symbols_for_ase[i] for i in sort_indices]

    atoms = Atoms(
        symbols=sorted_symbols,
        positions=sorted_positions,
        cell=cell,
        pbc=pbc
    )

    write(filename, atoms, format='vasp', direct=True, vasp5=True)

    with open(filename, 'r') as f:
        lines = f.readlines()

    if len(lines) > 5:
        lines[5] = lines[5].replace('He', 'XO')

    with open(filename, 'w') as f:
        f.writelines(lines)
