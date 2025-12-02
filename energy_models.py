"""
Energy model abstraction for flexible ML potential integration

Supports multiple ML models:
- CHGNet
- MACE
- Other custom models
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from ase import Atoms
import torch

class EnergyModel(ABC):
    """Abstract base class for energy models

    This allows flexible switching between different ML potentials
    while maintaining a consistent interface.
    """

    @abstractmethod
    def predict_energy(self,
                       structures: Union[Atoms, List[Atoms]],
                       batch_size: int = 64) -> np.ndarray:
        """Predict energies for one or more structures

        Args:
            structures: Single structure or list of structures (ASE Atoms format)
            batch_size: Batch size for prediction

        Returns:
            energies: Array of energies in eV
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model"""
        pass


class CHGNetModel(EnergyModel):
    """CHGNet energy model wrapper"""

    def __init__(self, device='cpu', graph_cutoff: float = 10.0):
        """Initialize CHGNet model

        Args:
            device: Device for computation ('cpu' or 'cuda:0', etc.)
            graph_cutoff: Graph cutoff radius for CHGNet
        """
        from chgnet.model import CHGNet
        from pymatgen.io.ase import AseAtomsAdaptor

        self.model = CHGNet.load(use_device=device)
        self.model.graph_converter.atom_graph_cutoff = graph_cutoff
        
        # NOTE: calculat the total energy, not intensive property
        self.model.is_intensive = False 
        self.device = device
        self.adaptor = AseAtomsAdaptor()

    def predict_energy(self,
                       structures: Union[Atoms, List[Atoms]],
                       batch_size: int = 64) -> np.ndarray:
        """Predict energies using CHGNet

        Args:
            structures: Single structure or list of structures (ASE Atoms)
            batch_size: Batch size for prediction

        Returns:
            energies: Array of energies in eV
        """

        # Convert ASE Atoms to pymatgen Structure for CHGNet
        if not isinstance(structures, list):
            structures = [structures]

        pmg_structures = [self.adaptor.get_structure(s) for s in structures]

        # Use no_grad since we only need energy, not forces/gradients
        with torch.no_grad():
            results = self.model.predict_structure(
                pmg_structures,
                task='e',
                batch_size=batch_size
            )

        # Extract energies
        if isinstance(results, list):
            return np.array([r['e'] for r in results])
        else:
            return np.array([results['e']])

    def get_model_name(self) -> str:
        return "CHGNet"

class MACEModel(EnergyModel):
    """MACE energy model wrapper"""

    def __init__(self, device='cpu', model_path: str = None):
        """Initialize MACE model

        Args:
            device: Device for computation ('cpu', 'cuda', or torch.device object)
            model_path: Path to MACE model file (e.g., /path/to/mace-mh-1.model)
        """
        from mace.calculators import MACECalculator
        import torch

        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            device_str = str(device)
        else:
            device_str = device

        # Load MACE model from file
        self.calculator = MACECalculator(
            model_paths=model_path,
            device=device_str,
            default_dtype='float32',
            head="omat_pbe"  
        )
        self.device = device_str

    def predict_energy(self,
                       structures: Union[Atoms, List[Atoms]],
                       batch_size: int = 64) -> np.ndarray:
        """Predict energies using MACE

        Args:
            structures: Single structure or list of structures (ASE Atoms)
            batch_size: Batch size for prediction (note: MACE processes one at a time)

        Returns:
            energies: Array of energies in eV
        """
        if not isinstance(structures, list):
            structures = [structures]

        energies = []
        for atoms in structures:
            # Attach calculator and compute energy
            atoms.calc = self.calculator
            energy = atoms.get_potential_energy()
            energies.append(energy)

        return np.array(energies)

    def get_model_name(self) -> str:
        return "MACE"


# Factory function for easy model creation
def create_energy_model(model_type: str, **kwargs) -> EnergyModel:
    """Factory function to create energy models

    Args:
        model_type: Type of model ('chgnet', 'mace')
        **kwargs: Model-specific arguments

    Returns:
        EnergyModel instance

    Examples:
        >>> model = create_energy_model('chgnet', device='cuda:0')
        >>> model = create_energy_model('mace', model_path='path/to/mace.model')
    """
    model_type = model_type.lower()

    if model_type == 'chgnet':
        return CHGNetModel(**kwargs)
    elif model_type == 'mace':
        return MACEModel(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'chgnet','mace'"
        )
