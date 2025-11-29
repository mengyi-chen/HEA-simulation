# Cavity Healing Kinetic Monte Carlo (KMC) Simulation for HEA Spinel

A refactored Python implementation of Kinetic Monte Carlo simulations for cavity healing in High-Entropy Alloy (HEA) spinel structures. This code simulates atomic diffusion through cation and oxygen vacancy hopping mechanisms, using machine learning potentials (CHGNet, MACE) for energy barrier calculations.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for structure, neighbors, barriers, events, and logging
- **Multiple Energy Models**: Support for CHGNet, M3GNet, and MACE machine learning potentials
- **Dual Vacancy Dynamics**: Handles both cation and oxygen vacancy diffusion
- **Element-Specific Barriers**: Configurable base energy barriers for different elements
- **Efficient Event Catalog**: Optimized data structures for managing KMC events
- **Short-Range Order (SRO) Analysis**: Built-in calculation of Warren-Cowley SRO parameters
- **Checkpoint/Resume**: Save and resume simulations from checkpoints
- **Batch Energy Calculations**: Efficient batched computation of energy barriers

## Project Structure

```
cavity_kmc/
├── main.py                    # Main entry point and CavityHealingKMC class
├── structure.py               # Spinel structure management (SpinelStructure)
├── neighbors.py               # Neighbor list management (NeighborManager)
├── barriers.py                # Energy barrier calculations (BarrierCalculator)
├── events.py                  # Event catalog management (EventManager)
├── energy_models.py           # Energy model wrappers (CHGNet, M3GNet, MACE)
├── sro.py                     # Short-range order calculations (SROCalculator)
├── logging_utils.py           # Simulation logging (SimulationLogger)
├── checkpoint.py              # Checkpoint save/load functionality
├── utils.py                   # Utilities and data structures
├── optimized_structures.py    # EventCatalog implementation
├── config_uniform.json        # Example barrier configuration
└── generate_config/           # Structure generation scripts
```

## Requirements

- Python 3.7+
- NumPy
- PyTorch
- ASE (Atomic Simulation Environment)
- CHGNet (for CHGNet energy model)
- matgl (for M3GNet energy model, optional)
- MACE (for MACE energy model, optional)
- tqdm (for progress bars)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cavity_kmc
```

2. Install dependencies:
```bash
pip install numpy torch ase chgnet tqdm
```

3. (Optional) Install additional energy models:
```bash
pip install matgl  # For M3GNet
pip install mace-torch  # For MACE
```

## Usage

### Basic Usage

Run a KMC simulation with default CHGNet model:

```bash
python main.py --poscar_path ./generate_config/POSCAR_6x6x6_with_cavity.vasp \
               --temp 1000 \
               --steps 1000000 \
               --device 0
```

### Command Line Arguments

- `--device`: CUDA device ID or 'cpu' (default: '2')
- `--temp`: Temperature in Kelvin (default: 1000)
- `--cutoff`: Cutoff radius for neighbor list in Angstrom (default: 6.0)
- `--steps`: Number of KMC steps (default: 1,000,000)
- `--seed`: Random seed for reproducibility (default: 0)
- `--log_interval`: Save configurations every N steps (default: 1000)
- `--log_file`: CSV file for step logging (default: 'kmc_steps.csv')
- `--sro_interval`: Calculate SRO every N steps (default: 1000)
- `--sro_log_file`: CSV file for SRO data (default: 'sro_log.csv')
- `--checkpoint_interval`: Save checkpoint every N steps (default: 10000)
- `--batch_size`: Batch size for energy model (default: 64)
- `--energy_model`: Energy model to use: 'chgnet', 'm3gnet', or 'mace' (default: 'chgnet')
- `--mace_model_path`: Path to MACE model file (required if using MACE)
- `--barriers_config`: JSON config file for energy barriers (default: 'config_uniform.json')

### Using Different Energy Models

**CHGNet** (default):
```bash
python main.py --energy_model chgnet
```

**MACE**:
```bash
python main.py --energy_model mace --mace_model_path /path/to/mace-model.model
```

### Configuring Energy Barriers

Create a JSON configuration file (e.g., `config_custom.json`):

```json
{
  "base_barriers": {
    "Cu": 0.6,
    "Fe": 0.9,
    "Co": 1.1,
    "Ni": 1.4,
    "Al": 2.0,
    "Cr": 2.4,
    "O": 2.5
  }
}
```

Then run:
```bash
python main.py --barriers_config config_custom.json
```

### Resuming from Checkpoint

```bash
python main.py --resume_from ./configs_2025_11_26_14_20_16/checkpoint.npz \
               --steps 500000
```

## Output Files

Simulations create a timestamped directory (e.g., `configs_2025_11_26_14_20_16/`) containing:

- `kmc.log`: Detailed simulation log
- `kmc_steps.csv`: Step-by-step data (time, barriers, rates, vacancy counts)
- `sro_log.csv`: Warren-Cowley SRO parameters over time
- `POSCAR_step_*.vasp`: Structure snapshots at specified intervals
- `checkpoint.npz`: Checkpoint file for resuming

### Step Log Format

The `kmc_steps.csv` file contains:
- `step`: KMC step number
- `time`: Simulation time (seconds)
- `dt`: Time increment for this step
- `vac_idx`: Vacancy index (before hop)
- `cat_idx`: Cation/oxygen index (before hop)
- `barrier`: Energy barrier (eV)
- `rate`: Hopping rate (Hz)
- `n_cation_vacancies`: Number of cation vacancies
- `n_oxygen_vacancies`: Number of oxygen vacancies
- `n_events`: Number of available events

### SRO Log Format

The `sro_log.csv` file contains Warren-Cowley SRO parameters for all element pairs.

## Algorithm Overview

### KMC Algorithm

1. **Initialization**: Build event catalog with all possible vacancy hops
2. **Event Selection**: Select event based on rates (proportional to exp(-E/kT))
3. **Time Update**: Advance time by dt = -ln(r)/total_rate
4. **Execute Hop**: Swap vacancy with neighboring atom
5. **Update Catalog**: Remove invalidated events, add new events
6. **Repeat**: Continue until target steps reached

### Energy Barrier Model

```
E_barrier = E_base + max(0, ΔE_MLP)
```

where:
- `E_base`: Element-specific base barrier
- `ΔE_MLP`: Energy difference from machine learning potential (E_final - E_initial)

## Spinel Structure

The code handles normal spinel structures (AB₂O₄) with:
- **A-sites**: Tetrahedral coordination (8a Wyckoff position)
- **B-sites**: Octahedral coordination (16d Wyckoff position)
- **Oxygen**: 32e Wyckoff position

Supported atom types:
- `X`: Cation vacancy
- `XO`: Oxygen vacancy
- Regular elements: Ni, Co, Cu, Fe, Cr, Al, etc.
- `O`: Oxygen

## Performance Considerations

- **Batch Size**: Larger batch sizes improve GPU utilization but require more memory
- **Cutoff Radius**: Smaller cutoffs reduce neighbor list size and computation
- **Checkpoint Interval**: More frequent checkpointing increases I/O overhead
- **Log Interval**: Reduce logging frequency for faster simulations

## License

This project is licensed under the MIT License. 