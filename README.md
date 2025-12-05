# Chromatin Polymer Modeling: Loop Extrusion with Extruder Dynamics and Noise

This repository contains the code and data used in the article:

**"Effects of Extruder Dynamics and Noise on Simulated Chromatin Contact Probability Curves"**

## Overview

This project implements a computational framework for simulating chromatin folding via loop extrusion by SMC complexes (condensins). The framework combines:

- **1D simulation**: Loop extrusion dynamics on a 1D lattice with SMC complexes
- **3D simulation**: Molecular dynamics using [Polychrom](https://github.com/open2c/polychrom) for polymer folding
- **Contact analysis**: Hi-C-like contact maps and P(s) curves from 3D conformations

The code supports two types of repulsive potentials for non-bonded interactions:
- **Step potential**: Steep polynomial repulsion (hard sphere-like)
- **DPD potential**: Dissipative Particle Dynamics (soft sphere-like)

## Repository Structure

```
article_code/
├── README.md                           # This file
│
├── Core Simulation Scripts
├── fullsimulation_stepforce.py         # Full workflow with Step potential
├── fullsimulation_dpdforce.py          # Full workflow with DPD potential
├── ensemble_director.py                # 1D simulation orchestration
├── file_handlers.py                    # Data I/O utilities
│
├── Force and Geometry Modules
├── spatial_functions.py                # 3D forces, starting conformations
├── smc.py                              # SMC complex classes (Condensin, Cohesin, etc.)
├── bead.py                             # Basic bead classes
│
├── Analysis and Visualization
├── draw_and_plot.py                    # Plotting functions for figures
├── loop_statistics.py                  # Loop size distribution analysis
├── spiral_metrics.py                   # Spiral correlation C(s) analysis
├── cuda_utils.py                       # GPU-accelerated contact detection
│
├── Data Processing
├── map_creation_example.py             # Generate .mcool contact maps
├── prepare_goloborodko_data.py         # Process reference data from Samejima et al.
│
├── Jupyter Notebooks
├── tutorial_chromatin_modeling.ipynb   # Step-by-step tutorial
├── article_figures_visualization.ipynb # Generate all article figures
│
├── Data Directories
├── data_for_figs/                      # P(s) curves (CSV format)
├── cells_cond1_gol_p_plen3/            # Example simulation data
├── figures/                            # Generated figure outputs
└── reference_data/                     # Data from Samejima et al. (optional)
```

## Requirements

### Python Version
- Python 3.7 or higher

### Core Dependencies
```bash
# Polymer simulation
polychrom>=0.1.0
openmm>=7.5.0

# Scientific computing
numpy>=1.19.0
scipy>=1.5.0
pandas>=1.1.0
h5py>=2.10.0

# Visualization
matplotlib>=3.3.0
seaborn>=0.11.0

# Contact map analysis
cooler>=0.8.11
cooltools>=0.5.1

# GPU acceleration (optional, for faster contact detection)
numba>=0.53.0
cudatoolkit>=11.0  # If using CUDA-enabled GPU
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chromatin-modeling.git
cd chromatin-modeling/article_code
```

2. Install dependencies:
```bash
# Using conda (recommended)
conda create -n chromatin python=3.9
conda activate chromatin
conda install -c conda-forge polychrom openmm numpy scipy pandas h5py matplotlib seaborn
pip install cooler cooltools

# For GPU acceleration
conda install numba cudatoolkit=11.2

# Or using pip
pip install -r requirements.txt  # (create this file with above packages)
```

## Quick Start

### 1. Tutorial Notebook

The easiest way to get started is with the tutorial notebook:

```bash
jupyter notebook tutorial_chromatin_modeling.ipynb
```

This notebook demonstrates:
- Creating SMC complexes (static and dynamic extruders)
- Running 1D loop extrusion simulation
- Running 3D Polychrom simulation with custom forces
- Generating contact maps and P(s) curves

### 2. Reproducing Simulations

To reproduce the full simulation data used in the article:

#### Step Potential Simulations
```bash
python fullsimulation_stepforce.py
```

#### DPD Potential Simulations
```bash
python fullsimulation_dpdforce.py
```

**Note:** Full simulations can take several hours to days depending on your hardware. Parameters can be adjusted in the scripts:
- `N`: Number of beads (default: 200,000 for 40 Mbp)
- `num_cells`: Number of independent replicates (default: 5)
- `n_lifetimes`: Number of condensin lifetimes (default: 5)

### 3. Generating Contact Maps

After running simulations, create .mcool contact maps:

```bash
python map_creation_example.py
```

This script:
1. Adds optional Gaussian noise to 3D conformations
2. Detects contacts using GPU-accelerated functions
3. Generates multi-resolution .mcool files

### 4. Creating Figures

Generate all figures from the article:

```bash
jupyter notebook article_figures_visualization.ipynb
```

## Reference Data

This project uses reference data from:

**Samejima, K., et al. (2024).** *"Mitotic chromosomes are self-entangled and disentangle through a topoisomerase-II-dependent two-stage exit from mitosis."* Science, 385(6714). DOI: [10.1126/science.adq1709](https://doi.org/10.1126/science.adq1709)

The reference simulations model condensin I as static extruders in mitotic chromosomes. To process this data for comparison:

```bash
python prepare_goloborodko_data.py --input path/to/samejima/data --output reference_data/
```

## Key Features

### Loop Extrusion Models

- **Static extruders**: SMCs with fixed positions (no turnover)
- **Dynamic extruders**: SMCs with finite lifetime and turnover
- **Directional blocking**: SMC complexes can block passage of other complexes
- **Variable processivity**: Control loop extrusion length

### Repulsive Potentials

1. **Step Potential** (`polynomial_repulsive`):
   - Steep repulsion at bead radius
   - Minimizes bead overlaps

2. **DPD Potential** (`dpd_repulsive`):
   - Smoother, softer repulsion

### Contact Map Analysis

- **GPU acceleration**: CUDA kernels for fast contact detection
- **Multiple resolutions**: 1-10 kb bins
- **Gaussian noise**: Optional smoothing (σ = 0-400 nm)
- **.mcool format**: Standard cooler format for compatibility

### Visualization

- **P(s) curves**: Contact probability vs genomic distance
- **Contact maps**: 2D Hi-C-like matrices
- **Loop distributions**: Size distributions for inner/outer loops
- **Spiral metrics**: C(s) correlation for helical structures

## File Formats

### Input
- **JSON**: Simulation parameters, bead sizes, SMC positions
- **HDF5**: 1D simulation trajectories (SMC positions over time)

### Output
- **HDF5**: 3D polymer conformations (Polychrom format)
- **.mcool**: Multi-resolution contact maps (cooler format)
- **CSV**: P(s) curves, loop statistics, spiral correlations
- **SVG/PNG**: Publication-quality figures

## Performance Notes

- **GPU acceleration**: Strongly recommended for contact detection
  - CPU: ~10 min per 200k bead conformation
  - GPU (CUDA): ~10 sec per 200k bead conformation
  
- **Memory requirements**:
  - 1D simulation: ~1 GB
  - 3D simulation: ~8 GB per trajectory
  - Contact map generation: ~16 GB (for 200k beads)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024chromatin,
  title={Effects of Extruder Dynamics and Noise on Simulated Chromatin Contact Probability Curves},
  author={Konstantinov et al.},
  journal={In process},
  year={2025-2026},
  doi={...}
}
```

And reference data from:
```bibtex
@article{samejima2024mitotic,
  title={Mitotic chromosomes are self-entangled and disentangle through a topoisomerase-II-dependent two-stage exit from mitosis},
  author={Samejima, Kumiko and others},
  journal={Science},
  volume={385},
  number={6714},
  year={2024},
  doi={10.1126/science.adq1709}
}
```

This code was prepared and structured with the support of Cursor AI.

**Last updated:** December 2025

