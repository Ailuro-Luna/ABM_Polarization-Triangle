# ABM Polarization Triangle

An Agent-Based Model (ABM) implementation for studying opinion polarization dynamics using the Polarization Triangle framework. This project simulates opinion formation and propagation in social networks with various configurations including zealot advocacy, moralization processes, and temporal dynamics.

## Features

- **Agent-Based Modeling**: Simulate opinion dynamics in social networks with configurable agent behaviors
- **Polarization Triangle Framework**: Implementation of the theoretical framework for studying opinion polarization
- **LFR Network Generation**: Generate realistic social networks using the Lancichinetti-Fortunato-Radicchi benchmark
- **Zealot Analysis**: Study the impact of zealot agents on opinion propagation and moralization
- **Sensitivity Analysis**: Comprehensive Sobol sensitivity analysis for parameter exploration
- **Network Pool System**: Efficient caching and reuse of generated networks
- **Multiple Experiment Types**: Support for various experimental configurations and analyses

## Installation

1. Clone the repository:
```bash
git clone git@github.com:Ailuro-Luna/ABM_Polarization-Triangle.git
cd ABM_Polarization-Triangle
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Network Pool

Before running any experiments, you need to generate the LFR network pool:

```bash
python polarization_triangle/scripts/generate_network_pool.py
```

This command creates a collection of pre-generated LFR networks that will be cached and reused across experiments for consistency and performance.

### 2. Run Experiments

#### Zealot Advocacy and Moralization in Issue Propagation

To study how zealot agents influence opinion formation and moralization processes:

```bash
python -m polarization_triangle.experiments.zealot_morality_analysis
```

This experiment examines:
- Impact of zealot agents number on opinion propagation
- Impact of zealot advocacy on opinion propagation
- Impact of moralization on opinion propagation

#### Temporal Dynamics of Issue Propagation

To analyze how opinion dynamics evolve over time with different parameter configurations:

```bash
python -m polarization_triangle.experiments.zealot_parameter_sweep --runs 20 --steps 300
```

Parameters:
- `--runs`: Number of simulation runs (default: 20)
- `--steps`: Number of time steps per simulation (default: 300)

This experiment explores:
- Temporal evolution of opinion distributions
- Long-term stability of opinion states

#### Sensitivity Analysis

To perform comprehensive Sobol sensitivity analysis:

```bash
python polarization_triangle/scripts/run_sobol_analysis.py --config full
```

Configuration options:
- `--config full`: Complete sensitivity analysis across all parameters
- `--config limited`: Focused analysis on key parameters

This analysis provides:
- First-order and total-effect sensitivity indices
- Parameter importance rankings
- Interaction effects between parameters

## Project Structure

```
polarization_triangle/
├── core/                   # Core simulation components
│   ├── config.py          # Configuration management
│   ├── dynamics.py        # Opinion dynamics implementation
│   └── simulation.py      # Main simulation engine
├── experiments/            # Experimental configurations
│   ├── zealot_morality_analysis.py
│   ├── zealot_parameter_sweep.py
│   └── multi_zealot_experiment.py
├── utils/                  # Utility functions
│   ├── network.py         # Network utilities
│   ├── lfr_generator.py   # LFR network generation
│   └── network_pool.py    # Network caching system
├── analysis/               # Analysis tools
│   ├── sobol_analysis.py  # Sensitivity analysis
│   ├── statistics.py      # Statistical analysis
│   └── trajectory.py      # Temporal analysis
├── visualization/          # Visualization tools
│   ├── opinion_viz.py     # Opinion distribution plots
│   ├── network_viz.py     # Network visualization
│   └── activation_viz.py  # Activation pattern plots
└── scripts/               # Utility scripts
    ├── generate_network_pool.py
    └── run_sobol_analysis.py
```

## Key Parameters

The simulation behavior is controlled through `SimulationConfig` in `polarization_triangle/core/config.py`:

### Network Parameters
- `num_agents`: Number of agents in the simulation (default: 500)
- `network_params`: Network-specific parameters (degree, community structure, etc.)

### Polarization Triangle Framework
- `alpha`: Self-activation coefficient (default: 0.4)
- `beta`: Social influence coefficient (default: 0.12)
- `gamma`: Moralization influence coefficient (default: 1.0)
- `delta`: Opinion decay rate (default: 1.0)

### Zealot Configuration
- `zealot_count`: Number of zealot agents
- `zealot_opinion`: Fixed opinion value for zealots
- `zealot_morality`: Whether zealots are moralized
- `zealot_identity_allocation`: Whether zealots are allocated by identity

## Output

Results are stored in the `results/` directory and include:
- Opinion trajectory data (CSV format)
- Network statistics and metrics
- Visualization plots (PNG/PDF format)
- Sensitivity analysis reports
- Configuration files for reproducibility

## Contributing

This project implements the Polarization Triangle theoretical framework for studying opinion dynamics in social networks. For questions about the theoretical foundation or implementation details, please refer to im.jiawei.liao@gmail.com.

## AI Assistance Acknowledgment

This project utilized AI assistance for code organization and development, including:
- **File Integration**: Consolidating and organizing code across multiple modules
- **Structure Optimization**: Refactoring code architecture and improving modularity
- **Variable and Function Naming**: Standardizing naming conventions throughout the codebase
- **Documentation Generation**: Creating comprehensive comments, docstrings, and documentation
- **Code Quality Improvements**: Enhancing code readability and maintainability

The core research methodology, theoretical framework, and experimental design remain the work of the primary researchers.

## Dependencies

- Python 3.8+
- NetworkX 3.1+ (network analysis)
- NumPy 1.24+ (numerical computing)
- SciPy 1.10+ (scientific computing)
- Matplotlib 3.7+ (visualization)
- Seaborn 0.12+ (statistical visualization)
- SALib 1.4+ (sensitivity analysis)
- Pandas 2.0+ (data manipulation)
- tqdm 4.65+ (progress bars)

See `requirements.txt` for the complete list of dependencies with specific versions.
