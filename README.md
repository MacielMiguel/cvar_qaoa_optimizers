# cvar-quantum

A Python framework for experimenting with QAOA/VQE using a CVaR objective on Max-Cut and Max-Multicut instances.

## Project Goal

This project provides:

- **Problem instances**: MaxCut and MaxMulticut graph generators with multiple topologies.
- **Hamiltonian builders**: Quantum cost operators for combinatorial optimization.
- **Quantum solvers**: QAOA and VQE implementations with CVaR support.
- **Classical optimizers**: Unified interface for scipy and pdfo-based optimizers (COBYLA, BOBYQA, NEWUOA, etc.).
- **Experiments**: Automated benchmarking suites to compare architectures and algorithms.

## Structure

```text
.
├── pyproject.toml
├── README.md
├── main.py                          # Entry point - choose MaxCut or MaxMulticut suites
├── run.sh
├── src/
│   ├── quantum_core/                # Core quantum & optimization abstractions
│   │   ├── hamiltonian.py           # Base Hamiltonian class
│   │   ├── optimizers.py            # Classical optimizer wrappers
│   │   ├── experiment.py            # VQE/QAOA experiment runner
│   │   ├── cvar_cost.py             # CVaR cost function
│   │   ├── graphs.py                # GraphInstanceGenerator base class
│   │   ├── display.py               # Plotting utilities
│   │   ├── results_io.py            # Results serialization
│   │   └── solver.py                # Brute force solver
│   └── problems/                    # Problem-specific implementations
│       ├── maxcut/
│       │   ├── hamiltonian.py       # MaxCutHamiltonian & Ising_Hamiltonian
│       │   ├── instances.py         # MaxCutInstanceGenerator
│       │   └── solvers.py           # MaxCut solvers
│       └── maxmulticut/
│           ├── hamiltonian.py       # MaxMulticutHamiltonian
│           ├── instances.py         # MaxMulticutInstanceGenerator
│           └── README.md            # MaxMulticut documentation
├── suites/                          # Experiment configurations
│   ├── maxcut/
│   │   ├── suite1.py                # QAOA vs VQE comparison
│   │   ├── suite2.py                # Optimizer benchmarking
│   │   └── suite3.py                # Scaling analysis
│   └── maxmulticut/
│       ├── suite1.py                # QAOA vs VQE comparison (with debug prints)
│       ├── suite2.py                # Optimizer benchmarking
│       ├── suite3.py                # Number of colors scaling
│       └── README.md                # MaxMulticut suite guide
├── results/                         # Output directory for experiments
└── tests/
    ├── test_brute_force.py
    ├── test_hamiltonian.py
    └── test_solvers.py
```

## Prerequisites

- Python >= 3.10
- Anaconda/conda (recommended) or venv

Main dependencies:

- qiskit >= 1.0.0
- qiskit-algorithms >= 0.3.0
- scipy >= 1.10.0
- numpy >= 1.24.0
- networkx >= 3.0
- rustworkx >= 0.14.0
- matplotlib >= 3.7.0
- pdfo (optional, for Powell solvers)

## Installation

### Setup with conda (recommended)

```bash
# Create environment
conda create -n quantum python=3.10

# Activate environment
conda activate quantum

# Install from project root
pip install -e .

# Install optional pdfo for advanced optimizers
pip install pdfo
```

### Install development dependencies

```bash
pip install -e .[dev]
```

## Run Tests

```bash
pytest
```

## Run Experiments

### Configure Experiments

Edit the configuration dataclass in each suite file:

**MaxCut Suites**: `suites/maxcut/suite{1,2,3}.py`
**MaxMulticut Suites**: `suites/maxmulticut/suite{1,2,3}.py`

Each suite has a `Suite{N}Config` dataclass with parameters:
- `num_qubits` / `num_vertices`: Problem size
- `num_colors`: For MaxMulticut only
- `alphas`: CVaR alpha values to sweep
- `reps`: QAOA circuit depth
- `maxiter`: Classical optimizer iterations
- `shots`: Shots per circuit evaluation
- `optimizers`: List of optimizers to benchmark

### Quick Start (Fast)

```python
# In main.py, uncomment one suite:

# MaxCut - Suite 1
maxcut_suite1.run(output_run_dir)
```

### Reduced Parameters for Testing

MaxMulticut Suite 1 has friendly defaults for fast debugging:
```python
alphas: [0.1, 0.5, 1.0]      # 3 alphas instead of 7
maxiter: 15                   # reduced from 90
shots: 256                    # reduced from 1024
# ~144x faster for initial testing
```

For full benchmark, modify config in suite file:
```python
alphas: [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
maxiter: 90
shots: 1024
```

## Usage Examples

### Quick Python Usage

```python
from problems.maxcut.instances import MaxCutInstanceGenerator
from problems.maxmulticut.instances import MaxMulticutInstanceGenerator
from quantum_core.experiment import run_experiment

# MaxCut instance
ham_cut = MaxCutInstanceGenerator.grid(3, 3)

# MaxMulticut instance (3 colors)
ham_multi = MaxMulticutInstanceGenerator.k_partite(
    num_vertices=9, 
    num_partitions=3, 
    num_colors=3
)

# Run single experiment
result = run_experiment(
    hamiltonian=ham_multi,
    ansatz_type="QAOA",
    reps=1,
    cvar_alpha=0.5,
    optimizer_name="COBYLA",
    maxiter=50,
    shots=1024
)
```

### Available Instance Generators

**GraphInstanceGenerator** (shared base):
- `random(num_vertices, edge_density, seed)` - Random graph
- `complete(num_vertices)` - Complete graph
- `grid(rows, cols)` - Grid graph
- `bipartite(num_vertices, partition_ratio)` - Bipartite graph
- `k_partite(num_vertices, num_partitions, density, seed)` - k-partite graph

**MaxCutInstanceGenerator**:
- All above methods
- `ising_random(num_qubits, h_strength, J_strength, coupling_density, seed)` - Ising model
- `fixed(n)` - Fixed benchmarks for n ∈ {4, 6, 7, 8, 10, 20}

**MaxMulticutInstanceGenerator**:
- All shared methods
- Wraps graphs with arbitrary number of colors (communities)

## Available Optimizers

### SciPy-based (no bounds)
- `COBYLA` - Constrained Optimization (recommended for QAOA)
- `COBYQA` - Modern alternative to COBYLA
- `NelderMead` - Simplex method
- `BFGS` - Quasi-Newton
- `Powell` - Derivative-free
- `ConjugateGradient` - First-order method
- `SimulatedAnnealing` - Stochastic global optimization

### PDFO-based (Powell's Derivative-Free Optimization, requires `pip install pdfo`)
- `NEWUOA` - Unconstrained, trust-region (fast & robust)
- `COBYLA_PDFO` - Powell's original COBYLA
- `BOBYQA` - Bound-constrained
- `LINCOA` - Linear-constrained
- `UOBYQA` - Simple unconstrained (for very low-dim)

## Suite Descriptions

### MaxCut Suites

**Suite 1: Architecture Comparison**
- Compares QAOA vs VQE across multiple CVaR alpha values
- Tests on fixed 6-node instance
- Outputs convergence curves and success probability plots

**Suite 2: Optimizer Benchmark**
- Tests 5 different classical optimizers on QAOA
- Fixed alpha=0.25, measures convergence speed
- Compares scipy vs pdfo solvers

**Suite 3: Scaling Analysis**  
- Studies how problem size affects convergence
- Varies problem size (4-20 nodes)
- Outputs scaling curves

### MaxMulticut Suites

**Suite 1: Architecture Comparison** (with debug prints)
- QAOA vs VQE on 6-node graph with 3-coloring
- Includes detailed timing breakdown:
  - Hamiltonian creation
  - Pauli operator generation (can be slow for many qubits)
  - Brute-force reference computation
  - Per-iteration timing statistics

**Suite 2: Optimizer Benchmark**
- Tests pdfo optimizers on MaxMulticut
- Helps identify best solver for the problem

**Suite 3: Color Scaling**
- Studies effect of number of colors (2-4)  
- Different graph sizes (4-7 vertices)

## Output

Experiment results saved to `results/{timestamp}/`:
- `suite{N}_summary.csv` - Summary statistics
- `suite{N}_full.json` - Full experimental data
- `*.png` - Plots (cost evolution, distributions)

## Performance Notes

- **Statevector simulation**: Circuits simulated classically on CPU
  - ~10 qubits: Fast (seconds)
  - 12+ qubits: Slow (minutes per experiment)
  - 16+ qubits: Very slow (hours)

- **Bottlenecks** (typical MaxMulticut 12-qubit):
  1. Pauli operator has many terms → expensive evaluations
  2. Brute force reference (exponential in qubits)
  3. Multiple iterations × multiple alphas × 2 architectures

- **Optimization tips**:
  - Reduce `maxiter` for quick testing
  - Reduce `shots` (256 instead of 1024)
  - Reduce `alphas` list (3 values instead of 7)
  - Use 4-6 vertex problems for fast iteration

## Reproducibility

- All experiments use fixed random seeds
- Instance generators support seeded randomness
- Save full configs in results

## References

- MaxCut: [Garey & Johnson, 1979]
- QAOA: [Farhi et al., 2014]
- CVaR for QAOA: [Barkoutsos et al., 2020]
- PDFO: https://www.pdfo.net

## License

TBD
- The matplotlib backend is set to non-interactive mode in the experiment script (`Agg`) to support non-notebook execution.
