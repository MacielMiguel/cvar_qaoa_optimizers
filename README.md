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
├── run.sh                           # ./run.sh <suite> [--workers N]
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
- scipy >= 1.10.0
- numpy >= 1.24.0
- rustworkx >= 0.14.0
- matplotlib >= 3.9
- pdfo (optional, for the PDFO optimizers; building it needs a Fortran compiler on some platforms)

## Installation

### Setup with conda (recommended)

```bash
# Create environment
conda create -n quantum python=3.10

# Activate environment
conda activate quantum

# Install from project root
pip install -e .

# Install optional pdfo for the PDFO optimizers
pip install -e .[pdfo]
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

```bash
python main.py maxcut-suite1                 # one suite
python main.py paper --workers 4             # the three suites used in the paper
python main.py maxcut-suite3 --quick         # tiny configuration, for testing
python main.py --help                        # all suites and options
```

Each run writes to `results/<timestamp>/`. Parameters live in the `Suite{N}Config`
dataclass at the top of each suite file (`suites/maxcut/suite{1,2,3}.py`,
`suites/maxmulticut/suite{1,2,3}.py`):

- `num_qubits` / `num_nodes` / `num_vertices`: problem size
- `alphas` / `alpha`: CVaR level(s), 0 < alpha <= 1
- `reps` / `reps_list`: circuit depth
- `max_evals`: budget of **circuit evaluations** per run, identical for every optimizer
- `shots`: shots per circuit evaluation
- `seeds` / `num_instances`: independent runs (initial point and shot noise are seeded)

## Reproducing the Paper

The default configurations of the three MaxCut suites are exactly those used in the paper.

```bash
pip install -e .[pdfo]
python main.py paper --workers 4
```

| Paper figure | Suite | Output file |
|---|---|---|
| Fig. 2 (effect of alpha) | `maxcut-suite1` | `suite1_alpha_sweep.png` |
| Fig. 3 (outcome distribution) | `maxcut-suite1` | `suite1_distribution.png` |
| Fig. 4 (optimizer benchmark) | `maxcut-suite2` | `suite2_scipy.png`, `suite2_pdfo.png` |
| Fig. 5 (success probability) | `maxcut-suite3` | `suite3_success_probability.png` |

Every suite also writes `suite{N}_config.json` (configuration and library versions),
`suite{N}_summary.csv` (one row per run) and `suite{N}_full.json` (full history).
The results used in the paper are kept in `results/paper/`.

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

**Suite 1: Effect of the CVaR level**
- Fixed 6-node graph, depth p = 1, COBYLA, 100 circuit evaluations, 1024 shots
- alpha in {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1}, QAOA and VQE, 10 seeds each
- Outputs: CVaR cost versus evaluations (median and inter-quartile range over seeds),
  outcome distribution before/after optimization

**Suite 2: Optimizer benchmark**
- Same graph, p = 1, alpha = 0.5, 200 circuit evaluations for every optimizer, 10 seeds
- SciPy: COBYLA, Nelder-Mead, Powell, BFGS, conjugate gradient, dual annealing
- PDFO: COBYLA, NEWUOA, BOBYQA, LINCOA, UOBYQA
- Output: best CVaR found so far versus evaluations

**Suite 3: Success probability (Fig. 4 of Barkoutsos et al.)**
- 60 random graphs, 8 nodes, edge density 0.5 (seeds 800 to 859)
- p in {1, 2}, alpha in {0.01, 0.25, 0.5, 1}, COBYLA, 100 circuit evaluations
- Output: fraction of instances whose success probability reached 1 % and 10 %,
  versus normalized evaluations t / n

### MaxMulticut Suites

**Suite 1: QAOA vs VQE** on a 6-vertex graph with 3 colors.
**Suite 2: Optimizer benchmark** (PDFO optimizers, QAOA).
**Suite 3: Scaling** with the number of vertices (4 to 6) and colors (2 and 3).

> Note: with a number of colors that is not a power of 2 (e.g. k = 3), some
> binary codes do not correspond to a color. The classical `evaluate()` penalizes
> them, but the quantum operator does not yet include a penalty term, so the
> MaxMulticut results are exploratory.

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

- Each run is fully determined by its `seed`: the initial point and the shot noise of
  the sampler are both drawn from seeded generators, so re-running a suite gives
  identical numbers (tested in `tests/test_experiment.py`).
- Random instances are generated from explicit seeds.
- Every results folder contains the configuration and the library versions used.
- Bit-ordering convention: measured states are integers in Qiskit order (bit i = qubit i);
  `quantum_core.solver.state_to_bitstring` converts them to bitstrings indexed by variable.

## References

- MaxCut: [Garey & Johnson, 1979]
- QAOA: [Farhi et al., 2014]
- CVaR for QAOA: [Barkoutsos et al., 2020]
- PDFO: https://www.pdfo.net

## License

GNU General Public License v3.0 (see `LICENSE`).
