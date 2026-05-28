# MaxMulticut Problem

## Overview

MaxMulticut is a generalization of the MaxCut problem where vertices of a graph are partitioned into **k > 2** communities (colors) instead of just 2 partitions. The goal is to maximize the number of edges between different communities.

## Problem Definition

**Input:**
- Graph G = (V, E) with |V| vertices and |E| edges
- Number of colors k ≥ 2

**Objective:**
Partition vertices into k communities to maximize:
$$\sum_{(i,j) \in E} \mathbb{1}(\text{color}_i \neq \text{color}_j)$$

where $\mathbb{1}(\cdot)$ is the indicator function.

## Encoding

MaxMulticut uses **one-hot encoding** to represent color assignments:
- Each vertex requires $\lceil \log_2(k) \rceil$ qubits
- Total qubits: $n \times \lceil \log_2(k) \rceil$, where n is the number of vertices

Example for k=3 colors:
- 2 qubits per vertex encode colors 0, 1, 2
- A 6-vertex graph requires 12 qubits

## Hamiltonian

The cost operator uses Z-terms to detect color differences:

$$H = \sum_{(i,j) \in E} h_{ij}$$

where each $h_{ij}$ is constructed from the qubits encoding vertices i and j.

## Usage

### Creating Instances

```python
from problems.maxmulticut.instances import MaxMulticutInstanceGenerator

# Fixed benchmark instance
ham = MaxMulticutInstanceGenerator.fixed_simple(num_colors=3)

# Random instance
ham = MaxMulticutInstanceGenerator.random(
    num_vertices=6, 
    num_colors=3,
    edge_density=0.5,
    seed=42
)

# Grid graph
ham = MaxMulticutInstanceGenerator.grid(rows=3, cols=3, num_colors=3)

# k-partite graph
ham = MaxMulticutInstanceGenerator.k_partite(
    num_vertices=10,
    num_partitions=5,
    num_colors=3,
    inter_partition_density=0.7,
    seed=42
)
```

### Running Experiments

See the suite files for complete examples:
- [suite1.py](suite1.py): QAOA vs VQE across CVaR alpha values
- [suite2.py](suite2.py): Optimizer comparison
- [suite3.py](suite3.py): Scaling analysis with varying vertices and colors

### Running with Main

To run MaxMulticut suites, update [main.py](../../main.py):

```python
import suites.maxmulticut.suite1 as suite1
import suites.maxmulticut.suite2 as suite2
import suites.maxmulticut.suite3 as suite3

# Then in main():
suite1.run(output_run_dir)
suite2.run(output_run_dir)
suite3.run(output_run_dir)
```

## Available Instance Generators

| Method | Description |
|--------|-------------|
| `random()` | Random graph with configurable edge density |
| `complete()` | Complete graph (all edges) |
| `grid()` | Grid graph (rows × cols) |
| `bipartite()` | Bipartite graph |
| `k_partite()` | k-partite graph with tunable inter-partition edges |
| `fixed_simple()` | Simple 6-vertex benchmark |
| `fixed_grid_3x3()` | 3×3 grid benchmark |
| `fixed_grid_4x4()` | 4×4 grid benchmark |

## Properties

| Property | Value |
|----------|-------|
| Vertex count | 4-16 (configurable) |
| Colors | 2-4 (configurable) |
| Qubits per vertex | $\lceil \log_2(k) \rceil$ |
| NP-hardness | NP-hard (generalizes MaxCut) |

## Related Problems

- **MaxCut** (k=2): 2-coloring special case
- **Graph Coloring**: Finding minimum k for proper coloring (different problem)
- **MaxSAT**: Related combinatorial optimization

## References

- MaxCut: [Garey & Johnson, 1979]
- Quantum QAOA: [Farhi et al., 2014]
- CVaR: [Barkoutsos et al., 2020]
