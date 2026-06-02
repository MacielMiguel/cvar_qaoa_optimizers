import numpy as np
from typing import List, Optional

from problems.maxcut.hamiltonian import MaxCutHamiltonian, Ising_Hamiltonian
from quantum_core.graphs import GraphInstanceGenerator


class MaxCutInstanceGenerator:
    """Generate MaxCut (and Ising) problem instances."""

    # ------------------------------------------------------------------
    # Random instances
    # ------------------------------------------------------------------

    @staticmethod
    def random(num_qubits: int, edge_density: float = 0.5, seed: Optional[int] = None) -> MaxCutHamiltonian:
        """Random MaxCut instance with reproducible seed."""
        if seed is None:
            edges = GraphInstanceGenerator.random(num_qubits, edge_density)
        else:
            edges = GraphInstanceGenerator.random(num_qubits, edge_density, seed)
        return MaxCutHamiltonian(edges, num_qubits)

    @staticmethod
    def complete(num_qubits: int) -> MaxCutHamiltonian:
        """Complete graph (all edges present)."""
        edges = GraphInstanceGenerator.complete(num_qubits)
        return MaxCutHamiltonian(edges, num_qubits)

    @staticmethod
    def grid(rows: int, cols: int) -> MaxCutHamiltonian:
        """Grid graph."""
        edges = GraphInstanceGenerator.grid(rows, cols)
        return MaxCutHamiltonian(edges, rows * cols)

    @staticmethod
    def bipartite(num_qubits: int, partition_ratio: float = 0.5) -> MaxCutHamiltonian:
        """Bipartite graph (edges only between the two partitions)."""
        edges = GraphInstanceGenerator.bipartite(num_qubits, partition_ratio)
        return MaxCutHamiltonian(edges, num_qubits)

    @staticmethod
    def ising_random(
        num_qubits: int,
        h_strength: float = 1.0,
        J_strength: float = 1.0,
        coupling_density: float = 0.5,
        seed: Optional[int] = None,
    ) -> Ising_Hamiltonian:
        """Random Ising model."""
        rng = np.random.RandomState(seed)
        h = {i: rng.randn() * h_strength for i in range(num_qubits)}
        J = {
            (i, j): rng.randn() * J_strength
            for i in range(num_qubits)
            for j in range(i + 1, num_qubits)
            if rng.random() < coupling_density
        }
        return Ising_Hamiltonian(h, J, num_qubits)

    # ------------------------------------------------------------------
    # Fixed benchmark instances
    # ------------------------------------------------------------------

    @staticmethod
    def fixed(n: int = 6) -> MaxCutHamiltonian:
        """
        Return a deterministic benchmark instance for n nodes.

        Available sizes: 4, 6, 7, 8, 10, 20.
        """
        _instances = {
            4:  MaxCutInstanceGenerator._fixed_4,
            6:  MaxCutInstanceGenerator._fixed_6,
            7:  MaxCutInstanceGenerator._fixed_7,
            8:  MaxCutInstanceGenerator._fixed_8,
            10: MaxCutInstanceGenerator._fixed_10,
            20: MaxCutInstanceGenerator._fixed_20,
        }
        if n not in _instances:
            raise ValueError(f"n={n} not available. Choose from: {list(_instances)}")
        return _instances[n]()

    @staticmethod
    def _fixed_4() -> MaxCutHamiltonian:
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        return MaxCutHamiltonian(edges, 4)

    @staticmethod
    def _fixed_6() -> MaxCutHamiltonian:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
            (0, 3), (1, 4), (2, 5), (0, 2),
        ]
        return MaxCutHamiltonian(edges, 6)

    @staticmethod
    def _fixed_7() -> MaxCutHamiltonian:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0),
            (0, 2), (0, 3), (1, 3), (1, 4), (2, 4), (2, 5),
            (3, 5), (3, 6), (4, 6), (5, 0),
        ]
        return MaxCutHamiltonian(edges, 7)

    @staticmethod
    def _fixed_8() -> MaxCutHamiltonian:
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (0, 3), (1, 2), (4, 7), (5, 6),
        ]
        return MaxCutHamiltonian(edges, 8)

    @staticmethod
    def _fixed_10() -> MaxCutHamiltonian:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
            (0, 2), (1, 3), (2, 4), (3, 0), (4, 1),
            (5, 7), (6, 8), (7, 9), (8, 5), (9, 6),
        ]
        return MaxCutHamiltonian(edges, 10)

    @staticmethod
    def _fixed_20() -> MaxCutHamiltonian:
        edges = []
        for i in range(10):
            edges.append((i, (i + 1) % 10))
            edges.append((i, (i + 2) % 10))
        for i in range(10, 20):
            edges.append((i, ((i + 1 - 10) % 10) + 10))
            edges.append((i, ((i + 2 - 10) % 10) + 10))
        for i in range(5):
            edges.append((i, i + 10))
            edges.append(((i + 5) % 10, (i + 10 + 5) % 10))
        return MaxCutHamiltonian(list(set(edges)), 20)
