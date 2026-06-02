import numpy as np
from typing import List, Dict, Optional, Tuple

import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp

from quantum_core.hamiltonian import Hamiltonian


class MaxCutHamiltonian(Hamiltonian):
    """
    MaxCut Problem Hamiltonian.

    For a graph G=(V,E), minimises: -sum_{(i,j) in E} (1 - Z_i*Z_j)/2
    which is equivalent to maximising the cut size.
    """

    def __init__(self, edges: List[Tuple[int, int]], num_qubits: Optional[int] = None):
        self.edges = edges
        self.num_qubits = num_qubits or (max(max(e) for e in edges) + 1)
        self.instance = {
            "type": "MaxCut",
            "edges": edges,
            "num_qubits": self.num_qubits,
        }

    def evaluate(self, bitstring: str) -> float:
        """Return negative cut size (for minimisation)."""
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != {self.num_qubits}")
        cut_size = sum(1 for i, j in self.edges if bitstring[i] != bitstring[j])
        return -cut_size

    def evaluate_cut_size(self, bitstring: str) -> int:
        """Return the actual cut size (positive)."""
        return int(-self.evaluate(bitstring))

    def _build_pauli_op(self) -> SparsePauliOp:
        """Convert to a Qiskit SparsePauliOp for use with QAOA/VQE."""
        graph = rx.PyGraph()
        graph.add_nodes_from(range(self.num_qubits))
        seen = set()
        for i, j in self.edges:
            if (i, j) not in seen and (j, i) not in seen:
                graph.add_edge(i, j, 1.0)
                seen.add((i, j))

        pauli_list = []
        for u, v in graph.edge_list():
            paulis = ["I"] * len(graph)
            paulis[u], paulis[v] = "Z", "Z"
            weight = graph.get_edge_data(u, v)
            pauli_list.append(("".join(paulis)[::-1], weight))

        return SparsePauliOp.from_list(pauli_list)

    def to_dict(self) -> Dict:
        return self.instance


class Ising_Hamiltonian(Hamiltonian):
    """
    General Ising Model Hamiltonian.

    H = sum_i h_i * Z_i + sum_{i,j} J_{ij} * Z_i * Z_j
    """

    def __init__(self, h: Dict[int, float], J: Dict[Tuple[int, int], float], num_qubits: Optional[int] = None):
        self.h = h
        self.J = J
        if num_qubits is None:
            all_indices = set(h.keys()) | {i for pair in J for i in pair}
            num_qubits = max(all_indices) + 1 if all_indices else 0
        self.num_qubits = num_qubits
        self.instance = {"type": "Ising", "h": h, "J": J, "num_qubits": num_qubits}

    def evaluate(self, bitstring: str) -> float:
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != {self.num_qubits}")
        z = np.array([1 if b == "0" else -1 for b in bitstring])
        energy = sum(coeff * z[i] for i, coeff in self.h.items())
        energy += sum(coeff * z[i] * z[j] for (i, j), coeff in self.J.items())
        return energy

    def to_dict(self) -> Dict:
        return self.instance


class HamiltonianBuilder:
    """Factory for creating Hamiltonian instances."""

    @staticmethod
    def maxcut(edges: List[Tuple[int, int]], num_qubits: Optional[int] = None) -> MaxCutHamiltonian:
        return MaxCutHamiltonian(edges, num_qubits)

    @staticmethod
    def ising(h: Dict[int, float], J: Dict[Tuple[int, int], float], num_qubits: Optional[int] = None) -> Ising_Hamiltonian:
        return Ising_Hamiltonian(h, J, num_qubits)
