import numpy as np
from typing import List, Dict, Optional, Tuple

from qiskit.quantum_info import SparsePauliOp

from quantum_core.hamiltonian import Hamiltonian


class MaxCutHamiltonian(Hamiltonian):
    """
    MaxCut Problem Hamiltonian.

    For a graph G=(V,E), the cost Hamiltonian is

        H_C = -1/2 * sum_{(i,j) in E} (I - Z_i Z_j)
            =  1/2 * sum_{(i,j) in E} Z_i Z_j  -  |E|/2 * I,

    whose eigenvalue on a basis state equals minus the size of the
    corresponding cut. Minimising H_C is therefore maximising the cut, and
    the quantum operator returned by to_pauli_op() gives exactly the same
    value as evaluate().
    """

    def __init__(self, edges: List[Tuple[int, int]], num_qubits: Optional[int] = None):
        if num_qubits is None:
            if not edges:
                raise ValueError("num_qubits must be given for a graph without edges")
            num_qubits = max(max(e) for e in edges) + 1
        self.edges = edges
        self.num_qubits = num_qubits
        self.instance = {
            "type": "MaxCut",
            "edges": edges,
            "num_qubits": self.num_qubits,
        }

    def _unique_edges(self) -> List[Tuple[int, int]]:
        seen = set()
        unique = []
        for i, j in self.edges:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                unique.append(key)
        return unique

    def evaluate(self, bitstring: str) -> float:
        """Return minus the cut size. Character i of the bitstring is vertex i."""
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != {self.num_qubits}")
        cut_size = sum(1 for i, j in self._unique_edges() if bitstring[i] != bitstring[j])
        return -cut_size

    def evaluate_cut_size(self, bitstring: str) -> int:
        """Return the actual cut size (positive)."""
        return int(-self.evaluate(bitstring))

    def _build_pauli_op(self) -> SparsePauliOp:
        """Return H_C = 1/2 sum Z_i Z_j - |E|/2 I as a Qiskit SparsePauliOp."""
        edges = self._unique_edges()
        n = self.num_qubits
        pauli_list = [("I" * n, -0.5 * len(edges))]
        for u, v in edges:
            paulis = ["I"] * n
            paulis[u], paulis[v] = "Z", "Z"
            # Qiskit labels are written qubit n-1 ... qubit 0, hence the reversal
            pauli_list.append(("".join(paulis)[::-1], 0.5))
        return SparsePauliOp.from_list(pauli_list).simplify()

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
