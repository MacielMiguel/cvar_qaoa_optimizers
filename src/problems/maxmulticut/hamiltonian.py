import numpy as np
from typing import List, Dict, Tuple

from qiskit.quantum_info import SparsePauliOp

from quantum_core.hamiltonian import Hamiltonian


class MaxMulticutHamiltonian(Hamiltonian):
    """
    MaxMulticut Problem Hamiltonian.

    For a graph G=(V,E) and k colors, partitions vertices into k communities
    to maximize the number of edges crossing partition boundaries.

    The Hamiltonian uses a one-hot encoding: each vertex v requires log2(k) qubits
    to represent its color assignment (0 to k-1).

    Cost: minimizes -sum_{(i,j) in E} I(color_i != color_j)
    where I(·) is the indicator function.
    """

    def __init__(self, edges: List[Tuple[int, int]], num_colors: int = 3, num_vertices: int = None):
        """
        Initialize MaxMulticut Hamiltonian.

        Args:
            edges: List of (u, v) pairs representing graph edges.
            num_colors: Number of communities (k >= 2).
            num_vertices: Number of vertices. If None, inferred from edges.
        """
        if num_colors < 2:
            raise ValueError("num_colors must be >= 2")

        self.edges = edges
        self.num_colors = num_colors
        self.num_vertices = num_vertices or (max(max(e) for e in edges) + 1 if edges else 0)
        
        # Each vertex needs ceil(log2(num_colors)) qubits to encode its color
        self.qubits_per_vertex = int(np.ceil(np.log2(num_colors))) if num_colors > 1 else 1
        self.num_qubits = self.num_vertices * self.qubits_per_vertex

        self.instance = {
            "type": "MaxMulticut",
            "edges": edges,
            "num_vertices": self.num_vertices,
            "num_colors": num_colors,
            "num_qubits": self.num_qubits,
        }

    def _bitstring_to_colors(self, bitstring: str) -> List[int]:
        """Convert a bitstring to vertex color assignments.

        Returns None for a vertex whose b-bit block encodes an integer >= num_colors
        (invalid state). evaluate() penalises such states.
        """
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Bitstring length {len(bitstring)} != {self.num_qubits}")

        colors = []
        for v in range(self.num_vertices):
            start = v * self.qubits_per_vertex
            end = start + self.qubits_per_vertex
            color = int(bitstring[start:end], 2)
            colors.append(color if color < self.num_colors else None)
        return colors

    def evaluate(self, bitstring: str) -> float:
        """Return negative cut size (for minimisation).

        States where any vertex has an invalid color encoding (integer >= num_colors)
        are penalised with a large positive energy so brute_force() never picks them.
        """
        colors = self._bitstring_to_colors(bitstring)
        if any(c is None for c in colors):
            return float(len(self.edges))  # worse than any valid cut (max cut = |E|)
        cut_size = sum(1 for i, j in self.edges if colors[i] != colors[j])
        return -cut_size

    def evaluate_cut_size(self, bitstring: str) -> int:
        """Return the actual cut size (positive, number of edges between communities)."""
        return -self.evaluate(bitstring)

    def _build_pauli_op(self) -> SparsePauliOp:
        """Convert to a Qiskit SparsePauliOp for use with QAOA/VQE.

        Correct derivation
        ------------------
        Let b = qubits_per_vertex.  For two vertices u, v, define the
        "same-color indicator":

            S_uv = prod_{i=0}^{b-1}  (I + Z_{u_i} Z_{v_i}) / 2
                 = (1/2^b) * sum_{S ⊆ {0..b-1}}  prod_{i in S} Z_{u_i} Z_{v_i}

        S_uv = 1 iff both vertices share the same b-bit binary encoding,
        S_uv = 0 otherwise (for valid, distinct colors).

        The cost Hamiltonian to minimise is:
            H_C = -sum_{(u,v) in E} (1 - S_uv)
                = const + sum_{(u,v) in E}  (1/2^b)
                          * sum_{non-empty S ⊆ {0..b-1}} prod_{i in S} Z_{u_i}Z_{v_i}

        The constant shift does not affect the optimum, so we emit only the
        non-trivial Pauli terms (non-empty subsets).

        For b=1 this reduces to (1/2) Z_u Z_v — identical to MaxCut.
        For b=2 each edge contributes 3 terms:
            (1/4) Z_{u0}Z_{v0}  +  (1/4) Z_{u1}Z_{v1}
            +  (1/4) Z_{u0}Z_{u1}Z_{v0}Z_{v1}
        """
        b = self.qubits_per_vertex
        weight = 1.0 / (1 << b)   # 1 / 2^b
        pauli_list = []

        for u, v in self.edges:
            # Iterate over all non-empty subsets of {0, ..., b-1} via bitmask
            for mask in range(1, 1 << b):
                paulis = ["I"] * self.num_qubits
                for i in range(b):
                    if mask & (1 << i):
                        paulis[u * b + i] = "Z"
                        paulis[v * b + i] = "Z"
                pauli_list.append(("".join(paulis)[::-1], weight))

        if not pauli_list:
            pauli_list = [("I" * self.num_qubits, 0.0)]

        return SparsePauliOp.from_list(pauli_list)

    def to_dict(self) -> Dict:
        return self.instance
