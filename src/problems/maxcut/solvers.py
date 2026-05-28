"""
Classical solvers for MaxCut.

BruteForceSolver  — exact, exponential time, feasible up to ~20 qubits.
GreedySolver      — O(n·|E|) approximation (≥ 1/2 of optimal cut guaranteed).

Usage:
    from problems.maxcut.solvers import BruteForceSolver, GreedySolver

    hamiltonian = MaxCutInstanceGenerator.fixed(6)

    result = BruteForceSolver().solve(hamiltonian)
    result = GreedySolver().solve(hamiltonian)

    # result dict:
    # {
    #   'bitstring' : str,   — best assignment found
    #   'energy'    : float, — Hamiltonian value (negative cut size for MaxCut)
    #   'cut_size'  : int,   — number of edges in the cut
    #   'time_s'    : float, — wall-clock seconds
    #   'method'    : str,
    #   'exact'     : bool,
    # }
"""
import time
from typing import Dict

from quantum_core.solver import ClassicalSolver, brute_force
from problems.maxcut.hamiltonian import MaxCutHamiltonian


class BruteForceSolver(ClassicalSolver):
    """
    Exact MaxCut solver by exhaustive search over all 2^n bitstrings.

    Complexity: O(2^n · |E|).
    Practical limit: ~20 qubits before wall-clock time becomes unreasonable.
    """

    def solve(self, hamiltonian: MaxCutHamiltonian) -> Dict:
        start = time.time()
        optimal_states = brute_force(hamiltonian)
        # Pick the lowest integer as a canonical representative
        representative = min(optimal_states)
        bitstring = format(representative, f"0{hamiltonian.num_qubits}b")
        energy = hamiltonian.evaluate(bitstring)
        return {
            "bitstring": bitstring,
            "energy": energy,
            "cut_size": hamiltonian.evaluate_cut_size(bitstring),
            "num_optimal_states": len(optimal_states),
            "time_s": time.time() - start,
            "method": "brute_force",
            "exact": True,
        }


class GreedySolver(ClassicalSolver):
    """
    Greedy MaxCut approximation.

    Algorithm:
      1. Process vertices in order 0…n-1.
      2. Assign each vertex to the partition that maximises the cut with
         already-assigned neighbours.
      3. One pass of local search: flip any vertex that strictly improves
         the cut, repeat until stable.

    Complexity: O(n · |E|).
    Approximation guarantee: cut_size ≥ optimal / 2.
    """

    def solve(self, hamiltonian: MaxCutHamiltonian) -> Dict:
        start = time.time()
        n = hamiltonian.num_qubits
        edges = hamiltonian.edges

        # Build adjacency list for fast neighbour lookups
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # --- Phase 1: greedy assignment ---
        assignment = [-1] * n
        for v in range(n):
            # Count neighbours already assigned to each partition
            cuts_if_0 = sum(1 for u in adj[v] if assignment[u] == 1)
            cuts_if_1 = sum(1 for u in adj[v] if assignment[u] == 0)
            assignment[v] = 0 if cuts_if_0 >= cuts_if_1 else 1

        # --- Phase 2: local search improvement ---
        def _cut_delta(v: int) -> int:
            """Return change in cut size if we flip vertex v (positive = improvement)."""
            same = sum(1 for u in adj[v] if assignment[u] == assignment[v])
            diff = len(adj[v]) - same
            return same - diff  # flipping gains `same` edges, loses `diff` edges

        improved = True
        while improved:
            improved = False
            for v in range(n):
                if _cut_delta(v) > 0:
                    assignment[v] ^= 1
                    improved = True

        bitstring = "".join(str(b) for b in assignment)
        energy = hamiltonian.evaluate(bitstring)
        return {
            "bitstring": bitstring,
            "energy": energy,
            "cut_size": hamiltonian.evaluate_cut_size(bitstring),
            "time_s": time.time() - start,
            "method": "greedy",
            "exact": False,
        }
