"""
Classical solver interface for quantum_core.

ClassicalSolver defines the contract that all problem-specific classical
solvers must implement. The brute_force() utility is also used internally
by experiment.py to find the reference optimal state.
"""
from abc import ABC, abstractmethod
from typing import Dict


class ClassicalSolver(ABC):
    """
    Base class for classical solvers.

    Each problem package (e.g. problems/maxcut/solvers.py) provides its own
    concrete implementations. quantum_core never imports from problems/ — only
    the other way around.

    solve() must return a dict with at least:
        bitstring  : str    — optimal (or best-found) bitstring
        energy     : float  — Hamiltonian value at that bitstring
        time_s     : float  — wall-clock seconds
        method     : str    — human-readable solver name
        exact      : bool   — True if the result is provably optimal
    """

    @abstractmethod
    def solve(self, hamiltonian) -> Dict:
        pass


# ---------------------------------------------------------------------------
# Internal utility — used by experiment.py and wrapped by BruteForceSolver
# ---------------------------------------------------------------------------

def brute_force(hamiltonian) -> frozenset:
    """
    Exhaustively search all 2^n states and return ALL states achieving the
    minimum Hamiltonian energy as a frozenset of integers.

    Returning the full degenerate set is important for problems like MaxCut,
    where a bitstring and its bitwise complement represent the same cut and
    both achieve the same (minimum) energy.

    Feasible up to ~20 qubits (2^20 ≈ 1 M evaluations).
    Used internally by run_experiment() for success-probability tracking.
    """
    n = hamiltonian.num_qubits
    best_val = float("inf")
    optimal_states = []
    for state in range(2**n):
        val = hamiltonian.evaluate(format(state, f"0{n}b"))
        if val < best_val:
            best_val = val
            optimal_states = [state]
        elif val == best_val:
            optimal_states.append(state)
    return frozenset(optimal_states)
