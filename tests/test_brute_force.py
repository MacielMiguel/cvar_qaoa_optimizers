"""Brute-force optimal states: values and consistency with the quantum operator."""
import numpy as np
import pytest

from problems.maxcut.instances import MaxCutInstanceGenerator
from problems.maxcut.solvers import BruteForceSolver
from quantum_core.cvar_cost import evaluate_sparse_pauli
from quantum_core.solver import brute_force, state_to_bitstring


def _quantum_optima(h):
    op = h.to_pauli_op()
    energies = {s: float(np.real(evaluate_sparse_pauli(s, op))) for s in range(2 ** h.num_qubits)}
    best = min(energies.values())
    return frozenset(s for s, e in energies.items() if abs(e - best) < 1e-9), best


@pytest.mark.parametrize("n, max_cut", [(4, 4), (6, 9)])
def test_fixed_instances_max_cut(n, max_cut):
    result = BruteForceSolver().solve(MaxCutInstanceGenerator.fixed(n))
    assert result["cut_size"] == max_cut
    assert result["energy"] == -max_cut


def test_complete_graph_k4():
    h = MaxCutInstanceGenerator.complete(4)
    states = brute_force(h)
    # K4: best cut is 2|2 -> 4 edges; C(4,2) = 6 optimal bitstrings
    assert len(states) == 6
    assert h.evaluate(state_to_bitstring(min(states), 4)) == -4


def test_optimal_states_come_in_complementary_pairs():
    h = MaxCutInstanceGenerator.random(8, 0.5, seed=801)
    states = brute_force(h)
    assert all(((1 << 8) - 1) ^ s in states for s in states)


@pytest.mark.parametrize("seed", range(800, 810))
def test_brute_force_matches_quantum_operator(seed):
    """Regression test for the bit-ordering bug: random graphs are not symmetric
    under vertex relabelling i -> n-1-i, so a wrong convention is detected."""
    h = MaxCutInstanceGenerator.random(8, 0.5, seed=seed)
    quantum_states, quantum_energy = _quantum_optima(h)
    assert brute_force(h) == quantum_states
    assert quantum_energy == h.evaluate(state_to_bitstring(min(quantum_states), 8))


def test_operator_energy_equals_minus_cut_for_every_state():
    h = MaxCutInstanceGenerator.random(6, 0.6, seed=3)
    op = h.to_pauli_op()
    for s in range(2 ** 6):
        assert np.isclose(np.real(evaluate_sparse_pauli(s, op)), h.evaluate(state_to_bitstring(s, 6)))
