import pytest
import numpy as np
from problems.maxcut.hamiltonian import MaxCutHamiltonian, HamiltonianBuilder
from problems.maxcut.instances import MaxCutInstanceGenerator


def test_maxcut_evaluate_cut_size():
    """Test standard evaluation for a known graph."""
    edges = [(0, 1), (1, 2)]
    hamiltonian = MaxCutHamiltonian(edges, num_qubits=3)

    # 010: edges (0,1) and (1,2) are both cut → cut_size = 2
    assert hamiltonian.evaluate_cut_size("010") == 2
    # 000: no cuts
    assert hamiltonian.evaluate_cut_size("000") == 0


def test_problem_generator_seed_consistency():
    """Random instances with the same seed must be identical."""
    h1 = MaxCutInstanceGenerator.random(num_qubits=4, edge_density=0.5, seed=42)
    h2 = MaxCutInstanceGenerator.random(num_qubits=4, edge_density=0.5, seed=42)
    assert h1.edges == h2.edges
    assert h1.num_qubits == h2.num_qubits


def test_hamiltonian_builder_factory():
    """Factory method must return the correct class."""
    h = HamiltonianBuilder.maxcut([(0, 1)], 2)
    assert isinstance(h, MaxCutHamiltonian)
    assert h.num_qubits == 2
