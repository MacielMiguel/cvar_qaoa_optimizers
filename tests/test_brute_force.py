"""Test brute force optimal state search."""

from problems.maxcut.instances import MaxCutInstanceGenerator
from quantum_core.solver import brute_force


def _solve(hamiltonian):
    optimal_states = brute_force(hamiltonian)
    # Pick canonical representative (lowest int) for display
    representative = min(optimal_states)
    optimal_bitstring = format(representative, f"0{hamiltonian.num_qubits}b")
    optimal_value = hamiltonian.evaluate(optimal_bitstring)
    return optimal_states, optimal_bitstring, optimal_value


def test_brute_force_fixed_4():
    h = MaxCutInstanceGenerator.fixed(4)
    states, bs, val = _solve(h)
    print(f"\n4-qubit: repr={bs}, energy={val}, num_optimal={len(states)}")


def test_brute_force_fixed_6():
    h = MaxCutInstanceGenerator.fixed(6)
    states, bs, val = _solve(h)
    print(f"\n6-qubit: repr={bs}, energy={val}, num_optimal={len(states)}")


def test_brute_force_fixed_10():
    h = MaxCutInstanceGenerator.fixed(10)
    states, bs, val = _solve(h)
    print(f"\n10-qubit: repr={bs}, energy={val}, num_optimal={len(states)}")


def test_brute_force_complete_graph():
    h = MaxCutInstanceGenerator.complete(4)
    states, bs, val = _solve(h)
    print(f"\nK4: repr={bs}, energy={val}, cut={-val}, num_optimal={len(states)}")


def test_brute_force_grid():
    h = MaxCutInstanceGenerator.grid(rows=2, cols=2)
    states, bs, val = _solve(h)
    print(f"\n2x2 grid: repr={bs}, energy={val}, num_optimal={len(states)}")


if __name__ == "__main__":
    for fn in [
        test_brute_force_fixed_4,
        test_brute_force_fixed_6,
        test_brute_force_fixed_10,
        test_brute_force_complete_graph,
        test_brute_force_grid,
    ]:
        fn()
    print("\nAll tests completed!")
