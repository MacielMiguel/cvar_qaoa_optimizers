import numpy as np

from quantum_core.optimizers import COBYLA, SimulatedAnnealing
from quantum_core.cvar_cost import _get_cvar_aggregation


def test_cobyla_classical_optimization():
    """COBYLA must converge to the minimum of (x - 2)^2."""
    def obj(x):
        return (x[0] - 2.0) ** 2

    optimizer = COBYLA(maxiter=100)
    result = optimizer.optimize(obj, initial_point=np.array([0.0]))

    assert result["success"] is True
    assert np.isclose(result["x"][0], 2.0, atol=1e-2)


def test_simulated_annealing():
    """dual_annealing must converge to the global minimum of x^2 + y^2."""
    def obj(x):
        return x[0] ** 2 + x[1] ** 2

    optimizer = SimulatedAnnealing(maxiter=50)
    result = optimizer.optimize(obj, initial_point=np.array([5.0, -5.0]))

    assert result["success"] is True
    assert np.isclose(result["x"][0], 0.0, atol=1e-1)
    assert np.isclose(result["x"][1], 0.0, atol=1e-1)


def test_cvar_aggregation():
    """CVaR maths: alpha=0.5 must return the mean of the best 50% of outcomes."""
    # Four outcomes with equal probability 0.25, energies -10, -5, 0, +5
    outcomes = [(0.25, -10.0), (0.25, -5.0), (0.25, 0.0), (0.25, 5.0)]

    # alpha=1.0 → plain expectation value: (-10 - 5 + 0 + 5) / 4 = -2.5
    assert np.isclose(_get_cvar_aggregation(1.0)(outcomes), -2.5)

    # alpha=0.5 → mean of best 50%: (-10 + -5) / 2 = -7.5
    assert np.isclose(_get_cvar_aggregation(0.5)(outcomes), -7.5)

    # alpha=0.25 → best 25%: just -10
    assert np.isclose(_get_cvar_aggregation(0.25)(outcomes), -10.0)
