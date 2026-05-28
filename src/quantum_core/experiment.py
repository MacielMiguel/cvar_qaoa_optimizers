import time
import logging
import numpy as np
from typing import Dict, Any

from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler as Sampler

from quantum_core.solver import brute_force
from quantum_core.cvar_cost import cvar_sampler_cost_fun
from quantum_core.optimizers import (
    COBYLA,
    COBYQA,
    NelderMead,
    BFGS,
    Powell,
    ConjugateGradient,
    SimulatedAnnealing,
    NEWUOA,
    COBYLA_PDFO,
    BOBYQA,
    LINCOA,
    UOBYQA,
)

logger = logging.getLogger(__name__)

_OPTIMIZERS = {
    # scipy-based
    "COBYLA": COBYLA,
    "COBYQA": COBYQA,
    "Nelder-Mead": NelderMead,
    "BFGS": BFGS,
    "Powell": Powell,
    "ConjugateGradient": ConjugateGradient,
    "SimulatedAnnealing": SimulatedAnnealing,
    # pdfo-based (Powell's Derivative-Free Optimization)
    "NEWUOA": NEWUOA,
    "COBYLA_PDFO": COBYLA_PDFO,
    "BOBYQA": BOBYQA,
    "LINCOA": LINCOA,
    "UOBYQA": UOBYQA,
}


def run_experiment(
    hamiltonian,
    ansatz_type: str = "QAOA",
    reps: int = 1,
    cvar_alpha: float = 1.0,
    optimizer_name: str = "COBYLA",
    maxiter: int = 50,
    seed: int = 42,
    shots: int = 1024,
) -> Dict[str, Any]:
    """Run a single QAOA/VQE optimization experiment.

    Args:
        hamiltonian:    Problem Hamiltonian (must implement to_pauli_op()).
                        Created and owned by the caller (suite file).
        ansatz_type:    "QAOA" or "VQE".
        reps:           Circuit depth.
        cvar_alpha:     CVaR parameter (0 < alpha <= 1).
        optimizer_name: Classical optimizer name.
        maxiter:        Maximum optimizer iterations.
        seed:           Random seed for initial parameters.
        shots:          Number of circuit executions per optimizer call.
    """
    np.random.seed(seed)

    num_qubits = hamiltonian.num_qubits
    
    logger.debug(f"[EXPERIMENT] Creating cost operator for {num_qubits} qubits...")
    t_cost_op = time.time()
    cost_operator = hamiltonian.to_pauli_op()
    logger.debug(f"[EXPERIMENT] Cost operator created in {time.time() - t_cost_op:.2f}s ({len(cost_operator)} terms)")

    logger.debug(f"[EXPERIMENT] Creating {ansatz_type} ansatz (reps={reps})...")
    t_ansatz = time.time()
    if ansatz_type == "QAOA":
        ansatz = QAOAAnsatz(cost_operator, reps=reps)
        num_params = 2 * reps
    elif ansatz_type == "VQE":
        ansatz = efficient_su2(num_qubits, reps=reps)
        num_params = ansatz.num_parameters
    else:
        raise ValueError(f"Invalid ansatz_type '{ansatz_type}'. Use 'QAOA' or 'VQE'.")
    logger.debug(f"[EXPERIMENT] Ansatz created in {time.time() - t_ansatz:.2f}s ({num_params} parameters)")

    ansatz.measure_all()

    if optimizer_name not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Available: {list(_OPTIMIZERS.keys())}"
        )

    optimizer = _OPTIMIZERS[optimizer_name](maxiter=maxiter)
    sampler = Sampler()
    optimization_history = []

    # brute_force is cached on the hamiltonian object — free on repeated calls
    # with the same instance (common in suites where one hamiltonian is reused)
    if not hasattr(hamiltonian, "_optimal_states_cache"):
        logger.debug(f"[EXPERIMENT] Computing brute force solution...")
        t_brute = time.time()
        hamiltonian._optimal_states_cache = brute_force(hamiltonian)
        logger.debug(f"[EXPERIMENT] Brute force completed in {time.time() - t_brute:.2f}s")
    optimal_states = hamiltonian._optimal_states_cache
    
    initial_point = np.random.rand(num_params) * 2 * np.pi

    def objective(params):
        return cvar_sampler_cost_fun(
            params,
            ansatz,
            cost_operator,
            sampler,
            cvar_alpha,
            optimization_history,
            optimal_states=optimal_states,
            shots=shots,
        )

    logger.info(f"[EXPERIMENT] Starting {optimizer_name} optimization ({maxiter} max iterations)...")
    start_time = time.time()
    result = optimizer.optimize(objective_function=objective, initial_point=initial_point)
    elapsed = time.time() - start_time
    
    logger.info(f"[EXPERIMENT] Optimization completed in {elapsed:.2f}s ({len(optimization_history)} evaluations)")
    if len(optimization_history) > 0:
        avg_eval_time = elapsed / len(optimization_history)
        logger.debug(f"[EXPERIMENT] Average evaluation time: {avg_eval_time:.3f}s per iteration")

    return {
        "ansatz_type": ansatz_type,
        "cvar_alpha": cvar_alpha,
        "optimizer_name": optimizer_name,
        "num_qubits": num_qubits,
        "reps": reps,
        "shots": shots,
        "history": optimization_history,
        "result": result,
        "time": elapsed,
        "optimal_states": optimal_states,
    }
