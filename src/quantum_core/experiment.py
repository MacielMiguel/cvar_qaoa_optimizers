import time
import logging
import numpy as np
from typing import Dict, Any, Optional

from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit.library import efficient_su2
from qiskit import transpile
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
    max_evals: int = 100,
    seed: int = 0,
    shots: int = 1024,
    maxiter: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single QAOA/VQE optimization experiment.

    The budget is expressed in **circuit evaluations** (calls to the cost
    function), so that optimizers with different notions of "iteration" are
    compared on equal terms. Only the first ``max_evals`` evaluations are
    executed on the (simulated) quantum device and recorded; if the optimizer
    asks for more, it receives the last recorded cost without any circuit
    execution, and those calls are not part of the results.

    Args:
        hamiltonian:    Problem Hamiltonian (must implement to_pauli_op()).
        ansatz_type:    "QAOA" or "VQE".
        reps:           Circuit depth.
        cvar_alpha:     CVaR parameter (0 < alpha <= 1).
        optimizer_name: Classical optimizer name (see _OPTIMIZERS).
        max_evals:      Budget of circuit evaluations.
        seed:           Seed for the initial point and for shot sampling.
        shots:          Number of circuit executions per evaluation.
        maxiter:        Deprecated alias of max_evals (kept for old suites).
    """
    if maxiter is not None:
        max_evals = maxiter

    rng = np.random.default_rng(seed)
    num_qubits = hamiltonian.num_qubits
    cost_operator = hamiltonian.to_pauli_op()

    if ansatz_type == "QAOA":
        ansatz = QAOAAnsatz(cost_operator, reps=reps)
    elif ansatz_type == "VQE":
        ansatz = efficient_su2(num_qubits, reps=reps)
    else:
        raise ValueError(f"Invalid ansatz_type '{ansatz_type}'. Use 'QAOA' or 'VQE'.")
    num_params = ansatz.num_parameters
    ansatz.measure_all()
    # Decompose once into elementary gates: simulating the high-level
    # PauliEvolutionGate of QAOA at every evaluation is ~50x slower.
    # Parameter order (and hence the meaning of the parameter vector) is kept.
    ansatz = transpile(ansatz, basis_gates=["rz", "rx", "ry", "h", "cx", "rzz", "sx", "x"],
                       optimization_level=0)
    assert ansatz.num_parameters == num_params

    if optimizer_name not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{optimizer_name}'. "
            f"Available: {list(_OPTIMIZERS.keys())}"
        )
    optimizer = _OPTIMIZERS[optimizer_name](maxiter=max_evals)

    # A Generator (not an int) so that successive evaluations draw fresh,
    # yet reproducible, shot noise.
    sampler = Sampler(seed=np.random.default_rng(seed + 1_000_003))
    optimization_history = []
    calls = {"total": 0}

    if not hasattr(hamiltonian, "_optimal_states_cache"):
        hamiltonian._optimal_states_cache = brute_force(hamiltonian)
    optimal_states = hamiltonian._optimal_states_cache

    initial_point = rng.uniform(0.0, 2 * np.pi, size=num_params)

    def objective(params):
        calls["total"] += 1
        if len(optimization_history) >= max_evals:
            return optimization_history[-1]["cvar_mean_energy"]
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

    start_time = time.time()
    raw_result = optimizer.optimize(objective_function=objective, initial_point=initial_point)
    elapsed = time.time() - start_time

    best = min(optimization_history, key=lambda h: h["cvar_mean_energy"])
    result = {
        "x": np.asarray(best["params"]),
        "fun": float(best["cvar_mean_energy"]),
        "nfev": len(optimization_history),
        "nfev_requested": calls["total"],
        "optimizer_fun": float(raw_result.get("fun", np.nan)),
        "success": bool(raw_result.get("success", False)),
        "message": str(raw_result.get("message", "")),
    }
    logger.info(
        "[EXPERIMENT] %s/%s alpha=%.2f seed=%d: %d evaluations (%d requested) in %.2fs",
        ansatz_type, optimizer_name, cvar_alpha, seed,
        result["nfev"], result["nfev_requested"], elapsed,
    )

    return {
        "ansatz_type": ansatz_type,
        "cvar_alpha": cvar_alpha,
        "optimizer_name": optimizer_name,
        "num_qubits": num_qubits,
        "num_params": num_params,
        "reps": reps,
        "shots": shots,
        "seed": seed,
        "max_evals": max_evals,
        "instance": hamiltonian.to_dict(),
        "history": optimization_history,
        "result": result,
        "time": elapsed,
        "optimal_states": optimal_states,
    }


def _run_task(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return run_experiment(**kwargs)


def run_many(tasks, workers: int = 1):
    """Run several experiments, optionally in parallel processes.

    Each task is a dict of keyword arguments for run_experiment(). Results are
    returned in the same order as the tasks, so parallelism never changes them.
    """
    tasks = list(tasks)
    for t in tasks:  # compute the reference optimum once, before pickling
        h = t["hamiltonian"]
        if not hasattr(h, "_optimal_states_cache"):
            h._optimal_states_cache = brute_force(h)
    if workers <= 1:
        return [run_experiment(**t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_task, tasks, chunksize=4))
