import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Iterable, Union

from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


# Lookup table: parity of the number of set bits in each byte (0–255).
# Used to compute ±1 eigenvalues of Pauli-Z products via bitwise XOR.
_PARITY = np.array(
    [-1 if bin(i).count("1") % 2 else 1 for i in range(256)],
    dtype=np.complex128,
)


def evaluate_sparse_pauli(state: int, observable: SparsePauliOp) -> complex:
    """Evaluate ⟨state|H|state⟩ for a computational basis state.

    Instead of building the full 2^n matrix, exploits the fact that for a
    Z-diagonal Hamiltonian H = Σ c_k P_k the expectation value of a basis
    state |s⟩ is Σ c_k · (-1)^{popcount(s & mask_k)}, where mask_k encodes
    which qubits carry a Z in Pauli string k.

    Args:
        state:      Measured bitstring encoded as an integer.
        observable: Hamiltonian as a SparsePauliOp (only Z terms matter here;
                    X/Y terms vanish in the Z basis).

    Returns:
        Real part of the expectation value (imaginary part is always 0 for
        Hamiltonians with real coefficients).
    """
    # Pack the Z masks of every Pauli term into bytes (one row per term)
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    # Encode the measured state as the same byte layout
    state_bytes = np.frombuffer(
        state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8
    )
    # For each term: AND selects the relevant qubits, XOR-reduce gives parity
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    coeffs = observable.coeffs
    if coeffs is None:
        coeffs = np.ones(len(reduced), dtype=np.complex128)
    return np.sum(coeffs * _PARITY[reduced])


def _get_cvar_aggregation(alpha: float) -> Callable:
    """Return a CVaR aggregation function for the given alpha.

    CVaR_alpha = mean energy of the best alpha-fraction of outcomes.
    alpha=1.0 reduces to the plain expectation value.

    Args:
        alpha: Quantile level in (0, 1].

    Returns:
        A function (iterable of (probability, value)) -> float.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")

    def cvar_aggregation(objective_dict: Iterable[Tuple[float, float]]) -> float:
        sorted_measurements = sorted(objective_dict, key=lambda x: x[1])
        accumulated = 0.0
        cvar = 0.0
        for probability, value in sorted_measurements:
            cvar += value * min(probability, alpha - accumulated)
            accumulated += probability
            if accumulated >= alpha:
                break
        return cvar / alpha

    return cvar_aggregation


def cvar_sampler_cost_fun(
    params: np.ndarray,
    ansatz,
    hamiltonian: SparsePauliOp,
    sampler,
    aggregation: Optional[Union[Callable, float]] = None,
    optimization_history: Optional[List[Dict]] = None,
    optimal_states: Optional[frozenset] = None,
    shots: int = 1024,
) -> float:
    """CVaR cost function for any parametrized ansatz (QAOA, VQE, …).

    Called at every iteration of the classical optimizer. Runs the quantum
    circuit, evaluates the Hamiltonian on each measured outcome, then returns
    the CVaR (or expectation value if aggregation=None) as a scalar for the
    optimizer to minimize.

    Args:
        params:               Current circuit parameters.
        ansatz:               Parametrized quantum circuit (QAOA or VQE).
        hamiltonian:          Cost Hamiltonian as a SparsePauliOp.
        sampler:              Qiskit StatevectorSampler (or compatible).
        aggregation:          None → expectation value; float → CVaR with that
                              alpha; callable → custom aggregation.
        optimization_history: If provided, a dict is appended each call with
                              cvar_mean_energy, mean_energy, success_prob and distribution.
        optimal_state:        Reference optimal state (int) used to track
                              success probability across iterations.
        shots:                Number of circuit executions per call (default 1024).

    Returns:
        Scalar cost value for the optimizer.
    """
    # --- 1. Run the circuit ---
    job = sampler.run([(ansatz, params)], shots=shots)
    sampler_result = job.result()

    # Extract {state_int: count} (supports SamplerV2 and older interfaces)
    if hasattr(sampler_result[0].data, "c"):
        sampled_int_counts = sampler_result[0].data.c.get_int_counts()
    else:
        pub_result = sampler_result[0]
        bitstrings = pub_result.data.meas.get_counts()
        sampled_int_counts = {int(k, 2): v for k, v in bitstrings.items()}

    shots = sum(sampled_int_counts.values())
    probabilities = {s: c / shots for s, c in sampled_int_counts.items()}

    # --- 2. Evaluate H on each measured outcome ---
    evaluated = {
        state: (prob, float(np.real(evaluate_sparse_pauli(state, hamiltonian))))
        for state, prob in probabilities.items()
    }

    # --- 3. Aggregate into a scalar cost ---
    if aggregation is None:
        result = sum(p * v for p, v in evaluated.values())
    elif isinstance(aggregation, (int, float, np.integer, np.floating)):
        result = _get_cvar_aggregation(float(aggregation))(evaluated.values())
    else:
        result = aggregation(evaluated.values())

    # --- 4. Record history (optional) ---
    if optimization_history is not None:
        optimization_history.append({
            "params":           params.tolist(),
            "cvar_mean_energy": result,
            "mean_energy":      sum(p * v for p, v in evaluated.values()),
            "success_prob":     sum(probabilities.get(s, 0.0) for s in optimal_states) if optimal_states else 0.0,
            "distribution":     sampled_int_counts,
            # Per-state energy — used by plot_distribution_evolution to shade the CVaR region
            "state_energies":   {state: val for state, (_, val) in evaluated.items()},
        })

    return result
