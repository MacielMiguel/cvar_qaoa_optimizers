"""
Suite 1 — Architecture Comparison: QAOA vs VQE across CVaR alpha values (MaxMulticut).

Edit the config dataclass below to change experiment parameters, then run main.py.
"""
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_experiment
from quantum_core.display import plot_cost_evolution, plot_distribution_evolution
from quantum_core.results_io import save_results
from problems.maxmulticut.instances import MaxMulticutInstanceGenerator

# Suppress scipy sparse matrix efficiency warnings (they're from internal operations)
warnings.filterwarnings("ignore", category=Warning, module="scipy.sparse")

logger = logging.getLogger(__name__)


@dataclass
class Suite1Config:
    # Number of vertices/qubits (actual qubit count = num_vertices * ceil(log2(num_colors)))
    num_vertices: int = 5

    # Number of colors (communities)
    num_colors: int = 3

    # Circuit depth (number of QAOA layers / VQE reps)
    reps: int = 1

    # CVaR alpha values to sweep (0 < alpha <= 1)
    alphas: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])

    # Classical optimizer used for both QAOA and VQE
    optimizer: str = "COBYLA"

    # Maximum optimizer iterations per experiment
    maxiter: int = 15

    # Number of circuit executions (shots) per optimizer call
    # More shots → better probability estimates, slower runtime
    shots: int = 256


config = Suite1Config()


def run(output_dir: str) -> None:
    # Enable debug logging to see detailed timing info
    logging.getLogger('quantum_core.experiment').setLevel(logging.DEBUG)
    
    logger.info(f"\n{'='*60}")
    logger.info("[Suite 1 MaxMulticut] Starting... (QAOA vs VQE)")
    logger.info(f"{'='*60}")
    
    logger.info(f"[DEBUG] Creating hamiltonian (vertices={config.num_vertices}, colors={config.num_colors})")
    t0 = time.time()
    hamiltonian = MaxMulticutInstanceGenerator.fixed_simple(num_colors=config.num_colors)
    logger.info(f"[DEBUG] Hamiltonian created in {time.time() - t0:.2f}s")
    logger.info(f"[DEBUG]   → num_qubits: {hamiltonian.num_qubits}")
    logger.info(f"[DEBUG]   → num_edges: {len(hamiltonian.edges)}")
    
    logger.info(f"[DEBUG] Creating Pauli operator...")
    t0 = time.time()
    pauli_op = hamiltonian.to_pauli_op()
    logger.info(f"[DEBUG] Pauli operator created in {time.time() - t0:.2f}s")
    logger.info(f"[DEBUG]   → num_terms: {len(pauli_op)}")
    
    methods = ["QAOA", "VQE"]
    total = len(methods) * len(config.alphas)

    logger.info(
        f"\n[Suite 1 MaxMulticut] QAOA vs VQE (vertices={config.num_vertices}, colors={config.num_colors}, "
        f"reps={config.reps}, alphas={config.alphas})"
    )
    logger.info(f"[DEBUG] Total experiments: {total}")
    logger.info(f"[DEBUG] Max iterations per experiment: {config.maxiter}")

    results = {}
    i = 0
    suite_start = time.time()
    
    for method in methods:
        logger.info(f"\n[DEBUG] === Starting {method} experiments ===")
        method_start = time.time()
        
        for alpha in config.alphas:
            i += 1
            exp_start = time.time()
            logger.info(f"  [{i}/{total}] {method} α={alpha:.2f} → Starting...")
            
            res = run_experiment(
                hamiltonian=hamiltonian,
                ansatz_type=method,
                reps=config.reps,
                cvar_alpha=alpha,
                optimizer_name=config.optimizer,
                maxiter=config.maxiter,
                shots=config.shots,
            )
            
            exp_time = time.time() - exp_start
            results[f"{method}_alpha_{alpha}"] = res
            logger.info(f"  [{i}/{total}] {method} α={alpha:.2f} ✓ ({exp_time:.1f}s, best_cost={res['result']['fun']:.4f})")
        
        method_time = time.time() - method_start
        logger.info(f"[DEBUG] {method} completed in {method_time:.1f}s")

    suite_time = time.time() - suite_start
    logger.info(f"\n[DEBUG] Suite completed in {suite_time:.1f}s total")
    
    logger.info(f"[DEBUG] Creating plots...")
    plot_start = time.time()
    plot_cost_evolution(results, output_dir)
    # For each method/alpha, plot how the solution distribution evolves over iterations
    for key, res in results.items():
        plot_distribution_evolution(res, output_dir, filename=f"distribution_{key}.png")
    plot_time = time.time() - plot_start
    logger.info(f"[DEBUG] Plots created in {plot_time:.1f}s")
    
    logger.info(f"[DEBUG] Saving results...")
    save_start = time.time()
    save_results(results, output_dir, suite_name="suite1_maxmulticut", config=config)
    save_time = time.time() - save_start
    logger.info(f"[DEBUG] Results saved in {save_time:.1f}s")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Suite 1 Complete! Total time: {suite_time:.1f}s")
    logger.info(f"{'='*60}")
