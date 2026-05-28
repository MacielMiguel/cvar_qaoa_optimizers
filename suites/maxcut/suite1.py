"""
Suite 1 — Architecture Comparison: QAOA vs VQE across CVaR alpha values.

Edit the config dataclass below to change experiment parameters, then run main.py.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_experiment
from quantum_core.display import plot_cost_evolution, plot_distribution_evolution
from quantum_core.results_io import save_results
from problems.maxcut.instances import MaxCutInstanceGenerator

logger = logging.getLogger(__name__)


@dataclass
class Suite1Config:
    # Number of graph nodes (= number of qubits)
    num_qubits: int = 6

    # Circuit depth (number of QAOA layers / VQE reps)
    reps: int = 1

    # CVaR alpha values to sweep (0 < alpha <= 1)
    # Small alpha → focus on best outcomes; 1.0 → plain expectation value
    alphas: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])

    # Classical optimizer used for both QAOA and VQE
    optimizer: str = "COBYLA"

    # Maximum optimizer iterations per experiment
    maxiter: int = 30*3

    # Number of circuit executions (shots) per optimizer call
    # More shots → better probability estimates, slower runtime
    shots: int = 1024


config = Suite1Config()


def run(output_dir: str) -> None:
    hamiltonian = MaxCutInstanceGenerator.fixed(config.num_qubits)
    methods = ["QAOA", "VQE"]
    total = len(methods) * len(config.alphas)

    logger.info(f"\n[Suite 1] QAOA vs VQE (reps={config.reps}, alphas={config.alphas})")

    results = {}
    i = 0
    for method in methods:
        for alpha in config.alphas:
            i += 1
            res = run_experiment(
                hamiltonian=hamiltonian,
                ansatz_type=method,
                reps=config.reps,
                cvar_alpha=alpha,
                optimizer_name=config.optimizer,
                maxiter=config.maxiter,
                shots=config.shots,
            )
            results[f"{method}_alpha_{alpha}"] = res
            logger.info(f"  [{i}/{total}] {method} α={alpha:.2f} ({res['time']:.1f}s)")

    plot_cost_evolution(results, output_dir)
    # For each method/alpha, plot how the solution distribution evolves over iterations
    for key, res in results.items():
        plot_distribution_evolution(res, output_dir, filename=f"distribution_{key}.png")
    save_results(results, output_dir, suite_name="suite1")
