"""
Suite 2 — Optimizer Benchmark: compare classical optimizers on QAOA and VQE.

Edit the config dataclass below to change experiment parameters, then run main.py.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_experiment
from quantum_core.display import plot_optimizer_benchmark, plot_distribution_evolution
from quantum_core.results_io import save_results
from problems.maxcut.instances import MaxCutInstanceGenerator

logger = logging.getLogger(__name__)


@dataclass
class Suite2Config:
    # Number of graph nodes (= number of qubits)
    num_qubits: int = 6

    # Circuit depth
    reps: int = 1

    # Fixed CVaR alpha for this benchmark
    alpha: float = 1.0

    # Classical optimizers to compare
    # scipy-based:  COBYLA, COBYQA, Nelder-Mead, BFGS, Powell, ConjugateGradient, SimulatedAnnealing
    # pdfo-based:   NEWUOA, COBYLA_PDFO, BOBYQA, LINCOA, UOBYQA
    optimizers: List[str] = field(default_factory=lambda: ["COBYLA", "Nelder-Mead", "BFGS", "Powell", "ConjugateGradient", "SimulatedAnnealing"])

    # Maximum optimizer iterations per experiment
    maxiter: int = 30*3

    # Number of circuit executions (shots) per optimizer call
    shots: int = 1024


config = Suite2Config()


def run(output_dir: str) -> None:
    hamiltonian = MaxCutInstanceGenerator.fixed(config.num_qubits)
    methods = ["QAOA", "VQE"]
    total = len(methods) * len(config.optimizers)

    logger.info(f"\n[Suite 2] Optimizer Benchmark (reps={config.reps}, alpha={config.alpha})")

    results = {}
    i = 0
    for method in methods:
        for opt_name in config.optimizers:
            i += 1
            try:
                res = run_experiment(
                    hamiltonian=hamiltonian,
                    ansatz_type=method,
                    reps=config.reps,
                    cvar_alpha=config.alpha,
                    optimizer_name=opt_name,
                    maxiter=config.maxiter,
                    shots=config.shots,
                )
                results[f"{method}_{opt_name}"] = res
                logger.info(f"  [{i}/{total}] {method} #{opt_name} ({res['time']:.1f}s)")
            except Exception as e:
                logger.error(f"  [{i}/{total}] {method} #{opt_name} failed: {e}")

    plot_optimizer_benchmark(results, output_dir)
    # For each method/optimizer, plot how the solution distribution evolves over iterations
    for key, res in results.items():
        plot_distribution_evolution(res, output_dir, filename=f"distribution_{key}.png")
    save_results(results, output_dir, suite_name="suite2")
