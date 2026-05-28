"""
Suite 2 — Optimizer Comparison across MaxMulticut instances.

Benchmark different classical optimizers (scipy-based and pdfo-based) on MaxMulticut.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_experiment
from quantum_core.display import plot_cost_evolution, plot_distribution_evolution
from quantum_core.results_io import save_results
from problems.maxmulticut.instances import MaxMulticutInstanceGenerator

logger = logging.getLogger(__name__)


@dataclass
class Suite2Config:
    # Number of vertices
    num_vertices: int = 6

    # Number of colors
    num_colors: int = 3

    # Circuit depth
    reps: int = 1

    # CVaR alpha (fixed)
    cvar_alpha: float = 0.25

    # Optimizers to benchmark
    # scipy-based:  COBYLA, COBYQA, Nelder-Mead, BFGS, Powell, ConjugateGradient, SimulatedAnnealing
    # pdfo-based:   NEWUOA, COBYLA_PDFO, BOBYQA, LINCOA, UOBYQA
    optimizers: List[str] = field(default_factory=lambda: ["NEWUOA", "COBYLA_PDFO", "BOBYQA", "LINCOA", "UOBYQA"])

    # Maximum optimizer iterations
    maxiter: int = 30

    # Shots per evaluation
    shots: int = 256


config = Suite2Config()


def run(output_dir: str) -> None:
    hamiltonian = MaxMulticutInstanceGenerator.fixed_simple(num_colors=config.num_colors)
    total = len(config.optimizers)

    logger.info(
        f"\n[Suite 2 MaxMulticut] Optimizer Comparison (vertices={config.num_vertices}, "
        f"colors={config.num_colors}, reps={config.reps}, alpha={config.cvar_alpha})"
    )

    results = {}
    for i, opt_name in enumerate(config.optimizers):
        res = run_experiment(
            hamiltonian=hamiltonian,
            ansatz_type="QAOA",
            reps=config.reps,
            cvar_alpha=config.cvar_alpha,
            optimizer_name=opt_name,
            maxiter=config.maxiter,
            shots=config.shots,
        )
        results[opt_name] = res
        logger.info(f"  [{i + 1}/{total}] {opt_name} ({res['time']:.1f}s)")

    plot_cost_evolution(results, output_dir)
    for key, res in results.items():
        plot_distribution_evolution(res, output_dir, filename=f"distribution_{key}.png")
    save_results(results, output_dir, suite_name="suite2_maxmulticut")
