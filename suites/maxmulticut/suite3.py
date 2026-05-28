"""
Suite 3 — Scaling & Number of Colors Analysis.

Benchmark how MaxMulticut performance scales with number of vertices and colors.
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
class Suite3Config:
    # Vertex counts to sweep
    num_vertices_list: List[int] = field(default_factory=lambda: [4, 5, 6])

    # Color counts to sweep
    num_colors_list: List[int] = field(default_factory=lambda: [2, 3])

    # Circuit depth
    reps: int = 1

    # CVaR alpha
    cvar_alpha: float = 0.5

    # Classical optimizer
    optimizer: str = "COBYLA"

    # Maximum iterations
    maxiter: int = 30

    # Shots
    shots: int = 256


config = Suite3Config()


def run(output_dir: str) -> None:
    total = len(config.num_vertices_list) * len(config.num_colors_list)

    logger.info(
        f"\n[Suite 3 MaxMulticut] Scaling Analysis "
        f"(vertices={config.num_vertices_list}, colors={config.num_colors_list}, "
        f"reps={config.reps}, alpha={config.cvar_alpha})"
    )

    results = {}
    i = 0
    for num_v in config.num_vertices_list:
        for num_c in config.num_colors_list:
            i += 1
            hamiltonian = MaxMulticutInstanceGenerator.fixed_grid_3x3(num_colors=num_c) if num_v == 9 else MaxMulticutInstanceGenerator.random(
                num_vertices=num_v, num_colors=num_c, edge_density=0.5, seed=42
            )
            res = run_experiment(
                hamiltonian=hamiltonian,
                ansatz_type="QAOA",
                reps=config.reps,
                cvar_alpha=config.cvar_alpha,
                optimizer_name=config.optimizer,
                maxiter=config.maxiter,
                shots=config.shots,
            )
            key = f"v{num_v}_c{num_c}"
            results[key] = res
            logger.info(f"  [{i}/{total}] {key} ({res['time']:.1f}s)")

    plot_cost_evolution(results, output_dir)
    for key, res in results.items():
        plot_distribution_evolution(res, output_dir, filename=f"distribution_{key}.png")
    save_results(results, output_dir, suite_name="suite3_maxmulticut")
