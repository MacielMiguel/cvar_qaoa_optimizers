"""
Suite 3 — Success Probability: fraction of instances reaching a success threshold
as a function of normalised iterations, for different circuit depths (reps).

Reproduces the structure of Figure 4 in Barkoutsos et al. (2020):
  - Rows: circuit depth (reps_list)
  - Columns: VQE (left) / QAOA (right)
  - Curves: CVaR alpha values × two thresholds (1 %, 10 %)
"""
import logging
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_experiment
from quantum_core.display import plot_success_probability
from quantum_core.results_io import save_results
from problems.maxcut.instances import MaxCutInstanceGenerator

logger = logging.getLogger(__name__)


@dataclass
class Suite3Config:
    # Fixed graph size for all instances (= number of qubits)
    # Larger n → rarer optimal states → more meaningful success curves
    # n=8 recommended; n=6 for faster runs
    num_nodes: int = 8

    # Edge density — keep fixed across all instances so graphs are comparable
    edge_density: float = 0.5

    # Number of random instances (different seeds, same topology parameters)
    num_instances: int = 20

    # Circuit depths to compare (one row per depth in the figure)
    reps_list: List[int] = field(default_factory=lambda: [1, 2])

    # CVaR alpha values to sweep (0 < alpha <= 1)
    alphas: List[float] = field(default_factory=lambda: [0.01, 0.25, 0.5, 1.0])

    # Classical optimizer
    optimizer: str = "COBYLA"

    # Maximum optimizer iterations per experiment
    # More iterations → longer x-axis range in the figure
    maxiter: int = 100

    # Number of circuit executions (shots) per optimizer call
    shots: int = 1024


config = Suite3Config()


def run(output_dir: str) -> None:
    methods = ["QAOA", "VQE"]
    total = (
        len(config.reps_list)
        * config.num_instances
        * len(methods)
        * len(config.alphas)
    )

    logger.info(f"\n[Suite 3] Success Probability")
    logger.info(
        f"  {len(config.reps_list)} depths × {config.num_instances} instances "
        f"× 2 ansätze × {len(config.alphas)} alphas  =  {total} experiments"
    )
    logger.info(
        f"  num_nodes={config.num_nodes}, edge_density={config.edge_density}, "
        f"maxiter={config.maxiter}, shots={config.shots}"
    )

    results = []
    i = 0

    for reps in config.reps_list:
        for instance_id in range(config.num_instances):
            # Deterministic seed per (instance_id) — same graph across all reps/alphas
            instance_seed = config.num_nodes * 100 + instance_id
            hamiltonian = MaxCutInstanceGenerator.random(
                config.num_nodes,
                edge_density=config.edge_density,
                seed=instance_seed,
            )
            for method in methods:
                for alpha in config.alphas:
                    i += 1
                    res = run_experiment(
                        hamiltonian=hamiltonian,
                        ansatz_type=method,
                        reps=reps,
                        cvar_alpha=alpha,
                        optimizer_name=config.optimizer,
                        maxiter=config.maxiter,
                        shots=config.shots,
                    )
                    results.append(res)
                    logger.info(
                        f"  [{i}/{total}] reps={reps} {method} α={alpha:.2f} "
                        f"inst={instance_id} ({res['time']:.1f}s)"
                    )

    plot_success_probability(results, output_dir)
    save_results(results, output_dir, suite_name="suite3")
