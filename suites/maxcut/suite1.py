"""
Suite 1 — Effect of the CVaR level alpha for QAOA and VQE (paper Figs. 2 and 3).

Edit the config dataclass below to change experiment parameters, then run
`python main.py maxcut-suite1`.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_many
from quantum_core.display import plot_cost_evolution, plot_distribution_evolution
from quantum_core.results_io import save_results
from problems.maxcut.instances import MaxCutInstanceGenerator

logger = logging.getLogger(__name__)


@dataclass
class Suite1Config:
    # Fixed benchmark graph: number of nodes (= qubits)
    num_qubits: int = 6
    # Circuit depth (QAOA layers / EfficientSU2 reps)
    reps: int = 1
    # CVaR levels (alpha = 1 is the plain expectation value)
    alphas: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    optimizer: str = "COBYLA"
    # Budget of circuit evaluations per run
    max_evals: int = 100
    shots: int = 1024
    # Independent runs (initial point + shot noise) per configuration
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    # Run whose distribution is plotted (paper Fig. 3)
    distribution_ansatz: str = "VQE"
    distribution_alpha: float = 0.25
    workers: int = 1


config = Suite1Config()


def run(output_dir: str) -> list:
    hamiltonian = MaxCutInstanceGenerator.fixed(config.num_qubits)
    tasks = [
        dict(hamiltonian=hamiltonian, ansatz_type=method, reps=config.reps, cvar_alpha=alpha,
             optimizer_name=config.optimizer, max_evals=config.max_evals,
             shots=config.shots, seed=seed)
        for method in ("QAOA", "VQE") for alpha in config.alphas for seed in config.seeds
    ]
    logger.info("[Suite 1] %d runs (alpha sweep, %d seeds)", len(tasks), len(config.seeds))
    results = run_many(tasks, config.workers)

    plot_cost_evolution(results, output_dir, filename="suite1_alpha_sweep.png")
    first_seed = config.seeds[0]
    for res in results:
        if (res["ansatz_type"] == config.distribution_ansatz
                and res["cvar_alpha"] == config.distribution_alpha and res["seed"] == first_seed):
            plot_distribution_evolution(res, output_dir, filename="suite1_distribution.png")
    save_results(results, output_dir, suite_name="suite1", config=config)
    return results
