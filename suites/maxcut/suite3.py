"""
Suite 3 — Success probability on random graphs (paper Fig. 5).

Reproduces the structure of Fig. 4 in Barkoutsos et al. (Quantum 4, 256, 2020):
  - Rows: circuit depth (reps_list)
  - Columns: QAOA (left) / VQE (right)
  - Curves: CVaR alpha values x two success thresholds (1 %, 10 %)

Edit the config dataclass below, then run `python main.py maxcut-suite3`.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_many
from quantum_core.display import plot_success_probability
from quantum_core.results_io import save_results
from problems.maxcut.instances import MaxCutInstanceGenerator

logger = logging.getLogger(__name__)


@dataclass
class Suite3Config:
    # Graph size (= number of qubits) and edge density, fixed for all instances
    num_nodes: int = 8
    edge_density: float = 0.5
    # Random instances; instance i uses seed base_seed + i for the graph,
    # the initial point and the shot noise
    num_instances: int = 60
    base_seed: int = 800
    reps_list: List[int] = field(default_factory=lambda: [1, 2])
    alphas: List[float] = field(default_factory=lambda: [0.01, 0.25, 0.5, 1.0])
    optimizer: str = "COBYLA"
    # Budget of circuit evaluations per run (x-axis reaches max_evals / num_nodes)
    max_evals: int = 100
    shots: int = 1024
    workers: int = 1


config = Suite3Config()


def run(output_dir: str) -> list:
    instances = [
        (config.base_seed + i,
         MaxCutInstanceGenerator.random(config.num_nodes, edge_density=config.edge_density,
                                        seed=config.base_seed + i))
        for i in range(config.num_instances)
    ]
    tasks = [
        dict(hamiltonian=h, ansatz_type=method, reps=reps, cvar_alpha=alpha,
             optimizer_name=config.optimizer, max_evals=config.max_evals,
             shots=config.shots, seed=seed)
        for reps in config.reps_list for seed, h in instances
        for method in ("QAOA", "VQE") for alpha in config.alphas
    ]
    logger.info("[Suite 3] %d runs (%d instances)", len(tasks), config.num_instances)
    results = run_many(tasks, config.workers)

    plot_success_probability(results, output_dir, filename="suite3_success_probability.png")
    save_results(results, output_dir, suite_name="suite3", config=config)
    return results
