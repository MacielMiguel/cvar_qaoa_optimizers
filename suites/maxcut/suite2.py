"""
Suite 2 — Classical optimizer benchmark on QAOA and VQE (paper Fig. 4).

All optimizers get the same budget of circuit evaluations, the same CVaR
level and the same initial points (one per seed). Edit the config dataclass
below, then run `python main.py maxcut-suite2`.
"""
import logging
from dataclasses import dataclass, field
from typing import List

from quantum_core.experiment import run_many
from quantum_core.display import plot_optimizer_benchmark
from quantum_core.results_io import save_results
from problems.maxcut.instances import MaxCutInstanceGenerator

logger = logging.getLogger(__name__)


@dataclass
class Suite2Config:
    num_qubits: int = 6
    reps: int = 1
    # Same CVaR level for every optimizer
    alpha: float = 0.5
    # SciPy solvers (BFGS and CG use finite-difference gradients)
    scipy_optimizers: List[str] = field(default_factory=lambda: [
        "COBYLA", "Nelder-Mead", "Powell", "BFGS", "ConjugateGradient", "SimulatedAnnealing"])
    # PDFO solvers (Powell's derivative-free methods); requires `pip install pdfo`
    pdfo_optimizers: List[str] = field(default_factory=lambda: [
        "COBYLA_PDFO", "NEWUOA", "BOBYQA", "LINCOA", "UOBYQA"])
    # Budget of circuit evaluations, identical for all optimizers
    max_evals: int = 200
    shots: int = 1024
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    workers: int = 1


config = Suite2Config()


def run(output_dir: str) -> list:
    hamiltonian = MaxCutInstanceGenerator.fixed(config.num_qubits)
    optimizers = config.scipy_optimizers + config.pdfo_optimizers
    tasks = [
        dict(hamiltonian=hamiltonian, ansatz_type=method, reps=config.reps, cvar_alpha=config.alpha,
             optimizer_name=opt, max_evals=config.max_evals, shots=config.shots, seed=seed)
        for method in ("QAOA", "VQE") for opt in optimizers for seed in config.seeds
    ]
    logger.info("[Suite 2] %d runs (%d optimizers, %d seeds)", len(tasks), len(optimizers), len(config.seeds))
    results = run_many(tasks, config.workers)

    scipy_res = [r for r in results if r["optimizer_name"] in config.scipy_optimizers]
    pdfo_res = [r for r in results if r["optimizer_name"] in config.pdfo_optimizers]
    if scipy_res:
        plot_optimizer_benchmark(scipy_res, output_dir, filename="suite2_scipy.png",
                                 titles={"QAOA": "QAOA (SciPy)", "VQE": "VQE (SciPy)"})
    if pdfo_res:
        plot_optimizer_benchmark(pdfo_res, output_dir, filename="suite2_pdfo.png",
                                 titles={"QAOA": "QAOA (PDFO)", "VQE": "VQE (PDFO)"})
    save_results(results, output_dir, suite_name="suite2", config=config)
    return results
