"""End-to-end behaviour of run_experiment, results I/O and plotting."""
import json
import logging

import pytest

from problems.maxcut.instances import MaxCutInstanceGenerator
from quantum_core import display
from quantum_core.experiment import run_experiment, run_many
from quantum_core.results_io import save_results

logging.disable(logging.CRITICAL)
H6 = MaxCutInstanceGenerator.fixed(6)


def test_same_seed_gives_identical_runs():
    a = run_experiment(H6, "QAOA", 1, 0.5, "COBYLA", max_evals=15, seed=7)
    b = run_experiment(H6, "QAOA", 1, 0.5, "COBYLA", max_evals=15, seed=7)
    assert [h["cvar_mean_energy"] for h in a["history"]] == [h["cvar_mean_energy"] for h in b["history"]]


def test_different_seeds_differ():
    a = run_experiment(H6, "VQE", 1, 0.5, "COBYLA", max_evals=15, seed=1)
    b = run_experiment(H6, "VQE", 1, 0.5, "COBYLA", max_evals=15, seed=2)
    assert a["history"][0]["params"] != b["history"][0]["params"]


@pytest.mark.parametrize("opt", ["COBYLA", "Nelder-Mead", "Powell", "BFGS", "SimulatedAnnealing"])
def test_evaluation_budget_is_respected(opt):
    res = run_experiment(H6, "VQE", 1, 0.5, opt, max_evals=30, seed=0)
    assert len(res["history"]) <= 30
    assert res["result"]["nfev"] == len(res["history"])


def test_parallel_runs_match_serial_runs():
    tasks = [dict(hamiltonian=H6, ansatz_type="QAOA", reps=1, cvar_alpha=0.5,
                  optimizer_name="COBYLA", max_evals=10, seed=s) for s in (0, 1)]
    serial = run_many(tasks, workers=1)
    parallel = run_many(tasks, workers=2)
    assert [r["result"]["fun"] for r in serial] == [r["result"]["fun"] for r in parallel]


def test_results_and_figures_are_written(tmp_path):
    runs = [run_experiment(H6, a, 1, alpha, "COBYLA", max_evals=10, seed=s)
            for a in ("QAOA", "VQE") for alpha in (0.25, 1.0) for s in (0, 1)]
    save_results(runs, str(tmp_path), "t", config={"note": "test"})
    full = json.loads((tmp_path / "t_full.json").read_text())
    assert full[0]["optimizer_result"]["nfev"] == len(full[0]["history"])
    assert full[0]["seed"] == 0 and full[0]["shots"] == 1024
    cfg = json.loads((tmp_path / "t_config.json").read_text())
    assert "qiskit" in cfg["versions"]
    display.plot_cost_evolution(runs, str(tmp_path), filename="a.png")
    display.plot_optimizer_benchmark(runs, str(tmp_path), filename="b.png")
    display.plot_success_probability(runs, str(tmp_path), filename="c.png")
    display.plot_distribution_evolution(runs[0], str(tmp_path), filename="d.png")
    for name in "abcd":
        assert (tmp_path / f"{name}.png").stat().st_size > 0
