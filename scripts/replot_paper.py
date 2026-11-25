#!/usr/bin/env python
"""Regenerate the paper figures from saved results, without re-running the suites.

Usage:
    python scripts/replot_paper.py results/paper

Reads suite{1,2,3}_full.json and suite{1,2}_config.json from the folder and
rewrites the PNG files next to them. The outcome-distribution figure needs the
per-state counts, which are not stored in the JSON, so that single run is
recomputed (runs are deterministic given their seed, so it is identical).
"""
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from quantum_core import display  # noqa: E402
from quantum_core.experiment import run_experiment  # noqa: E402
from problems.maxcut.instances import MaxCutInstanceGenerator  # noqa: E402


def _load(folder, name):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):  # full.json: restore the keys the plots expect
        for r in data:
            r["time"] = r.get("time_s", 0.0)
    return data


def main(folder: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)

    s1, c1 = _load(folder, "suite1_full.json"), _load(folder, "suite1_config.json")
    if s1:
        display.plot_cost_evolution(s1, folder, filename="suite1_alpha_sweep.png")
        cfg = c1["config"] if c1 else {}
        res = run_experiment(
            MaxCutInstanceGenerator.fixed(cfg.get("num_qubits", 6)),
            ansatz_type=cfg.get("distribution_ansatz", "VQE"), reps=cfg.get("reps", 1),
            cvar_alpha=cfg.get("distribution_alpha", 0.25), optimizer_name=cfg.get("optimizer", "COBYLA"),
            max_evals=cfg.get("max_evals", 100), shots=cfg.get("shots", 1024),
            seed=cfg.get("seeds", [0])[0],
        )
        display.plot_distribution_evolution(res, folder, filename="suite1_distribution.png")

    s2, c2 = _load(folder, "suite2_full.json"), _load(folder, "suite2_config.json")
    if s2 and c2:
        for family in ("scipy", "pdfo"):
            names = set(c2["config"][f"{family}_optimizers"])
            label = "SciPy" if family == "scipy" else "PDFO"
            display.plot_optimizer_benchmark(
                [r for r in s2 if r["optimizer_name"] in names], folder,
                filename=f"suite2_{family}.png",
                titles={"QAOA": f"QAOA ({label})", "VQE": f"VQE ({label})"})

    s3 = _load(folder, "suite3_full.json")
    if s3:
        display.plot_success_probability(s3, folder, filename="suite3_success_probability.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "paper"))
