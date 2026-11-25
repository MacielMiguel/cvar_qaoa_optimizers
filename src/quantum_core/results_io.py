"""
Persist experiment results to CSV (summary) and JSON (full history).

Usage:
    from quantum_core.results_io import save_results

    save_results(results, output_dir, suite_name="suite1", config=config)

`results` can be either:
  - a dict  {key: result_dict}   (suites 1 & 2)
  - a list  [result_dict, ...]   (suite 3)

Output files:
  {output_dir}/{suite_name}_summary.csv   — one row per experiment
  {output_dir}/{suite_name}_full.json     — full iteration history
  {output_dir}/{suite_name}_config.json   — suite configuration + library versions
"""

import csv
import dataclasses
import json
import logging
import os
import platform
from importlib import metadata
from typing import Any, Dict, List, Optional, Union

import numpy as np

from quantum_core.solver import state_to_bitstring

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

class _NumpyEncoder(json.JSONEncoder):
    """Make numpy scalars and arrays JSON-serialisable."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def _to_list(results: Union[Dict, List]) -> List[Dict]:
    """Normalise dict or list of result dicts to a flat list."""
    if isinstance(results, dict):
        return list(results.values())
    return results


def _serialize_optimizer_result(res: Dict) -> Dict:
    """Extract JSON-safe fields from the result dict built by run_experiment()."""
    r = res.get("result") or {}
    out = {}
    for key, value in r.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        out[key] = value
    return out


def _versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for pkg in ("qiskit", "numpy", "scipy", "matplotlib", "pdfo"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def save_results(
    results: Union[Dict, List],
    output_dir: str,
    suite_name: str,
    config: Optional[Any] = None,
) -> None:
    """
    Save experiment results to CSV and JSON.

    Args:
        results:     dict or list of result dicts returned by run_experiment().
        output_dir:  Directory where files are written (must already exist).
        suite_name:  Used as a filename prefix, e.g. "suite1".
        config:      Suite configuration (dataclass or dict) saved alongside the
                     results so that every figure can be regenerated.
    """
    records = _to_list(results)
    _save_csv(records, output_dir, suite_name)
    _save_json(records, output_dir, suite_name)
    if config is not None:
        cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
        path = os.path.join(output_dir, f"{suite_name}_config.json")
        with open(path, "w") as f:
            json.dump({"suite": suite_name, "config": cfg, "versions": _versions()},
                      f, indent=2, cls=_NumpyEncoder)
        logger.info("Config saved to %s", path)


def _save_csv(records: List[Dict], output_dir: str, suite_name: str) -> None:
    """One row per experiment with hyperparams and aggregate metrics."""
    path = os.path.join(output_dir, f"{suite_name}_summary.csv")
    fieldnames = [
        "ansatz_type",
        "cvar_alpha",
        "optimizer_name",
        "num_qubits",
        "reps",
        "shots",
        "seed",
        "max_evals",
        "num_evaluations",
        "initial_cvar_mean_energy",
        "final_cvar_mean_energy",
        "initial_success_prob",
        "final_success_prob",
        "time_s",
        "num_optimal_states",
        "optimal_states",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in records:
            history = res.get("history", [])
            optimal_states = res.get("optimal_states", frozenset())
            writer.writerow({
                "ansatz_type":              res["ansatz_type"],
                "cvar_alpha":               res["cvar_alpha"],
                "optimizer_name":           res["optimizer_name"],
                "num_qubits":               res["num_qubits"],
                "reps":                     res["reps"],
                "shots":                    res.get("shots", ""),
                "seed":                     res.get("seed", ""),
                "max_evals":                res.get("max_evals", ""),
                "num_evaluations":          len(history),
                "initial_cvar_mean_energy": history[0]["cvar_mean_energy"] if history else "",
                "final_cvar_mean_energy":   history[-1]["cvar_mean_energy"] if history else "",
                "initial_success_prob":     history[0].get("success_prob", "") if history else "",
                "final_success_prob":       history[-1].get("success_prob", "") if history else "",
                "time_s":                   round(res["time"], 3),
                "num_optimal_states":       len(optimal_states),
                "optimal_states":           " ".join(
                    state_to_bitstring(s, res["num_qubits"]) for s in sorted(optimal_states)
                ),
            })

    logger.info("Results CSV saved to %s", path)


def _save_json(records: List[Dict], output_dir: str, suite_name: str) -> None:
    """Full iteration history (cost + success_prob per step), hyperparams, optimizer result."""
    path = os.path.join(output_dir, f"{suite_name}_full.json")

    serializable = []
    for res in records:
        history = res.get("history", [])
        serializable.append({
            "ansatz_type":      res["ansatz_type"],
            "cvar_alpha":       res["cvar_alpha"],
            "optimizer_name":   res["optimizer_name"],
            "num_qubits":       res["num_qubits"],
            "reps":             res["reps"],
            "shots":            res.get("shots"),
            "seed":             res.get("seed"),
            "max_evals":        res.get("max_evals"),
            "num_params":       res.get("num_params"),
            "instance":         res.get("instance"),
            "time_s":           res["time"],
            "optimal_states":   sorted(res.get("optimal_states", [])),
            "optimizer_result": _serialize_optimizer_result(res),
            "history": [
                {
                    "evaluation":   i + 1,
                    "cvar_mean_energy":    step["cvar_mean_energy"],
                    "mean_energy":  step["mean_energy"],
                    "success_prob": step.get("success_prob", 0.0),
                }
                for i, step in enumerate(history)
            ],
        })

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, cls=_NumpyEncoder)

    logger.info("Results JSON saved to %s", path)
