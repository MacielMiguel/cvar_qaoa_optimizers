#!/usr/bin/env python
"""Entry point for the CVaR quantum optimization experiments.

Examples:
    python main.py maxcut-suite1 maxcut-suite2 maxcut-suite3   # paper figures
    python main.py paper --workers 4                           # same, in parallel
    python main.py maxcut-suite3 --quick                       # fast smoke test
"""
import argparse
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import suites.maxcut.suite1 as maxcut_suite1
import suites.maxcut.suite2 as maxcut_suite2
import suites.maxcut.suite3 as maxcut_suite3
import suites.maxmulticut.suite1 as maxmulticut_suite1
import suites.maxmulticut.suite2 as maxmulticut_suite2
import suites.maxmulticut.suite3 as maxmulticut_suite3

SUITES = {
    "maxcut-suite1": maxcut_suite1,
    "maxcut-suite2": maxcut_suite2,
    "maxcut-suite3": maxcut_suite3,
    "maxmulticut-suite1": maxmulticut_suite1,
    "maxmulticut-suite2": maxmulticut_suite2,
    "maxmulticut-suite3": maxmulticut_suite3,
}
PAPER = ["maxcut-suite1", "maxcut-suite2", "maxcut-suite3"]


def _quick(cfg) -> None:
    """Shrink a suite configuration for a fast smoke test."""
    for attr, value in (("seeds", [0, 1]), ("num_instances", 3), ("max_evals", 20),
                        ("maxiter", 5), ("shots", 256)):
        if hasattr(cfg, attr):
            setattr(cfg, attr, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="+", choices=list(SUITES) + ["paper"])
    parser.add_argument("--output", default="results", help="parent folder for the run")
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    parser.add_argument("--quick", action="store_true", help="tiny configuration for testing")
    parser.add_argument("--verbose", action="store_true", help="log every experiment")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    if not args.verbose:
        logging.getLogger("quantum_core").setLevel(logging.WARNING)
    logger = logging.getLogger("main")

    names = [n for s in args.suites for n in (PAPER if s == "paper" else [s])]
    run_dir = os.path.join(args.output, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Output -> %s", run_dir)

    for name in names:
        module = SUITES[name]
        if hasattr(module.config, "workers"):
            module.config.workers = args.workers
        if args.quick:
            _quick(module.config)
        logger.info("Running %s", name)
        module.run(run_dir)

    logger.info("Done. Output: %s", run_dir)


if __name__ == "__main__":
    main()
