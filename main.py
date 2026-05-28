#!/usr/bin/env python
"""Main entry point for CVaR quantum optimization experiments."""

import os
import logging
from datetime import datetime

# Import suites with descriptive names for easy switching
import suites.maxcut.suite1 as maxcut_suite1
import suites.maxcut.suite2 as maxcut_suite2
import suites.maxcut.suite3 as maxcut_suite3

import suites.maxmulticut.suite1 as maxmulticut_suite1
import suites.maxmulticut.suite2 as maxmulticut_suite2
import suites.maxmulticut.suite3 as maxmulticut_suite3

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

#logging.getLogger('scipy').setLevel(logging.ERROR)
#logging.getLogger('scipy.sparse').setLevel(logging.ERROR)
#logging.getLogger('scipy.optimize').setLevel(logging.ERROR)
#logging.getLogger('qiskit').setLevel(logging.WARNING)

OUTPUT_DIR = "results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_run_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_run_dir, exist_ok=True)

    logger.info(f"Output → {output_run_dir}")

    # ============================================================
    # MaxCut Suites
    # ============================================================
    #maxcut_suite1.run(output_run_dir)
    #maxcut_suite2.run(output_run_dir)
    maxcut_suite3.run(output_run_dir)

    # ============================================================
    # MaxMulticut Suites
    # ============================================================
    #maxmulticut_suite1.run(output_run_dir)
    #maxmulticut_suite2.run(output_run_dir)
    #maxmulticut_suite3.run(output_run_dir)

    logger.info(f"\n✓ Done! Output: {output_run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
