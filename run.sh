#!/bin/bash
# Simple wrapper to run main.py with the quantum environment

# Get the directory this script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run with quantum env Python explicitly
/Users/hugoleroy/radioconda/envs/quantum/bin/python "$SCRIPT_DIR/main.py" "$@"
