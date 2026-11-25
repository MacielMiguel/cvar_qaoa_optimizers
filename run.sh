#!/bin/bash
# Run main.py with whichever Python is active (conda env, venv, ...).
# Example: ./run.sh paper --workers 4
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python "$SCRIPT_DIR/main.py" "$@"
