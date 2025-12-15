#!/bin/bash

# This script runs the strategy unit backtest.
# It can be run from the root directory of the 'quant' project.

# If you are using a virtual environment, make sure to activate it before running this script.
# Example: source path/to/your/venv/bin/activate

# ./run_backtest.sh [market] [interval] 
# ex) ./run_backtest.sh coin minute240

# --- Configuration ---
ROOT_DIR=$(pwd)
MARKET=${1:-"coin"}      # Default: coin. Or pass as first argument.
INTERVAL=${2:-"minute1"} # Default: minute1. Or pass as second argument.
BACKTEST_SCRIPT="${ROOT_DIR}/sbin/strategy_unit_backtest/02_strategy_unit_backtest.py"


echo "Starting backtest for Market: ${MARKET}, Interval: ${INTERVAL}"
echo "Project root: ${ROOT_DIR}"
echo "-------------------------------------"

# Check if the python script exists
if [ ! -f "$BACKTEST_SCRIPT" ]; then
    echo "Error: Python script not found at ${BACKTEST_SCRIPT}"
    exit 1
fi

# Run the backtest script
python3 "${BACKTEST_SCRIPT}" \
    --root_dir "${ROOT_DIR}" \
    --market "${MARKET}" \
    --interval "${INTERVAL}"

echo "Backtest finished."

