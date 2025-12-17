#!/bin/bash

# This script downloads OHLCV data for the 'coin' market.
# It can be run from the root directory of the 'quant' project.

# If you are using a virtual environment, make sure to activate it before running this script.
# Example: source path/to/your/venv/bin/activate

# --- Configuration ---
ROOT_DIR=$(pwd)
# The first argument is the date (YYYYMMDD). If not provided, it defaults to yesterday.
DATE_ARG=${1:-$(date -d "yesterday" +%Y%m%d)}
MARKET="coin"
OUTPUT_DIR="${ROOT_DIR}/var/data"
PIPELINE_SCRIPT="${ROOT_DIR}/sbin/data_pipeline/01_get_daily_ohlcv_data.py"

echo "Starting data pipeline for date: ${DATE_ARG}"
echo "Project root: ${ROOT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "-------------------------------------"

# Check if the python script exists
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "Error: Python script not found at ${PIPELINE_SCRIPT}"
    exit 1
fi

# A loop to get data for all intervals
for interval in day minute1 minute3 minute5 minute10 minute15 minute30 minute60 minute240 week month; do
    echo "Fetching data for interval: ${interval}..."
    python3 "${PIPELINE_SCRIPT}" \
        --root_dir "${ROOT_DIR}" \
        --date "${DATE_ARG}" \
        --market "${MARKET}" \
        --interval "${interval}" \
        --output_dir "${OUTPUT_DIR}"
    echo "-------------------------------------"
done

echo "Data pipeline finished."

