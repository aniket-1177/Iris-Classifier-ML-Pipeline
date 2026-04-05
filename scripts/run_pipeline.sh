#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh
#
# Wrapper script for running the ML training pipeline.
# Designed to be invoked by cron or a CI/CD system.
#
# Usage:
#   bash scripts/run_pipeline.sh
#
# Cron example (runs every Monday at 02:00):
#   0 2 * * 1 /app/scripts/run_pipeline.sh >> /var/log/ml_pipeline.log 2>&1
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/train_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

echo "========================================================"
echo " ML Pipeline Training Run"
echo " Timestamp : $TIMESTAMP"
echo " Project   : $PROJECT_ROOT"
echo " Log       : $LOG_FILE"
echo "========================================================"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "[INFO] Virtual environment activated."
fi

# Run training and capture results
python scripts/train.py \
    --output-json "logs/results_$TIMESTAMP.json" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Training completed at $(date). Log: $LOG_FILE"
else
    echo "[ERROR] Training failed with exit code $EXIT_CODE at $(date)."
    exit $EXIT_CODE
fi
