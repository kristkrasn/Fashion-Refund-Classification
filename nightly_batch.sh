#!/bin/bash

ROOT="/Users/kriskrasnopolska/Desktop/Fashion_Refund_Classification"
LOG="$ROOT/logs/nightly_batch.log"
API_URL="http://127.0.0.1:5055/health"

cd "$ROOT" || exit 1

echo "================================" >> "$LOG"
echo "Nightly batch started: $(date)" >> "$LOG"

source "$ROOT/.venv/bin/activate"

# Check whether the API is already running
if ! curl -s --fail "$API_URL" > /dev/null; then
    echo "Starting REST API..." >> "$LOG"

    nohup python "$ROOT/src/app.py" >> "$ROOT/logs/api_cron.log" 2>&1 &

    # Wait up to 60 seconds for the API
    for i in {1..30}; do
        if curl -s --fail "$API_URL" > /dev/null; then
            break
        fi
        sleep 2
    done
fi

# Verify that the API is ready
if ! curl -s --fail "$API_URL" > /dev/null; then
    echo "ERROR: REST API did not start." >> "$LOG"
    exit 1
fi

echo "REST API is ready." >> "$LOG"

# Run batch predictions
python "$ROOT/src/batch_predict.py" >> "$LOG" 2>&1

if [ $? -eq 0 ]; then
    echo "Nightly batch completed successfully: $(date)" >> "$LOG"
else
    echo "ERROR: Batch processing failed: $(date)" >> "$LOG"
    exit 1
fi
