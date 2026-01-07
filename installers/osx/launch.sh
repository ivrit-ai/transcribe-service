#!/bin/bash
set -e

# Transcribe Service Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
APP_DIR="$SCRIPT_DIR/transcribe-service"
MODELS_DIR="$SCRIPT_DIR/models"
DATA_DIR="$HOME/Library/ivrit.ai/transcribe-service"
LOG_FILE="$DATA_DIR/app.log"
PID_FILE="$DATA_DIR/app.pid"

# Ensure data directory exists
mkdir -p "$DATA_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Transcribe service is already running (PID: $OLD_PID)"
        echo "Opening browser..."
        sleep 1
        open "http://localhost:4500"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

# Activate virtual environment and launch app
echo "Starting transcribe service..."
source "$VENV_DIR/bin/activate"
cd "$APP_DIR"
nohup python app.py --local --data-dir "$DATA_DIR" --models-dir "$MODELS_DIR" > "$LOG_FILE" 2>&1 &
APP_PID=$!
echo $APP_PID > "$PID_FILE"

echo "Transcribe service started (PID: $APP_PID)"
echo "Log file: $LOG_FILE"

# Wait a moment for the server to start
echo "Waiting for server to start..."
sleep 3

# Launch browser
echo "Opening browser..."
open "http://localhost:4500"

echo "Done! The service is running in the background."
echo "To stop the service, run: kill $APP_PID"

