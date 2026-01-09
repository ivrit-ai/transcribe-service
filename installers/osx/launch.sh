#!/bin/bash
set -e

# Transcribe Service Launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
VENV_DIR="$SCRIPT_DIR/venv"
APP_DIR="$SCRIPT_DIR/transcribe-service"
MODELS_DIR="$SCRIPT_DIR/models"
DATA_DIR="$HOME/Library/ivrit.ai/transcribe-service"
LOG_FILE="$DATA_DIR/app.log"
LAUNCH_LOG_FILE="$DATA_DIR/launch.log"
PID_FILE="$DATA_DIR/app.pid"

# Add bin directory to PATH for ffmpeg
export PATH="$BIN_DIR:$PATH"

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
nohup python app.py --local --data-dir "$DATA_DIR" --models-dir "$MODELS_DIR" > "$LOG_FILE" 2> "$LAUNCH_LOG_FILE" &
APP_PID=$!
echo $APP_PID > "$PID_FILE"

echo "Transcribe service started (PID: $APP_PID)"
echo "Log file: $LOG_FILE"
echo "Error log file: $LAUNCH_LOG_FILE"

# Wait for server to start (poll for up to 20 seconds)
echo "Waiting for server to start..."
MAX_WAIT=20
ELAPSED=0
SERVER_UP=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:4500 > /dev/null 2>&1; then
        SERVER_UP=true
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ "$SERVER_UP" = false ]; then
    echo "ERROR: Server failed to start after $MAX_WAIT seconds"
    echo "Check the error log for details: $LAUNCH_LOG_FILE"
    echo "Server process (PID: $APP_PID) may still be running"
    exit 1
fi

echo "Server is up after $ELAPSED seconds"

# Launch browser
echo "Opening browser..."
open "http://localhost:4500"

echo "Done! The service is running in the background."
echo "To stop the service, run: kill $APP_PID"

