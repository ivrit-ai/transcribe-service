#!/bin/bash
set -e

# Transcribe Service OSX Installer
# Usage: 
#   Install latest release:
#     curl -fsSL https://raw.githubusercontent.com/ivrit-ai/transcribe-service/main/installers/osx/install-osx.sh | bash
#   Install specific tag:
#     curl -fsSL https://raw.githubusercontent.com/ivrit-ai/transcribe-service/main/installers/osx/install-osx.sh | TRANSCRIBE_VERSION=v1.0.0 bash
#   Install from branch:
#     curl -fsSL https://raw.githubusercontent.com/ivrit-ai/transcribe-service/main/installers/osx/install-osx.sh | TRANSCRIBE_VERSION=develop bash

echo "==================================="
echo "Transcribe Service OSX Installer"
echo "==================================="
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            TRANSCRIBE_VERSION="$2"
            shift 2
            ;;
        --branch)
            TRANSCRIBE_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--version TAG] [--branch BRANCH]"
            exit 1
            ;;
    esac
done

# Configuration
INSTALL_DIR="$(pwd)"
UV_DIR="$INSTALL_DIR/uv"
APP_DIR="$INSTALL_DIR/transcribe-service"
MODELS_DIR="$INSTALL_DIR/models"
VENV_DIR="$INSTALL_DIR/venv"
TRANSCRIBE_VERSION="${TRANSCRIBE_VERSION:-latest}"
MODEL_URL="https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    UV_ARCH="aarch64"
elif [ "$ARCH" = "x86_64" ]; then
    UV_ARCH="x86_64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

# Installation directory
echo "Installing to: $INSTALL_DIR"

# Step 1: Download and install uv
echo ""
echo "Step 1/7: Downloading uv..."
mkdir -p "$UV_DIR"
UV_DOWNLOAD_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}-apple-darwin.tar.gz"
echo "Downloading from: $UV_DOWNLOAD_URL"
curl -fsSL "$UV_DOWNLOAD_URL" | tar -xzf - -C "$UV_DIR" --strip-components=1
chmod +x "$UV_DIR/uv"
echo "✓ uv installed successfully"

# Step 2: Download transcribe-service release
echo ""
echo "Step 2/7: Downloading transcribe-service..."
if [ "$TRANSCRIBE_VERSION" = "latest" ]; then
    # Get the latest release tag
    echo "Fetching latest release..."
    RELEASE_URL=$(curl -fsSL https://api.github.com/repos/ivrit-ai/transcribe-service/releases/latest | grep "tarball_url" | cut -d '"' -f 4)
    if [ -z "$RELEASE_URL" ]; then
        echo "Error: Could not fetch latest release"
        exit 1
    fi
    # Extract the ref for the raw content URL
    GITHUB_REF=$(curl -fsSL https://api.github.com/repos/ivrit-ai/transcribe-service/releases/latest | grep '"tag_name"' | cut -d '"' -f 4)
    if [ -z "$GITHUB_REF" ]; then
        GITHUB_REF="main"
    fi
else
    # Check if this is a release tag or a branch
    echo "Checking if $TRANSCRIBE_VERSION is a release tag..."
    # Temporarily disable exit on error for validation checks
    set +e
    TAG_EXISTS=$(curl -fsSL -o /dev/null -w "%{http_code}" https://api.github.com/repos/ivrit-ai/transcribe-service/releases/tags/$TRANSCRIBE_VERSION 2>/dev/null)
    set -e
    
    if [ "$TAG_EXISTS" = "200" ]; then
        # It's a release tag
        echo "Using release tag: $TRANSCRIBE_VERSION"
        RELEASE_URL="https://api.github.com/repos/ivrit-ai/transcribe-service/tarball/refs/tags/$TRANSCRIBE_VERSION"
        GITHUB_REF="$TRANSCRIBE_VERSION"
    else
        # Check if it's a valid branch
        echo "Checking if $TRANSCRIBE_VERSION is a branch..."
        set +e
        BRANCH_EXISTS=$(curl -fsSL -o /dev/null -w "%{http_code}" https://api.github.com/repos/ivrit-ai/transcribe-service/branches/$TRANSCRIBE_VERSION 2>/dev/null)
        set -e
        
        if [ "$BRANCH_EXISTS" = "200" ]; then
            echo "Using branch: $TRANSCRIBE_VERSION"
            RELEASE_URL="https://api.github.com/repos/ivrit-ai/transcribe-service/tarball/$TRANSCRIBE_VERSION"
            GITHUB_REF="$TRANSCRIBE_VERSION"
        else
            echo "Error: '$TRANSCRIBE_VERSION' is not a valid release tag or branch"
            echo "Please check the version/branch name and try again"
            exit 1
        fi
    fi
fi

echo "Downloading from GitHub (ref: $GITHUB_REF)..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR"
if ! curl -fsSL "$RELEASE_URL" | tar -xzf - -C "$APP_DIR" --strip-components=1; then
    echo "Error: Failed to download or extract the release"
    echo "URL attempted: $RELEASE_URL"
    exit 1
fi
echo "✓ transcribe-service downloaded successfully"

# Step 3: Create virtual environment with Python 3.13
echo ""
echo "Step 3/7: Creating virtual environment with Python 3.13..."
"$UV_DIR/uv" venv "$VENV_DIR" --python 3.13
echo "✓ Virtual environment created successfully"

# Step 4: Install requirements
echo ""
echo "Step 4/7: Installing requirements..."
"$UV_DIR/uv" pip install --python "$VENV_DIR/bin/python" -r "$APP_DIR/requirements.txt"
echo "✓ Requirements installed successfully"

# Step 5: Install ivrit[all]
echo ""
echo "Step 5/7: Installing ivrit[all]..."
"$UV_DIR/uv" pip install --python "$VENV_DIR/bin/python" "ivrit[all]"
echo "✓ ivrit[all] installed successfully"

# Step 6: Download model
echo ""
echo "Step 6/7: Downloading ivrit-ai model..."
mkdir -p "$MODELS_DIR"
MODEL_FILE="$MODELS_DIR/ggml-model.bin"
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE"
    read -p "Do you want to re-download it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping model download"
    else
        echo "Downloading model (this may take a while)..."
        curl -fL --progress-bar "$MODEL_URL" -o "$MODEL_FILE"
        echo "✓ Model downloaded successfully"
    fi
else
    echo "Downloading model (this may take a while)..."
    curl -fL --progress-bar "$MODEL_URL" -o "$MODEL_FILE"
    echo "✓ Model downloaded successfully"
fi

# Step 7: Create ivrit.ai.app in Applications folder
echo ""
echo "Step 7/7: Creating ivrit.ai application..."

APP_BUNDLE="$HOME/Applications/ivrit.ai.app"
APP_CONTENTS="$APP_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"

# Create the .app bundle structure
mkdir -p "$APP_MACOS"

# Create Info.plist
cat > "$APP_CONTENTS/Info.plist" << 'PLIST_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>ivrit.ai</string>
    <key>CFBundleIdentifier</key>
    <string>ai.ivrit.transcribe</string>
    <key>CFBundleName</key>
    <string>ivrit.ai</string>
    <key>CFBundleDisplayName</key>
    <string>ivrit.ai Transcribe</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>
PLIST_EOF

# Create the launcher script inside the app bundle
cat > "$APP_MACOS/ivrit.ai" << LAUNCHER_EOF
#!/bin/bash
set -e

# Installation directory (embedded at install time)
INSTALL_DIR="$INSTALL_DIR"
VENV_DIR="\$INSTALL_DIR/venv"
APP_DIR="\$INSTALL_DIR/transcribe-service"
LOG_FILE="\$INSTALL_DIR/app.log"
PID_FILE="\$INSTALL_DIR/app.pid"

# Check if already running
if [ -f "\$PID_FILE" ]; then
    OLD_PID=\$(cat "\$PID_FILE")
    if ps -p "\$OLD_PID" > /dev/null 2>&1; then
        echo "Transcribe service is already running (PID: \$OLD_PID)"
        echo "Opening browser..."
        sleep 1
        open "http://localhost:4500"
        exit 0
    else
        rm -f "\$PID_FILE"
    fi
fi

# Activate virtual environment and launch app
echo "Starting transcribe service..."
source "\$VENV_DIR/bin/activate"
cd "\$APP_DIR"
nohup python app.py --local > "\$LOG_FILE" 2>&1 &
APP_PID=\$!
echo \$APP_PID > "\$PID_FILE"

echo "Transcribe service started (PID: \$APP_PID)"
echo "Log file: \$LOG_FILE"

# Wait a moment for the server to start
echo "Waiting for server to start..."
sleep 3

# Launch browser
echo "Opening browser..."
open "http://localhost:4500"

echo "Done! The service is running in the background."
echo "To stop the service, run: kill \$APP_PID"
LAUNCHER_EOF

chmod +x "$APP_MACOS/ivrit.ai"

# Set the application icon using the favicon.png from the downloaded source
ICON_PATH="$APP_DIR/static/favicon.png"
if [ -f "$ICON_PATH" ]; then
    echo "Setting application icon..."
    osascript -e "use framework \"AppKit\"" \
      -e "set img to (current application's NSImage's alloc()'s initWithContentsOfFile:\"$ICON_PATH\")" \
      -e "(current application's NSWorkspace's sharedWorkspace())'s setIcon:img forFile:\"$APP_BUNDLE\" options:0"
    echo "✓ Application icon set successfully"
else
    echo "Warning: Icon file not found at $ICON_PATH, skipping icon setup"
fi

echo "✓ ivrit.ai application created successfully"

# Installation complete
echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo ""
echo "Installation directory: $INSTALL_DIR"
echo "Application: $APP_BUNDLE"
echo ""
echo "To start the transcribe service:"
echo "  - Open ivrit.ai from your Applications folder, or"
echo "  - Run: open \"$APP_BUNDLE\""
echo ""

