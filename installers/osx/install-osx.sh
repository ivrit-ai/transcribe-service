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
LAUNCH_SCRIPT="$INSTALL_DIR/launch.sh"
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

# Step 7: Download launch script
echo ""
echo "Step 7/7: Downloading launch script..."
LAUNCH_SCRIPT_URL="https://raw.githubusercontent.com/ivrit-ai/transcribe-service/$GITHUB_REF/installers/osx/launch.sh"
echo "Downloading from: $LAUNCH_SCRIPT_URL"

# Try to download the launch script, fall back to main branch if not found
if ! curl -fsSL "$LAUNCH_SCRIPT_URL" -o "$LAUNCH_SCRIPT" 2>/dev/null; then
    echo "Launch script not found in ref: $GITHUB_REF"
    echo "Falling back to main branch..."
    LAUNCH_SCRIPT_URL="https://raw.githubusercontent.com/ivrit-ai/transcribe-service/main/installers/osx/launch.sh"
    echo "Downloading from: $LAUNCH_SCRIPT_URL"
    curl -fsSL "$LAUNCH_SCRIPT_URL" -o "$LAUNCH_SCRIPT"
fi

chmod +x "$LAUNCH_SCRIPT"
echo "✓ Launch script downloaded successfully"

# Installation complete
echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo ""
echo "Installation directory: $INSTALL_DIR"
echo "Launch script: $LAUNCH_SCRIPT"
echo ""
echo "To start the transcribe service, run:"
echo "  $LAUNCH_SCRIPT"
echo ""
echo "You can also add an alias to your shell profile:"
echo "  alias transcribe='$LAUNCH_SCRIPT'"
echo ""

