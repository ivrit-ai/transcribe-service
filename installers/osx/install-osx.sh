#!/bin/bash
set -e

# Transcribe Service OSX Installer
# Usage: 
#   Install latest release from default repo:
#     curl -fsSL https://raw.githubusercontent.com/ivrit-ai/transcribe-service/main/installers/osx/install-osx.sh | bash
#
#   Install from any branch or tag (specify repo once):
#     REPO_PATH=ivrit-ai/transcribe-service/main; curl -fsSL https://raw.githubusercontent.com/$REPO_PATH/installers/osx/install-osx.sh | REPO_PATH=$REPO_PATH bash
#     REPO_PATH=ivrit-ai/transcribe-service/yairl/onprem; curl -fsSL https://raw.githubusercontent.com/$REPO_PATH/installers/osx/install-osx.sh | REPO_PATH=$REPO_PATH bash
#     REPO_PATH=ivrit-ai/transcribe-service/v1.0.0; curl -fsSL https://raw.githubusercontent.com/$REPO_PATH/installers/osx/install-osx.sh | REPO_PATH=$REPO_PATH bash

echo "==================================="
echo "Transcribe Service OSX Installer"
echo "==================================="
echo ""

# Parse REPO_PATH environment variable
# Format: org/repo/ref-name (where ref-name can be a branch or tag)
# If not specified, defaults to latest release from ivrit-ai/transcribe-service
if [ -n "$REPO_PATH" ]; then
    # Split REPO_PATH into parts
    IFS='/' read -ra PATH_PARTS <<< "$REPO_PATH"
    
    if [ ${#PATH_PARTS[@]} -lt 3 ]; then
        echo "Error: Invalid REPO_PATH format"
        echo "Expected: org/repo/ref-name (e.g., ivrit-ai/transcribe-service/main)"
        echo "Got: $REPO_PATH"
        exit 1
    fi
    
    GITHUB_REPO="${PATH_PARTS[0]}/${PATH_PARTS[1]}"
    
    # Join remaining parts (in case ref name contains slashes like feature/my-branch)
    REF_NAME=""
    for ((i=2; i<${#PATH_PARTS[@]}; i++)); do
        if [ -z "$REF_NAME" ]; then
            REF_NAME="${PATH_PARTS[$i]}"
        else
            REF_NAME="$REF_NAME/${PATH_PARTS[$i]}"
        fi
    done
    
    GITHUB_REF="$REF_NAME"
    
    # Check if this is a tag or branch by querying GitHub API
    echo "Checking if $GITHUB_REF is a tag or branch..."
    set +e
    TAG_EXISTS=$(curl -fsSL -o /dev/null -w "%{http_code}" https://api.github.com/repos/$GITHUB_REPO/git/refs/tags/$GITHUB_REF 2>/dev/null)
    set -e
    
    if [ "$TAG_EXISTS" = "200" ]; then
        IS_TAG=true
        echo "Detected as tag: $GITHUB_REF"
    else
        IS_TAG=false
        echo "Detected as branch: $GITHUB_REF"
    fi
else
    # Default to latest release from ivrit-ai/transcribe-service
    GITHUB_REPO="ivrit-ai/transcribe-service"
    GITHUB_REF="latest"
    IS_TAG=true
fi

echo "Using GitHub repository: $GITHUB_REPO"
echo "Using reference: $GITHUB_REF"
echo ""

# Check architecture first - only arm64 (Apple Silicon) is supported
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "Error: This installer only supports Apple Silicon (arm64) Macs."
    echo "Detected architecture: $ARCH"
    exit 1
fi

# Configuration
INSTALL_DIR="$(pwd)"
UV_DIR="$INSTALL_DIR/uv"
BIN_DIR="$INSTALL_DIR/bin"
APP_DIR="$INSTALL_DIR/transcribe-service"
VENV_DIR="$INSTALL_DIR/venv"
MODELS_DIR="$INSTALL_DIR/models"
DATA_DIR="$HOME/Library/ivrit.ai/transcribe-service"
MODEL_URL="https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin"
INSTALL_LOG="$INSTALL_DIR/install.log"

# Redirect all output to install.log while also showing on terminal
exec > >(tee -a "$INSTALL_LOG") 2>&1

# Check if installation already exists
if [ -d "$APP_DIR" ] || [ -d "$UV_DIR" ] || [ -d "$VENV_DIR" ] || [ -d "$BIN_DIR" ]; then
    echo ""
    echo "==================================="
    echo "Existing Installation Detected"
    echo "==================================="
    echo ""
    echo "An installation already exists in this directory:"
    echo "  $INSTALL_DIR"
    echo ""
    [ -d "$APP_DIR" ] && echo "  - transcribe-service/"
    [ -d "$UV_DIR" ] && echo "  - uv/"
    [ -d "$BIN_DIR" ] && echo "  - bin/"
    [ -d "$VENV_DIR" ] && echo "  - venv/"
    echo ""
    read -p "Do you want to uninstall and reinstall? (y/N): " -n 1 -r < /dev/tty
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation aborted."
        exit 0
    fi
    
    echo "Removing existing installation..."
    [ -d "$APP_DIR" ] && rm -rf "$APP_DIR"
    [ -d "$UV_DIR" ] && rm -rf "$UV_DIR"
    [ -d "$BIN_DIR" ] && rm -rf "$BIN_DIR"
    [ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"
    [ -f "$INSTALL_DIR/VERSION" ] && rm -f "$INSTALL_DIR/VERSION"
    echo "✓ Existing installation removed"
    echo ""
fi

# Set UV architecture (arm64 already validated at startup)
UV_ARCH="aarch64"

# Installation directory
echo "Installing to: $INSTALL_DIR"

# Step 1: Download and install uv
echo ""
echo "Step 1/8: Downloading uv..."
mkdir -p "$UV_DIR"
UV_DOWNLOAD_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_ARCH}-apple-darwin.tar.gz"
echo "Downloading from: $UV_DOWNLOAD_URL"
curl -fsSL "$UV_DOWNLOAD_URL" | tar -xzf - -C "$UV_DIR" --strip-components=1
chmod +x "$UV_DIR/uv"
echo "✓ uv installed successfully"

# Step 2: Download transcribe-service release
echo ""
echo "Step 2/8: Downloading transcribe-service..."
if [ "$GITHUB_REF" = "latest" ]; then
    # Get the latest release tag
    echo "Fetching latest release..."
    RELEASE_URL=$(curl -fsSL https://api.github.com/repos/$GITHUB_REPO/releases/latest | grep "tarball_url" | cut -d '"' -f 4)
    if [ -z "$RELEASE_URL" ]; then
        echo "Error: Could not fetch latest release"
        exit 1
    fi
    # Extract the actual tag name for the VERSION file
    GITHUB_REF=$(curl -fsSL https://api.github.com/repos/$GITHUB_REPO/releases/latest | grep '"tag_name"' | cut -d '"' -f 4)
    if [ -z "$GITHUB_REF" ]; then
        echo "Error: Could not determine release tag"
        exit 1
    fi
    echo "Found latest release: $GITHUB_REF"
elif [ "$IS_TAG" = true ]; then
    # It's a tag
    echo "Using tag: $GITHUB_REF"
    RELEASE_URL="https://api.github.com/repos/$GITHUB_REPO/tarball/refs/tags/$GITHUB_REF"
else
    # It's a branch
    echo "Using branch: $GITHUB_REF"
    RELEASE_URL="https://api.github.com/repos/$GITHUB_REPO/tarball/$GITHUB_REF"
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

# Fetch commit hash for the version
echo "Fetching commit hash for $GITHUB_REF..."
if [ "$IS_TAG" = true ]; then
    # For tags, try to get the commit hash from the tag reference
    COMMIT_HASH=$(curl -fsSL "https://api.github.com/repos/$GITHUB_REPO/git/refs/tags/$GITHUB_REF" 2>/dev/null | grep '"sha"' | head -1 | cut -d '"' -f 4)
    # If it's an annotated tag, we need to dereference it
    if [ -z "$COMMIT_HASH" ]; then
        COMMIT_HASH=$(curl -fsSL "https://api.github.com/repos/$GITHUB_REPO/commits/$GITHUB_REF" 2>/dev/null | grep '"sha"' | head -1 | cut -d '"' -f 4)
    fi
else
    # For branches, get the latest commit hash
    COMMIT_HASH=$(curl -fsSL "https://api.github.com/repos/$GITHUB_REPO/commits/$GITHUB_REF" 2>/dev/null | grep '"sha"' | head -1 | cut -d '"' -f 4)
fi

if [ -z "$COMMIT_HASH" ]; then
    echo "Warning: Could not fetch commit hash, using 'unknown'"
    COMMIT_HASH="unknown"
else
    # Truncate to first 8 characters for brevity
    COMMIT_HASH="${COMMIT_HASH:0:8}"
    echo "✓ Commit hash: $COMMIT_HASH"
fi

# Create VERSION file
VERSION_FILE="$INSTALL_DIR/VERSION"
if [ "$IS_TAG" = true ]; then
    REF_TYPE="tag"
else
    REF_TYPE="branch"
fi
echo "${GITHUB_REPO}/${GITHUB_REF}@${COMMIT_HASH} (${REF_TYPE})" > "$VERSION_FILE"
echo "✓ Version file created: $VERSION_FILE"
echo "   ${GITHUB_REPO}/${GITHUB_REF}@${COMMIT_HASH} (${REF_TYPE})"

# Step 3: Create virtual environment with Python 3.13
echo ""
echo "Step 3/8: Creating virtual environment with Python 3.13..."
"$UV_DIR/uv" venv "$VENV_DIR" --python 3.13
echo "✓ Virtual environment created successfully"

# Step 4: Install requirements
echo ""
echo "Step 4/8: Installing requirements..."
"$UV_DIR/uv" pip install --python "$VENV_DIR/bin/python" -r "$APP_DIR/requirements.txt"
echo "✓ Requirements installed successfully"

# Step 5: Install ivrit[all]
echo ""
echo "Step 5/8: Installing ivrit[all]..."
"$UV_DIR/uv" pip install --python "$VENV_DIR/bin/python" "ivrit[all]"
echo "✓ ivrit[all] installed successfully"

# Step 6: Create symlinks for ffmpeg
echo ""
echo "Step 6/8: Setting up ffmpeg symlinks..."
mkdir -p "$BIN_DIR"

# Get ffmpeg path from imageio-ffmpeg
FFMPEG_PATH=$("$VENV_DIR/bin/python" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
if [ -z "$FFMPEG_PATH" ] || [ ! -f "$FFMPEG_PATH" ]; then
    echo "Error: Could not find ffmpeg binary from imageio-ffmpeg"
    exit 1
fi

echo "Found ffmpeg at: $FFMPEG_PATH"

# Create symlink
ln -sf "$FFMPEG_PATH" "$BIN_DIR/ffmpeg"
echo "✓ Created symlink: $BIN_DIR/ffmpeg -> $FFMPEG_PATH"

# Step 7: Create data directory and download model
echo ""
echo "Step 7/8: Setting up data directory and downloading model..."
mkdir -p "$DATA_DIR"
mkdir -p "$MODELS_DIR"
echo "Data directory: $DATA_DIR"
MODEL_FILE="$MODELS_DIR/ivrit-ggml-model.bin"
if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists at $MODEL_FILE"
    read -p "Do you want to re-download it? (y/N): " -n 1 -r < /dev/tty
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

# Step 8: Create ivrit.ai.app in Applications folder
echo ""
echo "Step 8/8: Creating ivrit.ai application..."

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
    <string>11.0</string>
    <key>LSArchitecturePriority</key>
    <array>
        <string>arm64</string>
    </array>
    <key>LSRequiresNativeExecution</key>
    <true/>
</dict>
</plist>
PLIST_EOF

# Create the launcher script inside the app bundle by copying and patching launch.sh
cp "$APP_DIR/installers/osx/launch.sh" "$APP_MACOS/ivrit.ai"

# Patch the launcher script to use absolute paths
# Replace SCRIPT_DIR line with absolute INSTALL_DIR
sed -i '' "s|SCRIPT_DIR=\"\$(cd \"\$(dirname \"\${BASH_SOURCE\[0\]}\")\" \&\& pwd)\"|SCRIPT_DIR=\"$INSTALL_DIR\"|g" "$APP_MACOS/ivrit.ai"

# Replace DATA_DIR to use absolute path
sed -i '' "s|\$HOME/Library/ivrit.ai/transcribe-service|$DATA_DIR|g" "$APP_MACOS/ivrit.ai"

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
echo "Models directory: $MODELS_DIR"
echo "Data directory: $DATA_DIR"
echo "Application: $APP_BUNDLE"
echo "Installation log: $INSTALL_LOG"
echo ""
echo "To start the transcribe service:"
echo "  - Open ivrit.ai from your Applications folder, or"
echo "  - Run: open \"$APP_BUNDLE\""
echo ""

